//! The todo API (bench/SPECS.md "api", GET/POST only) on axum + tokio, multi-threaded; readers share an RwLock. Port 8103.
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Serialize)]
struct Todo {
    done: bool,
    id: i64,
    tags: Vec<String>,
    title: String,
}

#[derive(Deserialize)]
struct NewTodo {
    title: String,
    tags: Vec<String>,
}

struct Store {
    todos: Vec<Todo>,
    next_id: i64,
}

type Shared = Arc<RwLock<Store>>;
const SEED: i64 = 100_000;

/// The store starts with SEED todos: "task i", tags t(i mod 10) and t(i/10 mod 10).
fn seeded() -> Store {
    let todos = (0..SEED)
        .map(|i| Todo {
            done: false,
            id: i + 1,
            tags: vec![format!("t{}", i % 10), format!("t{}", (i / 10) % 10)],
            title: format!("task {i}"),
        })
        .collect();
    Store { todos, next_id: SEED }
}

/// GET /search: how many todos have the tag and contain q in the title (scans the whole store).
fn search(store: &Store, q: &str, tag: &str) -> String {
    let n = store.todos.iter().filter(|t| t.tags.iter().any(|g| g == tag) && t.title.contains(q)).count();
    format!("{{\"count\":{n}}}")
}


fn json(status: StatusCode, body: String) -> impl IntoResponse {
    (status, [(header::CONTENT_TYPE, "application/json")], body)
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

fn query_int(q: &HashMap<String, String>, key: &str, default: i64) -> i64 {
    q.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

async fn list_todos(
    State(store): State<Shared>,
    Query(q): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let limit = query_int(&q, "limit", 20).max(0) as usize;
    let offset = query_int(&q, "offset", 0).max(0) as usize;
    let body = {
        let store = store.read().expect("store lock");
        let page: Vec<&Todo> = store.todos.iter().skip(offset).take(limit).collect();
        serde_json::to_string(&page).expect("serializable")
    };
    json(StatusCode::OK, body)
}

async fn create_todo(State(store): State<Shared>, body: Bytes) -> impl IntoResponse {
    let Ok(new) = serde_json::from_slice::<NewTodo>(&body) else {
        return json(StatusCode::BAD_REQUEST, error_json("invalid body"));
    };
    let body = {
        let mut store = store.write().expect("store lock");
        store.next_id += 1;
        let todo = Todo { done: false, id: store.next_id, tags: new.tags, title: new.title };
        let body = serde_json::to_string(&todo).expect("serializable");
        store.todos.push(todo);
        body
    };
    json(StatusCode::CREATED, body)
}

async fn search_todos(
    State(store): State<Shared>,
    Query(q): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let get = |k: &str| q.get(k).map(String::as_str).unwrap_or("").to_string();
    let body = search(&store.read().expect("store lock"), &get("q"), &get("tag"));
    json(StatusCode::OK, body)
}

async fn not_found() -> impl IntoResponse {
    json(StatusCode::NOT_FOUND, error_json("not found"))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/todos", get(list_todos).post(create_todo))
        .route("/search", get(search_todos))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(65536))
        .with_state(Arc::new(RwLock::new(seeded())));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8103")
        .await
        .expect("bind 127.0.0.1:8103");
    println!("rust axum listening on 127.0.0.1:8103");
    axum::serve(listener, app).await.expect("server");
}
