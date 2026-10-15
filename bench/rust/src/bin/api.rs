//! In-process REST API: routing + JSON parse/serialize over a todo store.

use serde::{Deserialize, Serialize};
use serde_json::Value;

const REQUESTS: usize = 200_000;
const SEED: u64 = 777;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state * 48271 % 2_147_483_647;
        self.state
    }

    fn rand_below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Field order here is the serialized key order.
#[derive(Serialize)]
struct Todo {
    done: bool,
    id: u64,
    tags: Vec<String>,
    title: String,
}

#[derive(Deserialize)]
struct NewTodo {
    title: String,
    tags: Vec<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: &'static str,
}

struct Store {
    /// Kept in insertion order. Ids are assigned monotonically, so this is also sorted by id.
    todos: Vec<Todo>,
    next_id: u64,
}

impl Store {
    fn new() -> Self {
        Self { todos: Vec::new(), next_id: 1 }
    }

    fn position(&self, id: u64) -> Option<usize> {
        self.todos.binary_search_by_key(&id, |t| t.id).ok()
    }
}

struct Response {
    status: u16,
    body: String,
}

impl Response {
    fn json<T: Serialize + ?Sized>(status: u16, value: &T) -> Self {
        let body = serde_json::to_string(value).expect("serializing a response cannot fail");
        Self { status, body }
    }

    fn error(status: u16, error: &'static str) -> Self {
        Self::json(status, &ErrorBody { error })
    }

    fn not_found() -> Self {
        Self::error(404, "not found")
    }

    fn no_content() -> Self {
        Self { status: 204, body: String::new() }
    }
}

struct Query<'a> {
    raw: &'a str,
}

impl<'a> Query<'a> {
    fn get(&self, key: &str) -> Option<&'a str> {
        self.raw
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
    }

    fn get_usize(&self, key: &str, default: usize) -> usize {
        self.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
}

fn handle(store: &mut Store, method: &str, path: &str, body: &str) -> Response {
    let (path, query) = path.split_once('?').unwrap_or((path, ""));
    let query = Query { raw: query };
    let mut segments = path.trim_start_matches('/').split('/');

    match (segments.next(), segments.next(), segments.next()) {
        (Some("todos"), None, None) => match method {
            "POST" => create_todo(store, body),
            "GET" => list_todos(store, &query),
            _ => Response::not_found(),
        },
        (Some("todos"), Some(id), None) => {
            let Ok(id) = id.parse::<u64>() else {
                return Response::not_found();
            };
            match method {
                "PATCH" => patch_todo(store, id, body),
                "DELETE" => delete_todo(store, id),
                _ => Response::not_found(),
            }
        }
        _ => Response::not_found(),
    }
}

fn create_todo(store: &mut Store, body: &str) -> Response {
    let Ok(NewTodo { title, tags }) = serde_json::from_str::<NewTodo>(body) else {
        return Response::error(400, "invalid body");
    };
    let todo = Todo { done: false, id: store.next_id, tags, title };
    store.next_id += 1;
    let response = Response::json(201, &todo);
    store.todos.push(todo);
    response
}

fn list_todos(store: &Store, query: &Query) -> Response {
    let limit = query.get_usize("limit", 20);
    let offset = query.get_usize("offset", 0);
    let start = offset.min(store.todos.len());
    let end = start.saturating_add(limit).min(store.todos.len());
    Response::json(200, &store.todos[start..end])
}

fn patch_todo(store: &mut Store, id: u64, body: &str) -> Response {
    let Some(pos) = store.position(id) else {
        return Response::not_found();
    };
    let done = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("done").and_then(Value::as_bool));
    let todo = &mut store.todos[pos];
    if let Some(done) = done {
        todo.done = done;
    }
    Response::json(200, todo)
}

fn delete_todo(store: &mut Store, id: u64) -> Response {
    match store.position(id) {
        Some(pos) => {
            store.todos.remove(pos);
            Response::no_content()
        }
        None => Response::not_found(),
    }
}

fn main() {
    let mut rng = Rng::new(SEED);
    let mut store = Store::new();
    let mut counts = [0u64; 4]; // 200, 201, 204, 404
    let mut bytes = 0usize;

    for k in 0..REQUESTS {
        let next_id_hint = store.next_id;
        let op = rng.rand_below(100);
        let response = if op < 45 {
            let a = rng.rand_below(10);
            let b = rng.rand_below(10);
            let body = format!(r#"{{"title":"task {k}","tags":["t{a}","t{b}"]}}"#);
            handle(&mut store, "POST", "/todos", &body)
        } else if op < 75 {
            let l = 1 + rng.rand_below(50);
            let o = rng.rand_below(next_id_hint);
            let path = format!("/todos?limit={l}&offset={o}");
            handle(&mut store, "GET", &path, "")
        } else if op < 90 {
            let id = 1 + rng.rand_below(next_id_hint);
            let path = format!("/todos/{id}");
            handle(&mut store, "PATCH", &path, r#"{"done":true}"#)
        } else {
            let id = 1 + rng.rand_below(next_id_hint);
            let path = format!("/todos/{id}");
            handle(&mut store, "DELETE", &path, "")
        };

        match response.status {
            200 => counts[0] += 1,
            201 => counts[1] += 1,
            204 => counts[2] += 1,
            404 => counts[3] += 1,
            other => panic!("unexpected status {other}"),
        }
        bytes += response.body.len();
    }

    println!("status 200 {}", counts[0]);
    println!("status 201 {}", counts[1]);
    println!("status 204 {}", counts[2]);
    println!("status 404 {}", counts[3]);
    println!("bytes {bytes}");
}
