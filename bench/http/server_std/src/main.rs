//! The todo API (bench/SPECS.md "api", GET/POST only) on std::net, single-threaded,
//! one request per connection. Same architecture as the ail server. Port 8102.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

const MAX_BODY: usize = 65536;
const MAX_HEADER_BYTES: usize = 16 * 1024;

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

#[derive(Default)]
struct Store {
    todos: Vec<Todo>,
    next_id: i64,
}

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

struct Request {
    method: String,
    path: String,
    query: HashMap<String, String>,
    body: Vec<u8>,
}

struct Response {
    status: u16,
    body: String,
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_request(stream: &TcpStream) -> io::Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let mut total = reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(bad("malformed request line"));
    };
    let (path, qs) = target.split_once('?').unwrap_or((target, ""));
    let query = qs
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (k.to_string(), v.to_string())
        })
        .collect();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        let n = reader.read_line(&mut header)?;
        total += n;
        if total > MAX_HEADER_BYTES {
            return Err(bad("headers too large"));
        }
        let header = header.trim_end();
        if n == 0 || header.is_empty() {
            break;
        }
        if let Some((k, v)) = header.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().map_err(|_| bad("bad content-length"))?;
            }
        }
    }
    if content_length > MAX_BODY {
        return Err(bad("body too large"));
    }
    let mut body = vec![0; content_length];
    reader.read_exact(&mut body)?;
    Ok(Request { method: method.to_string(), path: path.to_string(), query, body })
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

fn query_int(req: &Request, key: &str, default: i64) -> i64 {
    req.query.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn handle(store: &mut Store, req: &Request) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/todos") => {
            let limit = query_int(req, "limit", 20).max(0) as usize;
            let offset = query_int(req, "offset", 0).max(0) as usize;
            let page: Vec<&Todo> = store.todos.iter().skip(offset).take(limit).collect();
            Response { status: 200, body: serde_json::to_string(&page).expect("serializable") }
        }
        ("GET", "/search") => {
            let get = |k: &str| req.query.get(k).map(String::as_str).unwrap_or("");
            Response { status: 200, body: search(store, get("q"), get("tag")) }
        }
        ("POST", "/todos") => match serde_json::from_slice::<NewTodo>(&req.body) {
            Ok(new) => {
                store.next_id += 1;
                let todo = Todo { done: false, id: store.next_id, tags: new.tags, title: new.title };
                let body = serde_json::to_string(&todo).expect("serializable");
                store.todos.push(todo);
                Response { status: 201, body }
            }
            Err(_) => Response { status: 400, body: error_json("invalid body") },
        },
        _ => Response { status: 404, body: error_json("not found") },
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Status",
    }
}

fn write_response(mut stream: &TcpStream, resp: &Response) -> io::Result<()> {
    let mut out = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        resp.status,
        reason(resp.status),
        resp.body.len()
    );
    out.push_str(&resp.body);
    stream.write_all(out.as_bytes())?;
    stream.flush()
}

fn serve(store: &mut Store, stream: TcpStream) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let resp = match read_request(&stream) {
        Ok(req) => handle(store, &req),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => {
            Response { status: 400, body: error_json("bad request") }
        }
        Err(e) => return Err(e),
    };
    write_response(&stream, &resp)
}

fn main() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8102")?;
    println!("rust std listening on 127.0.0.1:8102");
    let mut store = seeded();
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = serve(&mut store, stream) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
