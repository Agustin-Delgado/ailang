//! HTTP load generator for bench/http.
//!
//! usage: loadgen <port> <concurrency> [duration_s=10] [warmup_s=2] [mix=todos|search]
//!
//! C client threads, each in a closed loop: open a fresh TCP connection, send one request with
//! `Connection: close`, read the response to EOF, verify the status, close. 30% POST /todos with a
//! small body, 70% GET /todos?limit=20&offset=0. Mix "search" is CPU-bound instead: 10% POST, 90%
//! GET /search?q=<n mod 1000>&tag=t<n mod 10>, each of which scans the whole store. Only requests that *start* after the warmup and
//! *finish* before the deadline are counted. The client closes with SO_LINGER=0 (RST) so the
//! ephemeral port range is not exhausted by TIME_WAIT sockets over many back-to-back runs.
//! Prints one JSON line: {"rps":..,"p50_us":..,"p99_us":..,"ok":..,"errors":..}.
use socket2::{Domain, Protocol, Socket, Type};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn connect(addr: &SocketAddr) -> std::io::Result<TcpStream> {
    let s = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
    s.set_linger(Some(Duration::ZERO))?;
    s.set_nodelay(true)?;
    s.connect(&(*addr).into())?;
    let s: TcpStream = s.into();
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    Ok(s)
}

/// Sends one request on a fresh connection; returns Ok(()) if the status and body shape are right.
fn one_request(addr: &SocketAddr, port: u16, post: bool, search: bool, n: u64, buf: &mut Vec<u8>) -> Result<(), String> {
    let req = if post {
        let body = format!(r#"{{"title":"task {n}","tags":["t{}","t{}"]}}"#, n % 10, (n / 10) % 10);
        format!(
            "POST /todos HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    } else if search {
        format!("GET /search?q={}&tag=t{} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n", n % 1000, n % 10)
    } else {
        format!("GET /todos?limit=20&offset=0 HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
    };
    let mut s = connect(addr).map_err(|e| format!("connect: {e}"))?;
    s.write_all(req.as_bytes()).map_err(|e| format!("write: {e}"))?;
    buf.clear();
    s.read_to_end(buf).map_err(|e| format!("read: {e}"))?;
    let text = std::str::from_utf8(buf).map_err(|_| "non-utf8 response".to_string())?;
    let want = if post { "HTTP/1.1 201" } else { "HTTP/1.1 200" };
    if !text.starts_with(want) {
        return Err(format!("unexpected status line: {:?}", text.lines().next().unwrap_or("")));
    }
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    let ok_shape = if post {
        body.starts_with("{\"done\":false,\"id\":")
    } else if search {
        body.starts_with("{\"count\":") && body.ends_with('}')
    } else {
        body.starts_with('[') && body.ends_with(']')
    };
    if !ok_shape {
        return Err(format!("unexpected body: {:?}", &body[..body.len().min(80)]));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: loadgen <port> <concurrency> [duration_s=10] [warmup_s=2]");
        std::process::exit(2);
    }
    let port: u16 = args[1].parse().expect("port");
    let conc: usize = args[2].parse().expect("concurrency");
    let duration = Duration::from_secs_f64(args.get(3).map_or(10.0, |s| s.parse().expect("duration")));
    let warmup = Duration::from_secs_f64(args.get(4).map_or(2.0, |s| s.parse().expect("warmup")));
    let search = match args.get(5).map(String::as_str) {
        None | Some("todos") => false,
        Some("search") => true,
        Some(m) => panic!("unknown mix {m}"),
    };
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    let start = Instant::now();
    let measure_from = start + warmup;
    let deadline = measure_from + duration;
    let errors = Arc::new(AtomicU64::new(0));
    let first_error = Arc::new(std::sync::Mutex::new(None::<String>));

    let handles: Vec<_> = (0..conc)
        .map(|t| {
            let errors = Arc::clone(&errors);
            let first_error = Arc::clone(&first_error);
            std::thread::spawn(move || {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((t as u64 + 1) * 0x1000_0001));
                let mut lat_us: Vec<u32> = Vec::with_capacity(1 << 16);
                let mut buf = Vec::with_capacity(4096);
                loop {
                    let t0 = Instant::now();
                    if t0 >= deadline {
                        break;
                    }
                    let r = rng.next();
                    let post = r % 100 < if search { 10 } else { 30 };
                    let res = one_request(&addr, port, post, search, r >> 8, &mut buf);
                    let t1 = Instant::now();
                    let counted = t0 >= measure_from && t1 <= deadline;
                    match res {
                        Ok(()) => {
                            if counted {
                                lat_us.push((t1 - t0).as_micros().min(u32::MAX as u128) as u32);
                            }
                        }
                        Err(e) => {
                            if counted {
                                errors.fetch_add(1, Ordering::Relaxed);
                            }
                            first_error.lock().unwrap().get_or_insert(e);
                        }
                    }
                }
                lat_us
            })
        })
        .collect();

    let mut all: Vec<u32> = handles.into_iter().flat_map(|h| h.join().expect("client thread")).collect();
    all.sort_unstable();
    let pct = |p: f64| -> u32 {
        if all.is_empty() {
            0
        } else {
            all[((all.len() as f64 * p).ceil() as usize).clamp(1, all.len()) - 1]
        }
    };
    let ok = all.len();
    let errs = errors.load(Ordering::Relaxed);
    let rps = ok as f64 / duration.as_secs_f64();
    if let Some(e) = first_error.lock().unwrap().as_ref() {
        eprintln!("first error: {e}");
    }
    println!(
        "{{\"concurrency\":{conc},\"rps\":{rps:.1},\"p50_us\":{},\"p99_us\":{},\"ok\":{ok},\"errors\":{errs}}}",
        pct(0.50),
        pct(0.99)
    );
}
