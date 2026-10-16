//! Fan-out client (bench/SPECS.md "fanout"): N GET requests to a backend, 64 in flight at a time,
//! summing the `count` of every answer. Idiomatic async Rust: tokio + reqwest (with its connection
//! pool) + buffer_unordered.
//!
//! usage: fanout <port> <requests> <search|todos>   (see fanout.ail for the two modes)
use futures::stream::{self, StreamExt};
use std::time::Instant;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = args[1].parse().expect("port");
    let n: i64 = args[2].parse().expect("requests");
    let search = args[3] == "search";
    let client = reqwest::Client::new();
    let t0 = Instant::now();
    let counts: Vec<i64> = stream::iter(0..n)
        .map(|q| {
            let client = client.clone();
            async move {
                let url = if search {
                    format!("http://127.0.0.1:{port}/search?q={q}&tag=t{}", q % 10)
                } else {
                    format!("http://127.0.0.1:{port}/todos?limit=1&offset={q}")
                };
                match client.get(url).send().await {
                    Ok(r) if r.status() == 200 => {
                        let body = r.text().await.unwrap_or_default();
                        if search {
                            body.trim_start_matches("{\"count\":").trim_end_matches('}').parse().unwrap_or(-1)
                        } else {
                            body.len() as i64
                        }
                    }
                    _ => -1,
                }
            }
        })
        .buffered(64)
        .collect()
        .await;
    let ms = t0.elapsed().as_millis();
    let failed = counts.iter().filter(|c| **c < 0).count();
    println!("sum {} failed {failed} ms {ms}", counts.iter().filter(|c| **c >= 0).sum::<i64>());
}
