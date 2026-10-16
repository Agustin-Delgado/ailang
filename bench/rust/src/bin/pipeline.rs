//! pipeline: three stages on three threads connected by std mpsc channels (bench/SPECS.md).
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;

const N: i64 = 2_000_000;

struct Hit {
    path: String,
    status: i64,
    ms: i64,
}

fn parse_hit(line: &str) -> Hit {
    let mut parts = line.split(' ');
    parts.next();
    let path = parts.next().expect("path").to_string();
    let status = parts.next().expect("status").parse().expect("status");
    let ms = parts.next().expect("ms").parse().expect("ms");
    Hit { path, status, ms }
}

fn main() {
    let (tx_lines, rx_lines) = mpsc::channel::<String>();
    let (tx_hits, rx_hits) = mpsc::channel::<Hit>();
    let producer = thread::spawn(move || {
        for i in 0..N {
            let status = if i % 13 == 0 { 500 } else if i % 7 == 0 { 404 } else { 200 };
            tx_lines.send(format!("GET /item/{} {} {}", i % 1000, status, (i * 7919) % 1000)).unwrap();
        }
    });
    let parser = thread::spawn(move || {
        for line in rx_lines {
            tx_hits.send(parse_hit(&line)).unwrap();
        }
    });
    let aggregator = thread::spawn(move || {
        let mut by_path: HashMap<String, (i64, i64)> = HashMap::new();
        let (mut errors, mut missing) = (0i64, 0i64);
        for h in rx_hits {
            if h.status >= 500 {
                errors += 1;
            }
            if h.status == 404 {
                missing += 1;
            }
            let e = by_path.entry(h.path).or_insert((0, 0));
            e.0 += 1;
            e.1 += h.ms;
        }
        (by_path, errors, missing)
    });
    producer.join().unwrap();
    parser.join().unwrap();
    let (by_path, errors, missing) = aggregator.join().unwrap();
    let mut paths: Vec<&String> = by_path.keys().collect();
    paths.sort();
    let (mut total, mut total_ms, mut best, mut best_ms) = (0i64, 0i64, "", -1i64);
    for p in paths {
        let (c, ms) = by_path[p];
        total += c;
        total_ms += ms;
        if ms > best_ms {
            best = p;
            best_ms = ms;
        }
    }
    println!("hits {total} errors {errors} missing {missing}");
    println!("paths {} ms {total_ms}", by_path.len());
    println!("slowest {best} {best_ms}");
}
