//! json — build, serialize, parse and aggregate dynamic JSON documents.

use serde_json::{json, Value};

const ROUNDS: usize = 8;
const RECORDS: u64 = 40_000;
const SEED: u64 = 4242;

/// Park–Miller minimal standard generator.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state * 48_271 % 2_147_483_647;
        self.state
    }

    fn rand_below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn build_document(rng: &mut Rng) -> Value {
    let records = (0..RECORDS)
        .map(|i| {
            let n = rng.rand_below(1_000_000);
            let s = rng.rand_below(10_000);
            let t = rng.rand_below(5);
            let tags: Vec<Value> = (0..=t).map(|k| Value::String(format!("tag{k}"))).collect();
            json!({
                "id": i,
                "name": format!("user{n}"),
                "score": s as f64 / 4.0 + 0.125,
                "active": n % 2 == 0,
                "tags": tags,
                "pos": { "x": n % 1000, "y": n / 1000 },
            })
        })
        .collect();
    Value::Array(records)
}

#[derive(Default)]
struct Totals {
    bytes: usize,
    score_sum: f64,
    actives: u64,
    tag_count: u64,
    pos_sum: i64,
}

impl Totals {
    fn aggregate(&mut self, doc: &Value) {
        let records = doc.as_array().expect("document is an array");
        for record in records {
            self.score_sum += record["score"].as_f64().expect("score is a number");
            if record["active"].as_bool() == Some(true) {
                self.actives += 1;
            }
            self.tag_count += record["tags"].as_array().map_or(0, |t| t.len()) as u64;
            let pos = &record["pos"];
            let x = pos["x"].as_i64().expect("x is an integer");
            let y = pos["y"].as_i64().expect("y is an integer");
            self.pos_sum += x + y;
        }
    }
}

fn main() {
    let mut rng = Rng::new(SEED);
    let mut totals = Totals::default();

    for _ in 0..ROUNDS {
        let doc = build_document(&mut rng);
        let text = serde_json::to_string(&doc).expect("serialize");
        totals.bytes += text.len();
        let parsed: Value = serde_json::from_str(&text).expect("parse");
        totals.aggregate(&parsed);
    }

    println!("bytes {}", totals.bytes);
    println!("score {:.3}", totals.score_sum);
    println!("actives {}", totals.actives);
    println!("tags {} pos {}", totals.tag_count, totals.pos_sum);
}
