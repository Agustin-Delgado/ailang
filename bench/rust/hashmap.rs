use std::collections::HashMap;

fn main() {
    let mut m: HashMap<i64, i64> = HashMap::new();
    let n = 5_000_000i64;
    for i in 0..n {
        m.insert(i * 7919, i);
    }
    let mut hits = 0i64;
    for i in 0..2 * n {
        if let Some(v) = m.get(&(i * 7919)) {
            hits += v;
        }
    }
    println!("{} {}", m.len(), hits);
}
