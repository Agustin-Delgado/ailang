use std::collections::HashMap;

fn main() {
    let words: Vec<String> = (0..200_000i64).map(|i| format!("w{}", (i * 31) % 5003)).collect();
    let text = words.join(" ");
    let mut counts: HashMap<String, i64> = HashMap::new();
    for _ in 0..20 {
        for w in text.split(' ') {
            *counts.entry(w.to_string()).or_insert(0) += 1;
        }
    }
    let mut keys: Vec<&String> = counts.keys().collect();
    keys.sort();
    let (mut best, mut best_n) = (String::new(), 0);
    for k in keys {
        let c = counts[k];
        if c > best_n {
            best = k.clone();
            best_n = c;
        }
    }
    println!("{} {} {} {}", counts.len(), best, best_n, text.len());
}
