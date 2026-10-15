//! wordfreq — text processing: tokenize, count in a hash map, sort.

use std::collections::HashMap;

const SEED: u64 = 99;
const VOCAB: usize = 5000;
const WORDS: usize = 3_000_000;
const TOP: usize = 15;
const CONS: &[u8] = b"bcdfghjklmnprstv";
const VOW: &[u8] = b"aeiou";

/// Park–Miller minimal standard generator (multiplier 48271).
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

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn build_vocab(rng: &mut Rng) -> Vec<String> {
    (0..VOCAB)
        .map(|_| {
            let syllables = 2 + rng.below(3);
            let mut word = String::with_capacity(2 * syllables as usize);
            for _ in 0..syllables {
                word.push(CONS[rng.below(CONS.len() as u64) as usize] as char);
                word.push(VOW[rng.below(VOW.len() as u64) as usize] as char);
            }
            if rng.below(4) == 0 {
                word[..1].make_ascii_uppercase();
            }
            word
        })
        .collect()
}

fn build_text(rng: &mut Rng, vocab: &[String]) -> String {
    let mut text = String::with_capacity(WORDS * 9);
    for _ in 0..WORDS {
        let a = rng.below(VOCAB as u64);
        let b = rng.below(VOCAB as u64);
        text.push_str(&vocab[a.min(b) as usize]);
        text.push_str(match rng.below(10) {
            0 => ", ",
            1 => ". ",
            _ => " ",
        });
    }
    text
}

fn count_words(text: &str) -> HashMap<String, u64> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut lower = String::new();
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|c| c == ',' || c == '.');
        if token.is_empty() {
            continue;
        }
        lower.clear();
        lower.push_str(token);
        lower.make_ascii_lowercase();
        match counts.get_mut(lower.as_str()) {
            Some(n) => *n += 1,
            None => {
                counts.insert(lower.clone(), 1);
            }
        }
    }
    counts
}

fn main() {
    let mut rng = Rng::new(SEED);
    let vocab = build_vocab(&mut rng);
    let text = build_text(&mut rng, &vocab);
    let counts = count_words(&text);

    let mut ranked: Vec<(&str, u64)> = counts.iter().map(|(w, &n)| (w.as_str(), n)).collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    println!("distinct {}", counts.len());
    for (word, n) in ranked.iter().take(TOP) {
        println!("{word} {n}");
    }
}
