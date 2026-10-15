fn count_primes(n: usize) -> i64 {
    let mut is_p = vec![true; n + 1];
    is_p[0] = false;
    is_p[1] = false;
    let mut i = 2;
    while i * i <= n {
        if is_p[i] {
            let mut j = i * i;
            while j <= n {
                is_p[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is_p.iter().filter(|&&p| p).count() as i64
}

fn main() {
    let mut total = 0;
    for _ in 0..10 {
        total += count_primes(50_000_000);
    }
    println!("{total}");
}
