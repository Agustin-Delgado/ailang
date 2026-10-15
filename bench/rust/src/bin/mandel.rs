//! Mandelbrot set, rows computed in parallel with rayon.

use rayon::prelude::*;

const W: usize = 2000;
const H: usize = 2000;
const MAX_ITER: u32 = 1000;
const MODULUS: u64 = 1_000_000_007;

/// Escape-iteration count for the point `(cr, ci)`.
fn escape_time(cr: f64, ci: f64) -> u32 {
    let (mut zr, mut zi) = (0.0f64, 0.0f64);
    let mut i = 0;
    while i < MAX_ITER && zr * zr + zi * zi <= 4.0 {
        let t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        i += 1;
    }
    i
}

/// Returns `(row_sum, row_inside)` for row `y`.
fn compute_row(y: usize) -> (u64, u64) {
    let ci = 2.4 * y as f64 / H as f64 - 1.2;
    (0..W).fold((0, 0), |(sum, inside), x| {
        let cr = 3.0 * x as f64 / W as f64 - 2.2;
        let v = escape_time(cr, ci);
        (sum + u64::from(v), inside + u64::from(v == MAX_ITER))
    })
}

fn main() {
    let rows: Vec<(u64, u64)> = (0..H).into_par_iter().map(compute_row).collect();

    let total: u64 = rows.iter().map(|&(s, _)| s).sum();
    let inside: u64 = rows.iter().map(|&(_, n)| n).sum();
    let checksum = rows
        .iter()
        .enumerate()
        .fold(0u64, |c, (y, &(s, _))| (c + s * (y as u64 + 1)) % MODULUS);

    println!("total {total}");
    println!("inside {inside}");
    println!("checksum {checksum}");
}
