use std::cmp::Reverse;
use std::collections::BinaryHeap;

const W: usize = 640;
const H: usize = 640;
const SEED: u64 = 2024;
const WALL_PERCENT: u64 = 28;
const QUERIES: usize = 40;

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

    fn rand_below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Grid {
    width: usize,
    height: usize,
    walls: Vec<bool>,
}

impl Grid {
    fn generate(width: usize, height: usize, rng: &mut Rng) -> Self {
        // Row-major: y outer, x inner, so the draw order matches the index order.
        let mut walls: Vec<bool> = (0..width * height)
            .map(|_| rng.rand_below(100) < WALL_PERCENT)
            .collect();
        walls[0] = false;
        walls[width * height - 1] = false;
        Self { width, height, walls }
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn is_wall(&self, x: usize, y: usize) -> bool {
        self.walls[self.index(x, y)]
    }
}

/// Reusable A* state, so repeated queries don't reallocate the cost table.
struct Pathfinder {
    g: Vec<u32>,
    open: BinaryHeap<Reverse<(u32, u32, u32)>>,
}

impl Pathfinder {
    fn new(cells: usize) -> Self {
        Self {
            g: vec![u32::MAX; cells],
            open: BinaryHeap::new(),
        }
    }

    /// Length of the shortest 4-connected path, or `None` if the goal is unreachable.
    fn shortest_path(
        &mut self,
        grid: &Grid,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<u32> {
        self.g.fill(u32::MAX);
        self.open.clear();

        let heuristic =
            |x: usize, y: usize| (x.abs_diff(goal.0) + y.abs_diff(goal.1)) as u32;

        let start_idx = grid.index(start.0, start.1);
        let goal_idx = grid.index(goal.0, goal.1);
        self.g[start_idx] = 0;
        self.open
            .push(Reverse((heuristic(start.0, start.1), 0, start_idx as u32)));

        while let Some(Reverse((_, g, idx))) = self.open.pop() {
            let idx = idx as usize;
            if idx == goal_idx {
                return Some(g);
            }
            if g > self.g[idx] {
                continue; // stale entry
            }
            let (x, y) = (idx % grid.width, idx / grid.width);
            let neighbors = [
                (x > 0).then(|| (x - 1, y)),
                (x + 1 < grid.width).then(|| (x + 1, y)),
                (y > 0).then(|| (x, y - 1)),
                (y + 1 < grid.height).then(|| (x, y + 1)),
            ];
            let next_g = g + 1;
            for (nx, ny) in neighbors.into_iter().flatten() {
                let n = grid.index(nx, ny);
                if grid.walls[n] || next_g >= self.g[n] {
                    continue;
                }
                self.g[n] = next_g;
                self.open
                    .push(Reverse((next_g + heuristic(nx, ny), next_g, n as u32)));
            }
        }
        None
    }
}

fn main() {
    let mut rng = Rng::new(SEED);
    let grid = Grid::generate(W, H, &mut rng);
    let mut finder = Pathfinder::new(W * H);

    let mut total: i64 = 0;
    for q in 0..QUERIES {
        let sx = rng.rand_below(W as u64) as usize;
        let sy = rng.rand_below(H as u64) as usize;
        let gx = rng.rand_below(W as u64) as usize;
        let gy = rng.rand_below(H as u64) as usize;

        let result: i64 = if grid.is_wall(sx, sy) || grid.is_wall(gx, gy) {
            -2
        } else {
            finder
                .shortest_path(&grid, (sx, sy), (gx, gy))
                .map_or(-1, i64::from)
        };
        if result >= 0 {
            total += result;
        }
        println!("q{q} {result}");
    }
    println!("total {total}");
}
