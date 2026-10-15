// Idiomatic Rust baseline for the "game" benchmark (see bench/SPECS.md):
// headless arena simulation with entity movement, a uniform spatial grid and pairwise collisions.

const W: f64 = 1600.0;
const H: f64 = 1200.0;
const N: usize = 4000;
const FRAMES: usize = 1500;
const DT: f64 = 1.0 / 60.0;
const CELL: f64 = 32.0;
const SEED: u64 = 12345;
const TAU: f64 = 6.283185307179586;

/// Park–Miller minimal standard generator (multiplier 48271).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state * 48271 % 2147483647;
        self.state
    }

    fn rand_f64(&mut self) -> f64 {
        self.next() as f64 / 2147483647.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Entity {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    r: f64,
    hp: i32,
}

impl Entity {
    fn spawn(rng: &mut Rng) -> Self {
        let r = 3.0 + rng.rand_f64() * 9.0;
        let x = r + rng.rand_f64() * (W - 2.0 * r);
        let y = r + rng.rand_f64() * (H - 2.0 * r);
        let angle = rng.rand_f64() * TAU;
        let speed = 40.0 + rng.rand_f64() * 160.0;
        Self {
            x,
            y,
            vx: angle.cos() * speed,
            vy: angle.sin() * speed,
            r,
            hp: 3,
        }
    }

    fn step(&mut self) {
        self.x = self.x + self.vx * DT;
        self.y = self.y + self.vy * DT;
        if self.x < self.r {
            self.x = self.r;
            self.vx = -self.vx;
        } else if self.x > W - self.r {
            self.x = W - self.r;
            self.vx = -self.vx;
        }
        if self.y < self.r {
            self.y = self.r;
            self.vy = -self.vy;
        } else if self.y > H - self.r {
            self.y = H - self.r;
            self.vy = -self.vy;
        }
    }
}

/// Uniform grid stored as compressed buckets: `start[c]..start[c + 1]` indexes `items`.
struct Grid {
    cols: usize,
    rows: usize,
    cell_of: Vec<usize>,
    start: Vec<usize>,
    items: Vec<usize>,
}

impl Grid {
    fn new(n: usize) -> Self {
        let cols = (W / CELL).floor() as usize;
        let rows = (H / CELL).floor() as usize;
        Self {
            cols,
            rows,
            cell_of: vec![0; n],
            start: vec![0; cols * rows + 1],
            items: vec![0; n],
        }
    }

    fn cell_coords(&self, e: &Entity) -> (usize, usize) {
        // Positions are clamped to [r, W - r] x [r, H - r], so they are never negative.
        let cx = ((e.x / CELL).floor() as usize).min(self.cols - 1);
        let cy = ((e.y / CELL).floor() as usize).min(self.rows - 1);
        (cx, cy)
    }

    fn rebuild(&mut self, entities: &[Entity]) {
        self.start.fill(0);
        for (i, e) in entities.iter().enumerate() {
            let (cx, cy) = self.cell_coords(e);
            let c = cy * self.cols + cx;
            self.cell_of[i] = c;
            self.start[c + 1] += 1;
        }
        for c in 1..self.start.len() {
            self.start[c] += self.start[c - 1];
        }
        let mut fill = self.start.clone();
        for (i, &c) in self.cell_of.iter().enumerate() {
            self.items[fill[c]] = i;
            fill[c] += 1;
        }
    }

    fn bucket(&self, cx: usize, cy: usize) -> &[usize] {
        let c = cy * self.cols + cx;
        &self.items[self.start[c]..self.start[c + 1]]
    }
}

/// Counts colliding pairs (i < j) in neighbouring cells and applies the hp damage.
fn collide(entities: &mut [Entity], grid: &Grid) -> u64 {
    let mut collisions = 0;
    for i in 0..entities.len() {
        let c = grid.cell_of[i];
        let (cx, cy) = (c % grid.cols, c / grid.cols);
        let (xi, yi, ri) = (entities[i].x, entities[i].y, entities[i].r);
        for ny in cy.saturating_sub(1)..=(cy + 1).min(grid.rows - 1) {
            for nx in cx.saturating_sub(1)..=(cx + 1).min(grid.cols - 1) {
                for &j in grid.bucket(nx, ny) {
                    if j <= i {
                        continue;
                    }
                    let other = &entities[j];
                    let ddx = other.x - xi;
                    let ddy = other.y - yi;
                    let rr = ri + other.r;
                    if ddx * ddx + ddy * ddy < rr * rr {
                        collisions += 1;
                        entities[i].hp -= 1;
                        entities[j].hp -= 1;
                    }
                }
            }
        }
    }
    collisions
}

fn main() {
    let mut rng = Rng::new(SEED);
    let mut entities: Vec<Entity> = (0..N).map(|_| Entity::spawn(&mut rng)).collect();
    let mut grid = Grid::new(N);

    let mut collisions: u64 = 0;
    let mut respawns: u64 = 0;

    for _ in 0..FRAMES {
        entities.iter_mut().for_each(Entity::step);
        grid.rebuild(&entities);
        collisions += collide(&mut entities, &grid);
        for e in entities.iter_mut().filter(|e| e.hp <= 0) {
            *e = Entity::spawn(&mut rng);
            respawns += 1;
        }
    }

    let checksum: i64 = entities
        .iter()
        .map(|e| e.x.floor() as i64 + e.y.floor() as i64)
        .sum();

    println!("collisions {collisions}");
    println!("respawns {respawns}");
    println!("checksum {checksum}");
}
