//! The `game` benchmark as a browser game compiled to WebAssembly (same world and rules as
//! bench/rust/src/bin/game.rs). The host calls `tick(dt)` then `view()` every frame and reads
//! the draw commands from linear memory: `[2, x, y, r, color]` per circle, colors are palette
//! indices the host maps to CSS strings.
use std::cell::RefCell;
use std::fmt::Write;

const W: f64 = 1600.0;
const H: f64 = 1200.0;
const N: usize = 4000;
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

    fn step(&mut self, dt: f64) {
        self.x = self.x + self.vx * dt;
        self.y = self.y + self.vy * dt;
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

struct Game {
    rng: Rng,
    entities: Vec<Entity>,
    grid: Grid,
    collisions: u64,
    respawns: u64,
    frames: u64,
    cmds: Vec<f64>,
    hud: String,
}

impl Game {
    fn new() -> Self {
        let mut rng = Rng::new(SEED);
        let entities = (0..N).map(|_| Entity::spawn(&mut rng)).collect();
        Self { rng, entities, grid: Grid::new(N), collisions: 0, respawns: 0, frames: 0, cmds: Vec::new(), hud: String::new() }
    }

    fn tick(&mut self, dt: f64) {
        self.entities.iter_mut().for_each(|e| e.step(dt));
        self.grid.rebuild(&self.entities);
        self.collisions += collide(&mut self.entities, &self.grid);
        for e in self.entities.iter_mut().filter(|e| e.hp <= 0) {
            *e = Entity::spawn(&mut self.rng);
            self.respawns += 1;
        }
        self.frames += 1;
    }

    fn view(&mut self) {
        const BACKGROUND: f64 = 0.0;
        const HEALTHY: f64 = 1.0;
        const HURT: f64 = 2.0;
        const DYING: f64 = 3.0;
        const TEXT: f64 = 4.0;
        self.cmds.clear();
        self.cmds.extend_from_slice(&[0.0, BACKGROUND]);
        for e in &self.entities {
            let color = match e.hp {
                3 => HEALTHY,
                2 => HURT,
                _ => DYING,
            };
            self.cmds.extend_from_slice(&[2.0, e.x, e.y, e.r, color]);
        }
        self.hud.clear();
        let _ = write!(self.hud, "frame {}  collisions {}  respawns {}", self.frames, self.collisions, self.respawns);
        self.cmds.extend_from_slice(&[4.0, 16.0, 14.0, 22.0, TEXT]);
    }
}

thread_local! {
    static GAME: RefCell<Option<Game>> = const { RefCell::new(None) };
}

fn with_game<R>(f: impl FnOnce(&mut Game) -> R) -> R {
    GAME.with(|g| f(g.borrow_mut().as_mut().expect("init must run first")))
}

#[no_mangle]
pub extern "C" fn init() {
    GAME.with(|g| *g.borrow_mut() = Some(Game::new()));
}

#[no_mangle]
pub extern "C" fn tick(dt: f64) {
    with_game(|g| g.tick(dt));
}

/// Builds the frame's draw commands and returns how many f64 values they take.
#[no_mangle]
pub extern "C" fn view() -> u32 {
    with_game(|g| {
        g.view();
        g.cmds.len() as u32
    })
}

#[no_mangle]
pub extern "C" fn cmds() -> *const f64 {
    with_game(|g| g.cmds.as_ptr())
}

#[no_mangle]
pub extern "C" fn hud_ptr() -> *const u8 {
    with_game(|g| g.hud.as_ptr())
}

#[no_mangle]
pub extern "C" fn hud_len() -> u32 {
    with_game(|g| g.hud.len() as u32)
}
