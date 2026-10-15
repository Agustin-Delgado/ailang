// Idiomatic Rust baseline for the `raytrace` benchmark (see bench/SPECS.md).
use std::ops::{Add, Mul, Sub};

const WIDTH: usize = 480;
const HEIGHT: usize = 360;
const MAX_DEPTH: u32 = 3;
const EPS: f64 = 0.0001;
const BIAS: f64 = 0.001;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    fn len(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalize(self) -> Vec3 {
        let l = self.len();
        Vec3::new(self.x / l, self.y / l, self.z / l)
    }

    fn reflect(self, n: Vec3) -> Vec3 {
        self - n * (2.0 * self.dot(n))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, k: f64) -> Vec3 {
        Vec3::new(self.x * k, self.y * k, self.z * k)
    }
}

type Color = Vec3;

#[derive(Clone, Copy, Debug)]
enum Shape {
    Sphere { center: Vec3, radius: f64 },
    /// The plane `y = 0` with normal (0, 1, 0).
    Plane,
}

impl Shape {
    /// Nearest hit distance `t > EPS` along the ray, if any.
    fn intersect(&self, o: Vec3, d: Vec3) -> Option<f64> {
        match *self {
            Shape::Sphere { center, radius } => {
                let oc = o - center;
                let b = oc.dot(d);
                let cc = oc.dot(oc) - radius * radius;
                let disc = b * b - cc;
                if disc < 0.0 {
                    return None;
                }
                let s = disc.sqrt();
                let mut t = -b - s;
                if t <= EPS {
                    t = -b + s;
                }
                (t > EPS).then_some(t)
            }
            Shape::Plane => {
                let denom = d.y;
                if denom.abs() < 1e-9 {
                    return None;
                }
                let t = -o.y / denom;
                (t > EPS).then_some(t)
            }
        }
    }

    fn normal_at(&self, p: Vec3) -> Vec3 {
        match *self {
            Shape::Sphere { center, .. } => (p - center).normalize(),
            Shape::Plane => Vec3::new(0.0, 1.0, 0.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Object {
    shape: Shape,
    color: Color,
    reflectivity: f64,
}

struct Scene {
    objects: Vec<Object>,
    light: Vec3,
}

impl Scene {
    fn new() -> Self {
        let sphere = |c: Vec3, r: f64, color: Color, reflectivity: f64| Object {
            shape: Shape::Sphere { center: c, radius: r },
            color,
            reflectivity,
        };
        let objects = vec![
            sphere(Vec3::new(0.0, 1.0, 3.0), 1.0, Vec3::new(1.0, 0.2, 0.2), 0.3),
            sphere(Vec3::new(-2.2, 0.7, 4.0), 0.7, Vec3::new(0.2, 1.0, 0.2), 0.5),
            sphere(Vec3::new(2.0, 0.5, 2.5), 0.5, Vec3::new(0.2, 0.4, 1.0), 0.0),
            sphere(Vec3::new(0.8, 0.3, 1.2), 0.3, Vec3::new(1.0, 1.0, 0.2), 0.6),
            Object { shape: Shape::Plane, color: Vec3::new(0.8, 0.8, 0.8), reflectivity: 0.4 },
        ];
        Scene { objects, light: Vec3::new(-4.0, 6.0, -3.0).normalize() }
    }

    fn nearest_hit(&self, o: Vec3, d: Vec3) -> Option<(&Object, f64)> {
        let mut best: Option<(&Object, f64)> = None;
        for obj in &self.objects {
            if let Some(t) = obj.shape.intersect(o, d) {
                if best.map_or(true, |(_, bt)| t < bt) {
                    best = Some((obj, t));
                }
            }
        }
        best
    }

    fn occluded(&self, o: Vec3, d: Vec3) -> bool {
        self.objects.iter().any(|obj| obj.shape.intersect(o, d).is_some())
    }

    fn trace(&self, o: Vec3, d: Vec3, depth: u32) -> Color {
        let Some((obj, t)) = self.nearest_hit(o, d) else {
            let k = 0.5 + 0.5 * d.y;
            return Vec3::new(0.6 * k, 0.8 * k, 1.0 * k);
        };

        let p = o + d * t;
        let n = obj.shape.normal_at(p);
        let origin = p + n * BIAS;
        let diff = if self.occluded(origin, self.light) {
            0.0
        } else {
            n.dot(self.light).max(0.0)
        };
        let base = obj.color * (0.1 + 0.9 * diff);

        let refl = obj.reflectivity;
        if refl > 0.0 && depth < MAX_DEPTH {
            let rc = self.trace(origin, d.reflect(n), depth + 1);
            base * (1.0 - refl) + rc * refl
        } else {
            base
        }
    }
}

fn to_byte(c: f64) -> u64 {
    (c.max(0.0).min(1.0) * 255.0).floor() as u64
}

fn main() {
    let scene = Scene::new();
    let camera = Vec3::new(0.0, 1.0, -4.0);

    let mut checksum: u64 = 0;
    let mut center = [0u64; 3];

    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let u = (px as f64 + 0.5) / WIDTH as f64 * 2.0 - 1.0;
            let v = 1.0 - (py as f64 + 0.5) / HEIGHT as f64 * 2.0;
            let dir = Vec3::new(u * 1.3333333333333333, v, 1.5).normalize();
            let c = scene.trace(camera, dir, 0);
            let rgb = [to_byte(c.x), to_byte(c.y), to_byte(c.z)];
            checksum += rgb.iter().sum::<u64>();
            if (px, py) == (240, 180) {
                center = rgb;
            }
        }
    }

    println!("checksum {checksum}");
    println!("center {} {} {}", center[0], center[1], center[2]);
}
