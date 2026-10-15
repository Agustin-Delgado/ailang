//! Runtime support appended to every generated program.

pub const RUNTIME: &str = r####"
#[global_allocator]
static AIL_ALLOC: ail_alloc::SizeClassAlloc = ail_alloc::SizeClassAlloc;

/// Size-class allocator: small blocks come from per-class free lists carved out of large chunks,
/// so allocating and freeing a small value is a couple of instructions. ail programs are
/// single-threaded (v0.1), which lets the free lists live without locks.
#[allow(dead_code)]
mod ail_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::UnsafeCell;
    use std::ptr::null_mut;

    const CLASS: usize = 16;
    const MAX_SMALL: usize = 1024;
    const NCLASSES: usize = MAX_SMALL / CLASS;
    const CHUNK: usize = 1 << 20;

    struct State { free: [*mut u8; NCLASSES], bump: *mut u8, end: *mut u8 }
    struct Cell(UnsafeCell<State>);
    unsafe impl Sync for Cell {}
    static STATE: Cell = Cell(UnsafeCell::new(State { free: [null_mut(); NCLASSES], bump: null_mut(), end: null_mut() }));

    pub struct SizeClassAlloc;

    #[inline(always)]
    fn class_of(size: usize) -> usize { (size.max(1) + CLASS - 1) / CLASS - 1 }

    #[cold]
    unsafe fn refill(st: &mut State, bytes: usize) -> *mut u8 {
        let chunk = System.alloc(Layout::from_size_align_unchecked(CHUNK, 4096));
        if chunk.is_null() { return null_mut(); }
        st.bump = chunk.add(bytes);
        st.end = chunk.add(CHUNK);
        chunk
    }

    unsafe impl GlobalAlloc for SizeClassAlloc {
        #[inline(always)]
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if l.size() > MAX_SMALL || l.align() > CLASS { return System.alloc(l); }
            let st = &mut *STATE.0.get();
            let c = class_of(l.size());
            let head = st.free[c];
            if !head.is_null() {
                st.free[c] = *(head as *mut *mut u8);
                return head;
            }
            let bytes = (c + 1) * CLASS;
            let p = st.bump;
            if !p.is_null() && (st.end as usize - p as usize) >= bytes {
                st.bump = p.add(bytes);
                return p;
            }
            refill(st, bytes)
        }
        #[inline(always)]
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if l.size() > MAX_SMALL || l.align() > CLASS { return System.dealloc(p, l); }
            let st = &mut *STATE.0.get();
            let c = class_of(l.size());
            *(p as *mut *mut u8) = st.free[c];
            st.free[c] = p;
        }
        #[inline]
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
            let small_old = l.size() <= MAX_SMALL && l.align() <= CLASS;
            let small_new = new_size <= MAX_SMALL && l.align() <= CLASS;
            if !small_old && !small_new { return System.realloc(p, l, new_size); }
            if small_old && small_new && class_of(l.size()) == class_of(new_size) { return p; }
            let nl = Layout::from_size_align_unchecked(new_size, l.align());
            let q = self.alloc(nl);
            if !q.is_null() {
                std::ptr::copy_nonoverlapping(p, q, l.size().min(new_size));
                self.dealloc(p, l);
            }
            q
        }
    }
}
#[allow(dead_code)]
mod ail_rt {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::hash::{BuildHasherDefault, Hasher};
    use std::io::{BufWriter, Read, Write};

    #[derive(Clone, Copy)] pub struct Console;
    #[derive(Clone, Copy)] pub struct Fs;
    #[derive(Clone, Copy)] pub struct Clock;
    #[derive(Clone, Copy)] pub struct Rand;
    #[derive(Clone, Copy)] pub struct Env;

    // Fast non-cryptographic hasher (word-at-a-time multiply-rotate).
    #[derive(Default, Clone, Copy)]
    pub struct AilHasher { h: u64 }
    const K: u64 = 0x517c_c1b7_2722_0a95;
    impl AilHasher {
        #[inline(always)]
        fn add(&mut self, w: u64) { self.h = (self.h.rotate_left(5) ^ w).wrapping_mul(K); }
    }
    impl Hasher for AilHasher {
        #[inline]
        fn write(&mut self, bytes: &[u8]) {
            let mut b = bytes;
            while b.len() >= 8 { self.add(u64::from_le_bytes(b[..8].try_into().unwrap())); b = &b[8..]; }
            if b.len() >= 4 { self.add(u32::from_le_bytes(b[..4].try_into().unwrap()) as u64); b = &b[4..]; }
            for &x in b { self.add(x as u64); }
        }
        #[inline(always)] fn write_u8(&mut self, i: u8) { self.add(i as u64) }
        #[inline(always)] fn write_u16(&mut self, i: u16) { self.add(i as u64) }
        #[inline(always)] fn write_u32(&mut self, i: u32) { self.add(i as u64) }
        #[inline(always)] fn write_u64(&mut self, i: u64) { self.add(i) }
        #[inline(always)] fn write_usize(&mut self, i: usize) { self.add(i as u64) }
        #[inline(always)] fn write_i64(&mut self, i: i64) { self.add(i as u64) }
        #[inline(always)] fn finish(&self) -> u64 { self.h }
    }
    pub type AilMap<K, V> = HashMap<K, V, BuildHasherDefault<AilHasher>>;

    static mut LINES: &[(u32, u32, u32)] = &[];
    static mut FILE: &str = "";

    thread_local! {
        static OUT: RefCell<BufWriter<std::io::Stdout>> = RefCell::new(BufWriter::with_capacity(1 << 16, std::io::stdout()));
        static RNG: RefCell<u64> = RefCell::new(seed());
        static LAST_TRAP: RefCell<Option<String>> = RefCell::new(None);
    }

    fn seed() -> u64 {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        let a = &t as *const u64 as u64;
        (t ^ a.rotate_left(32)) | 1
    }

    pub fn ail_location(rust_line: u32) -> Option<(u32, u32)> {
        let lines = unsafe { LINES };
        let i = lines.partition_point(|r| r.0 <= rust_line);
        if i == 0 { None } else { Some((lines[i - 1].1, lines[i - 1].2)) }
    }

    fn describe(info: &std::panic::PanicHookInfo) -> String {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() { s.to_string() }
            else if let Some(s) = info.payload().downcast_ref::<String>() { s.clone() }
            else { "trap".to_string() };
        let at = info.location().and_then(|l| ail_location(l.line())).map(|(l, c)| format!("{}:{}:{}", unsafe { FILE }, l, c));
        match at { Some(a) => format!("{msg}\n  at {a}"), None => msg }
    }

    pub fn init(lines: &'static [(u32, u32, u32)], file: &'static str) {
        unsafe { LINES = lines; FILE = file; }
        std::panic::set_hook(Box::new(|info| {
            flush();
            eprintln!("trap: {}", describe(info));
            std::process::exit(101);
        }));
    }

    pub fn flush() { OUT.with(|o| { let _ = o.borrow_mut().flush(); }); }
    pub fn print(s: &String) { OUT.with(|o| { let _ = o.borrow_mut().write_all(s.as_bytes()); }); }
    pub fn println(s: &String) { OUT.with(|o| { let mut o = o.borrow_mut(); let _ = o.write_all(s.as_bytes()); let _ = o.write_all(b"\n"); }); }
    pub fn eprintln(s: &String) { flush(); eprintln!("{s}"); }

    pub fn main_error<E: std::fmt::Display>(e: &E) -> ! {
        flush();
        eprintln!("error: {e}");
        std::process::exit(1)
    }

    #[track_caller] #[cold]
    pub fn fail(msg: &String) -> ! { panic!("fail: {msg}") }
    #[track_caller] #[cold]
    pub fn assert_fail(msg: &String) -> ! { panic!("assertion failed: {msg}") }
    #[track_caller] #[cold]
    pub fn contract(kind: &str, f: &str, src: &str) -> ! { panic!("contract violated: {kind} {src} (in fn {f})") }

    #[track_caller] #[inline(always)]
    pub fn ix(i: i64) -> usize {
        if i < 0 { neg_index(i) }
        i as usize
    }
    #[track_caller] #[inline(always)]
    pub fn ix_incl(i: i64) -> usize { ix(i) }
    #[track_caller] #[inline(always)]
    pub fn count(n: i64) -> usize {
        if n < 0 { panic!("negative count {n}") }
        n as usize
    }
    #[track_caller] #[cold]
    fn neg_index(i: i64) -> ! { panic!("negative index {i}") }

    pub trait ShiftAil: Sized { fn checked_shl_ail(self, s: u32) -> Self; fn checked_shr_ail(self, s: u32) -> Self; }
    macro_rules! shift_impl { ($($t:ty),*) => { $(
        impl ShiftAil for $t {
            #[track_caller] #[inline(always)]
            fn checked_shl_ail(self, s: u32) -> Self { self.checked_shl(s).unwrap_or_else(|| panic!("shift amount {s} out of range")) }
            #[track_caller] #[inline(always)]
            fn checked_shr_ail(self, s: u32) -> Self { self.checked_shr(s).unwrap_or_else(|| panic!("shift amount {s} out of range")) }
        }
    )* } }
    shift_impl!(i8, i16, i32, i64, u8, u16, u32, u64);

    #[track_caller] #[inline(always)]
    pub fn icast<A: Copy + std::fmt::Display, B: TryFrom<A>>(a: A) -> B {
        match B::try_from(a) { Ok(b) => b, Err(_) => panic!("value {a} does not fit in {}", std::any::type_name::<B>()) }
    }
    macro_rules! f2i { ($($name:ident $t:ty),*) => { $(
        #[track_caller] #[inline(always)]
        pub fn $name(f: f64) -> $t {
            if !(f >= (<$t>::MIN as f64) && f < (<$t>::MAX as f64 + 1.0)) { panic!("float {f} does not fit in {}", stringify!($t)) }
            f as $t
        }
    )* } }
    f2i!(f2i_i8 i8, f2i_i16 i16, f2i_i32 i32, f2i_i64 i64, f2i_u8 u8, f2i_u16 u16, f2i_u32 u32, f2i_u64 u64);

    #[track_caller]
    pub fn slice<T: Clone>(v: &Vec<T>, a: i64, b: i64) -> Vec<T> {
        let (a, b) = (ix(a), ix(b));
        if a > b || b > v.len() { panic!("slice [{a}, {b}) out of bounds for length {}", v.len()) }
        v[a..b].to_vec()
    }
    pub fn concat<T: Clone>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
        let mut r = Vec::with_capacity(a.len() + b.len());
        r.extend_from_slice(a);
        r.extend_from_slice(b);
        r
    }
    pub fn index_of<T: PartialEq>(v: &Vec<T>, x: &T) -> Option<i64> { v.iter().position(|e| e == x).map(|i| i as i64) }
    pub fn keys<K: Clone + Ord, V>(m: &AilMap<K, V>) -> Vec<K> {
        let mut k: Vec<K> = m.keys().cloned().collect();
        k.sort_unstable();
        k
    }
    pub fn values<K: Clone + Ord, V: Clone>(m: &AilMap<K, V>) -> Vec<V> {
        let mut kv: Vec<(&K, &V)> = m.iter().collect();
        kv.sort_unstable_by(|a, b| a.0.cmp(b.0));
        kv.into_iter().map(|(_, v)| v.clone()).collect()
    }
    #[track_caller]
    pub fn expect_opt<T>(o: Option<T>, msg: &String) -> T { match o { Some(v) => v, None => panic!("expect failed: {msg}") } }
    #[track_caller]
    pub fn expect_res<T, E: std::fmt::Display>(r: Result<T, E>, msg: &String) -> T {
        match r { Ok(v) => v, Err(e) => panic!("expect failed: {msg}: {e}") }
    }
    #[track_caller]
    pub fn substr(s: &String, a: i64, b: i64) -> String {
        let (a, b) = (ix(a), ix(b));
        match s.get(a..b) { Some(x) => x.to_string(), None => panic!("substr [{a}, {b}) invalid for string of {} bytes", s.len()) }
    }
    pub fn split(s: &String, sep: &String) -> Vec<String> {
        if sep.is_empty() { return chars(s); }
        s.split(sep.as_str()).map(String::from).collect()
    }
    #[track_caller]
    pub fn from_code(n: i64) -> String {
        match u32::try_from(n).ok().and_then(char::from_u32) { Some(c) => c.to_string(), None => panic!("{n} is not a valid code point") }
    }
    pub fn chars(s: &String) -> Vec<String> { s.chars().map(|c| c.to_string()).collect() }
    #[track_caller]
    pub fn byte_at(s: &String, i: i64) -> i64 {
        let i = ix(i);
        match s.as_bytes().get(i) { Some(b) => *b as i64, None => panic!("byte index {i} out of bounds for string of {} bytes", s.len()) }
    }

    pub fn read_line() -> Option<String> {
        flush();
        let mut s = String::new();
        match std::io::stdin().read_line(&mut s) {
            Ok(0) | Err(_) => None,
            Ok(_) => {
                while s.ends_with('\n') || s.ends_with('\r') { s.pop(); }
                Some(s)
            }
        }
    }
    pub fn read_file(path: &String, max: i64) -> Result<String, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut buf = Vec::new();
        let limit = if max < 0 { 0 } else { max as u64 };
        f.take(limit + 1).read_to_end(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        if buf.len() as u64 > limit { return Err(format!("{path}: larger than the limit of {max} bytes")); }
        String::from_utf8(buf).map_err(|_| format!("{path}: not valid UTF-8"))
    }
    pub fn write_file(path: &String, content: &String) -> Result<(), String> {
        std::fs::write(path, content).map_err(|e| format!("{path}: {e}"))
    }
    pub fn now_ms() -> i64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
    }
    pub fn now_ns() -> i64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as i64
    }
    fn next_u64() -> u64 {
        RNG.with(|r| {
            let mut x = *r.borrow();
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            *r.borrow_mut() = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        })
    }
    #[track_caller]
    pub fn rand_int(lo: i64, hi: i64) -> i64 {
        if hi <= lo { panic!("rand-int needs low < high, got [{lo}, {hi})") }
        let span = (hi as i128 - lo as i128) as u128;
        (lo as i128 + ((next_u64() as u128 * span) >> 64) as i128) as i64
    }
    pub fn rand_f64() -> f64 { (next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) }
    pub fn args() -> Vec<String> { std::env::args().skip(1).collect() }

    pub fn run_tests(tests: &[(&str, fn())]) {
        std::panic::set_hook(Box::new(|info| {
            let d = describe(info);
            LAST_TRAP.with(|t| *t.borrow_mut() = Some(d));
        }));
        let mut failed = 0;
        for (name, f) in tests {
            let ok = std::panic::catch_unwind(*f).is_ok();
            flush();
            if ok {
                println!("PASS {name}");
            } else {
                failed += 1;
                let why = LAST_TRAP.with(|t| t.borrow_mut().take()).unwrap_or_default();
                println!("FAIL {name}\n  {}", why.replace('\n', "\n  "));
            }
        }
        println!("{} passed, {} failed", tests.len() - failed, failed);
        if failed > 0 { std::process::exit(1) }
    }
}
"####;
