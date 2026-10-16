//! Runtime support appended to every generated program.

pub const RUNTIME: &str = r####"
#[global_allocator]
static AIL_ALLOC: ail_alloc::SizeClassAlloc = ail_alloc::SizeClassAlloc;

/// Size-class allocator: small blocks come from per-class free lists carved out of large chunks,
/// so allocating and freeing a small value is a couple of instructions. Each thread has its own
/// free lists (no locks); a block freed on another thread joins that thread's list, which is safe
/// because chunks are never returned and every list holds blocks of one size class.
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
    thread_local! {
        static STATE: UnsafeCell<State> = const { UnsafeCell::new(State { free: [null_mut(); NCLASSES], bump: null_mut(), end: null_mut() }) };
    }
    #[inline(always)]
    fn state() -> *mut State { STATE.with(|s| s.get()) }

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

    // Medium blocks (1 KB to 1 MB) come in power-of-two classes with one free list per class
    // shared by all threads, behind a spin lock. They never go back to the system: a freed
    // block waits for the next allocation of its class. This keeps rows of a grid built on one
    // thread and dropped on another from piling up, and keeps them out of the system heap,
    // which on Windows serializes threads on blocks above 16 KB.
    const MEDIUM_MAX: usize = 1 << 20;
    const NMEDIUM: usize = 10;
    struct MediumList { lock: std::sync::atomic::AtomicBool, head: UnsafeCell<*mut u8> }
    unsafe impl Sync for MediumList {}
    static MEDIUM: [MediumList; NMEDIUM] = [const { MediumList { lock: std::sync::atomic::AtomicBool::new(false), head: UnsafeCell::new(null_mut()) } }; NMEDIUM];

    #[inline(always)]
    fn medium_class(size: usize) -> usize { (usize::BITS - (size - 1).leading_zeros()) as usize - 11 }

    #[derive(PartialEq, Clone, Copy)]
    enum Kind { Small(usize), Medium(usize), Large }

    #[inline(always)]
    fn kind(size: usize, align: usize) -> Kind {
        if align > CLASS || size > MEDIUM_MAX { Kind::Large }
        else if size <= MAX_SMALL { Kind::Small(class_of(size)) }
        else { Kind::Medium(medium_class(size)) }
    }

    fn lock(m: &MediumList) {
        use std::sync::atomic::Ordering;
        while m.lock.swap(true, Ordering::Acquire) {
            while m.lock.load(Ordering::Relaxed) { std::hint::spin_loop(); }
        }
    }
    fn unlock(m: &MediumList) { m.lock.store(false, std::sync::atomic::Ordering::Release); }

    unsafe fn medium_alloc(c: usize) -> *mut u8 {
        let m = &MEDIUM[c];
        lock(m);
        let head = *m.head.get();
        if !head.is_null() {
            *m.head.get() = *(head as *mut *mut u8);
        }
        unlock(m);
        if !head.is_null() { return head; }
        System.alloc(Layout::from_size_align_unchecked(1 << (c + 11), CLASS))
    }

    unsafe fn medium_free(p: *mut u8, c: usize) {
        let m = &MEDIUM[c];
        lock(m);
        *(p as *mut *mut u8) = *m.head.get();
        *m.head.get() = p;
        unlock(m);
    }

    unsafe impl GlobalAlloc for SizeClassAlloc {
        #[inline(always)]
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if l.size() > MAX_SMALL || l.align() > CLASS {
                return match kind(l.size(), l.align()) {
                    Kind::Medium(c) => medium_alloc(c),
                    _ => System.alloc(l),
                };
            }
            let st = &mut *state();
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
            if l.size() > MAX_SMALL || l.align() > CLASS {
                return match kind(l.size(), l.align()) {
                    Kind::Medium(c) => medium_free(p, c),
                    _ => System.dealloc(p, l),
                };
            }
            let st = &mut *state();
            let c = class_of(l.size());
            *(p as *mut *mut u8) = st.free[c];
            st.free[c] = p;
        }
        #[inline]
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
            let (old, new) = (kind(l.size(), l.align()), kind(new_size, l.align()));
            if old == Kind::Large && new == Kind::Large { return System.realloc(p, l, new_size); }
            if old == new && old != Kind::Large { return p; }
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
    #[derive(Clone, Copy)] pub struct Net;
    #[derive(Clone, Copy)] pub struct Gfx;

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

    #[cfg(target_arch = "wasm32")]
    fn seed() -> u64 { (super::ail_web::now_ms() * 1000.0) as u64 ^ 0x9E37_79B9_7F4A_7C15 | 1 }
    #[cfg(not(target_arch = "wasm32"))]
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
        // Source lines carry the file index in their top byte; FILE lists the files by line.
        let at = info.location().and_then(|l| ail_location(l.line())).map(|(l, c)| {
            let files = unsafe { FILE };
            let name = files.split('\n').nth((l >> 24) as usize).unwrap_or(files);
            format!("{}:{}:{}", name, l & 0xFF_FFFF, c)
        });
        match at { Some(a) => format!("{msg}\n  at {a}"), None => msg }
    }

    // Set while a read handler of http-serve runs: a trap there answers that request with 500
    // and the server keeps going (the handler could not have changed anything).
    #[cfg(not(target_arch = "wasm32"))]
    thread_local! { pub static SOFT_TRAP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn init(lines: &'static [(u32, u32, u32)], file: &'static str) {
        unsafe { LINES = lines; FILE = file; }
        std::panic::set_hook(Box::new(|info| {
            if SOFT_TRAP.with(|s| s.get()) {
                eprintln!("trap in a read handler (answered 500): {}", describe(info));
                return;
            }
            flush();
            eprintln!("trap: {}", describe(info));
            std::process::exit(101);
        }));
    }
    #[cfg(target_arch = "wasm32")]
    pub fn init(lines: &'static [(u32, u32, u32)], file: &'static str) {
        unsafe { LINES = lines; FILE = file; }
        std::panic::set_hook(Box::new(|info| {
            flush();
            super::ail_web::trap(&describe(info));
        }));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush() { OUT.with(|o| { let _ = o.borrow_mut().flush(); }); }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn print(s: &str) { OUT.with(|o| { let _ = o.borrow_mut().write_all(s.as_bytes()); }); }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn println(s: &str) { OUT.with(|o| { let mut o = o.borrow_mut(); let _ = o.write_all(s.as_bytes()); let _ = o.write_all(b"\n"); }); }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn eprintln(s: &str) { flush(); eprintln!("{s}"); }

    // In the browser, console output goes to the developer console, one message per line.
    #[cfg(target_arch = "wasm32")]
    thread_local! { static LINE: RefCell<String> = RefCell::new(String::new()); }
    #[cfg(target_arch = "wasm32")]
    pub fn flush() { LINE.with(|l| { let mut l = l.borrow_mut(); if !l.is_empty() { super::ail_web::log(&l, false); l.clear(); } }); }
    #[cfg(target_arch = "wasm32")]
    pub fn print(s: &str) { LINE.with(|l| l.borrow_mut().push_str(s)); }
    #[cfg(target_arch = "wasm32")]
    pub fn println(s: &str) { LINE.with(|l| { let mut l = l.borrow_mut(); l.push_str(s); super::ail_web::log(&l, false); l.clear(); }); }
    #[cfg(target_arch = "wasm32")]
    pub fn eprintln(s: &str) { flush(); super::ail_web::log(s, true); }

    pub fn main_error<E: std::fmt::Display>(e: &E) -> ! {
        flush();
        eprintln!("error: {e}");
        std::process::exit(1)
    }

    #[track_caller] #[cold]
    pub fn fail(msg: &str) -> ! { panic!("fail: {msg}") }
    #[track_caller] #[cold]
    pub fn assert_fail(msg: &str) -> ! { panic!("assertion failed: {msg}") }
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
    #[track_caller]
    pub fn slice_ref<T>(v: &Vec<T>, a: i64, b: i64) -> &[T] {
        let (a, b) = (ix(a), ix(b));
        if a > b || b > v.len() { panic!("slice [{a}, {b}) out of bounds for length {}", v.len()) }
        &v[a..b]
    }
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
    pub fn expect_opt<T>(o: Option<T>, msg: &str) -> T { match o { Some(v) => v, None => panic!("expect failed: {msg}") } }
    #[track_caller]
    pub fn expect_res<T, E: std::fmt::Display>(r: Result<T, E>, msg: &str) -> T {
        match r { Ok(v) => v, Err(e) => panic!("expect failed: {msg}: {e}") }
    }
    #[track_caller]
    pub fn substr(s: &str, a: i64, b: i64) -> String {
        let (a, b) = (ix(a), ix(b));
        match s.get(a..b) { Some(x) => x.to_string(), None => panic!("substr [{a}, {b}) invalid for string of {} bytes", s.len()) }
    }
    pub fn lower(s: &str) -> String { if s.is_ascii() { s.to_ascii_lowercase() } else { s.to_lowercase() } }
    /// A borrowed piece of `s`: no copy (string views).
    pub fn substr_ref<'a>(s: &'a str, a: i64, b: i64) -> &'a str {
        let (a, b) = (ix(a), ix(b));
        match s.get(a..b) { Some(x) => x, None => panic!("substr [{a}, {b}) invalid for string of {} bytes", s.len()) }
    }
    /// Lower-cases a view, copying only when some character actually changes.
    pub fn lower_cow<'a>(s: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'a, str> {
        if s.bytes().all(|b| b < 128 && !b.is_ascii_uppercase()) { s } else { std::borrow::Cow::Owned(lower_owned(s.into_owned())) }
    }
    pub fn upper_cow<'a>(s: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'a, str> {
        if s.bytes().all(|b| b < 128 && !b.is_ascii_lowercase()) { s } else { std::borrow::Cow::Owned(upper_owned(s.into_owned())) }
    }
    /// `par-run`: `f` over `xs` with up to 64 threads, one element at a time, for work that waits
    /// on I/O (requests to other services, files). Results come back in order.
    pub fn par_run<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(xs: &[T], f: &F) -> Vec<R> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = xs.len();
        let threads = n.min(64);
        if threads <= 1 {
            return xs.iter().map(f).collect();
        }
        flush();
        let next = AtomicUsize::new(0);
        let mut parts: Vec<(usize, R)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n {
                                break;
                            }
                            mine.push((i, f(&xs[i])));
                        }
                        mine
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap_or_else(|_| std::process::exit(101))).collect()
        });
        parts.sort_unstable_by_key(|p| p.0);
        parts.into_iter().map(|p| p.1).collect()
    }

    /// `f` over `xs` on all cores; the result is in order, the same as a sequential map. The
    /// work runs on a pool of threads started once (not per call), and each result is written
    /// straight into its slot of the output. `f` cannot have effects (the checker forbids
    /// capabilities): the result never depends on timing.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn par_map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(xs: &[T], f: &F) -> Vec<R> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = xs.len();
        let threads = pool::size().min(n);
        if threads <= 1 || pool::inside() {
            return xs.iter().map(f).collect();
        }
        // A trap in a worker exits the process: make sure what main printed is out first.
        flush();
        let block = (n / (threads * 16)).max(1);
        let blocks = n.div_ceil(block);
        let next = AtomicUsize::new(0);
        let mut out: Vec<R> = Vec::with_capacity(n);
        struct Slots<R>(*mut R);
        unsafe impl<R: Send> Sync for Slots<R> {}
        impl<R> Slots<R> {
            // A method, so closures capture the whole Slots (which is Sync), not the raw field.
            fn at(&self, i: usize) -> *mut R { unsafe { self.0.add(i) } }
        }
        let slots = Slots(out.as_mut_ptr());
        let work = || loop {
            let b = next.fetch_add(1, Ordering::Relaxed);
            if b >= blocks {
                break;
            }
            let lo = b * block;
            for i in lo..(lo + block).min(n) {
                // Each index is claimed by exactly one block, so each slot is written once.
                unsafe { slots.at(i).write(f(&xs[i])) };
            }
        };
        pool::run(&work);
        unsafe { out.set_len(n) };
        out
    }
    #[cfg(target_arch = "wasm32")]
    pub fn par_map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(xs: &[T], f: &F) -> Vec<R> {
        xs.iter().map(f).collect()
    }

    /// The threads behind par-map, started on first use and kept for the whole run.
    #[cfg(not(target_arch = "wasm32"))]
    pub mod pool {
        use std::cell::Cell;
        use std::sync::{Condvar, Mutex, OnceLock};

        struct Job(*const (dyn Fn() + Sync));
        unsafe impl Send for Job {}
        struct State { generation: u64, job: Option<Job>, running: usize, trapped: bool }
        struct Pool { state: Mutex<State>, start: Condvar, end: Condvar, busy: Mutex<()>, workers: usize }

        thread_local! { static INSIDE: Cell<bool> = const { Cell::new(false) }; }

        fn get() -> &'static Pool {
            static POOL: OnceLock<&'static Pool> = OnceLock::new();
            POOL.get_or_init(|| {
                // AIL_THREADS caps the threads (for measuring); by default, one per core.
                let cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
                let wanted = std::env::var("AIL_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n >= 1);
                let workers = wanted.unwrap_or(cores).saturating_sub(1);
                let p: &'static Pool = Box::leak(Box::new(Pool {
                    state: Mutex::new(State { generation: 0, job: None, running: 0, trapped: false }),
                    start: Condvar::new(),
                    end: Condvar::new(),
                    busy: Mutex::new(()),
                    workers,
                }));
                for _ in 0..workers {
                    std::thread::spawn(move || worker(p));
                }
                p
            })
        }

        fn worker(p: &'static Pool) {
            INSIDE.with(|c| c.set(true));
            let mut seen = 0;
            loop {
                let job = {
                    let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
                    while st.generation == seen {
                        st = p.start.wait(st).unwrap_or_else(|e| e.into_inner());
                    }
                    seen = st.generation;
                    st.job.as_ref().map(|j| j.0)
                };
                let ok = match job {
                    Some(f) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*f)() })).is_ok(),
                    None => true,
                };
                let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
                st.trapped |= !ok;
                st.running -= 1;
                if st.running == 0 {
                    p.end.notify_all();
                }
            }
        }

        /// Threads that run par-map work, counting the caller.
        pub fn size() -> usize {
            get().workers + 1
        }

        /// True on a pool thread: a par-map inside a par-map runs sequentially there.
        pub fn inside() -> bool {
            INSIDE.with(|c| c.get())
        }

        /// Runs `work` on every pool thread and on the caller, and returns when all have
        /// finished, so `work` may borrow from the caller's stack.
        pub fn run(work: &(dyn Fn() + Sync)) {
            let p = get();
            // Another thread (a task) is already using the pool: do the work here instead.
            let Ok(_busy) = p.busy.try_lock() else {
                work();
                return;
            };
            {
                let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
                // The pointer outlives this call only in the pool's state, which is cleared below
                // before returning; workers use it only while `running` counts them.
                let job: *const (dyn Fn() + Sync) = work;
                st.job = Some(Job(unsafe { std::mem::transmute::<*const (dyn Fn() + Sync + '_), *const (dyn Fn() + Sync + 'static)>(job) }));
                st.running = p.workers;
                st.trapped = false;
                st.generation += 1;
            }
            p.start.notify_all();
            INSIDE.with(|c| c.set(true));
            let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
            INSIDE.with(|c| c.set(false));
            let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.running > 0 {
                st = p.end.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            st.job = None;
            let trapped = st.trapped;
            drop(st);
            if let Err(e) = mine {
                std::panic::resume_unwind(e);
            }
            if trapped {
                panic!("a parallel worker trapped");
            }
        }
    }
    /// Stable sort of `v` by precomputed keys (one per element): elements are moved, not copied.
    pub fn sort_by_keys<T, K>(v: &mut Vec<T>, keys: Vec<K>, cmp: impl Fn(&K, &K) -> std::cmp::Ordering) {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| cmp(&keys[a], &keys[b]));
        let mut old: Vec<Option<T>> = std::mem::take(v).into_iter().map(Some).collect();
        *v = idx.into_iter().map(|i| old[i].take().unwrap()).collect();
    }
    pub fn push_i64(s: &mut String, v: i64) {
        let mut buf = [0u8; 20];
        let mut i = buf.len();
        let mut n = v.unsigned_abs();
        loop { i -= 1; buf[i] = b'0' + (n % 10) as u8; n /= 10; if n == 0 { break; } }
        if v < 0 { s.push('-'); }
        s.push_str(unsafe { std::str::from_utf8_unchecked(&buf[i..]) });
    }
    pub fn lower_owned(mut s: String) -> String { if s.is_ascii() { s.make_ascii_lowercase(); s } else { s.to_lowercase() } }
    pub fn upper_owned(mut s: String) -> String { if s.is_ascii() { s.make_ascii_uppercase(); s } else { s.to_uppercase() } }
    pub fn upper(s: &str) -> String { if s.is_ascii() { s.to_ascii_uppercase() } else { s.to_uppercase() } }
    /// The pieces of `s` as borrowed slices (for loops over `split` without building a Vec).
    pub fn split_iter<'a>(s: &'a str, sep: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a> {
        if sep.is_empty() {
            Box::new(s.char_indices().map(move |(i, c)| &s[i..i + c.len_utf8()]))
        } else if sep.len() == 1 {
            Box::new(s.split(sep.as_bytes()[0] as char))
        } else {
            Box::new(s.split(sep))
        }
    }
    pub fn split(s: &str, sep: &str) -> Vec<String> {
        if sep.is_empty() { return chars(s); }
        if sep.len() == 1 { return s.split(sep.as_bytes()[0] as char).map(String::from).collect(); }
        s.split(sep).map(String::from).collect()
    }
    #[track_caller]
    pub fn from_code(n: i64) -> String {
        match u32::try_from(n).ok().and_then(char::from_u32) { Some(c) => c.to_string(), None => panic!("{n} is not a valid code point") }
    }
    pub fn chars(s: &str) -> Vec<String> { s.chars().map(|c| c.to_string()).collect() }
    #[track_caller]
    pub fn byte_at(s: &str, i: i64) -> i64 {
        let i = ix(i);
        match s.as_bytes().get(i) { Some(b) => *b as i64, None => panic!("byte index {i} out of bounds for string of {} bytes", s.len()) }
    }

    pub fn read_all() -> String {
        flush();
        let mut s = String::new();
        let _ = std::io::stdin().read_to_string(&mut s);
        s
    }
    /// `(for line (lines text) ...)`: the lines as views, no copies.
    pub fn lines_iter(s: &str) -> impl Iterator<Item = &str> {
        s.lines().map(|l| l.trim_end_matches('\r'))
    }
    pub fn lines(s: &str) -> Vec<String> {
        s.lines().map(|l| l.trim_end_matches('\r').to_string()).collect()
    }
    pub fn all_chars(s: &str, f: fn(char) -> bool) -> bool { !s.is_empty() && s.chars().all(f) }
    #[track_caller]
    pub fn code_of(s: &str) -> i64 {
        match s.chars().next() { Some(c) => c as i64, None => panic!("code-of on an empty string") }
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
    pub fn read_file(path: &str, max: i64) -> Result<String, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut buf = Vec::new();
        let limit = if max < 0 { 0 } else { max as u64 };
        f.take(limit + 1).read_to_end(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        if buf.len() as u64 > limit { return Err(format!("{path}: larger than the limit of {max} bytes")); }
        String::from_utf8(buf).map_err(|_| format!("{path}: not valid UTF-8"))
    }
    pub fn write_file(path: &str, content: &str) -> Result<(), String> {
        std::fs::write(path, content).map_err(|e| format!("{path}: {e}"))
    }
    #[cfg(target_arch = "wasm32")]
    pub fn now_ms() -> i64 { super::ail_web::now_ms() as i64 }
    #[cfg(target_arch = "wasm32")]
    pub fn now_ns() -> i64 { (super::ail_web::now_ms() * 1e6) as i64 }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn now_ms() -> i64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
    }
    #[cfg(not(target_arch = "wasm32"))]
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
