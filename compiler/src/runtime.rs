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

    // Set while a group of loop iterations runs with its GPU work queued (see ail_gpu::chunk_begin):
    // a trap there only rolls the group back, and the group runs again the ordinary way, which
    // reports the trap.
    thread_local! { pub static CHUNK_QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn init(lines: &'static [(u32, u32, u32)], file: &'static str) {
        unsafe { LINES = lines; FILE = file; }
        std::panic::set_hook(Box::new(|info| {
            let par = PAR_TRAP.with(|t| t.get());
            if !par.is_null() {
                unsafe { (*par).store(true, std::sync::atomic::Ordering::Release) };
                // A loop nested in another on this thread: the outer chunk stops here too.
                let up = PAR_TRAP_UP.with(|t| t.get());
                if !up.is_null() {
                    unsafe { (*up).store(true, std::sync::atomic::Ordering::Release) };
                }
                if cfg!(panic = "unwind") {
                    return;
                }
                pool::abandon();
                loop { std::thread::park(); }
            }
            if cfg!(panic = "unwind") && CHUNK_QUIET.with(|q| q.get()) {
                return;
            }
            if SOFT_TRAP.with(|s| s.get()) {
                eprintln!("trap in a read handler (answered 500): {}", describe(info));
                return;
            }
            release_all();
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

    // Output of a loop running on narrow copies is held until the loop finishes: if the run is
    // thrown away, so is what it printed. Runs nest (a held run may call a fn with its own).
    #[cfg(not(target_arch = "wasm32"))]
    thread_local! { static HOLD: RefCell<(String, usize)> = const { RefCell::new((String::new(), 0)) }; }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_begin() -> usize { HOLD.with(|h| { let mut h = h.borrow_mut(); h.1 += 1; h.0.len() }) }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_print(s: &str) { HOLD.with(|h| h.borrow_mut().0.push_str(s)); }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_println(s: &str) { HOLD.with(|h| { let mut h = h.borrow_mut(); h.0.push_str(s); h.0.push('\n'); }); }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_len() -> usize { HOLD.with(|h| h.borrow().0.len()) }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_commit() {
        let done = HOLD.with(|h| { let mut h = h.borrow_mut(); h.1 = h.1.saturating_sub(1); h.1 == 0 });
        if done { release_hold(); }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_discard(mark: usize) { HOLD.with(|h| { let mut h = h.borrow_mut(); h.1 = h.1.saturating_sub(1); h.0.truncate(mark); }); }
    #[cfg(not(target_arch = "wasm32"))]
    fn release_hold() {
        let s = HOLD.with(|h| std::mem::take(&mut h.borrow_mut().0));
        if !s.is_empty() { OUT.with(|o| { let _ = o.borrow_mut().write_all(s.as_bytes()); }); }
    }

    // A trap inside a held run is a real trap: what the run printed comes out before it.
    #[cfg(not(target_arch = "wasm32"))]
    fn release_all() { HOLD.with(|h| h.borrow_mut().1 = 0); release_hold(); }

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

    // `(at v i)`: one unsigned comparison covers both a negative index and one past the end (a
    // negative i64 is above any length as usize); only the message tells them apart.
    #[derive(Clone, Copy)]
    pub struct Ix(pub i64);
    #[track_caller] #[cold] #[inline(never)]
    fn out_of_bounds(i: i64, len: usize) -> ! {
        if i < 0 { neg_index(i) }
        panic!("index out of bounds: the len is {len} but the index is {i}")
    }
    impl<T> std::ops::Index<Ix> for [T] {
        type Output = T;
        #[track_caller] #[inline(always)]
        fn index(&self, i: Ix) -> &T {
            if (i.0 as u64) < self.len() as u64 { unsafe { self.get_unchecked(i.0 as usize) } } else { out_of_bounds(i.0, self.len()) }
        }
    }
    impl<T> std::ops::IndexMut<Ix> for [T] {
        #[track_caller] #[inline(always)]
        fn index_mut(&mut self, i: Ix) -> &mut T {
            if (i.0 as u64) < self.len() as u64 { unsafe { self.get_unchecked_mut(i.0 as usize) } } else { out_of_bounds(i.0, self.len()) }
        }
    }
    impl<T> std::ops::Index<Ix> for Vec<T> {
        type Output = T;
        #[track_caller] #[inline(always)]
        fn index(&self, i: Ix) -> &T { &self[..][i] }
    }
    impl<T> std::ops::IndexMut<Ix> for Vec<T> {
        #[track_caller] #[inline(always)]
        fn index_mut(&mut self, i: Ix) -> &mut T { &mut self[..][i] }
    }

    // Loops that run on 32-bit copies of their vectors (see `narrow_region` in the emitter):
    // every element stays within [lo, hi], a range that fits i32 and that the loop's index
    // reads rely on. A store outside it stops the fast run, which is then thrown away.
    pub fn narrow(v: &[i64], lo: i64, hi: i64) -> Option<Vec<i32>> {
        let mut ok = true;
        for &x in v {
            ok &= x >= lo && x <= hi;
        }
        if ok { Some(v.iter().map(|&x| x as i32).collect()) } else { None }
    }
    #[inline(always)]
    pub fn narrow_fill(n: usize, x: i64, lo: i64, hi: i64) -> Option<Vec<i32>> {
        if x >= lo && x <= hi { Some(vec![x as i32; n]) } else { None }
    }
    #[track_caller] #[inline(always)]
    pub fn nput(v: &mut [i32], i: usize, x: i64, lo: i64, hi: i64) -> bool {
        if x >= lo && x <= hi { v[i] = x as i32; true } else { false }
    }
    /// Safety: `i` is proven in range by the caller.
    #[inline(always)]
    pub unsafe fn nput_unchecked(v: &mut [i32], i: usize, x: i64, lo: i64, hi: i64) -> bool {
        if x >= lo && x <= hi { *v.get_unchecked_mut(i) = x as i32; true } else { false }
    }
    pub fn widen_into(dst: &mut [i64], src: &[i32]) {
        for (d, s) in dst.iter_mut().zip(src) { *d = *s as i64; }
    }

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

    // `(pow 2 e)`: below the width it is a shift; from there on the ordinary pow, which traps
    // exactly as before.
    pub trait Pow2: Sized { fn pow2_ail(e: u32) -> Self; }
    macro_rules! pow2_impl { ($($t:ty),*) => { $(
        impl Pow2 for $t {
            #[track_caller] #[inline(always)]
            fn pow2_ail(e: u32) -> Self {
                if e < <$t>::BITS - (<$t>::MIN != 0) as u32 { (1 as $t) << e } else { (2 as $t).pow(e) }
            }
        }
    )* } }
    pow2_impl!(i8, i16, i32, i64, u8, u16, u32, u64);

    // Arithmetic with the overflow check deferred: the result wraps and `of` records that some
    // operation overflowed. A loop built from these has no branch per operation, so LLVM can
    // vectorize it; when `of` comes back set, the loop runs again with the ordinary checks and
    // traps exactly where it would have. Only for loops whose second run gives the same result.
    pub trait Dov: Copy {
        fn dadd(self, b: Self, of: &mut bool) -> Self;
        fn dsub(self, b: Self, of: &mut bool) -> Self;
        fn dmul(self, b: Self, of: &mut bool) -> Self;
        /// Multiplication by a constant: the check folds into two comparisons.
        fn dmulc(self, c: Self, of: &mut bool) -> Self;
        fn ddiv(self, b: Self, of: &mut bool) -> Self;
        fn drem(self, b: Self, of: &mut bool) -> Self;
        fn dneg(self, of: &mut bool) -> Self;
        fn dabs(self, of: &mut bool) -> Self;
    }
    macro_rules! dov_signed { ($($t:ty),*) => { $(
        impl Dov for $t {
            #[inline(always)] fn dadd(self, b: Self, of: &mut bool) -> Self { let r = self.wrapping_add(b); *of |= ((self ^ r) & (b ^ r)) < 0; r }
            #[inline(always)] fn dsub(self, b: Self, of: &mut bool) -> Self { let r = self.wrapping_sub(b); *of |= ((self ^ b) & (self ^ r)) < 0; r }
            #[inline(always)] fn dmul(self, b: Self, of: &mut bool) -> Self { let (r, o) = self.overflowing_mul(b); *of |= o; r }
            #[inline(always)] fn dmulc(self, c: Self, of: &mut bool) -> Self {
                if c > 0 { *of |= self > <$t>::MAX / c || self < <$t>::MIN / c; self.wrapping_mul(c) } else { self.dmul(c, of) }
            }
            #[inline(always)] fn ddiv(self, b: Self, of: &mut bool) -> Self {
                if b == 0 || (b == -1 && self == <$t>::MIN) { *of = true; 0 } else { self / b }
            }
            #[inline(always)] fn drem(self, b: Self, of: &mut bool) -> Self {
                if b == 0 || (b == -1 && self == <$t>::MIN) { *of = true; 0 } else { self % b }
            }
            #[inline(always)] fn dneg(self, of: &mut bool) -> Self { *of |= self == <$t>::MIN; self.wrapping_neg() }
            #[inline(always)] fn dabs(self, of: &mut bool) -> Self { *of |= self == <$t>::MIN; self.wrapping_abs() }
        }
    )* } }
    dov_signed!(i8, i16, i32, i64);
    macro_rules! dov_unsigned { ($($t:ty),*) => { $(
        impl Dov for $t {
            #[inline(always)] fn dadd(self, b: Self, of: &mut bool) -> Self { let r = self.wrapping_add(b); *of |= r < self; r }
            #[inline(always)] fn dsub(self, b: Self, of: &mut bool) -> Self { *of |= b > self; self.wrapping_sub(b) }
            #[inline(always)] fn dmul(self, b: Self, of: &mut bool) -> Self { let (r, o) = self.overflowing_mul(b); *of |= o; r }
            #[inline(always)] fn dmulc(self, c: Self, of: &mut bool) -> Self {
                if c > 0 { *of |= self > <$t>::MAX / c; self.wrapping_mul(c) } else { 0 }
            }
            #[inline(always)] fn ddiv(self, b: Self, of: &mut bool) -> Self { if b == 0 { *of = true; 0 } else { self / b } }
            #[inline(always)] fn drem(self, b: Self, of: &mut bool) -> Self { if b == 0 { *of = true; 0 } else { self % b } }
            #[inline(always)] fn dneg(self, of: &mut bool) -> Self { *of |= self != 0; self.wrapping_neg() }
            #[inline(always)] fn dabs(self, _of: &mut bool) -> Self { self }
        }
    )* } }
    dov_unsigned!(u8, u16, u32, u64);

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
    #[track_caller]
    pub fn slice_bounds(len: usize, a: i64, b: i64) -> (usize, usize) {
        let (a, b) = (ix(a), ix(b));
        if a > b || b > len { panic!("slice [{a}, {b}) out of bounds for length {len}") }
        (a, b)
    }
    #[track_caller]
    pub fn slice_ref<T>(v: &Vec<T>, a: i64, b: i64) -> &[T] {
        let (a, b) = slice_bounds(v.len(), a, b);
        &v[a..b]
    }
    #[track_caller]
    pub fn slice<T: Clone>(v: &Vec<T>, a: i64, b: i64) -> Vec<T> {
        let (a, b) = slice_bounds(v.len(), a, b);
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
        if !pool::run(&work) {
            work();
        }
        unsafe { out.set_len(n) };
        out
    }
    #[cfg(target_arch = "wasm32")]
    pub fn par_map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(xs: &[T], f: &F) -> Vec<R> {
        xs.iter().map(f).collect()
    }

    /// The threads behind par-map and parallel loops, started on first use and kept for the
    /// whole run, one per core.
    #[cfg(not(target_arch = "wasm32"))]
    pub mod pool {
        use std::cell::Cell;
        use std::sync::{Condvar, Mutex, OnceLock};

        struct Job(*const (dyn Fn() + Sync));
        unsafe impl Send for Job {}
        // A parallel loop started on a pool thread (see `run_nested`): its work, and the idle
        // threads that joined it. Lives on the starting thread's stack; read under the lock.
        struct Nest { work: Job, helpers: Cell<usize> }
        struct NestPtr(*const Nest);
        unsafe impl Send for NestPtr {}
        // `base` threads are the ones a job normally runs on; a job may ask for more (`run_wide`),
        // which are started then and kept. `limit`: the threads (by index) the current job runs on.
        struct State { generation: u64, job: Option<Job>, running: usize, alive: usize, trapped: bool, base: usize, spawned: usize, limit: usize, nested: Vec<NestPtr> }
        struct Pool { state: Mutex<State>, start: Condvar, end: Condvar, busy: Mutex<()> }

        thread_local! { static INSIDE: Cell<bool> = const { Cell::new(false) }; }
        // On a pool thread: running a nested loop (as its starter or a helper), so loops inside it
        // stay sequential. HELPING: the nested loop this idle thread joined.
        thread_local! { static NESTED: Cell<bool> = const { Cell::new(false) }; }
        thread_local! { static HELPING: Cell<*const Nest> = const { Cell::new(std::ptr::null()) }; }
        // The nested loop this thread started and is running.
        thread_local! { static STARTED: Cell<*const Nest> = const { Cell::new(std::ptr::null()) }; }
        // Running a chunk of a loop with few iterations (each chunk is one or a few of them): the
        // threads that run out of chunks soon have nothing to do, so loops inside start nested.
        thread_local! { static FEW: Cell<bool> = const { Cell::new(false) }; }
        // Pool threads waiting for work.
        static IDLE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        fn get() -> &'static Pool {
            static POOL: OnceLock<&'static Pool> = OnceLock::new();
            POOL.get_or_init(|| {
                // AIL_THREADS caps the threads (for measuring); by default, one per core.
                let cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
                let wanted = std::env::var("AIL_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n >= 1);
                let workers = wanted.unwrap_or(cores);
                let workers = if workers <= 1 { 0 } else { workers };
                let p: &'static Pool = Box::leak(Box::new(Pool {
                    state: Mutex::new(State { generation: 0, job: None, running: 0, alive: workers, trapped: false, base: workers, spawned: workers, limit: workers, nested: Vec::new() }),
                    start: Condvar::new(),
                    end: Condvar::new(),
                    busy: Mutex::new(()),
                }));
                for i in 0..workers {
                    std::thread::spawn(move || worker(p, i, 0));
                }
                p
            })
        }

        fn worker(p: &'static Pool, idx: usize, mut seen: u64) {
            use std::sync::atomic::Ordering;
            INSIDE.with(|c| c.set(true));
            loop {
                let job = {
                    let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
                    let mut help: *const Nest = std::ptr::null();
                    while st.generation == seen {
                        if let Some(n) = st.nested.last() {
                            help = n.0;
                            unsafe { (*help).helpers.set((*help).helpers.get() + 1) };
                            break;
                        }
                        IDLE.fetch_add(1, Ordering::Relaxed);
                        st = p.start.wait(st).unwrap_or_else(|e| e.into_inner());
                        IDLE.fetch_sub(1, Ordering::Relaxed);
                    }
                    if !help.is_null() {
                        drop(st);
                        help_nested(p, help);
                        continue;
                    }
                    seen = st.generation;
                    if idx < st.limit { st.job.as_ref().map(|j| j.0) } else { None }
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

        // An idle thread runs the work of a nested loop with its starter; the work returns once
        // every chunk is taken, so the loop leaves the list then.
        fn help_nested(p: &'static Pool, n: *const Nest) {
            HELPING.with(|h| h.set(n));
            NESTED.with(|c| c.set(true));
            let f = unsafe { (*n).work.0 };
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*f)() }));
            NESTED.with(|c| c.set(false));
            HELPING.with(|h| h.set(std::ptr::null()));
            let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
            st.nested.retain(|x| x.0 != n);
            unsafe { (*n).helpers.set((*n).helpers.get() - 1) };
            p.end.notify_all();
        }

        /// On a pool thread outside a nested loop, with some thread idle or about to be: a loop
        /// here may start a nested one (`run_nested`).
        pub fn can_nest() -> bool {
            !NESTED.with(|c| c.get()) && (FEW.with(|c| c.get()) || IDLE.load(std::sync::atomic::Ordering::Relaxed) > 0)
        }

        /// Sets FEW for the chunks this thread runs; returns the value to put back.
        pub fn set_few(v: bool) -> bool {
            FEW.with(|c| c.replace(v))
        }

        /// Runs `work` on this pool thread, joined by the threads that are idle now or become
        /// idle before it ends; returns once every thread has left it. `work` must take its
        /// share of a common supply until it runs out (as the work of `par_for` does).
        pub fn run_nested(work: &(dyn Fn() + Sync)) {
            let p = get();
            let job: *const (dyn Fn() + Sync) = work;
            // As in `run_on`: the pointer is used only while this call waits below.
            let nest = Nest { work: Job(unsafe { std::mem::transmute::<*const (dyn Fn() + Sync + '_), *const (dyn Fn() + Sync + 'static)>(job) }), helpers: Cell::new(0) };
            let np: *const Nest = &nest;
            p.state.lock().unwrap_or_else(|e| e.into_inner()).nested.push(NestPtr(np));
            p.start.notify_all();
            NESTED.with(|c| c.set(true));
            STARTED.with(|c| c.set(np));
            work();
            STARTED.with(|c| c.set(std::ptr::null()));
            NESTED.with(|c| c.set(false));
            let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
            st.nested.retain(|x| x.0 != np);
            while nest.helpers.get() > 0 {
                st = p.end.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }

        /// Threads that run the work (the caller only waits).
        pub fn size() -> usize {
            let st = get().state.lock().unwrap_or_else(|e| e.into_inner());
            st.alive.min(st.base)
        }

        /// True on a pool thread: a parallel construct inside another runs sequentially there.
        pub fn inside() -> bool {
            INSIDE.with(|c| c.get())
        }

        /// Called by a worker that trapped inside a parallel loop, which then never returns: the
        /// pool stops waiting for it (the loop runs again on the caller, which traps the same way).
        pub fn abandon() {
            let p = get();
            let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
            let n = HELPING.with(|h| h.get());
            if !n.is_null() {
                // An idle thread helping a nested loop: it leaves that loop, not the pool's job.
                st.alive = st.alive.saturating_sub(1);
                st.nested.retain(|x| x.0 != n);
                unsafe { (*n).helpers.set((*n).helpers.get() - 1) };
                p.end.notify_all();
                return;
            }
            let n = STARTED.with(|h| h.get());
            if !n.is_null() {
                // The starter of a nested loop: its helpers may still use this thread's stack and
                // what the loop around it borrows, so they leave before the loop is given up.
                st.nested.retain(|x| x.0 != n);
                while unsafe { (*n).helpers.get() } > 0 {
                    st = p.end.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            }
            st.alive = st.alive.saturating_sub(1);
            st.running = st.running.saturating_sub(1);
            if st.running == 0 {
                p.end.notify_all();
            }
        }

        /// Runs `work` on every pool thread and returns when all have finished, so `work` may
        /// borrow from the caller's stack. False (and nothing ran) if the pool is in use by
        /// another thread or has no threads.
        pub fn run(work: &(dyn Fn() + Sync)) -> bool {
            run_on(work, 0)
        }

        /// `run` on `threads` threads (at least the usual ones), starting the missing ones.
        pub fn run_wide(work: &(dyn Fn() + Sync), threads: usize) -> bool {
            run_on(work, threads)
        }

        fn run_on(work: &(dyn Fn() + Sync), threads: usize) -> bool {
            let p = get();
            let Ok(_busy) = p.busy.try_lock() else {
                return false;
            };
            {
                let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
                if st.alive == 0 {
                    return false;
                }
                st.limit = threads.max(st.base);
                while st.spawned < st.limit {
                    let (i, g) = (st.spawned, st.generation);
                    std::thread::spawn(move || worker(p, i, g));
                    st.spawned += 1;
                    st.alive += 1;
                }
                // The pointer outlives this call only in the pool's state, which is cleared below
                // before returning; workers use it only while `running` counts them.
                let job: *const (dyn Fn() + Sync) = work;
                st.job = Some(Job(unsafe { std::mem::transmute::<*const (dyn Fn() + Sync + '_), *const (dyn Fn() + Sync + 'static)>(job) }));
                st.running = st.alive;
                st.trapped = false;
                st.generation += 1;
            }
            p.start.notify_all();
            let mut st = p.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.running > 0 {
                st = p.end.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            st.job = None;
            let trapped = st.trapped;
            drop(st);
            if trapped {
                panic!("a parallel worker trapped");
            }
            true
        }
    }

    // Set on a pool thread while it runs a chunk of a parallel loop: a trap there marks the loop
    // as failed and parks the thread instead of ending the program, and the caller runs the
    // whole loop again sequentially, which reaches the first trap in program order.
    thread_local! { pub static PAR_TRAP: std::cell::Cell<*const std::sync::atomic::AtomicBool> = const { std::cell::Cell::new(std::ptr::null()) }; }
    // The flag of the loop around a nested one on the same thread (see `pool::run_nested`).
    thread_local! { pub static PAR_TRAP_UP: std::cell::Cell<*const std::sync::atomic::AtomicBool> = const { std::cell::Cell::new(std::ptr::null()) }; }

    /// Worth running `lo..hi` in parallel: enough iterations (any count above one when the body
    /// itself loops or calls functions), threads to spare, and not already on a pool thread
    /// unless some thread is idle there (a nested loop, one level deep).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn par_worth(lo: i64, hi: i64, heavy: bool) -> bool {
        let n = hi as i128 - lo as i128;
        if n < 2 || (!heavy && n < 1 << 15) || SOFT_TRAP.with(|s| s.get()) {
            return false;
        }
        if pool::inside() {
            return pool::can_nest();
        }
        pool::size() > 1
    }
    #[cfg(target_arch = "wasm32")]
    pub fn par_worth(_lo: i64, _hi: i64, _heavy: bool) -> bool { false }

    /// Runs `f` over consecutive chunks of `lo..hi` on the pool and returns the chunk results in
    /// order, or None if the pool was not available or some chunk trapped (then nothing the
    /// caller can see has changed except writes the sequential run repeats identically).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn par_for<R: Send>(lo: i64, hi: i64, chunks: usize, f: &(dyn Fn(i64, i64) -> R + Sync)) -> Option<Vec<R>> {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let n = (hi - lo) as usize;
        let threads = pool::size();
        // chunks: 0 = several per thread (balances uneven work), 1 = one per thread (for big
        // private state), 2 = like 0 but never more threads than cores (big state per chunk).
        let k = (match chunks { 1 => threads, _ => threads * 8 }).clamp(1, n.max(1));
        // A few heavy iterations that do not split evenly over the cores (16 on 12): with one
        // thread each, the cores share them to the end instead of the last ones running alone.
        let wide = chunks == 0 && n > threads && n % threads != 0 && n <= threads * 4;
        let trapped = AtomicBool::new(false);
        let next = AtomicUsize::new(0);
        let mut out: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(k);
        unsafe { out.set_len(k) };
        let done: Vec<AtomicBool> = (0..k).map(|_| AtomicBool::new(false)).collect();
        struct Slots<R>(*mut std::mem::MaybeUninit<R>);
        unsafe impl<R: Send> Sync for Slots<R> {}
        impl<R> Slots<R> {
            #[inline(always)]
            unsafe fn put(&self, c: usize, r: R) { (*self.0.add(c)).write(r); }
        }
        let slots = Slots(out.as_mut_ptr());
        let nested = pool::inside();
        let few = !nested && n <= threads * 4;
        let work = || {
            let few_was = pool::set_few(few);
            let up = PAR_TRAP.with(|t| t.get());
            PAR_TRAP.with(|t| t.set(&trapped));
            PAR_TRAP_UP.with(|t| t.set(up));
            loop {
                if trapped.load(Ordering::Relaxed) {
                    break;
                }
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= k {
                    break;
                }
                let a = lo + (n * c / k) as i64;
                let b = lo + (n * (c + 1) / k) as i64;
                // Where traps unwind (tests, servers), the trap ends here; elsewhere the panic
                // hook parks this thread before any unwinding.
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(a, b))) {
                    Ok(r) => {
                        // Each chunk index is claimed once, so each slot is written once.
                        unsafe { slots.put(c, r) };
                        done[c].store(true, Ordering::Release);
                    }
                    Err(_) => trapped.store(true, Ordering::Release),
                }
            }
            PAR_TRAP.with(|t| t.set(up));
            PAR_TRAP_UP.with(|t| t.set(std::ptr::null()));
            pool::set_few(few_was);
        };
        let ran = if nested {
            pool::run_nested(&work);
            true
        } else if wide {
            pool::run_wide(&work, n)
        } else {
            pool::run(&work)
        };
        let ok = ran && !trapped.load(Ordering::Acquire);
        if !ok {
            for (c, d) in done.iter().enumerate() {
                if d.load(Ordering::Acquire) {
                    unsafe { out[c].assume_init_drop() };
                }
            }
            return None;
        }
        Some(out.into_iter().map(|x| unsafe { x.assume_init() }).collect())
    }
    #[cfg(target_arch = "wasm32")]
    pub fn par_for<R: Send>(_lo: i64, _hi: i64, _chunks: usize, _f: &(dyn Fn(i64, i64) -> R + Sync)) -> Option<Vec<R>> { None }

    /// A vector that a parallel loop writes from several threads, each element by one
    /// iteration only (the compiler proves the writes disjoint before using it).
    pub struct SharedSlice<T> { ptr: *mut T, len: usize }
    unsafe impl<T: Send> Sync for SharedSlice<T> {}
    unsafe impl<T: Send> Send for SharedSlice<T> {}
    impl<T> SharedSlice<T> {
        pub fn new(s: &mut [T]) -> Self { SharedSlice { ptr: s.as_mut_ptr(), len: s.len() } }
        #[track_caller] #[inline(always)] #[allow(clippy::mut_from_ref)]
        pub fn at(&self, i: i64) -> &mut T {
            if i as u64 >= self.len as u64 { out_of_bounds(i, self.len) }
            unsafe { &mut *self.ptr.add(i as usize) }
        }
        #[inline(always)]
        pub fn ptr(&self) -> *mut T { self.ptr }
        #[inline(always)]
        pub fn len(&self) -> usize { self.len }
    }

    /// Writes of iteration i fall in [i*s + lo, i*s + hi] for each window (lo, hi), empty when
    /// lo > hi: iterations write disjoint elements if the windows together span less than s.
    pub fn disjoint(s: i128, ws: &[(i128, i128)]) -> bool {
        let (mut lo, mut hi, mut any) = (i128::MAX, i128::MIN, false);
        for &(a, b) in ws {
            if a <= b {
                any = true;
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        !any || (s > 0 && hi - lo < s)
    }

    /// Magnitude of an integer, for the bound that proves a parallel sum cannot overflow.
    pub trait UAbs: Copy { fn uabs(self) -> u64; fn wide(self) -> i128; }
    macro_rules! uabs_impl { ($($t:ty),*) => { $(
        impl UAbs for $t {
            #[inline(always)] fn uabs(self) -> u64 { (self as i128).unsigned_abs() as u64 }
            #[inline(always)] fn wide(self) -> i128 { self as i128 }
        }
    )* } }
    uabs_impl!(i8, i16, i32, i64, u8, u16, u32, u64);

    /// Sorts `v` on all cores when it is long: chunks are sorted in parallel, then merged pairwise,
    /// each merge split among the threads at points found by binary search. Ties keep their
    /// order (an element of the left run goes first), so the result is the one a stable
    /// sequential sort gives; `stable` false allows an unstable sort of each chunk, for element
    /// types where equal means identical. `cmp` must not trap.
    pub fn psort<T: Send + Sync>(v: &mut Vec<T>, stable: bool, cmp: impl Fn(&T, &T) -> std::cmp::Ordering + Sync) {
        let n = v.len();
        #[cfg(not(target_arch = "wasm32"))]
        let threads = if n >= 1 << 16 && !pool::inside() { pool::size() } else { 1 };
        #[cfg(target_arch = "wasm32")]
        let threads = 1;
        if threads <= 1 {
            if stable { v.sort_by(cmp) } else { v.sort_unstable_by(cmp) }
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            struct P<T>(*mut T);
            unsafe impl<T> Sync for P<T> {}
            impl<T> P<T> {
                #[inline(always)]
                fn at(&self, i: usize) -> *mut T { unsafe { self.0.add(i) } }
            }
            // Runs task(0..count) on the pool, or here if the pool is taken.
            fn tasks(count: usize, task: &(dyn Fn(usize) + Sync)) {
                let next = AtomicUsize::new(0);
                let work = || loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= count {
                        break;
                    }
                    task(t);
                };
                if !pool::run(&work) {
                    work();
                }
            }
            let k = threads * 2;
            let bounds: Vec<usize> = (0..=k).map(|c| n * c / k).collect();
            let a = P(v.as_mut_ptr());
            tasks(k, &|c| {
                let s = unsafe { std::slice::from_raw_parts_mut(a.at(bounds[c]), bounds[c + 1] - bounds[c]) };
                if stable { s.sort_by(&cmp) } else { s.sort_unstable_by(&cmp) }
            });
            let mut tmp: Vec<T> = Vec::with_capacity(n);
            let b = P(tmp.as_mut_ptr());
            // Runs, as boundaries; each round merges neighbours from one buffer into the other.
            let mut runs = bounds;
            let mut in_v = true;
            while runs.len() > 2 {
                let (src, dst) = if in_v { (&a, &b) } else { (&b, &a) };
                let mut next_runs = vec![0];
                let mut jobs: Vec<(usize, usize, usize)> = Vec::new(); // (lo, mid, hi)
                let mut r = 0;
                while r + 1 < runs.len() {
                    if r + 2 < runs.len() {
                        jobs.push((runs[r], runs[r + 1], runs[r + 2]));
                        next_runs.push(runs[r + 2]);
                        r += 2;
                    } else {
                        jobs.push((runs[r], runs[r + 1], runs[r + 1]));
                        next_runs.push(runs[r + 1]);
                        r += 1;
                    }
                }
                // Each merge is split into pieces of about n / (2 * threads) output elements.
                let piece = (n / (threads * 2)).max(1 << 12);
                let mut pieces: Vec<(usize, usize, usize, usize)> = Vec::new(); // (job, k0, k1)
                for (jx, &(lo, _, hi)) in jobs.iter().enumerate() {
                    let len = hi - lo;
                    let m = len.div_ceil(piece).max(1);
                    for p in 0..m {
                        pieces.push((jx, len * p / m, len * (p + 1) / m, 0));
                    }
                }
                tasks(pieces.len(), &|t| {
                    let (jx, k0, k1, _) = pieces[t];
                    let (lo, mid, hi) = jobs[jx];
                    let x = unsafe { std::slice::from_raw_parts(src.at(lo) as *const T, mid - lo) };
                    let y = unsafe { std::slice::from_raw_parts(src.at(mid) as *const T, hi - mid) };
                    // How many of the first k outputs come from x: the largest i with
                    // x[i-1] not after y[k-i].
                    let co = |kk: usize| {
                        let (mut l, mut h) = (kk.saturating_sub(y.len()), kk.min(x.len()));
                        while l < h {
                            let i = (l + h + 1) / 2;
                            let j = kk - i;
                            if j < y.len() && cmp(&y[j], &x[i - 1]) == std::cmp::Ordering::Less { h = i - 1 } else { l = i }
                        }
                        l
                    };
                    let (mut i, i1) = (co(k0), co(k1));
                    let (mut j, j1) = (k0 - i, k1 - i1);
                    let mut o = lo + k0;
                    unsafe {
                        while i < i1 && j < j1 {
                            if cmp(&y[j], &x[i]) == std::cmp::Ordering::Less {
                                std::ptr::copy_nonoverlapping(&y[j], dst.at(o), 1);
                                j += 1;
                            } else {
                                std::ptr::copy_nonoverlapping(&x[i], dst.at(o), 1);
                                i += 1;
                            }
                            o += 1;
                        }
                        std::ptr::copy_nonoverlapping(x.as_ptr().add(i), dst.at(o), i1 - i);
                        o += i1 - i;
                        std::ptr::copy_nonoverlapping(y.as_ptr().add(j), dst.at(o), j1 - j);
                    }
                });
                runs = next_runs;
                in_v = !in_v;
            }
            if !in_v {
                // The elements were moved into tmp; move them back (tmp keeps length 0).
                unsafe { std::ptr::copy_nonoverlapping(b.0 as *const T, a.0, n) };
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
            Box::new(SplitByte::new(s, sep.as_bytes()[0]))
        } else {
            Box::new(s.split(sep))
        }
    }
    pub fn split(s: &str, sep: &str) -> Vec<String> {
        if sep.is_empty() { return chars(s); }
        if sep.len() == 1 { return SplitByte::new(s, sep.as_bytes()[0]).map(String::from).collect(); }
        s.split(sep).map(String::from).collect()
    }
    /// `split` as borrowed pieces of `s`, for a Vec of views (see `vec_views` in the emitter).
    pub fn split_ref<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
        if sep.is_empty() { return s.char_indices().map(|(i, c)| &s[i..i + c.len_utf8()]).collect(); }
        if sep.len() == 1 { return SplitByte::new(s, sep.as_bytes()[0]).collect(); }
        s.split(sep).collect()
    }
    /// The pieces of `s` between occurrences of the byte `b`, the same as `str::split` on a
    /// one-byte (so ASCII, so never inside a longer character) separator. Fields are usually a
    /// few bytes long, where a plain loop beats the searcher's set-up; a long one goes to it.
    pub struct SplitByte<'a> { s: &'a str, pos: usize, b: u8, done: bool }
    impl<'a> SplitByte<'a> {
        #[inline]
        pub fn new(s: &'a str, b: u8) -> Self { SplitByte { s, pos: 0, b, done: false } }
    }
    impl<'a> Iterator for SplitByte<'a> {
        type Item = &'a str;
        #[inline]
        fn next(&mut self) -> Option<&'a str> {
            if self.done { return None; }
            let by = self.s.as_bytes();
            let start = self.pos;
            let mut i = start;
            let near = by.len().min(start + 32);
            while i < near && by[i] != self.b { i += 1; }
            if i == near && near < by.len() {
                i = match self.s[near..].find(self.b as char) { Some(k) => near + k, None => by.len() };
            }
            // Safety: `b` is ASCII, so both ends are character boundaries.
            let piece = unsafe { self.s.get_unchecked(start..i) };
            if i >= by.len() { self.done = true } else { self.pos = i + 1 }
            Some(piece)
        }
    }
    #[track_caller]
    pub fn from_code(n: i64) -> String {
        match u32::try_from(n).ok().and_then(char::from_u32) { Some(c) => c.to_string(), None => panic!("{n} is not a valid code point") }
    }
    pub fn chars(s: &str) -> Vec<String> { s.chars().map(|c| c.to_string()).collect() }
    #[track_caller]
    pub fn byte_at(s: &str, i: i64) -> i64 {
        match s.as_bytes().get(usize::try_from(i as u64).unwrap_or(usize::MAX)) {
            Some(b) => *b as i64,
            None => {
                if i < 0 { neg_index(i) }
                panic!("byte index {i} out of bounds for string of {} bytes", s.len())
            }
        }
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
    /// `lines` as borrowed pieces of `s`, for a Vec of views.
    pub fn lines_ref(s: &str) -> Vec<&str> {
        s.lines().map(|l| l.trim_end_matches('\r')).collect()
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
            // A trap inside a parallel loop only makes the loop run again sequentially.
            if !PAR_TRAP.with(|t| t.get()).is_null() || CHUNK_QUIET.with(|q| q.get()) {
                return;
            }
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
