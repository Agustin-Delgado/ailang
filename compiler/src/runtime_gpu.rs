//! Runtime support for loops that run on the GPU: a small Vulkan compute layer loaded at run
//! time (the program has no link-time dependency on a GPU driver), device buffers that mirror
//! vectors, and kernel launches. Anything that goes wrong (no driver, no suitable device, an
//! allocation that fails) turns the GPU off for the rest of the run and the loops run on the CPU.

pub const RUNTIME_GPU: &str = r####"
#[allow(dead_code, non_snake_case, non_upper_case_globals, clippy::all)]
pub mod ail_gpu {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};

    type Ptr = *mut c_void;
    type Nd = u64;

    #[cfg(windows)]
    mod dl {
        use std::ffi::c_void;
        extern "system" {
            fn LoadLibraryA(n: *const u8) -> *mut c_void;
            fn GetProcAddress(m: *mut c_void, n: *const u8) -> *mut c_void;
        }
        pub fn open() -> *mut c_void { unsafe { LoadLibraryA(b"vulkan-1.dll\0".as_ptr()) } }
        pub fn sym(lib: *mut c_void, name: &[u8]) -> *mut c_void { unsafe { GetProcAddress(lib, name.as_ptr()) } }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    mod dl {
        use std::ffi::c_void;
        extern "C" {
            fn dlopen(f: *const u8, flags: i32) -> *mut c_void;
            fn dlsym(h: *mut c_void, s: *const u8) -> *mut c_void;
        }
        pub fn open() -> *mut c_void {
            unsafe {
                let h = dlopen(b"libvulkan.so.1\0".as_ptr(), 2);
                if h.is_null() { dlopen(b"libvulkan.so\0".as_ptr(), 2) } else { h }
            }
        }
        pub fn sym(lib: *mut c_void, name: &[u8]) -> *mut c_void { unsafe { dlsym(lib, name.as_ptr()) } }
    }
    #[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
    mod dl {
        use std::ffi::c_void;
        pub fn open() -> *mut c_void { std::ptr::null_mut() }
        pub fn sym(_lib: *mut c_void, _name: &[u8]) -> *mut c_void { std::ptr::null_mut() }
    }

    // ---- The part of the Vulkan API used here (layouts for 64-bit targets). ----
    #[repr(C)] struct AppInfo { s_type: u32, p_next: *const c_void, app: *const u8, app_v: u32, engine: *const u8, engine_v: u32, api: u32 }
    #[repr(C)] struct InstanceCI { s_type: u32, p_next: *const c_void, flags: u32, app: *const AppInfo, nl: u32, l: *const *const u8, ne: u32, e: *const *const u8 }
    #[repr(C)] struct QueueCI { s_type: u32, p_next: *const c_void, flags: u32, family: u32, count: u32, prio: *const f32 }
    #[repr(C)] struct DeviceCI { s_type: u32, p_next: *const c_void, flags: u32, nq: u32, q: *const QueueCI, nl: u32, l: *const *const u8, ne: u32, e: *const *const u8, features: *const c_void }
    #[repr(C)] struct Features2 { s_type: u32, p_next: *mut c_void, f: [u32; 55] }
    #[repr(C)] struct Props2 { s_type: u32, p_next: *mut c_void, p: [u64; 104] }
    #[repr(C)] struct AnyStruct { s_type: u32, p_next: *mut c_void, rest: [u64; 128] }
    #[repr(C)] struct BufferCI { s_type: u32, p_next: *const c_void, flags: u32, size: u64, usage: u32, sharing: u32, nq: u32, q: *const u32 }
    #[repr(C)] struct MemReq { size: u64, align: u64, bits: u32 }
    #[repr(C)] struct MemAI { s_type: u32, p_next: *const c_void, size: u64, index: u32 }
    #[repr(C)] struct ShaderCI { s_type: u32, p_next: *const c_void, flags: u32, size: usize, code: *const u32 }
    #[repr(C)] struct DslBinding { binding: u32, ty: u32, count: u32, stages: u32, samplers: *const c_void }
    #[repr(C)] struct DslCI { s_type: u32, p_next: *const c_void, flags: u32, n: u32, b: *const DslBinding }
    #[repr(C)] struct PlCI { s_type: u32, p_next: *const c_void, flags: u32, nsets: u32, sets: *const Nd, npc: u32, pc: *const c_void }
    #[repr(C)] struct StageCI { s_type: u32, p_next: *const c_void, flags: u32, stage: u32, module: Nd, name: *const u8, spec: *const c_void }
    #[repr(C)] struct ComputeCI { s_type: u32, p_next: *const c_void, flags: u32, stage: StageCI, layout: Nd, base: Nd, base_i: i32 }
    #[repr(C)] struct PoolSize { ty: u32, n: u32 }
    #[repr(C)] struct DpCI { s_type: u32, p_next: *const c_void, flags: u32, max: u32, n: u32, sizes: *const PoolSize }
    #[repr(C)] struct DsAI { s_type: u32, p_next: *const c_void, pool: Nd, n: u32, layouts: *const Nd }
    #[repr(C)] struct DescBuf { buffer: Nd, offset: u64, range: u64 }
    #[repr(C)] struct WriteDS { s_type: u32, p_next: *const c_void, set: Nd, binding: u32, elem: u32, n: u32, ty: u32, img: *const c_void, buf: *const DescBuf, texel: *const c_void }
    #[repr(C)] struct CmdPoolCI { s_type: u32, p_next: *const c_void, flags: u32, family: u32 }
    #[repr(C)] struct CbAI { s_type: u32, p_next: *const c_void, pool: Nd, level: u32, n: u32 }
    #[repr(C)] struct CbBI { s_type: u32, p_next: *const c_void, flags: u32, inherit: *const c_void }
    #[repr(C)] struct SubmitInfo { s_type: u32, p_next: *const c_void, nw: u32, w: *const Nd, ws: *const u32, ncb: u32, cb: *const Ptr, ns: u32, s: *const Nd }
    #[repr(C)] struct FenceCI { s_type: u32, p_next: *const c_void, flags: u32 }
    #[repr(C)] struct BufCopy { src: u64, dst: u64, size: u64 }
    #[repr(C)] struct MemBarrier { s_type: u32, p_next: *const c_void, src: u32, dst: u32 }

    const SHADER_READ: u32 = 0x20;
    const SHADER_WRITE: u32 = 0x40;
    const TRANSFER_READ: u32 = 0x800;
    const TRANSFER_WRITE: u32 = 0x1000;
    const HOST_READ: u32 = 0x2000;
    const HOST_WRITE: u32 = 0x4000;
    const ST_TRANSFER: u32 = 0x1000;
    const ST_COMPUTE: u32 = 0x800;
    const ST_HOST: u32 = 0x4000;

    macro_rules! vkfns { ($($name:ident: fn($($a:ty),*) $(-> $r:ty)?;)*) => {
        struct Fns { $($name: unsafe extern "system" fn($($a),*) $(-> $r)?,)* }
        impl Fns {
            unsafe fn load(get: &dyn Fn(&[u8]) -> Ptr) -> Option<Fns> {
                Some(Fns { $($name: {
                    let p = get(concat!(stringify!($name), "\0").as_bytes());
                    if p.is_null() { return None; }
                    std::mem::transmute::<Ptr, unsafe extern "system" fn($($a),*) $(-> $r)?>(p)
                },)* })
            }
        }
    } }
    vkfns! {
        vkEnumeratePhysicalDevices: fn(Ptr, *mut u32, *mut Ptr) -> i32;
        vkGetPhysicalDeviceProperties: fn(Ptr, *mut u64);
        vkGetPhysicalDeviceQueueFamilyProperties: fn(Ptr, *mut u32, *mut [u32; 6]);
        vkGetPhysicalDeviceMemoryProperties: fn(Ptr, *mut u64);
        vkGetPhysicalDeviceFeatures2: fn(Ptr, *mut Features2);
        vkGetPhysicalDeviceProperties2: fn(Ptr, *mut Props2);
        vkCreateDevice: fn(Ptr, *const DeviceCI, *const c_void, *mut Ptr) -> i32;
        vkGetDeviceProcAddr: fn(Ptr, *const u8) -> Ptr;
    }
    macro_rules! devfns { ($($name:ident: fn($($a:ty),*) $(-> $r:ty)?;)*) => {
        struct Dev { $($name: unsafe extern "system" fn($($a),*) $(-> $r)?,)* }
        impl Dev {
            unsafe fn load(get: &dyn Fn(&[u8]) -> Ptr) -> Option<Dev> {
                Some(Dev { $($name: {
                    let p = get(concat!(stringify!($name), "\0").as_bytes());
                    if p.is_null() { return None; }
                    std::mem::transmute::<Ptr, unsafe extern "system" fn($($a),*) $(-> $r)?>(p)
                },)* })
            }
        }
    } }
    devfns! {
        vkGetDeviceQueue: fn(Ptr, u32, u32, *mut Ptr);
        vkCreateBuffer: fn(Ptr, *const BufferCI, *const c_void, *mut Nd) -> i32;
        vkDestroyBuffer: fn(Ptr, Nd, *const c_void);
        vkGetBufferMemoryRequirements: fn(Ptr, Nd, *mut MemReq);
        vkAllocateMemory: fn(Ptr, *const MemAI, *const c_void, *mut Nd) -> i32;
        vkFreeMemory: fn(Ptr, Nd, *const c_void);
        vkBindBufferMemory: fn(Ptr, Nd, Nd, u64) -> i32;
        vkMapMemory: fn(Ptr, Nd, u64, u64, u32, *mut Ptr) -> i32;
        vkCreateShaderModule: fn(Ptr, *const ShaderCI, *const c_void, *mut Nd) -> i32;
        vkCreateDescriptorSetLayout: fn(Ptr, *const DslCI, *const c_void, *mut Nd) -> i32;
        vkCreatePipelineLayout: fn(Ptr, *const PlCI, *const c_void, *mut Nd) -> i32;
        vkCreateComputePipelines: fn(Ptr, Nd, u32, *const ComputeCI, *const c_void, *mut Nd) -> i32;
        vkCreateDescriptorPool: fn(Ptr, *const DpCI, *const c_void, *mut Nd) -> i32;
        vkAllocateDescriptorSets: fn(Ptr, *const DsAI, *mut Nd) -> i32;
        vkUpdateDescriptorSets: fn(Ptr, u32, *const WriteDS, u32, *const c_void);
        vkResetDescriptorPool: fn(Ptr, Nd, u32) -> i32;
        vkCreateCommandPool: fn(Ptr, *const CmdPoolCI, *const c_void, *mut Nd) -> i32;
        vkAllocateCommandBuffers: fn(Ptr, *const CbAI, *mut Ptr) -> i32;
        vkResetCommandBuffer: fn(Ptr, u32) -> i32;
        vkBeginCommandBuffer: fn(Ptr, *const CbBI) -> i32;
        vkEndCommandBuffer: fn(Ptr) -> i32;
        vkCmdBindPipeline: fn(Ptr, u32, Nd);
        vkCmdBindDescriptorSets: fn(Ptr, u32, Nd, u32, u32, *const Nd, u32, *const u32);
        vkCmdDispatch: fn(Ptr, u32, u32, u32);
        vkCmdCopyBuffer: fn(Ptr, Nd, Nd, u32, *const BufCopy);
        vkCmdFillBuffer: fn(Ptr, Nd, u64, u64, u32);
        vkCmdPipelineBarrier: fn(Ptr, u32, u32, u32, u32, *const MemBarrier, u32, *const c_void, u32, *const c_void);
        vkQueueSubmit: fn(Ptr, u32, *const SubmitInfo, Nd) -> i32;
        vkCreateFence: fn(Ptr, *const FenceCI, *const c_void, *mut Nd) -> i32;
        vkWaitForFences: fn(Ptr, u32, *const Nd, u32, u64) -> i32;
        vkGetFenceStatus: fn(Ptr, Nd) -> i32;
        vkResetFences: fn(Ptr, u32, *const Nd) -> i32;
    }

    /// A device buffer and its memory.
    struct Buf { buf: Nd, mem: Nd, size: u64, host: bool }

    struct Pipe { pipe: Nd, layout: Nd, set: Nd }

    struct Gpu {
        dev: Ptr,
        d: Dev,
        queue: Ptr,
        cmd: Ptr,
        fence: Nd,
        dsl_cache: std::collections::HashMap<u32, (Nd, Nd)>, // buffers -> (set layout, pipeline layout)
        pool: Nd,
        local_type: u32,
        host_type: u32,
        staging: Option<(Buf, Ptr)>,
        io: Option<(Buf, Ptr)>,
        io_dev: Option<Buf>,
        pipes: std::collections::HashMap<usize, Option<Pipe>>,
        f64_ok: bool,
        dead: bool,
        /// Freed device buffers kept for reuse (allocation is slow): at most POOL bytes.
        bufpool: RefCell<Vec<Buf>>,
        pool_bytes: std::cell::Cell<u64>,
        slab: std::cell::Cell<(Nd, u64)>,
        /// Work recorded but not yet submitted (only inside a chunk, see `chunk_begin`): each
        /// launch has its own segment of the argument buffers (`segs`, in words) and a
        /// descriptor set from `tpool`; buffers freed meanwhile wait in `deferred`.
        tpool: Nd,
        rec: bool,
        cursor: u64,
        segs: Vec<u64>,
        tsets: u32,
        deferred: Vec<Buf>,
    }
    const POOL: u64 = 1 << 30;
    /// Buffers up to SMALL bytes are carved out of blocks of SLAB bytes.
    const SMALL: u64 = 1 << 20;
    const SLAB: u64 = 64 << 20;
    unsafe impl Send for Gpu {}

    /// What a kernel needs to know about the device to use it.
    static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    static INIT: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

    /// AIL_GPU_LOG=1 prints what the GPU does (for measuring and debugging).
    pub fn log(msg: &str) {
        static L: OnceLock<bool> = OnceLock::new();
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        if *L.get_or_init(|| std::env::var("AIL_GPU_LOG").is_ok()) {
            let t = T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0;
            eprintln!("[gpu] {t:9.2}ms {msg}");
        }
    }

    fn mode() -> u8 {
        // AIL_GPU=0 turns the GPU off, AIL_GPU=force sends every loop that can go there.
        static M: OnceLock<u8> = OnceLock::new();
        *M.get_or_init(|| match std::env::var("AIL_GPU").as_deref() {
            Ok("0") | Ok("off") => 0,
            Ok("force") => 2,
            _ => 1,
        })
    }

    /// Starts opening the device on another thread, so that it overlaps with whatever the
    /// program does first (reading its input, usually).
    pub fn start() {
        if mode() == 0 {
            return;
        }
        STARTED.call_once(|| {
            let h = std::thread::spawn(|| {
                let t = std::time::Instant::now();
                let _ = GPU.get_or_init(|| unsafe { open() }.map(Mutex::new));
                log(&format!("opened in {:?}", t.elapsed()));
            });
            *INIT.lock().unwrap_or_else(|e| e.into_inner()) = Some(h);
        });
    }

    fn gpu() -> Option<&'static Mutex<Gpu>> {
        let r = gpu_inner();
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| log(if r.is_some() { "device open" } else { "no device" }));
        r
    }
    fn gpu_inner() -> Option<&'static Mutex<Gpu>> {
        if mode() == 0 {
            return None;
        }
        start();
        if let Some(h) = INIT.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
        GPU.get_or_init(|| unsafe { open() }.map(Mutex::new)).as_ref()
    }

    /// True once the device is open and usable (waits for the opening started by `start`).
    pub fn ready() -> bool {
        match gpu() {
            Some(g) => !g.lock().map(|g| g.dead).unwrap_or(true),
            None => false,
        }
    }

    /// Worth sending `n` iterations of a body of about `cost` operations to the GPU.
    pub fn worth(n: i64, cost: u64) -> bool {
        match mode() {
            0 => false,
            2 => n > 0,
            _ => n >= 4096 && (n as u64).saturating_mul(cost.max(1)) >= 1 << 22,
        }
    }

    /// `worth`, or yes when some of the kernel's vectors are current only on the device (running
    /// on the CPU would first download them).
    pub fn worth_at(n: i64, cost: u64, ms: &[&Mirror]) -> bool {
        if mode() == 0 || n <= 0 {
            return false;
        }
        ms.iter().any(|m| !m.on_cpu()) || worth(n, cost)
    }

    unsafe fn open() -> Option<Gpu> {
        let lib = dl::open();
        if lib.is_null() {
            return None;
        }
        let gipa_p = dl::sym(lib, b"vkGetInstanceProcAddr\0");
        if gipa_p.is_null() {
            return None;
        }
        let gipa: unsafe extern "system" fn(Ptr, *const u8) -> Ptr = std::mem::transmute(gipa_p);
        let create_p = gipa(std::ptr::null_mut(), b"vkCreateInstance\0".as_ptr());
        if create_p.is_null() {
            return None;
        }
        let create: unsafe extern "system" fn(*const InstanceCI, *const c_void, *mut Ptr) -> i32 = std::mem::transmute(create_p);
        let app = AppInfo { s_type: 0, p_next: std::ptr::null(), app: b"ail\0".as_ptr(), app_v: 0, engine: b"ail\0".as_ptr(), engine_v: 0, api: (1 << 22) | (2 << 12) };
        let ici = InstanceCI { s_type: 1, p_next: std::ptr::null(), flags: 0, app: &app, nl: 0, l: std::ptr::null(), ne: 0, e: std::ptr::null() };
        let mut inst: Ptr = std::ptr::null_mut();
        if create(&ici, std::ptr::null(), &mut inst) != 0 || inst.is_null() {
            return None;
        }
        let f = Fns::load(&|n| gipa(inst, n.as_ptr()))?;
        let mut count = 0u32;
        (f.vkEnumeratePhysicalDevices)(inst, &mut count, std::ptr::null_mut());
        if count == 0 {
            return None;
        }
        let mut pds = vec![std::ptr::null_mut(); count as usize];
        (f.vkEnumeratePhysicalDevices)(inst, &mut count, pds.as_mut_ptr());
        // A discrete GPU, or else an integrated one; never a software rasterizer.
        let mut best: Option<(Ptr, u32)> = None;
        for &pd in &pds {
            let mut props = [0u64; 104];
            (f.vkGetPhysicalDeviceProperties)(pd, props.as_mut_ptr());
            let ty = *(props.as_ptr() as *const u32).add(4);
            let rank = match ty { 2 => 3, 1 => 2, 3 => 1, _ => 0 };
            if rank > 0 && best.map_or(true, |b| rank > b.1) {
                best = Some((pd, rank));
            }
        }
        let (pd, _) = best?;
        let mut feats = Features2 { s_type: 1000059000, p_next: std::ptr::null_mut(), f: [0; 55] };
        (f.vkGetPhysicalDeviceFeatures2)(pd, &mut feats);
        let int64 = feats.f[40] != 0;
        if !int64 {
            return None;
        }
        // Floats only where results match the CPU exactly: denormals kept, round to nearest.
        let mut p12 = AnyStruct { s_type: 52, p_next: std::ptr::null_mut(), rest: [0; 128] };
        let mut props2 = Props2 { s_type: 1000059001, p_next: &mut p12 as *mut AnyStruct as *mut c_void, p: [0; 104] };
        (f.vkGetPhysicalDeviceProperties2)(pd, &mut props2);
        let p12b = &p12 as *const AnyStruct as *const u8;
        let rd = |off: usize| *(p12b.add(off) as *const u32) != 0;
        let f64_ok = feats.f[39] != 0 && rd(552) && rd(564) && rd(588);
        let mut nf = 0u32;
        (f.vkGetPhysicalDeviceQueueFamilyProperties)(pd, &mut nf, std::ptr::null_mut());
        let mut fams = vec![[0u32; 6]; nf as usize];
        (f.vkGetPhysicalDeviceQueueFamilyProperties)(pd, &mut nf, fams.as_mut_ptr());
        let family = fams.iter().position(|q| q[0] & 2 != 0 && q[1] > 0)? as u32;
        let prio = 1.0f32;
        let qci = QueueCI { s_type: 2, p_next: std::ptr::null(), flags: 0, family, count: 1, prio: &prio };
        let mut want = [0u32; 55];
        want[40] = 1;
        if f64_ok {
            want[39] = 1;
        }
        let dci = DeviceCI { s_type: 3, p_next: std::ptr::null(), flags: 0, nq: 1, q: &qci, nl: 0, l: std::ptr::null(), ne: 0, e: std::ptr::null(), features: want.as_ptr() as *const c_void };
        let mut dev: Ptr = std::ptr::null_mut();
        if (f.vkCreateDevice)(pd, &dci, std::ptr::null(), &mut dev) != 0 || dev.is_null() {
            return None;
        }
        let gdpa = f.vkGetDeviceProcAddr;
        let d = Dev::load(&|n| gdpa(dev, n.as_ptr()))?;
        let mut queue: Ptr = std::ptr::null_mut();
        (d.vkGetDeviceQueue)(dev, family, 0, &mut queue);
        // Memory: device-local for vectors, host-visible and coherent for transfers.
        let mut mp = [0u64; 65];
        (f.vkGetPhysicalDeviceMemoryProperties)(pd, mp.as_mut_ptr());
        let mpb = mp.as_ptr() as *const u32;
        let ntypes = *mpb;
        let flags = |i: u32| *mpb.add(1 + 2 * i as usize);
        let local_type = (0..ntypes).find(|&i| flags(i) & 1 != 0)?;
        let host_type = (0..ntypes).find(|&i| flags(i) & 6 == 6 && flags(i) & 8 != 0).or_else(|| (0..ntypes).find(|&i| flags(i) & 6 == 6))?;
        let mut pool: Nd = 0;
        let pci = CmdPoolCI { s_type: 39, p_next: std::ptr::null(), flags: 2, family };
        if (d.vkCreateCommandPool)(dev, &pci, std::ptr::null(), &mut pool) != 0 {
            return None;
        }
        let mut cmd: Ptr = std::ptr::null_mut();
        let cai = CbAI { s_type: 40, p_next: std::ptr::null(), pool, level: 0, n: 1 };
        if (d.vkAllocateCommandBuffers)(dev, &cai, &mut cmd) != 0 {
            return None;
        }
        let mut fence: Nd = 0;
        let fci = FenceCI { s_type: 8, p_next: std::ptr::null(), flags: 0 };
        if (d.vkCreateFence)(dev, &fci, std::ptr::null(), &mut fence) != 0 {
            return None;
        }
        let sizes = PoolSize { ty: 7, n: 4096 };
        let dpci = DpCI { s_type: 33, p_next: std::ptr::null(), flags: 0, max: 512, n: 1, sizes: &sizes };
        let mut dpool: Nd = 0;
        if (d.vkCreateDescriptorPool)(dev, &dpci, std::ptr::null(), &mut dpool) != 0 {
            return None;
        }
        let tsizes = PoolSize { ty: 7, n: TSETS * 16 };
        let tpci = DpCI { s_type: 33, p_next: std::ptr::null(), flags: 0, max: TSETS, n: 1, sizes: &tsizes };
        let mut tpool: Nd = 0;
        if (d.vkCreateDescriptorPool)(dev, &tpci, std::ptr::null(), &mut tpool) != 0 {
            return None;
        }
        Some(Gpu {
            dev, d, queue, cmd, fence, dsl_cache: Default::default(), pool: dpool, local_type, host_type,
            staging: None, io: None, io_dev: None, pipes: Default::default(), f64_ok, dead: false,
            bufpool: RefCell::new(Vec::new()), pool_bytes: std::cell::Cell::new(0), slab: std::cell::Cell::new((0, 0)),
            tpool, rec: false, cursor: 0, segs: Vec::new(), tsets: 0, deferred: Vec::new(),
        })
    }

    /// Launches queued before the work is submitted, and descriptor sets per submission.
    const MAX_QUEUED: usize = 512;
    const TSETS: u32 = 1024;
    /// Argument segments start at multiples of this many words (256 bytes, a valid offset).
    const SEG_ALIGN: u64 = 32;

    impl Gpu {
        /// Frees `b` now, or after the queued work if some is pending (it may use b).
        unsafe fn free_later(&mut self, b: Buf) {
            if self.rec {
                self.deferred.push(b);
            } else {
                self.free(b);
            }
        }
        /// Opens the command buffer for queued work, if it is not open.
        unsafe fn ensure_rec(&mut self) -> bool {
            if self.rec {
                return true;
            }
            if !self.begin() {
                return false;
            }
            self.rec = true;
            self.cursor = 0;
            self.segs.clear();
            self.tsets = 0;
            true
        }
        /// Submits the queued work and waits for it. False if the device failed or a launch in
        /// it trapped: the caller is inside a chunk, which is then rolled back.
        unsafe fn settle(&mut self) -> bool {
            if !self.rec {
                return true;
            }
            self.rec = false;
            let bytes = self.cursor * 8;
            if bytes > 0 {
                if let (Some(io), Some(io_dev)) = (self.io.as_ref().map(|x| x.0.buf), self.io_dev.as_ref().map(|x| x.buf)) {
                    self.barrier(ST_COMPUTE | ST_TRANSFER, SHADER_WRITE | TRANSFER_WRITE, ST_TRANSFER, TRANSFER_READ);
                    self.copy(io_dev, io, bytes);
                    self.barrier(ST_TRANSFER, TRANSFER_WRITE, ST_HOST, HOST_READ);
                }
            }
            let ok = self.run();
            let iop = self.io.as_ref().map(|x| x.1 as *const u64);
            let trapped = ok && iop.is_some_and(|p| self.segs.iter().any(|&off| std::ptr::read_volatile(p.add(off as usize)) != 0));
            (self.d.vkResetDescriptorPool)(self.dev, self.tpool, 0);
            self.tsets = 0;
            for b in std::mem::take(&mut self.deferred) {
                self.free(b);
            }
            ok && !trapped
        }
        /// Settles queued work before an operation that needs the device idle or its results.
        unsafe fn sync(&mut self) {
            if self.rec && !self.settle() {
                chunk_trap();
            }
        }
        unsafe fn alloc(&self, size: u64, host: bool) -> Option<Buf> {
            let size = size.max(16);
            if !host {
                let mut pool = self.bufpool.borrow_mut();
                if let Some(i) = pool.iter().position(|b| b.size == size) {
                    let b = pool.swap_remove(i);
                    self.pool_bytes.set(self.pool_bytes.get() - size);
                    return Some(b);
                }
            }
            let bci = BufferCI { s_type: 12, p_next: std::ptr::null(), flags: 0, size, usage: 0x20 | 1 | 2, sharing: 0, nq: 0, q: std::ptr::null() };
            let mut buf: Nd = 0;
            if (self.d.vkCreateBuffer)(self.dev, &bci, std::ptr::null(), &mut buf) != 0 {
                return None;
            }
            let mut req = MemReq { size: 0, align: 0, bits: 0 };
            (self.d.vkGetBufferMemoryRequirements)(self.dev, buf, &mut req);
            let ty = if host { self.host_type } else { self.local_type };
            if req.bits & (1 << ty) == 0 {
                (self.d.vkDestroyBuffer)(self.dev, buf, std::ptr::null());
                return None;
            }
            // Small device buffers share big blocks of memory: allocating memory is slow, and a
            // chunk may need a new buffer per iteration (a slice) before it can reuse freed ones.
            // Such a buffer has `mem` 0 and is never destroyed, only kept for reuse.
            if !host && req.size <= SMALL {
                let (mut smem, mut used) = self.slab.get();
                let at = used.div_ceil(req.align.max(1)) * req.align.max(1);
                if smem == 0 || at + req.size > SLAB {
                    let mai = MemAI { s_type: 5, p_next: std::ptr::null(), size: SLAB, index: ty };
                    smem = 0;
                    if (self.d.vkAllocateMemory)(self.dev, &mai, std::ptr::null(), &mut smem) != 0 {
                        (self.d.vkDestroyBuffer)(self.dev, buf, std::ptr::null());
                        return None;
                    }
                    used = 0;
                }
                let at = used.div_ceil(req.align.max(1)) * req.align.max(1);
                (self.d.vkBindBufferMemory)(self.dev, buf, smem, at);
                self.slab.set((smem, at + req.size));
                return Some(Buf { buf, mem: 0, size, host });
            }
            let mai = MemAI { s_type: 5, p_next: std::ptr::null(), size: req.size, index: ty };
            let mut mem: Nd = 0;
            if (self.d.vkAllocateMemory)(self.dev, &mai, std::ptr::null(), &mut mem) != 0 {
                (self.d.vkDestroyBuffer)(self.dev, buf, std::ptr::null());
                return None;
            }
            (self.d.vkBindBufferMemory)(self.dev, buf, mem, 0);
            Some(Buf { buf, mem, size, host })
        }
        unsafe fn free(&self, b: Buf) {
            if !b.host && (b.mem == 0 || self.pool_bytes.get() + b.size <= POOL) {
                self.pool_bytes.set(self.pool_bytes.get() + b.size);
                self.bufpool.borrow_mut().push(b);
                return;
            }
            (self.d.vkDestroyBuffer)(self.dev, b.buf, std::ptr::null());
            (self.d.vkFreeMemory)(self.dev, b.mem, std::ptr::null());
        }
        /// A host-visible buffer of at least `size` bytes, mapped; grows by replacement.
        unsafe fn host(&mut self, which: u8, size: u64) -> Option<(Nd, *mut u8)> {
            let slot = if which == 0 { &mut self.staging } else { &mut self.io };
            if slot.as_ref().map_or(true, |s| s.0.size < size) {
                if let Some((b, _)) = slot.take() {
                    (self.d.vkDestroyBuffer)(self.dev, b.buf, std::ptr::null());
                    (self.d.vkFreeMemory)(self.dev, b.mem, std::ptr::null());
                }
                let want = size.max(1 << 16).next_power_of_two();
                let b = self.alloc(want, true)?;
                let mut p: Ptr = std::ptr::null_mut();
                if (self.d.vkMapMemory)(self.dev, b.mem, 0, !0u64, 0, &mut p) != 0 {
                    self.free(b);
                    return None;
                }
                let slot = if which == 0 { &mut self.staging } else { &mut self.io };
                *slot = Some((b, p));
            }
            let slot = if which == 0 { &self.staging } else { &self.io };
            slot.as_ref().map(|(b, p)| (b.buf, *p as *mut u8))
        }
        unsafe fn begin(&self) -> bool {
            (self.d.vkResetCommandBuffer)(self.cmd, 0);
            let bi = CbBI { s_type: 42, p_next: std::ptr::null(), flags: 1, inherit: std::ptr::null() };
            (self.d.vkBeginCommandBuffer)(self.cmd, &bi) == 0
        }
        unsafe fn barrier(&self, src_stage: u32, src: u32, dst_stage: u32, dst: u32) {
            let b = MemBarrier { s_type: 46, p_next: std::ptr::null(), src, dst };
            (self.d.vkCmdPipelineBarrier)(self.cmd, src_stage, dst_stage, 0, 1, &b, 0, std::ptr::null(), 0, std::ptr::null());
        }
        unsafe fn copy(&self, src: Nd, dst: Nd, size: u64) {
            let c = BufCopy { src: 0, dst: 0, size };
            (self.d.vkCmdCopyBuffer)(self.cmd, src, dst, 1, &c);
        }
        /// Ends, submits and waits for the command buffer.
        unsafe fn run(&mut self) -> bool {
            if (self.d.vkEndCommandBuffer)(self.cmd) != 0 {
                return false;
            }
            let si = SubmitInfo { s_type: 4, p_next: std::ptr::null(), nw: 0, w: std::ptr::null(), ws: std::ptr::null(), ncb: 1, cb: &self.cmd, ns: 0, s: std::ptr::null() };
            if (self.d.vkQueueSubmit)(self.queue, 1, &si, self.fence) != 0 {
                self.dead = true;
                return false;
            }
            // Short waits are polled (waking a sleeping thread costs more than most kernels).
            let t = std::time::Instant::now();
            let mut ok = false;
            loop {
                match (self.d.vkGetFenceStatus)(self.dev, self.fence) {
                    0 => { ok = true; break; }
                    1 => {}
                    _ => break,
                }
                if t.elapsed().as_micros() > 2000 {
                    ok = (self.d.vkWaitForFences)(self.dev, 1, &self.fence, 1, !0u64) == 0;
                    break;
                }
                std::hint::spin_loop();
            }
            (self.d.vkResetFences)(self.dev, 1, &self.fence);
            if !ok {
                self.dead = true;
            }
            ok
        }
        unsafe fn upload(&mut self, dst: Nd, src: *const u8, size: u64) -> bool {
            self.sync();
            let Some((st, p)) = self.host(0, size) else { return false };
            std::ptr::copy_nonoverlapping(src, p, size as usize);
            if !self.begin() {
                return false;
            }
            self.copy(st, dst, size);
            self.run()
        }
        unsafe fn download(&mut self, src: Nd, dst: *mut u8, size: u64) -> bool {
            self.sync();
            let Some((st, p)) = self.host(0, size) else { return false };
            if !self.begin() {
                return false;
            }
            self.copy(src, st, size);
            self.barrier(ST_TRANSFER, TRANSFER_WRITE, ST_HOST, HOST_READ);
            if !self.run() {
                return false;
            }
            std::ptr::copy_nonoverlapping(p as *const u8, dst, size as usize);
            true
        }
        unsafe fn pipe(&mut self, k: &KernelDesc) -> Option<(Nd, Nd, Nd)> {
            if !self.pipes.contains_key(&k.id) {
                let t = std::time::Instant::now();
                let made = self.make_pipe(k);
                log(&format!("kernel {} built in {:?}", k.id, t.elapsed()));
                self.pipes.insert(k.id, made);
            }
            self.pipes[&k.id].as_ref().map(|p| (p.pipe, p.layout, p.set))
        }
        unsafe fn make_pipe(&mut self, k: &KernelDesc) -> Option<Pipe> {
            let nb = k.nbufs as u32 + 1;
            let (dsl, layout) = match self.dsl_cache.get(&nb) {
                Some(x) => *x,
                None => {
                    let binds: Vec<DslBinding> = (0..nb).map(|i| DslBinding { binding: i, ty: 7, count: 1, stages: 0x20, samplers: std::ptr::null() }).collect();
                    let dci = DslCI { s_type: 32, p_next: std::ptr::null(), flags: 0, n: nb, b: binds.as_ptr() };
                    let mut dsl: Nd = 0;
                    if (self.d.vkCreateDescriptorSetLayout)(self.dev, &dci, std::ptr::null(), &mut dsl) != 0 {
                        return None;
                    }
                    let pci = PlCI { s_type: 30, p_next: std::ptr::null(), flags: 0, nsets: 1, sets: &dsl, npc: 0, pc: std::ptr::null() };
                    let mut layout: Nd = 0;
                    if (self.d.vkCreatePipelineLayout)(self.dev, &pci, std::ptr::null(), &mut layout) != 0 {
                        return None;
                    }
                    self.dsl_cache.insert(nb, (dsl, layout));
                    (dsl, layout)
                }
            };
            let sci = ShaderCI { s_type: 16, p_next: std::ptr::null(), flags: 0, size: k.words.len() * 4, code: k.words.as_ptr() };
            let mut module: Nd = 0;
            if (self.d.vkCreateShaderModule)(self.dev, &sci, std::ptr::null(), &mut module) != 0 {
                return None;
            }
            let stage = StageCI { s_type: 18, p_next: std::ptr::null(), flags: 0, stage: 0x20, module, name: b"main\0".as_ptr(), spec: std::ptr::null() };
            let cci = ComputeCI { s_type: 29, p_next: std::ptr::null(), flags: 0, stage, layout, base: 0, base_i: -1 };
            let mut pipe: Nd = 0;
            if (self.d.vkCreateComputePipelines)(self.dev, 0, 1, &cci, std::ptr::null(), &mut pipe) != 0 {
                return None;
            }
            let dai = DsAI { s_type: 34, p_next: std::ptr::null(), pool: self.pool, n: 1, layouts: &dsl };
            let mut set: Nd = 0;
            if (self.d.vkAllocateDescriptorSets)(self.dev, &dai, &mut set) != 0 {
                return None;
            }
            Some(Pipe { pipe, layout, set })
        }
    }

    // ---- Mirrors: where the current contents of a vector are. ----

    /// `spare`: a second device buffer of the same size, kept for backups before kernels that
    /// write a vector only the device holds.
    struct MState { buf: Option<Buf>, spare: Option<Buf>, gpu_ok: bool, cpu_ok: bool }

    /// The companion of a vector the GPU may hold: its device copy and which copy is current.
    /// Dropped with the vector it belongs to.
    /// `birth`: the chunk it was made in (0 outside chunks); a chunk rolls back only the
    /// mirrors that existed before it.
    pub struct Mirror { st: RefCell<MState>, birth: u64 }

    impl Mirror {
        pub fn new() -> Mirror {
            let birth = CHUNK.with(|c| { let c = c.borrow(); if c.active { c.serial } else { 0 } });
            Mirror { st: RefCell::new(MState { buf: None, spare: None, gpu_ok: false, cpu_ok: true }), birth }
        }
        /// Makes the vector's own memory current (downloads it if the GPU changed it).
        pub fn to_cpu<T: Copy + 'static>(&self, v: &mut Vec<T>) {
            backup_cpu(self, v);
            let mut st = self.st.borrow_mut();
            if st.cpu_ok {
                return;
            }
            st.cpu_ok = true;
            let bytes = (v.len() * std::mem::size_of::<T>()) as u64;
            let Some(b) = st.buf.as_ref() else { return };
            let Some(g) = gpu() else { fatal() };
            let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
            if bytes > 0 && !unsafe { g.download(b.buf, v.as_mut_ptr() as *mut u8, bytes) } {
                drop(g);
                fatal();
            }
        }
        /// The vector's contents as the CPU can read them, for a vector the caller cannot
        /// write to: itself if current, else a copy downloaded from the device.
        pub fn view<'a, T: Copy>(&self, v: &'a Vec<T>) -> std::borrow::Cow<'a, Vec<T>> {
            touch(self);
            let st = self.st.borrow();
            if st.cpu_ok || st.buf.is_none() {
                return std::borrow::Cow::Borrowed(v);
            }
            let mut out: Vec<T> = Vec::with_capacity(v.len());
            let bytes = (v.len() * std::mem::size_of::<T>()) as u64;
            let Some(g) = gpu() else { fatal() };
            let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
            if bytes > 0 && !unsafe { g.download(st.buf.as_ref().unwrap().buf, out.as_mut_ptr() as *mut u8, bytes) } {
                drop(g);
                fatal();
            }
            unsafe { out.set_len(v.len()) };
            std::borrow::Cow::Owned(out)
        }
        /// The vector changed on the CPU: the device copy is stale.
        pub fn cpu_changed(&self) {
            touch(self);
            let mut st = self.st.borrow_mut();
            st.gpu_ok = false;
            st.cpu_ok = true;
        }
        pub fn on_cpu(&self) -> bool { self.st.borrow().cpu_ok }
    }
    impl Drop for Mirror {
        fn drop(&mut self) {
            let st = self.st.get_mut();
            let (a, b) = (st.buf.take(), st.spare.take());
            if a.is_some() || b.is_some() {
                if let Some(g) = GPU.get().and_then(|g| g.as_ref()) {
                    let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
                    for x in [a, b].into_iter().flatten() {
                        unsafe { g.free_later(x) };
                    }
                }
            }
        }
    }

    /// `(slice v a b)` of a vector whose current contents are on the device: the slice is copied
    /// there and stays there (its own memory holds zeros until something reads it).
    #[track_caller]
    pub fn slice_dev<T: Copy + Default>(v: &Vec<T>, src: &Mirror, a: i64, b: i64, dst: &Mirror) -> Vec<T> {
        let (ua, ub) = super::ail_rt::slice_bounds(v.len(), a, b);
        let sz = std::mem::size_of::<T>() as u64;
        let n = ub - ua;
        let copied = (|| -> Option<()> {
            let g = gpu()?;
            let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
            touch(src);
            let st = src.st.borrow();
            let from = st.buf.as_ref()?.buf;
            let buf = unsafe { g.alloc((n as u64 * sz).max(16), false)? };
            // Inside a chunk the copy is queued after the work that produces the source.
            let queued = in_chunk();
            let ok = unsafe {
                (if queued { g.ensure_rec() } else { g.begin() }) && {
                    if n > 0 {
                        g.barrier(ST_COMPUTE | ST_TRANSFER, SHADER_WRITE | TRANSFER_WRITE, ST_TRANSFER, TRANSFER_READ);
                        let c = BufCopy { src: ua as u64 * sz, dst: 0, size: n as u64 * sz };
                        (g.d.vkCmdCopyBuffer)(g.cmd, from, buf.buf, 1, &c);
                    }
                    g.barrier(ST_TRANSFER, TRANSFER_WRITE, ST_COMPUTE | ST_TRANSFER, SHADER_READ | TRANSFER_READ);
                    queued || g.run()
                }
            };
            if !ok {
                unsafe { g.free(buf) };
                return None;
            }
            let mut d = dst.st.borrow_mut();
            if let Some(old) = d.buf.replace(buf) {
                unsafe { g.free_later(old) };
            }
            d.gpu_ok = true;
            d.cpu_ok = false;
            Some(())
        })();
        match copied {
            Some(()) => vec![T::default(); n],
            None => src.view(v)[ua..ub].to_vec(),
        }
    }

    /// The device lost data only it held: nothing can recover the program's state.
    #[cold]
    fn fatal() -> ! {
        super::ail_rt::flush();
        eprintln!("trap: the GPU failed while it held the only copy of a vector");
        std::process::exit(101)
    }

    /// A compiled kernel: SPIR-V for one loop body. Binding 0 is the argument buffer (trap flag,
    /// loop bounds, scalars, then per-workgroup partial sums); bindings 1.. are the vectors.
    /// `exact`: no loops and no sums inside, so the kernel traps exactly when the loop on the CPU
    /// would; a vector it writes but does not read then needs no backup (after a trap the CPU
    /// run traps too, and what the vector held no longer matters).
    pub struct KernelDesc { pub id: usize, pub words: &'static [u32], pub nbufs: usize, pub nargs: usize, pub nred: usize, pub f64: bool, pub exact: bool }

    /// One vector a kernel uses.
    pub struct BufArg<'a> { ptr: *mut u8, bytes: u64, mirror: &'a Mirror, write: bool, reads: bool }
    impl<'a> BufArg<'a> {
        pub fn read<T: Copy>(v: &Vec<T>, m: &'a Mirror) -> Self {
            BufArg { ptr: v.as_ptr() as *mut u8, bytes: (v.len() * std::mem::size_of::<T>()) as u64, mirror: m, write: false, reads: true }
        }
        /// `reads`: the kernel also reads the vector it writes.
        pub fn write<T: Copy>(v: &mut Vec<T>, m: &'a Mirror, reads: bool) -> Self {
            BufArg { ptr: v.as_mut_ptr() as *mut u8, bytes: (v.len() * std::mem::size_of::<T>()) as u64, mirror: m, write: true, reads }
        }
    }

    /// Reached after the CPU run of a loop whose kernel trapped without backups (see
    /// `KernelDesc::exact`): that run should have trapped too; the program's state is lost, so
    /// stop rather than go on wrong.
    #[cold]
    pub fn fatal_mismatch() -> ! {
        super::ail_rt::flush();
        eprintln!("trap: internal error: the GPU and the CPU disagree on a trap");
        std::process::exit(101)
    }

    pub const WG: u32 = 256;

    /// Runs the kernel over iterations lo..hi and returns the final value of each sum (starting
    /// from `starts`), or None if the GPU could not run it, the kernel trapped, or the magnitudes
    /// of the terms do not prove the sequential sums free of overflow: then every vector is as
    /// before the call, and the caller runs the loop on the CPU (after making each vector's own
    /// memory current), which traps where the program would.
    /// Err(true): the kernel trapped and did not keep backups, so the CPU run must trap.
    pub fn launch(k: &KernelDesc, lo: i64, hi: i64, args: &[u64], bufs: &mut [BufArg], starts: &[i64]) -> Result<Vec<i64>, bool> {
        MUST_TRAP_LOCAL.with(|m| m.set(false));
        let r = launch_inner(k, lo, hi, args, bufs, starts);
        match r {
            Some(v) => Ok(v),
            None => Err(MUST_TRAP_LOCAL.with(|m| m.get())),
        }
    }
    thread_local! { static MUST_TRAP_LOCAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    fn launch_inner(k: &KernelDesc, lo: i64, hi: i64, args: &[u64], bufs: &mut [BufArg], starts: &[i64]) -> Option<Vec<i64>> {
        let n = hi.checked_sub(lo)?;
        if n <= 0 {
            return Some(starts.to_vec());
        }
        if DISABLED.lock().map(|d| d.contains(&k.id)).unwrap_or(true) {
            return None;
        }
        let g = gpu()?;
        let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
        if g.dead || (k.f64 && !g.f64_ok) {
            return None;
        }
        let t = std::time::Instant::now();
        if in_chunk() {
            let r = unsafe { launch_queued(&mut g, k, lo, n, args, bufs, starts) };
            if r.is_none() {
                log(&format!("kernel {} n={} not queued", k.id, n));
            }
            return r;
        }
        let r = unsafe { launch_locked(&mut g, k, lo, n, args, bufs, starts) };
        log(&format!("kernel {} n={} {} in {:?}", k.id, n, if r.is_some() { "ran" } else { "fell back" }, t.elapsed()));
        r
    }

    /// Kernels that failed to build or ran into their iteration budget: they stay on the CPU.
    static DISABLED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

    unsafe fn launch_locked(g: &mut Gpu, k: &KernelDesc, lo: i64, n: i64, args: &[u64], bufs: &mut [BufArg], starts: &[i64]) -> Option<Vec<i64>> {
        let Some((pipe, layout, set)) = g.pipe(k) else {
            log(&format!("kernel {} did not build", k.id));
            if let Ok(mut d) = DISABLED.lock() {
                d.push(k.id);
            }
            return None;
        };
        let groups = (n as u64).div_ceil(WG as u64);
        if groups > 65535 * 65535 {
            return None;
        }
        let gx = groups.min(65535) as u32;
        let gy = groups.div_ceil(gx as u64) as u32;
        let ngroups = gx as u64 * gy as u64;
        // Device copies: allocated, and current where the kernel reads them.
        for b in bufs.iter_mut() {
            if b.bytes / 4 > u32::MAX as u64 {
                return None;
            }
            let mut st = b.mirror.st.borrow_mut();
            let size = b.bytes.max(16);
            if st.buf.as_ref().map_or(true, |x| x.size != size) {
                if !st.cpu_ok {
                    return None;
                }
                if let Some(old) = st.buf.take() {
                    g.free(old);
                }
                if let Some(old) = st.spare.take() {
                    g.free(old);
                }
                st.buf = Some(g.alloc(size, false)?);
                st.gpu_ok = false;
            }
            if !st.gpu_ok {
                let dst = st.buf.as_ref().unwrap().buf;
                if b.bytes > 0 && !g.upload(dst, b.ptr, b.bytes) {
                    return None;
                }
                st.gpu_ok = true;
            }
        }
        // The argument buffer: flag, first partial slot, lo, n, the arguments, then the partial
        // sums. Written here, copied to device memory for the kernel, and back for the results.
        let pbase = 4 + args.len() as u64;
        let nparts = ngroups * 2 * k.nred as u64;
        let words = pbase + nparts;
        let (io, iop) = g.host(1, words * 8)?;
        if g.io_dev.as_ref().map_or(true, |b| b.size < words * 8) {
            if let Some(old) = g.io_dev.take() {
                g.free(old);
            }
            let want = (words * 8).max(1 << 16).next_power_of_two();
            g.io_dev = Some(g.alloc(want, false)?);
        }
        let io_dev = g.io_dev.as_ref().unwrap().buf;
        let iow = iop as *mut u64;
        *iow = 0;
        *iow.add(1) = pbase;
        *iow.add(2) = lo as u64;
        *iow.add(3) = n as u64;
        for (i, a) in args.iter().enumerate() {
            *iow.add(4 + i) = *a;
        }
        let mut infos: Vec<DescBuf> = vec![DescBuf { buffer: io_dev, offset: 0, range: !0u64 }];
        for b in bufs.iter() {
            infos.push(DescBuf { buffer: b.mirror.st.borrow().buf.as_ref().unwrap().buf, offset: 0, range: !0u64 });
        }
        let writes: Vec<WriteDS> = (0..infos.len())
            .map(|i| WriteDS { s_type: 35, p_next: std::ptr::null(), set, binding: i as u32, elem: 0, n: 1, ty: 7, img: std::ptr::null(), buf: &infos[i], texel: std::ptr::null() })
            .collect();
        (g.d.vkUpdateDescriptorSets)(g.dev, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null());
        if !g.begin() {
            return None;
        }
        // A written vector whose only current copy is on the device gets a backup first, so a
        // trap can put it back.
        let mut backups: Vec<usize> = Vec::new();
        let mut no_backup = false;
        for (i, b) in bufs.iter().enumerate() {
            let mut st = b.mirror.st.borrow_mut();
            if b.write && !st.cpu_ok && k.exact && !b.reads {
                no_backup = true;
                continue;
            }
            if b.write && !st.cpu_ok {
                let size = st.buf.as_ref().unwrap().size;
                if st.spare.is_none() {
                    match g.alloc(size, false) {
                        Some(x) => st.spare = Some(x),
                        None => {
                            drop(st);
                            let _ = g.run();
                            return None;
                        }
                    }
                }
                let (src, dst) = (st.buf.as_ref().unwrap().buf, st.spare.as_ref().unwrap().buf);
                g.copy(src, dst, size);
                backups.push(i);
            }
        }
        g.copy(io, io_dev, pbase * 8);
        g.barrier(ST_TRANSFER | ST_HOST, TRANSFER_WRITE | HOST_WRITE, ST_COMPUTE, SHADER_READ | SHADER_WRITE);
        (g.d.vkCmdBindPipeline)(g.cmd, 1, pipe);
        (g.d.vkCmdBindDescriptorSets)(g.cmd, 1, layout, 0, 1, &set, 0, std::ptr::null());
        (g.d.vkCmdDispatch)(g.cmd, gx, gy, 1);
        g.barrier(ST_COMPUTE, SHADER_WRITE, ST_TRANSFER, TRANSFER_READ);
        g.copy(io_dev, io, words * 8);
        g.barrier(ST_TRANSFER, TRANSFER_WRITE, ST_HOST, HOST_READ);
        let ran = g.run();
        let mut ok = ran && std::ptr::read_volatile(iow) == 0;
        let mut out = Vec::with_capacity(k.nred);
        if ok {
            for r in 0..k.nred {
                let mut total: i128 = starts[r] as i128;
                let mut bound: i128 = (starts[r] as i128).abs();
                for gi in 0..ngroups as usize {
                    let at = pbase as usize + (gi * k.nred + r) * 2;
                    total += (*iow.add(at)) as i64 as i128;
                    bound += (*iow.add(at + 1)) as i128;
                }
                if bound > i64::MAX as i128 {
                    ok = false;
                    break;
                }
                out.push(total as i64);
            }
        }
        if !ok {
            // Back to the state before the call.
            for b in bufs.iter() {
                if b.write {
                    let mut st = b.mirror.st.borrow_mut();
                    if st.cpu_ok {
                        st.gpu_ok = false;
                    }
                }
            }
            for i in backups {
                // The backup becomes the current copy (what the kernel wrote is dropped).
                let mut st = bufs[i].mirror.st.borrow_mut();
                let MState { buf, spare, .. } = &mut *st;
                std::mem::swap(buf, spare);
            }
            if !ran && bufs.iter().any(|b| !b.mirror.st.borrow().cpu_ok) {
                // Lost device: whatever only it held is gone.
                fatal();
            }
            if no_backup {
                MUST_TRAP_LOCAL.with(|m| m.set(true));
            }
            return None;
        }
        for b in bufs.iter() {
            if b.write {
                let mut st = b.mirror.st.borrow_mut();
                st.cpu_ok = false;
                st.gpu_ok = true;
            }
        }
        Some(out)
    }


    // ---- Chunks: loop iterations whose GPU work is queued and checked once at the end. ----
    //
    // Waiting for each launch costs more than most launches. A loop whose body only computes
    // (no output, no other effects) can instead run a group of its iterations with every launch
    // queued, and check them all at the end. Each mirror that existed before the group keeps,
    // from the moment the group first changes it, a copy of what it held (on the device or on
    // the CPU, wherever it was current); the variables the body assigns are saved by the
    // generated code. If anything traps inside the group, on the GPU or on the CPU, all of it
    // is put back and the group runs again the ordinary way, one launch at a time, which traps
    // exactly where the program does.

    struct CRec { m: *const Mirror, gpu_ok: bool, cpu_ok: bool, gpu_backed: bool, cpu: Option<Box<dyn FnOnce()>> }
    struct Chunk { active: bool, serial: u64, recs: Vec<CRec>, failures: u32 }
    thread_local! { static CHUNK: RefCell<Chunk> = const { RefCell::new(Chunk { active: false, serial: 0, recs: Vec::new(), failures: 0 }) }; }

    fn in_chunk() -> bool { CHUNK.with(|c| c.borrow().active) }

    /// The record of `m` in the running chunk (made on first use), if m predates the chunk.
    fn touch(m: &Mirror) -> Option<usize> {
        CHUNK.with(|c| {
            let mut c = c.borrow_mut();
            if !c.active || m.birth == c.serial {
                return None;
            }
            let p = m as *const Mirror;
            if let Some(i) = c.recs.iter().position(|r| r.m == p) {
                return Some(i);
            }
            let st = m.st.borrow();
            c.recs.push(CRec { m: p, gpu_ok: st.gpu_ok, cpu_ok: st.cpu_ok, gpu_backed: false, cpu: None });
            Some(c.recs.len() - 1)
        })
    }

    /// Before the CPU copy of `v` may change inside a chunk: keep what it held at the start.
    fn backup_cpu<T: Copy + 'static>(m: &Mirror, v: &mut Vec<T>) {
        let Some(i) = touch(m) else { return };
        CHUNK.with(|c| {
            let mut c = c.borrow_mut();
            let r = &mut c.recs[i];
            if r.cpu_ok && r.cpu.is_none() {
                let saved = v.clone();
                let p = v as *mut Vec<T>;
                // The vector outlives the chunk (it belongs to code outside it), and the restore
                // runs after everything inside has finished.
                r.cpu = Some(Box::new(move || unsafe { *p = saved }));
            }
        });
    }

    /// Before the device copy of `m` changes inside a chunk: queue a copy of what it held.
    unsafe fn backup_dev(g: &mut Gpu, m: &Mirror) -> bool {
        let Some(i) = touch(m) else { return true };
        let need = CHUNK.with(|c| { let c = c.borrow(); c.recs[i].gpu_ok && !c.recs[i].gpu_backed });
        if !need {
            return true;
        }
        let mut st = m.st.borrow_mut();
        let Some(size) = st.buf.as_ref().map(|b| b.size) else { return true };
        if st.spare.as_ref().map_or(true, |s| s.size != size) {
            if let Some(old) = st.spare.take() {
                g.free_later(old);
            }
            match g.alloc(size, false) {
                Some(x) => st.spare = Some(x),
                None => return false,
            }
        }
        if !g.ensure_rec() {
            return false;
        }
        g.barrier(ST_COMPUTE | ST_TRANSFER, SHADER_WRITE | TRANSFER_WRITE, ST_TRANSFER, TRANSFER_READ);
        g.copy(st.buf.as_ref().unwrap().buf, st.spare.as_ref().unwrap().buf, size);
        g.barrier(ST_TRANSFER, TRANSFER_WRITE | TRANSFER_READ, ST_COMPUTE | ST_TRANSFER, SHADER_READ | SHADER_WRITE | TRANSFER_WRITE);
        CHUNK.with(|c| c.borrow_mut().recs[i].gpu_backed = true);
        true
    }

    /// Ends the chunk the hard way: the generated code catches this and rolls back.
    #[cold]
    fn chunk_trap() -> ! {
        std::panic::panic_any(ChunkTrap)
    }
    struct ChunkTrap;

    /// How many iterations the next chunk of a loop takes, given how many it has done in chunks
    /// so far (`done`, -1 once a chunk of it failed) and how many are left.
    pub fn chunk_len(done: i64, left: i64) -> i64 {
        if done < 0 || mode() == 0 || CHUNK.with(|c| { let c = c.borrow(); c.active || c.failures >= 3 }) {
            return 1;
        }
        (done + 1).min(1 << 20).min(left).max(1)
    }

    /// Starts a chunk; false (and nothing started) if there is no device or one is running.
    pub fn chunk_begin() -> bool {
        if in_chunk() || !ready() {
            return false;
        }
        CHUNK.with(|c| {
            let mut c = c.borrow_mut();
            c.active = true;
            c.serial += 1;
            c.recs.clear();
        });
        super::ail_rt::CHUNK_QUIET.with(|q| q.set(true));
        true
    }

    /// Ends a chunk: true if it ran without trapping (its effects stay), false if it was rolled
    /// back (the caller restores its variables and runs the iterations again).
    pub fn chunk_end(r: Result<(), Box<dyn std::any::Any + Send>>) -> bool {
        super::ail_rt::CHUNK_QUIET.with(|q| q.set(false));
        let g = gpu();
        let settled = match g {
            Some(g) => unsafe { g.lock().unwrap_or_else(|e| e.into_inner()).settle() },
            None => true,
        };
        let recs = CHUNK.with(|c| {
            let mut c = c.borrow_mut();
            c.active = false;
            std::mem::take(&mut c.recs)
        });
        if r.is_ok() && settled {
            log(&format!("chunk of {} mirrors done", recs.len()));
            return true;
        }
        CHUNK.with(|c| c.borrow_mut().failures += 1);
        log("chunk rolled back");
        let dead = g.map_or(true, |g| g.lock().map(|g| g.dead).unwrap_or(true));
        for r in recs {
            // Every mirror in the list outlives the chunk (see `touch`).
            let m = unsafe { &*r.m };
            let mut st = m.st.borrow_mut();
            if r.gpu_backed {
                let MState { buf, spare, .. } = &mut *st;
                std::mem::swap(buf, spare);
            }
            if let Some(f) = r.cpu {
                f();
            }
            st.gpu_ok = r.gpu_ok;
            st.cpu_ok = r.cpu_ok;
            if dead && !st.cpu_ok {
                drop(st);
                fatal();
            }
        }
        false
    }

    /// `launch` inside a chunk: the kernel is queued, and the writes it makes are assumed to
    /// succeed (they are checked when the chunk ends). Kernels with sums wait for their result.
    /// None: it cannot run on the device; the caller runs the loop on the CPU (which waits for
    /// the queued work first).
    unsafe fn launch_queued(g: &mut Gpu, k: &KernelDesc, lo: i64, n: i64, args: &[u64], bufs: &mut [BufArg], starts: &[i64]) -> Option<Vec<i64>> {
        let Some((pipe, layout, _)) = g.pipe(k) else {
            if let Ok(mut d) = DISABLED.lock() {
                d.push(k.id);
            }
            return None;
        };
        let groups = (n as u64).div_ceil(WG as u64);
        if groups > 65535 * 65535 {
            return None;
        }
        let gx = groups.min(65535) as u32;
        let gy = groups.div_ceil(gx as u64) as u32;
        let ngroups = gx as u64 * gy as u64;
        for b in bufs.iter_mut() {
            if b.bytes / 4 > u32::MAX as u64 {
                return None;
            }
            touch(b.mirror);
            let size = b.bytes.max(16);
            let mut st = b.mirror.st.borrow_mut();
            if st.buf.as_ref().map_or(false, |x| x.size != size) {
                return None;
            }
            if st.buf.is_none() {
                st.buf = Some(g.alloc(size, false)?);
                st.gpu_ok = false;
            }
            if !st.gpu_ok {
                drop(st);
                if !backup_dev(g, b.mirror) {
                    return None;
                }
                let st = b.mirror.st.borrow();
                let dst = st.buf.as_ref().unwrap().buf;
                drop(st);
                if b.bytes > 0 && !g.upload(dst, b.ptr, b.bytes) {
                    return None;
                }
                b.mirror.st.borrow_mut().gpu_ok = true;
            }
        }
        for b in bufs.iter() {
            if b.write && !backup_dev(g, b.mirror) {
                return None;
            }
        }
        // This launch's segment of the argument buffers.
        let pbase = 4 + args.len() as u64;
        let words = pbase + ngroups * 2 * k.nred as u64;
        let span = words.div_ceil(SEG_ALIGN) * SEG_ALIGN;
        if g.rec && ((g.cursor + span) * 8 > g.io.as_ref().map_or(0, |x| x.0.size) || g.io_dev.as_ref().map_or(0, |x| x.size) < (g.cursor + span) * 8 || g.tsets >= TSETS || g.segs.len() >= MAX_QUEUED) {
            g.sync();
        }
        if !g.ensure_rec() {
            return None;
        }
        let need = (g.cursor + span) * 8;
        if g.io.as_ref().map_or(true, |x| x.0.size < need) || g.io_dev.as_ref().map_or(true, |x| x.size < need) {
            // Only at the start of a recording (see above): nothing queued uses them yet.
            let (io_ok, dev_ok) = (g.io.as_ref().map_or(false, |x| x.0.size >= need), g.io_dev.as_ref().map_or(false, |x| x.size >= need));
            if !io_ok {
                g.host(1, need.max(1 << 20))?;
            }
            if !dev_ok {
                if let Some(old) = g.io_dev.take() {
                    g.free_later(old);
                }
                g.io_dev = Some(g.alloc(need.max(1 << 20).next_power_of_two(), false)?);
            }
        }
        let off = g.cursor;
        g.cursor += span;
        let (io, iop) = (g.io.as_ref().unwrap().0.buf, g.io.as_ref().unwrap().1 as *mut u64);
        let io_dev = g.io_dev.as_ref().unwrap().buf;
        let iow = iop.add(off as usize);
        *iow = 0;
        *iow.add(1) = pbase;
        *iow.add(2) = lo as u64;
        *iow.add(3) = n as u64;
        for (i, a) in args.iter().enumerate() {
            *iow.add(4 + i) = *a;
        }
        g.segs.push(off);
        // A descriptor set of its own, pointing at the segment.
        let nb = k.nbufs as u32 + 1;
        let (dsl, _) = g.dsl_cache[&nb];
        let dai = DsAI { s_type: 34, p_next: std::ptr::null(), pool: g.tpool, n: 1, layouts: &dsl };
        let mut set: Nd = 0;
        if (g.d.vkAllocateDescriptorSets)(g.dev, &dai, &mut set) != 0 {
            chunk_trap();
        }
        g.tsets += 1;
        let mut infos: Vec<DescBuf> = vec![DescBuf { buffer: io_dev, offset: off * 8, range: words * 8 }];
        for b in bufs.iter() {
            infos.push(DescBuf { buffer: b.mirror.st.borrow().buf.as_ref().unwrap().buf, offset: 0, range: !0u64 });
        }
        let writes: Vec<WriteDS> = (0..infos.len())
            .map(|i| WriteDS { s_type: 35, p_next: std::ptr::null(), set, binding: i as u32, elem: 0, n: 1, ty: 7, img: std::ptr::null(), buf: &infos[i], texel: std::ptr::null() })
            .collect();
        (g.d.vkUpdateDescriptorSets)(g.dev, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null());
        let c = BufCopy { src: off * 8, dst: off * 8, size: pbase * 8 };
        (g.d.vkCmdCopyBuffer)(g.cmd, io, io_dev, 1, &c);
        g.barrier(ST_TRANSFER | ST_HOST, TRANSFER_WRITE | HOST_WRITE, ST_COMPUTE, SHADER_READ | SHADER_WRITE);
        (g.d.vkCmdBindPipeline)(g.cmd, 1, pipe);
        (g.d.vkCmdBindDescriptorSets)(g.cmd, 1, layout, 0, 1, &set, 0, std::ptr::null());
        (g.d.vkCmdDispatch)(g.cmd, gx, gy, 1);
        g.barrier(ST_COMPUTE, SHADER_WRITE, ST_COMPUTE | ST_TRANSFER, SHADER_READ | SHADER_WRITE | TRANSFER_READ | TRANSFER_WRITE);
        for b in bufs.iter() {
            if b.write {
                let mut st = b.mirror.st.borrow_mut();
                st.cpu_ok = false;
                st.gpu_ok = true;
            }
        }
        if k.nred == 0 {
            return Some(Vec::new());
        }
        // The sums are needed now: wait, and read them from the segment.
        g.sync();
        let mut out = Vec::with_capacity(k.nred);
        for r in 0..k.nred {
            let mut total: i128 = starts[r] as i128;
            let mut bound: i128 = (starts[r] as i128).abs();
            for gi in 0..ngroups as usize {
                let at = pbase as usize + (gi * k.nred + r) * 2;
                total += (*iow.add(at)) as i64 as i128;
                bound += (*iow.add(at + 1)) as i128;
            }
            if bound > i64::MAX as i128 {
                // The writes cannot be taken back one launch at a time here: roll back all.
                chunk_trap();
            }
            out.push(total as i64);
        }
        Some(out)
    }

    /// Adds the per-workgroup partials of reduction r (out of `nred`) to `start`; None if the
    /// magnitudes do not prove that the sequential sum could not overflow.
    pub fn sum_parts(parts: &[i64], nred: usize, r: usize, start: i64) -> Option<i64> {
        let mut total: i128 = start as i128;
        let mut bound: i128 = (start as i128).abs();
        for g in parts.chunks(2 * nred) {
            total += g[2 * r] as i128;
            bound += (g[2 * r + 1] as u64) as i128;
        }
        if bound > i64::MAX as i128 { None } else { Some(total as i64) }
    }
}
"####;
