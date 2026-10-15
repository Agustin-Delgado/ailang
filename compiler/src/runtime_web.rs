//! Web target: the generated Rust becomes a wasm module driven by a small JS host.
//! The app is Elm-shaped: (fn init) builds the model, (fn update) applies an Event to it,
//! and (fn view) draws with a Gfx capability or (fn page) returns HTML.

/// Exports appended to the generated program for `--target web`.
pub fn exports(model_ty: &str, has_view: bool, has_page: bool) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        r####"
thread_local! {{
    static AIL_MODEL: std::cell::RefCell<Option<{model_ty}>> = std::cell::RefCell::new(None);
}}

#[no_mangle]
pub extern "C" fn ail_init() {{
    ail_rt::init(&AIL_LINES, AIL_FILE);
    let m = f_init();
    AIL_MODEL.with(|c| *c.borrow_mut() = Some(m));
}}

#[no_mangle]
pub extern "C" fn ail_event(kind: u32, a: f64, b: f64, s1_len: u32, s2_len: u32) {{
    let (s1, s2) = ail_web::take_inputs(s1_len as usize, s2_len as usize);
    let ev = match kind {{
        0 => U_Event::Tick(a),
        1 => U_Event::KeyDown(s1),
        2 => U_Event::KeyUp(s1),
        3 => U_Event::MouseDown(a, b),
        4 => U_Event::MouseUp(a, b),
        5 => U_Event::MouseMove(a, b),
        6 => U_Event::Click(s1),
        7 => U_Event::Input(s1, s2),
        _ => U_Event::Resize(a, b),
    }};
    AIL_MODEL.with(|c| {{
        let mut c = c.borrow_mut();
        let m = c.as_mut().expect("ail_init must run first");
        f_update(m, &ev);
    }});
    ail_rt::flush();
}}
"####
    ));
    if has_view {
        s.push_str(
            r####"
#[no_mangle]
pub extern "C" fn ail_view() -> u32 {
    ail_web::begin_frame();
    AIL_MODEL.with(|c| {
        let c = c.borrow();
        f_view(c.as_ref().expect("ail_init must run first"), ail_rt::Gfx);
    });
    ail_rt::flush();
    ail_web::frame_len()
}
"####,
        );
    }
    if has_page {
        s.push_str(
            r####"
#[no_mangle]
pub extern "C" fn ail_page() -> u32 {
    let html = AIL_MODEL.with(|c| {
        let c = c.borrow();
        f_page(c.as_ref().expect("ail_init must run first"))
    });
    ail_rt::flush();
    ail_web::set_out(html)
}
"####,
        );
    }
    s
}

/// Runtime pieces only present in web builds.
pub const RUNTIME_WEB: &str = r####"
#[allow(dead_code)]
mod ail_web {
    use std::cell::RefCell;

    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn host_log(ptr: *const u8, len: usize, is_err: u32);
        pub fn host_now() -> f64;
        pub fn host_trap(ptr: *const u8, len: usize);
    }

    thread_local! {
        static IN: RefCell<Vec<u8>> = RefCell::new(Vec::new());
        static CMDS: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        static STRS: RefCell<Vec<String>> = RefCell::new(Vec::new());
        static STR_IDS: RefCell<std::collections::HashMap<String, u32>> = RefCell::new(std::collections::HashMap::new());
        static OUT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    }

    /// JS writes event strings into this buffer before calling ail_event.
    #[no_mangle]
    pub extern "C" fn ail_in_buf(len: u32) -> *mut u8 {
        IN.with(|b| {
            let mut b = b.borrow_mut();
            b.clear();
            b.resize(len as usize, 0);
            b.as_mut_ptr()
        })
    }

    pub fn take_inputs(l1: usize, l2: usize) -> (String, String) {
        IN.with(|b| {
            let b = b.borrow();
            let s1 = String::from_utf8_lossy(&b[..l1.min(b.len())]).into_owned();
            let s2 = String::from_utf8_lossy(&b[l1.min(b.len())..(l1 + l2).min(b.len())]).into_owned();
            (s1, s2)
        })
    }

    pub fn begin_frame() {
        CMDS.with(|c| c.borrow_mut().clear());
        STRS.with(|s| s.borrow_mut().clear());
        STR_IDS.with(|s| s.borrow_mut().clear());
    }
    pub fn frame_len() -> u32 { CMDS.with(|c| c.borrow().len() as u32) }

    #[no_mangle]
    pub extern "C" fn ail_cmds() -> *const f64 { CMDS.with(|c| c.borrow().as_ptr()) }
    #[no_mangle]
    pub extern "C" fn ail_str_ptr(i: u32) -> *const u8 { STRS.with(|s| s.borrow()[i as usize].as_ptr()) }
    #[no_mangle]
    pub extern "C" fn ail_str_len(i: u32) -> u32 { STRS.with(|s| s.borrow()[i as usize].len() as u32) }
    #[no_mangle]
    pub extern "C" fn ail_out_ptr() -> *const u8 { OUT.with(|o| o.borrow().as_ptr()) }

    pub fn set_out(s: String) -> u32 {
        OUT.with(|o| {
            let mut o = o.borrow_mut();
            *o = s.into_bytes();
            o.len() as u32
        })
    }

    /// Strings (colors, texts) are sent once per frame: a frame drawing 4000 circles in 3 colors
    /// carries 3 strings, and the host decodes each once.
    fn intern(s: &str) -> f64 {
        // Most frames use a handful of strings: a linear scan beats hashing.
        if let Some(i) = STRS.with(|v| { let v = v.borrow(); if v.len() <= 8 { v.iter().position(|x| x == s) } else { None } }) {
            return i as f64;
        }
        STR_IDS.with(|ids| {
            let mut ids = ids.borrow_mut();
            if let Some(&i) = ids.get(s) {
                return i as f64;
            }
            let i = STRS.with(|v| {
                let mut v = v.borrow_mut();
                v.push(s.to_string());
                (v.len() - 1) as u32
            });
            ids.insert(s.to_string(), i);
            i as f64
        })
    }

    fn push(xs: &[f64]) { CMDS.with(|c| c.borrow_mut().extend_from_slice(xs)); }

    // Draw commands: opcode followed by a fixed number of operands.
    pub fn clear(color: &str) { let c = intern(color); push(&[0.0, c]); }
    pub fn rect(x: f64, y: f64, w: f64, h: f64, color: &str) { let c = intern(color); push(&[1.0, x, y, w, h, c]); }
    pub fn circle(x: f64, y: f64, r: f64, color: &str) { let c = intern(color); push(&[2.0, x, y, r, c]); }
    pub fn line(x1: f64, y1: f64, x2: f64, y2: f64, width: f64, color: &str) { let c = intern(color); push(&[3.0, x1, y1, x2, y2, width, c]); }
    pub fn text(x: f64, y: f64, size: f64, color: &str, s: &str) { let c = intern(color); let t = intern(s); push(&[4.0, x, y, size, c, t]); }

    pub fn log(s: &str, is_err: bool) { unsafe { host_log(s.as_ptr(), s.len(), is_err as u32) } }
    pub fn now_ms() -> f64 { unsafe { host_now() } }
    pub fn trap(s: &str) { unsafe { host_trap(s.as_ptr(), s.len()) } }
}
"####;

/// Native stand-in so apps also build natively (tests): drawing is a no-op.
pub const RUNTIME_WEB_STUB: &str = r####"
#[allow(dead_code)]
mod ail_web {
    pub fn clear(_color: &str) {}
    pub fn rect(_x: f64, _y: f64, _w: f64, _h: f64, _color: &str) {}
    pub fn circle(_x: f64, _y: f64, _r: f64, _color: &str) {}
    pub fn line(_x1: f64, _y1: f64, _x2: f64, _y2: f64, _width: f64, _color: &str) {}
    pub fn text(_x: f64, _y: f64, _size: f64, _color: &str, _s: &str) {}
}
"####;

/// The HTML/JS host page. `{TITLE}` is replaced by the app name.
pub const HOST_HTML: &str = r####"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #eee; font-family: system-ui, sans-serif; }
  canvas { display: block; width: 100vw; height: 100vh; touch-action: none; }
  #page { padding: 16px; }
  #trap { position: fixed; left: 0; right: 0; bottom: 0; background: #a00; color: #fff; padding: 8px 12px; font-family: monospace; white-space: pre-wrap; display: none; }
</style>
</head>
<body>
<div id="page"></div>
<canvas id="screen" hidden></canvas>
<div id="trap"></div>
<script>
(async () => {
  const dec = new TextDecoder(), enc = new TextEncoder();
  let mem;
  const str = (p, n) => dec.decode(new Uint8Array(mem.buffer, p, n));
  const trapBox = document.getElementById("trap");
  let dead = false;
  const imports = { env: {
    host_log: (p, n, e) => (e ? console.error : console.log)(str(p, n)),
    host_now: () => performance.now(),
    host_trap: (p, n) => { dead = true; trapBox.textContent = "trap: " + str(p, n); trapBox.style.display = "block"; },
  }};
  const { instance } = await WebAssembly.instantiateStreaming(fetch("app.wasm"), imports).catch(async () =>
    WebAssembly.instantiate(await (await fetch("app.wasm")).arrayBuffer(), imports));
  const x = instance.exports;
  mem = x.memory;
  const call = (f) => { if (dead) return; try { f(); } catch (e) { dead = true; if (!trapBox.textContent) { trapBox.textContent = "trap: " + e; } trapBox.style.display = "block"; } };
  const send = (kind, a = 0, b = 0, s1 = "", s2 = "") => call(() => {
    const b1 = enc.encode(s1), b2 = enc.encode(s2);
    const p = x.ail_in_buf(b1.length + b2.length);
    const view = new Uint8Array(mem.buffer, p, b1.length + b2.length);
    view.set(b1, 0); view.set(b2, b1.length);
    x.ail_event(kind, a, b, b1.length, b2.length);
  });
  call(() => x.ail_init());

  if (x.ail_view) {
    const cv = document.getElementById("screen");
    cv.hidden = false;
    document.getElementById("page").remove();
    const g = cv.getContext("2d");
    const resize = () => { const r = devicePixelRatio || 1; cv.width = innerWidth * r; cv.height = innerHeight * r; g.setTransform(r, 0, 0, r, 0, 0); send(8, innerWidth, innerHeight); };
    addEventListener("resize", resize); resize();
    addEventListener("keydown", e => { if (!e.repeat) send(1, 0, 0, e.key); if (e.key.startsWith("Arrow") || e.key === " ") e.preventDefault(); });
    addEventListener("keyup", e => send(2, 0, 0, e.key));
    cv.addEventListener("pointerdown", e => send(3, e.clientX, e.clientY));
    cv.addEventListener("pointerup", e => send(4, e.clientX, e.clientY));
    cv.addEventListener("pointermove", e => send(5, e.clientX, e.clientY));
    const draw = () => {
      let n = 0;
      call(() => { n = x.ail_view(); });
      if (dead) return;
      const c = new Float64Array(mem.buffer, x.ail_cmds(), n);
      const strs = [];
      const s = (i) => strs[i] ?? (strs[i] = str(x.ail_str_ptr(i), x.ail_str_len(i)));
      for (let i = 0; i < n;) {
        switch (c[i]) {
          case 0: g.fillStyle = s(c[i + 1]); g.fillRect(0, 0, innerWidth, innerHeight); i += 2; break;
          case 1: g.fillStyle = s(c[i + 5]); g.fillRect(c[i + 1], c[i + 2], c[i + 3], c[i + 4]); i += 6; break;
          case 2: g.fillStyle = s(c[i + 4]); g.beginPath(); g.arc(c[i + 1], c[i + 2], c[i + 3], 0, Math.PI * 2); g.fill(); i += 5; break;
          case 3: g.strokeStyle = s(c[i + 6]); g.lineWidth = c[i + 5]; g.beginPath(); g.moveTo(c[i + 1], c[i + 2]); g.lineTo(c[i + 3], c[i + 4]); g.stroke(); i += 7; break;
          case 4: g.fillStyle = s(c[i + 4]); g.font = c[i + 3] + "px system-ui, sans-serif"; g.textBaseline = "top"; g.fillText(s(c[i + 5]), c[i + 1], c[i + 2]); i += 6; break;
          default: i = n;
        }
      }
    };
    let last = performance.now();
    const frame = (t) => { send(0, Math.min(0.1, (t - last) / 1000)); last = t; draw(); if (!dead) requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
  } else if (x.ail_page) {
    const root = document.getElementById("page");
    let lastHtml = null;
    const render = () => {
      let n = 0;
      call(() => { n = x.ail_page(); });
      if (dead) return;
      const html = str(x.ail_out_ptr(), n);
      if (html === lastHtml) return;
      const focused = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.on : null;
      const sel = focused ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : null;
      root.innerHTML = html;
      lastHtml = html;
      if (focused) {
        const el = root.querySelector('[data-on="' + CSS.escape(focused) + '"]');
        if (el) { el.focus(); if (sel && el.setSelectionRange) { try { el.setSelectionRange(sel[0], sel[1]); } catch (e) {} } }
      }
    };
    root.addEventListener("click", e => { const el = e.target.closest("[data-on]"); if (el && el.tagName !== "INPUT" && el.tagName !== "TEXTAREA") { send(6, 0, 0, el.dataset.on); render(); } });
    root.addEventListener("input", e => { const el = e.target.closest("[data-on]"); if (el) { send(7, 0, 0, el.dataset.on, el.value); render(); } });
    root.addEventListener("keydown", e => { if (e.key === "Enter") { const el = e.target.closest("[data-on]"); if (el && el.tagName === "INPUT") { send(6, 0, 0, el.dataset.on + ":enter"); render(); } } });
    let last = performance.now();
    setInterval(() => { const t = performance.now(); send(0, (t - last) / 1000); last = t; render(); }, 250);
    render();
  }
})();
</script>
</body>
</html>
"####;
