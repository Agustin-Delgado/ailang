// Runs the arena game headless in Node, ail wasm vs Rust wasm: per frame one update (Tick 1/60),
// one view, and the host decoding the draw commands the way the browser host does (into a fake
// canvas). Usage: node harness.mjs [frames]
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

const FRAMES = Number(process.argv[2] ?? 1500);
const DT = 1 / 60;
const dec = new TextDecoder();

// Fake 2D context: does the bookkeeping a canvas call would, so decoding is not optimized away.
function fakeCanvas() {
  return { fillStyle: "", calls: 0, sum: 0, arc(x, y, r) { this.calls++; this.sum += Math.floor(x) + Math.floor(y); }, rect() { this.calls++; }, text() { this.calls++; } };
}

async function runAil(path) {
  let mem;
  const str = (p, n) => dec.decode(new Uint8Array(mem.buffer, p, n));
  const env = { host_log: (p, n) => console.log(str(p, n)), host_now: () => performance.now(), host_trap: (p, n) => { throw new Error("trap: " + str(p, n)); } };
  const { instance } = await WebAssembly.instantiate(readFileSync(path), { env });
  const x = instance.exports;
  mem = x.memory;
  x.ail_init();
  const g = fakeCanvas();
  let tUpdate = 0, tView = 0, tHost = 0;
  for (let f = 0; f < FRAMES; f++) {
    const t0 = performance.now();
    x.ail_event(0, DT, 0, 0, 0);
    const t1 = performance.now();
    const n = x.ail_view();
    const t2 = performance.now();
    g.sum = 0;
    const c = new Float64Array(mem.buffer, x.ail_cmds(), n);
    const strs = [];
    const s = (i) => strs[i] ?? (strs[i] = str(x.ail_str_ptr(i), x.ail_str_len(i)));
    for (let i = 0; i < n;) {
      switch (c[i]) {
        case 0: g.fillStyle = s(c[i + 1]); g.rect(); i += 2; break;
        case 1: g.fillStyle = s(c[i + 5]); g.rect(); i += 6; break;
        case 2: g.fillStyle = s(c[i + 4]); g.arc(c[i + 1], c[i + 2], c[i + 3]); i += 5; break;
        case 3: g.fillStyle = s(c[i + 6]); g.rect(); i += 7; break;
        case 4: g.fillStyle = s(c[i + 4]); g.text(s(c[i + 5])); i += 6; break;
        default: i = n;
      }
    }
    const t3 = performance.now();
    tUpdate += t1 - t0; tView += t2 - t1; tHost += t3 - t2;
  }
  return { tUpdate, tView, tHost, checksum: g.sum, calls: g.calls };
}

async function runRust(path) {
  const { instance } = await WebAssembly.instantiate(readFileSync(path), {});
  const x = instance.exports;
  const palette = ["#10131a", "#4dd0e1", "#ffd54f", "#ff5d73", "#e6e6e6"];
  x.init();
  const g = fakeCanvas();
  let tUpdate = 0, tView = 0, tHost = 0;
  for (let f = 0; f < FRAMES; f++) {
    const t0 = performance.now();
    x.tick(DT);
    const t1 = performance.now();
    const n = x.view();
    const t2 = performance.now();
    g.sum = 0;
    const c = new Float64Array(x.memory.buffer, x.cmds(), n);
    for (let i = 0; i < n;) {
      switch (c[i]) {
        case 0: g.fillStyle = palette[c[i + 1]]; g.rect(); i += 2; break;
        case 2: g.fillStyle = palette[c[i + 4]]; g.arc(c[i + 1], c[i + 2], c[i + 3]); i += 5; break;
        case 4: g.fillStyle = palette[c[i + 4]]; g.text(dec.decode(new Uint8Array(x.memory.buffer, x.hud_ptr(), x.hud_len()))); i += 5; break;
        default: i = n;
      }
    }
    const t3 = performance.now();
    tUpdate += t1 - t0; tView += t2 - t1; tHost += t3 - t2;
  }
  return { tUpdate, tView, tHost, checksum: g.sum, calls: g.calls };
}

const fmt = (r) => `update ${(r.tUpdate / FRAMES).toFixed(3)} ms/frame  view ${(r.tView / FRAMES).toFixed(3)}  host ${(r.tHost / FRAMES).toFixed(3)}  total ${((r.tUpdate + r.tView + r.tHost) / FRAMES).toFixed(3)}  checksum ${r.checksum}`;
const best = async (f) => { let b; for (let k = 0; k < 3; k++) { const r = await f(); if (!b || r.tUpdate + r.tView + r.tHost < b.tUpdate + b.tView + b.tHost) b = r; } return b; };
const a = await best(() => runAil(new URL("./ail-out/app.wasm", import.meta.url)));
const r = await best(() => runRust(new URL("./rust/target/wasm32-unknown-unknown/release/arena.wasm", import.meta.url)));
console.log("ail ", fmt(a));
console.log("rust", fmt(r));
const tot = (x) => x.tUpdate + x.tView + x.tHost;
console.log(`ail/rust: update ${(a.tUpdate / r.tUpdate).toFixed(2)}  view+host ${((a.tView + a.tHost) / (r.tView + r.tHost)).toFixed(2)}  total ${(tot(a) / tot(r)).toFixed(2)}  same result: ${a.checksum === r.checksum}`);
