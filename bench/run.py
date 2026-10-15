"""Benchmark ail against idiomatic Rust: same algorithm, same machine, best of N runs.

Rust baseline: rustc -C opt-level=3 -C codegen-units=1 -C panic=abort (release, overflow checks OFF).
ail: `ail build` (opt-level 3, overflow checks ON, bounds checks ON - ail never has undefined behavior).
"""
import os, subprocess, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
AIL = os.path.join(HERE, "..", "compiler", "target", "release", "ail.exe")
RUNS = int(os.environ.get("RUNS", "3"))
NATIVE = "--native" in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(
    f[:-4] for f in os.listdir(os.path.join(HERE, "ail")) if f.endswith(".ail"))

def build(name):
    out = os.path.join(HERE, "bin")
    os.makedirs(out, exist_ok=True)
    ail_exe = os.path.join(out, name + "_ail.exe")
    rs_exe = os.path.join(out, name + "_rust.exe")
    flags = ["--native"] if NATIVE else []
    t = time.perf_counter()
    subprocess.run([AIL, "build", os.path.join(HERE, "ail", name + ".ail"), "-o", ail_exe] + flags, check=True, capture_output=True)
    ail_ct = time.perf_counter() - t
    rflags = ["-C", "opt-level=3", "-C", "codegen-units=1", "-C", "panic=abort", "--edition", "2021"]
    if NATIVE:
        rflags += ["-C", "target-cpu=native"]
    t = time.perf_counter()
    subprocess.run(["rustc", *rflags, "-o", rs_exe, os.path.join(HERE, "rust", name + ".rs")], check=True, capture_output=True)
    rs_ct = time.perf_counter() - t
    return ail_exe, rs_exe, ail_ct, rs_ct

def timeit(exe):
    best, out = None, None
    for _ in range(RUNS):
        t = time.perf_counter()
        p = subprocess.run([exe], capture_output=True, text=True)
        dt = time.perf_counter() - t
        if p.returncode != 0:
            raise SystemExit(f"{exe} failed: {p.stderr}")
        out = p.stdout
        best = dt if best is None else min(best, dt)
    return best, out

results = []
for n in names:
    a, r, act, rct = build(n)
    ta, oa = timeit(a)
    tr, orr = timeit(r)
    same = oa.strip() == orr.strip()
    results.append({"bench": n, "ail_s": round(ta, 3), "rust_s": round(tr, 3), "ratio": round(ta / tr, 3), "same_output": same})
    print(f"{n:10s} ail {ta:7.3f}s  rust {tr:7.3f}s  ail/rust {ta/tr:5.2f}  output {'OK' if same else 'DIFFERENT'}", flush=True)
    if not same:
        print("   ail :", oa.strip()[:200]); print("   rust:", orr.strip()[:200])
json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
