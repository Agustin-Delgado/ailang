"""Benchmark ail against idiomatic Rust: same task (bench/SPECS.md for the application ones),
same machine, best of N runs, outputs must be identical.

Rust baseline: cargo release profile in bench/rust (opt-level 3, codegen-units 1, fat LTO, panic=abort,
overflow checks OFF, serde_json where idiomatic).
ail: `ail build` (opt-level 3, overflow checks ON, bounds checks ON - ail never has undefined behavior).
"""
import os, subprocess, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
AIL = os.path.join(HERE, "..", "compiler", "target", "release", "ail.exe")
RUST = os.path.join(HERE, "rust")
RUNS = int(os.environ.get("RUNS", "5"))
NATIVE = "--native" in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(
    f[:-4] for f in os.listdir(os.path.join(HERE, "ail")) if f.endswith(".ail"))
names = [n for n in names if os.path.exists(os.path.join(RUST, "src", "bin", n + ".rs"))]

env = dict(os.environ)
if NATIVE:
    env["RUSTFLAGS"] = "-C target-cpu=native"
subprocess.run(["cargo", "build", "--release", "--quiet"] + [a for n in names for a in ("--bin", n)],
               cwd=RUST, check=True, env=env)

def build_ail(name):
    out = os.path.join(HERE, "bin")
    os.makedirs(out, exist_ok=True)
    exe = os.path.join(out, name + "_ail.exe")
    flags = ["--native"] if NATIVE else []
    subprocess.run([AIL, "build", os.path.join(HERE, "ail", name + ".ail"), "-o", exe] + flags, check=True, capture_output=True)
    return exe

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

path = os.path.join(HERE, "results.json")
results = {r["bench"]: r for r in json.load(open(path))} if os.path.exists(path) else {}
for n in names:
    a = build_ail(n)
    r = os.path.join(RUST, "target", "release", n + ".exe")
    ta, oa = timeit(a)
    tr, orr = timeit(r)
    same = oa.strip() == orr.strip()
    results[n] = {"bench": n, "ail_s": round(ta, 4), "rust_s": round(tr, 4), "ratio": round(ta / tr, 3), "same_output": same}
    print(f"{n:10s} ail {ta:7.3f}s  rust {tr:7.3f}s  ail/rust {ta/tr:5.2f}  output {'OK' if same else 'DIFFERENT'}", flush=True)
    if not same:
        print("   ail :", oa.strip()[:300]); print("   rust:", orr.strip()[:300])
json.dump(sorted(results.values(), key=lambda r: r["bench"]), open(path, "w"), indent=1)
