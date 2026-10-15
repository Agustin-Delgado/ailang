"""A/B timing of two ail compilers on the same benchmark programs, runs interleaved so machine
noise hits both sides equally.

usage: python tools/ab.py <old ail.exe> <new ail.exe> <bench name>... [RUNS=7]
"""
import os, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
old, new, names = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3:]
runs = int(os.environ.get("RUNS", "7"))
work = tempfile.mkdtemp(prefix="ab_")

def build(ail, name, tag):
    exe = os.path.join(work, f"{name}_{tag}.exe")
    src = os.path.join(ROOT, "bench", "ail", name + ".ail")
    subprocess.run([ail, "build", src, "-o", exe], check=True, capture_output=True)
    return exe

def once(exe):
    t = time.perf_counter()
    p = subprocess.run([exe], capture_output=True, text=True)
    return time.perf_counter() - t, p.stdout

for n in names:
    a, b = build(old, n, "old"), build(new, n, "new")
    ta, tb, outs = [], [], set()
    for _ in range(runs):
        t, o = once(a); ta.append(t); outs.add(o)
        t, o = once(b); tb.append(t); outs.add(o)
    print(f"{n:10s} old {min(ta):.3f}s  new {min(tb):.3f}s  new/old {min(tb)/min(ta):.2f}  "
          f"{'same output' if len(outs) == 1 else 'OUTPUT DIFFERS'}", flush=True)
