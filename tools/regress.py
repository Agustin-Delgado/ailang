"""Regression check for the compiler: every program in the repo must still build, the test files
in tests/ must pass, and every experiment solution must pass the same hidden tests as before.

usage: python tools/regress.py <ail.exe> <out.json> [baseline.json]
With a baseline, prints every program whose result changed and exits 1 if any got worse.
tests/gpu_traps/ only gets built here; tools/gputraps.py runs it on the CPU and the GPU.
"""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIL = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else None

def build(src):
    rel = os.path.relpath(src, ROOT).replace("\\", "/")
    work = tempfile.mkdtemp(prefix="regress_")
    web = "/web/" in rel or rel.endswith("dodge.ail")
    if rel.startswith("tests/") and rel.count("/") == 1:
        args = [AIL, "test", src]
    else:
        args = [AIL, "build", src] + (["--target", "web", "-o", work] if web else ["-o", os.path.join(work, "p.exe")])
    p = subprocess.run(args, capture_output=True, text=True)
    return rel, {"built": p.returncode == 0, "log": (p.stdout + p.stderr)[-1500:] if p.returncode else ""}

def grade(src):
    rel = os.path.relpath(src, ROOT).replace("\\", "/")
    task = rel.split("/")[-2]
    run = rel.split("/")[2]
    env = dict(os.environ, AIL=AIL, EXP_TASKS="tasks_" + run[:2] if run[:2] in ("r3", "r4", "r5", "r6", "r7") else "tasks")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "experiment", "grade.py"), "ail", task, src],
                       capture_output=True, text=True, env=env)
    r = json.loads(p.stdout)
    return rel, {"built": r["compiled"], "passed": r["passed"], "total": r["total"],
                 "log": "" if r["compiled"] else r["compile_output"][-1500:]}

progs, sols = [], []
for d, _, fs in os.walk(ROOT):
    if any(x in d for x in ("target", "node_modules", ".git", ".claude")):
        continue
    for f in fs:
        p = os.path.join(d, f)
        if f == "main.ail" and os.sep + "runs" + os.sep in p:
            sols.append(p)
        elif f.endswith(".ail") and not f.startswith("broken"):
            text = open(p, encoding="utf8").read()
            # Library files (used by another file, no entry point) are built through their user.
            if "(fn main" in text or "(fn init" in text or "(test " in text:
                progs.append(p)

with ThreadPoolExecutor(int(os.environ.get("REGRESS_JOBS", "4"))) as ex:
    res = dict(ex.map(build, progs))
    res.update(ex.map(grade, sols))
json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
bad = [k for k, v in res.items() if not v["built"]]
print(f"{len(res)} programs, {len(bad)} failed to build:", *bad, sep="\n  ")
if BASE:
    base = json.load(open(BASE))
    worse = 0
    for k, v in sorted(res.items()):
        b = base.get(k)
        if b is None:
            continue
        if (b["built"], b.get("passed")) != (v["built"], v.get("passed")):
            got_worse = (b["built"] and not v["built"]) or (v.get("passed") or 0) < (b.get("passed") or 0)
            worse += got_worse
            print(("WORSE " if got_worse else "better ") + k, b.get("passed"), "->", v.get("passed"), v["log"][:600])
    sys.exit(1 if worse else 0)
