"""Checks that loops run on the thread pool give the sequential output and trap.

usage: python tools/partraps.py [ail.exe]

Every program in tests/par_traps/ is built once and run with AIL_THREADS=1 (sequential) and with
every core. Both runs must print the same output, the same trap message and exit with the same
code. A line marked `; traps here` is where the trap must point; a program without the mark must
not trap. Each parallel run is repeated, since a race shows up only some of the time.
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIL = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "compiler", "target", "release", "ail.exe")
DIR = os.path.join(ROOT, "tests", "par_traps")
REPEAT = 5


def run(exe, threads):
    env = dict(os.environ)
    env.pop("AIL_THREADS", None)
    if threads:
        env["AIL_THREADS"] = threads
    p = subprocess.run([exe], capture_output=True, env=env, timeout=120)
    return p.returncode, p.stdout.decode(errors="replace").replace("\r\n", "\n"), p.stderr.decode(errors="replace").replace("\r\n", "\n")


failed = 0
names = sorted(f for f in os.listdir(DIR) if f.endswith(".ail"))
for name in names:
    src = os.path.join(DIR, name)
    lines = open(src, encoding="utf8").read().split("\n")
    want = [i + 1 for i, l in enumerate(lines) if l.rstrip().endswith("; traps here")]
    exe = os.path.join(tempfile.mkdtemp(prefix="partrap_"), "p.exe")
    b = subprocess.run([AIL, "build", src, "-o", exe], capture_output=True, text=True)
    bad = []
    if b.returncode:
        bad.append("build failed: " + (b.stdout + b.stderr)[-600:])
    else:
        seq = run(exe, "1")
        if want:
            at = re.search(r"at " + re.escape(name) + r":(\d+):", seq[2])
            if seq[0] == 0 or not at or int(at.group(1)) != want[0]:
                bad.append(f"sequential run should trap at line {want[0]}: exit {seq[0]}, stderr {seq[2][-300:]!r}")
        elif seq[0] != 0:
            bad.append(f"sequential run trapped: {seq[2][-300:]!r}")
        for _ in range(REPEAT):
            par = run(exe, None)
            for what, a, g in (("exit code", seq[0], par[0]), ("output", seq[1], par[1]), ("stderr", seq[2], par[2])):
                if a != g:
                    bad.append(f"{what} differs: sequential {a!r} parallel {g!r}"[:600])
            if bad:
                break
    print(("ok   " if not bad else "FAIL ") + name)
    for x in bad:
        print("     " + x)
    failed += bool(bad)
print(f"{len(names)} programs, {failed} failed")
sys.exit(1 if failed else 0)
