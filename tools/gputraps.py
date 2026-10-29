"""Checks that loops run on the GPU trap exactly as on the CPU.

usage: python tools/gputraps.py [ail.exe]

Every program in tests/gpu_traps/ is built once and run twice: with AIL_GPU=0 (CPU only) and with
AIL_GPU=force (every eligible loop on the GPU). Both runs must print the same output, the same trap
message and exit with the same code. A line marked `; traps here` is where the trap must point;
a program without the mark must not trap. The forced run must also have launched at least one
kernel, or the test proves nothing. With no Vulkan device the forced runs are skipped.
"""
import os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIL = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "compiler", "target", "release", "ail.exe")
DIR = os.path.join(ROOT, "tests", "gpu_traps")
LOG = re.compile(r"^\[gpu\].*$\n?", re.M)


def run(exe, gpu):
    env = dict(os.environ, AIL_GPU=gpu, AIL_GPU_LOG="1")
    p = subprocess.run([exe], capture_output=True, env=env, timeout=120)
    err = p.stderr.decode(errors="replace").replace("\r\n", "\n")
    launches = err.count(" ran in") + err.count(" fell back in")
    no_device = "no device" in err
    return p.returncode, p.stdout.decode(errors="replace").replace("\r\n", "\n"), LOG.sub("", err), launches, no_device


def check(name):
    src = os.path.join(DIR, name)
    lines = open(src, encoding="utf8").read().split("\n")
    want = [i + 1 for i, l in enumerate(lines) if l.rstrip().endswith("; traps here")]
    exe = os.path.join(tempfile.mkdtemp(prefix="gputrap_"), "p.exe")
    b = subprocess.run([AIL, "build", src, "-o", exe], capture_output=True, text=True)
    if b.returncode:
        return name, ["build failed: " + (b.stdout + b.stderr)[-600:]]
    cpu = run(exe, "0")
    gpu = run(exe, "force")
    bad = []
    if want:
        at = re.search(r"at " + re.escape(name) + r":(\d+):", cpu[2])
        if cpu[0] == 0 or not at or int(at.group(1)) != want[0]:
            bad.append(f"CPU run should trap at line {want[0]}: exit {cpu[0]}, stderr {cpu[2][-300:]!r}")
    elif cpu[0] != 0:
        bad.append(f"CPU run trapped: {cpu[2][-300:]!r}")
    if gpu[4]:
        return name, bad + ["(no Vulkan device: GPU run skipped)"]
    if gpu[3] == 0:
        bad.append("forced run launched no kernel")
    for what, a, g in (("exit code", cpu[0], gpu[0]), ("output", cpu[1], gpu[1]), ("stderr", cpu[2], gpu[2])):
        if a != g:
            bad.append(f"{what} differs: CPU {a!r} GPU {g!r}"[:600])
    return name, bad


names = sorted(f for f in os.listdir(DIR) if f.endswith(".ail"))
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(check, names))
failed = 0
for name, bad in results:
    real = [b for b in bad if not b.startswith("(")]
    failed += bool(real)
    print(("FAIL " if real else "ok   ") + name + "".join("\n     " + b for b in bad))
print(f"{len(names)} programs, {failed} failed")
sys.exit(1 if failed else 0)
