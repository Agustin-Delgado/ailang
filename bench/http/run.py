"""Real-HTTP benchmark (bench/SPECS.md "http"): ail vs Rust std (same architecture) vs Rust axum+tokio.

Builds the three servers and the load generator, then for each server and each concurrency level:
start the server alone (fresh store), wait until it answers, run loadgen (warmup + fixed duration,
fresh connection per request, 30% POST / 70% GET limit=20), kill the server. Writes results.json.

usage: python run.py [--duration S] [--warmup S] [--conc 1,8,64] [--only ail,std,axum]
"""
import argparse, json, os, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
AIL = os.path.join(HERE, "..", "..", "bin", "ail.exe")
REL = os.path.join(HERE, "target", "release")
BIN = os.path.join(HERE, "bin")

SERVERS = {
    "ail": (8101, os.path.join(BIN, "server_ail.exe")),
    "std": (8102, os.path.join(REL, "server_std.exe")),
    "axum": (8103, os.path.join(REL, "server_axum.exe")),
}


def build():
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=HERE, check=True)
    os.makedirs(BIN, exist_ok=True)
    subprocess.run([AIL, "build", os.path.join(HERE, "server.ail"), "-o", SERVERS["ail"][1]],
                   check=True, capture_output=True)


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_ready(port, proc, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit(f"server on {port} exited early with code {proc.returncode}")
        if port_open(port):
            return
        time.sleep(0.05)
    raise SystemExit(f"server on {port} did not start")


def run_one(name, conc, duration, warmup):
    port, exe = SERVERS[name]
    if port_open(port):
        raise SystemExit(f"port {port} is already in use; stop whatever is listening there")
    proc = subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(port, proc)
        out = subprocess.run([os.path.join(REL, "loadgen.exe"), str(port), str(conc), str(duration), str(warmup)],
                             capture_output=True, text=True, check=True)
        if out.stderr.strip():
            print(f"   loadgen: {out.stderr.strip()}", file=sys.stderr)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        r["server"] = name
        return r
    finally:
        proc.kill()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--conc", default="1,8,64")
    ap.add_argument("--only", default="ail,std,axum")
    a = ap.parse_args()
    concs = [int(c) for c in a.conc.split(",")]
    names = a.only.split(",")

    build()
    results = []
    for conc in concs:
        for name in names:
            r = run_one(name, conc, a.duration, a.warmup)
            results.append(r)
            print(f"  {name:5s} c={conc:<3d} {r['rps']:9.0f} req/s  p50 {r['p50_us']/1000:7.2f} ms  "
                  f"p99 {r['p99_us']/1000:7.2f} ms  errors {r['errors']}", flush=True)

    meta = {"duration_s": a.duration, "warmup_s": a.warmup, "mix": "30% POST /todos, 70% GET /todos?limit=20&offset=0",
            "connection": "new TCP connection per request, Connection: close", "platform": sys.platform}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)

    by = {(r["server"], r["concurrency"]): r for r in results}
    print()
    print(f"{'conc':>4} | " + " | ".join(f"{n + ' req/s':>12} {'p50 ms':>7} {'p99 ms':>7}" for n in names)
          + (" | ail/std  ail/axum" if "ail" in names else ""))
    for c in concs:
        row = f"{c:>4} | " + " | ".join(
            f"{by[(n, c)]['rps']:>12.0f} {by[(n, c)]['p50_us'] / 1000:>7.2f} {by[(n, c)]['p99_us'] / 1000:>7.2f}"
            for n in names)
        if "ail" in names:
            ail = by[("ail", c)]["rps"]
            ratios = [f"{ail / by[(n, c)]['rps']:7.2f}" if (n, c) in by else "      -" for n in ("std", "axum")]
            row += " | " + "  ".join(ratios)
        print(row)
    errs = sum(r["errors"] for r in results)
    print(f"\ntotal errors: {errs}")


if __name__ == "__main__":
    main()
