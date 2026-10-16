"""Fan-out benchmark (bench/SPECS.md "fanout"): one client sends N GET requests to a backend, 64 in
flight at a time, and adds up the answers. ail (`par-run` + `http-request`) vs Rust (tokio + reqwest
+ buffered(64)). The backend is the Rust axum server, the same for both clients.

Modes: "todos" (cheap answers: the client's own cost dominates) and "search" (each answer scans
100,000 todos: the backend dominates). Rounds alternate ail and Rust; the median time is kept.
Writes results_fanout.json.

usage: python fanout.py [--requests N] [--rounds R]
"""
import argparse, json, os, re, statistics, subprocess, sys

import run  # build(), port_open(), wait_ready(), paths

HERE = os.path.dirname(os.path.abspath(__file__))
AIL_CLIENT = os.path.join(run.BIN, "fanout_ail.exe")
RUST_CLIENT = os.path.join(run.REL, "fanout.exe")


def once(exe, port, n, mode):
    out = subprocess.run([exe, str(port), str(n), mode], capture_output=True, text=True, check=True).stdout
    m = re.search(r"sum (\d+) failed (\d+) ms (\d+)", out)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    run.build()
    subprocess.run([run.AIL, "build", os.path.join(HERE, "fanout.ail"), "-o", AIL_CLIENT], check=True, capture_output=True)
    port, exe = run.SERVERS["axum"]
    if run.port_open(port):
        raise SystemExit(f"port {port} is already in use")
    backend = subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    try:
        run.wait_ready(port, backend)
        for mode in ("todos", "search"):
            times = {"ail": [], "rust": []}
            sums = set()
            failed = 0
            once(AIL_CLIENT, port, a.requests, mode)  # warm the backend
            for _ in range(a.rounds):
                for name, client in (("ail", AIL_CLIENT), ("rust", RUST_CLIENT)):
                    s, f, ms = once(client, port, a.requests, mode)
                    sums.add(s)
                    failed += f
                    times[name].append(ms)
            row = {"mode": mode, "requests": a.requests, "ail_ms": statistics.median(times["ail"]),
                   "rust_ms": statistics.median(times["rust"]), "all_ms": times,
                   "same_sum": len(sums) == 1, "failed": failed}
            results.append(row)
            print(f"{mode:6s} ail {row['ail_ms']:6.0f} ms   rust {row['rust_ms']:6.0f} ms   "
                  f"ail/rust {row['ail_ms'] / row['rust_ms']:.2f}   same sum {row['same_sum']}   failed {failed}", flush=True)
    finally:
        backend.kill()
        backend.wait()
    meta = {"in_flight": 64, "backend": "Rust axum (bench/http/server_axum)", "rounds": a.rounds, "kept": "median",
            "platform": sys.platform}
    with open(os.path.join(HERE, "results_fanout.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
