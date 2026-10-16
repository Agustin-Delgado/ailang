"""Time the solutions of a run on each task's large input (tasks_r4/<task>/perf/00.in).

usage: python perf.py <run> [<run>...] [--runs N] [--ail path/to/ail.exe] [--out file.json]

Every solution that compiles is built once (the same way grade.py builds it), then run N times on
the large input with a 60 s limit; the best time counts, and only if the output is right. Runs of
the different languages alternate, so load from the rest of the machine hits all of them.
Prints a table and writes the rows as JSON.
"""
import json, os, shutil, subprocess, sys, tempfile, time

import grade

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(HERE, "tasks_r4")
LIMIT = 60


def main():
    args = sys.argv[1:]
    runs_n, out_path = 3, None
    if "--runs" in args:
        i = args.index("--runs")
        runs_n = int(args[i + 1])
        del args[i:i + 2]
    if "--ail" in args:
        i = args.index("--ail")
        grade.AIL = os.path.abspath(args[i + 1])
        del args[i:i + 2]
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        del args[i:i + 2]
    rows = []
    for run in args:
        for task in sorted(os.listdir(TASKS_DIR)):
            inp = open(os.path.join(TASKS_DIR, task, "perf", "00.in"), encoding="utf8", newline="").read()
            want = open(os.path.join(TASKS_DIR, task, "perf", "00.out"), encoding="utf8", newline="").read()
            built = {}
            for lang in ("ail", "rust", "ts"):
                src = os.path.join(HERE, "runs", run, lang, task, "main." + {"ail": "ail", "rust": "rs", "ts": "ts"}[lang])
                if not os.path.exists(src):
                    continue
                work = tempfile.mkdtemp(prefix="perf_")
                exe, _ = grade.compile_solution(lang, src, work)
                built[lang] = (exe, work)
            times = {lang: [] for lang in built}
            status = {lang: "ok" if built[lang][0] else "no build" for lang in built}
            for _ in range(runs_n):
                for lang, (exe, _) in built.items():
                    if status[lang] != "ok":
                        continue
                    cmd = exe if isinstance(exe, list) else [exe]
                    t = time.perf_counter()
                    try:
                        p = subprocess.run(cmd, input=inp, capture_output=True, text=True, encoding="utf8", timeout=LIMIT)
                    except subprocess.TimeoutExpired:
                        status[lang] = "timeout"
                        continue
                    dt = time.perf_counter() - t
                    if p.returncode != 0:
                        status[lang] = f"exit {p.returncode}: {p.stderr.strip()[-120:]}"
                    elif grade.norm(p.stdout) != grade.norm(want):
                        status[lang] = "wrong output"
                    else:
                        times[lang].append(dt)
            for lang, (_, work) in built.items():
                shutil.rmtree(work, ignore_errors=True)
                ok = status[lang] == "ok"
                row = {"run": run, "task": task, "lang": lang, "status": status[lang],
                       "seconds": round(min(times[lang]), 4) if ok and times[lang] else None}
                rows.append(row)
                print(f"{run:12s} {task:8s} {lang:4s} {row['seconds'] if row['seconds'] is not None else '-':>8}  {status[lang]}", flush=True)
    if out_path:
        json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
