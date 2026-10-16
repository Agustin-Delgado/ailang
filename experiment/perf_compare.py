"""Compare perf.py results: ail/Rust per task and model, for one or more measurements.

usage: python perf_compare.py <label>=<perf.json> [<label>=<perf.json> ...]
Prints, for every run and task, the ail and Rust times of each measurement and the ail/Rust
ratio, then the geometric mean of the ratios per measurement.
"""
import json, math, sys

sets = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    rows = json.load(open(path))
    sets.append((label, {(r["run"], r["task"], r["lang"]): r["seconds"] for r in rows}))

keys = sorted({(run, task) for _, d in sets for (run, task, _) in d})
head = f"{'run':10s} {'task':8s}" + "".join(f" | {label:>8s} ail   rust  ratio" for label, _ in sets)
print(head)
logs = {label: [] for label, _ in sets}
for run, task in keys:
    line = f"{run[3:]:10s} {task:8s}"
    for label, d in sets:
        a, r = d.get((run, task, "ail")), d.get((run, task, "rust"))
        if a and r:
            logs[label].append(math.log(a / r))
            line += f" | {'':>8s} {a:5.2f} {r:6.2f} {a / r:6.2f}"
        else:
            line += f" | {'':>8s} {'-':>5s} {'-':>6s} {'-':>6s}"
    print(line)
for label, ls in logs.items():
    if ls:
        print(f"{label}: geometric mean ail/rust = {math.exp(sum(ls) / len(ls)):.2f} over {len(ls)} programs; "
              f"ail faster in {sum(1 for x in ls if x < 0)}")
