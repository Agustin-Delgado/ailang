"""Experiment driver. The agents themselves are launched by the orchestrator (Claude Code);
this script writes their instruction files and grades what they produce.

  python exp.py prepare <run>                  write runs/<run>/<lang>/<task>/PROMPT.md for every pair
  python exp.py grade <run> <attempt>          grade every solution; record results for that attempt
  python exp.py repair <run> <attempt>         for failing pairs write REPAIR_<attempt>.md (feedback)
  python exp.py autofix <run>                  apply compiler auto-fixes to failing first attempts (copy)
  python exp.py report <run>                   print the summary table
"""
import json, os, shutil, subprocess, sys
import prompts

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["ail", "rust", "ts"]
TASKS = sorted(os.listdir(os.path.join(HERE, "tasks")))

def pair_dir(run, lang, task):
    return os.path.join(HERE, "runs", run, lang, task)

def sol_path(run, lang, task):
    return os.path.join(pair_dir(run, lang, task), "main." + prompts.EXT[lang])

def grade(lang, task, path):
    p = subprocess.run([sys.executable, os.path.join(HERE, "grade.py"), lang, task, path], capture_output=True, text=True)
    return json.loads(p.stdout)

def results_path(run):
    return os.path.join(HERE, "runs", run, "results.jsonl")

def load(run):
    rows = []
    if os.path.exists(results_path(run)):
        rows = [json.loads(l) for l in open(results_path(run), encoding="utf8")]
    return rows

def cmd_prepare(run, langs=LANGS, tasks=TASKS):
    for lang in langs:
        for task in tasks:
            d = pair_dir(run, lang, task)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "PROMPT.md"), "w", encoding="utf8", newline="\n").write(
                prompts.first_attempt_prompt(lang, task, sol_path(run, lang, task)))
    print("prepared", len(langs) * len(tasks), "pairs")

def cmd_grade(run, attempt, only_failing=False):
    attempt = int(attempt)
    prev = {(r["lang"], r["task"]): r for r in load(run) if r["attempt"] == attempt - 1}
    with open(results_path(run), "a", encoding="utf8") as out:
        for lang in LANGS:
            for task in TASKS:
                path = sol_path(run, lang, task)
                if not os.path.exists(path):
                    continue
                if attempt > 0:
                    p = prev.get((lang, task))
                    if p is None or p["passed"] == p["total"]:
                        continue
                g = grade(lang, task, path)
                shutil.copy(path, path + f".attempt{attempt}")
                row = {"attempt": attempt, "lang": lang, "task": task, "compiled": g["compiled"],
                       "passed": g["passed"], "total": g["total"]}
                out.write(json.dumps(row) + "\n")
                json.dump(g, open(os.path.join(pair_dir(run, lang, task), f"grade{attempt}.json"), "w"), indent=1)
                print(f"{lang:5s} {task:10s} compiled={g['compiled']!s:5s} {g['passed']}/{g['total']}")

def cmd_repair(run, attempt):
    attempt = int(attempt)
    n = 0
    for r in load(run):
        if r["attempt"] == attempt and r["passed"] < r["total"]:
            d = pair_dir(run, r["lang"], r["task"])
            g = json.load(open(os.path.join(d, f"grade{attempt}.json")))
            open(os.path.join(d, f"REPAIR_{attempt + 1}.md"), "w", encoding="utf8", newline="\n").write(
                prompts.repair_prompt(r["lang"], g))
            print(r["lang"], r["task"])
            n += 1
    print(n, "pairs need repair")

def cmd_autofix(run):
    rows = [r for r in load(run) if r["attempt"] == 0]
    with open(os.path.join(HERE, "runs", run, "autofix.jsonl"), "w", encoding="utf8") as out:
        for r in rows:
            src = os.path.join(pair_dir(run, r["lang"], r["task"]), "main." + prompts.EXT[r["lang"]] + ".attempt0")
            if r["compiled"] or r["lang"] == "ts":
                continue
            fixed = src.replace(".attempt0", ".autofixed")
            shutil.copy(src, fixed)
            p = subprocess.run([sys.executable, os.path.join(HERE, "autofix.py"), r["lang"], fixed], capture_output=True, text=True)
            nfix = int((p.stdout.strip() or "0").split()[-1])
            g = grade(r["lang"], r["task"], fixed)
            row = {"lang": r["lang"], "task": r["task"], "fixes": nfix, "compiled": g["compiled"], "passed": g["passed"], "total": g["total"]}
            out.write(json.dumps(row) + "\n")
            print(row)

def cmd_report(run):
    rows = load(run)
    best = {}
    for r in rows:
        k = (r["lang"], r["task"])
        best.setdefault(k, []).append(r)
    print(f"{'lang':5s} {'compiles@1':>10s} {'correct@1':>10s} {'tests@1':>8s} {'correct@final':>13s} {'avg attempts to correct':>24s}")
    for lang in LANGS:
        ks = [k for k in best if k[0] == lang]
        if not ks:
            continue
        a0 = [next(r for r in best[k] if r["attempt"] == 0) for k in ks]
        comp = sum(r["compiled"] for r in a0)
        corr = sum(r["passed"] == r["total"] for r in a0)
        tests = sum(r["passed"] for r in a0) / max(1, sum(r["total"] for r in a0))
        final, attempts = 0, []
        for k in ks:
            ok = [r["attempt"] for r in best[k] if r["passed"] == r["total"]]
            if ok:
                final += 1
                attempts.append(min(ok) + 1)
        avg = sum(attempts) / len(attempts) if attempts else float("nan")
        print(f"{lang:5s} {comp:>5d}/{len(ks):<4d} {corr:>5d}/{len(ks):<4d} {tests:8.1%} {final:>8d}/{len(ks):<4d} {avg:24.2f}")

if __name__ == "__main__":
    c, *a = sys.argv[1:]
    {"prepare": cmd_prepare, "grade": cmd_grade, "repair": cmd_repair, "autofix": cmd_autofix, "report": cmd_report}[c](*a)
