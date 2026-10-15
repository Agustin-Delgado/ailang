"""Grade one solution: compile it, run every hidden test, compare stdout exactly.

usage: python grade.py <lang: ail|rust|ts> <task> <solution file> [--json]
Prints a JSON object: {"compiled": bool, "compile_output": str, "passed": n, "total": n, "failures": [...]}
"""
import json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AIL = os.environ.get("AIL") or os.path.join(HERE, "..", "bin", "ail.exe")
NODE_BIN = os.path.join(HERE, "node_modules", ".bin")
TIMEOUT = 20

def norm(s):
    return "\n".join(l.rstrip() for l in s.replace("\r\n", "\n").rstrip("\n").split("\n"))

def compile_solution(lang, src, work):
    if lang == "ail":
        exe = os.path.join(work, "prog.exe")
        chk = subprocess.run([AIL, "check", src, "--json"], capture_output=True, text=True)
        if chk.returncode != 0:
            return None, chk.stdout + chk.stderr
        p = subprocess.run([AIL, "build", src, "-o", exe, "--json"], capture_output=True, text=True)
        return (exe if p.returncode == 0 else None), p.stdout + p.stderr
    if lang == "rust":
        exe = os.path.join(work, "prog.exe")
        p = subprocess.run(["rustc", "--edition", "2021", "-O", "-o", exe, src], capture_output=True, text=True)
        return (exe if p.returncode == 0 else None), p.stderr
    if lang == "ts":
        tsc = os.path.join(NODE_BIN, "tsc.cmd" if os.name == "nt" else "tsc")
        shutil.copy(src, os.path.join(work, "main.ts"))
        p = subprocess.run([tsc, "--strict", "--target", "es2022", "--lib", "es2022", "--module", "commonjs", "--types", "node",
                            "--typeRoots", os.path.join(HERE, "node_modules", "@types"), "--skipLibCheck",
                            "--outDir", os.path.join(work, "out"), os.path.join(work, "main.ts")],
                           capture_output=True, text=True, shell=False)
        js = os.path.join(work, "out", "main.js")
        if p.returncode != 0 or not os.path.exists(js):
            return None, p.stdout + p.stderr
        return ["node", js], p.stdout
    raise SystemExit("unknown lang " + lang)

def main():
    lang, task, src = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
    tdir = os.path.join(HERE, "tasks", task, "tests")
    work = tempfile.mkdtemp(prefix="grade_")
    res = {"lang": lang, "task": task, "compiled": False, "compile_output": "", "passed": 0, "total": 0, "failures": []}
    exe, out = compile_solution(lang, src, work)
    res["compile_output"] = out[-4000:]
    ins = sorted(f for f in os.listdir(tdir) if f.endswith(".in"))
    res["total"] = len(ins)
    if exe is not None:
        res["compiled"] = True
        cmd = exe if isinstance(exe, list) else [exe]
        for f in ins:
            inp = open(os.path.join(tdir, f), encoding="utf8", newline="").read()
            want = open(os.path.join(tdir, f[:-3] + ".out"), encoding="utf8", newline="").read()
            try:
                p = subprocess.run(cmd, input=inp, capture_output=True, text=True, encoding="utf8", timeout=TIMEOUT)
                got, err, code = p.stdout, p.stderr, p.returncode
            except subprocess.TimeoutExpired:
                got, err, code = "", "timeout", -1
            if norm(got) == norm(want) and code == 0:
                res["passed"] += 1
            else:
                gl, wl = norm(got).split("\n"), norm(want).split("\n")
                first = next((i for i in range(max(len(gl), len(wl))) if (gl[i] if i < len(gl) else None) != (wl[i] if i < len(wl) else None)), None)
                res["failures"].append({
                    "test": f, "exit_code": code, "stderr": err[-800:],
                    "first_diff_line": None if first is None else first + 1,
                    "expected_line": wl[first] if first is not None and first < len(wl) else None,
                    "got_line": gl[first] if first is not None and first < len(gl) else None,
                    "input_preview": inp[:300],
                })
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
