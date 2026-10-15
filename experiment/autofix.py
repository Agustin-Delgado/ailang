"""Apply the compiler's own machine-applicable fixes to a solution, without any model involved.

ail: `ail fix`. Rust: rustc's MachineApplicable suggestions (what `cargo fix` uses), iterated.
TypeScript: tsc has no command-line fix mode, so nothing is applied.
usage: python autofix.py <lang> <file>  -> rewrites the file in place, prints how many fixes applied
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AIL = os.path.join(HERE, "..", "bin", "ail.exe")

def fix_rust(path):
    total = 0
    for _ in range(10):
        src = open(path, encoding="utf8", newline="").read()
        out = tempfile.mktemp(suffix=".exe")
        p = subprocess.run(["rustc", "--edition", "2021", "--error-format=json", "-o", out, path],
                           capture_output=True, text=True)
        edits = []
        for line in p.stderr.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("level") != "error":
                continue
            for c in d.get("children", []):
                for sp in c.get("spans", []):
                    if sp.get("suggestion_applicability") == "MachineApplicable" and sp.get("suggested_replacement") is not None:
                        edits.append((sp["byte_start"], sp["byte_end"], sp["suggested_replacement"]))
        if os.path.exists(out):
            os.remove(out)
        if not edits:
            break
        b = src.encode("utf8")
        edits.sort(key=lambda e: -e[0])
        last = None
        for s, e, r in edits:
            if last is not None and e > last:
                continue
            b = b[:s] + r.encode("utf8") + b[e:]
            last = s
            total += 1
        open(path, "w", encoding="utf8", newline="").write(b.decode("utf8"))
    return total

def main():
    lang, path = sys.argv[1], sys.argv[2]
    if lang == "ail":
        p = subprocess.run([AIL, "fix", path], capture_output=True, text=True)
        line = next((l for l in p.stderr.splitlines() if l.startswith("applied")), "applied 0")
        print(int(line.split()[1]))
    elif lang == "rust":
        print(fix_rust(path))
    else:
        print(0)

if __name__ == "__main__":
    main()
