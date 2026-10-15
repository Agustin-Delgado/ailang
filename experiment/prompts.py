"""Prompts for the experiment: identical task text and rules for every language."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LANG_NOTES = {
    "ail": "Write the program in ail. The complete language specification follows; it is the only "
           "documentation that exists, ail did not exist before today.\n\n" + open(os.path.join(ROOT, "SPEC.md"), encoding="utf8").read(),
    "rust": "Write the program in Rust (edition 2021, standard library only, single file with fn main). "
            "It is compiled with `rustc --edition 2021 -O main.rs`.",
    "ts": "Write the program in TypeScript (single file). It is type-checked and compiled with "
          "`tsc --strict --target es2022 --lib es2022 --module commonjs --types node` and run with Node.js 25. "
          "Read all of stdin with `require(\"fs\").readFileSync(0, \"utf8\")` or "
          "`import * as fs from \"fs\"`. No npm packages besides Node built-ins.",
}

EXT = {"ail": "ail", "rust": "rs", "ts": "ts"}

def first_attempt_prompt(lang, task, out_path):
    statement = open(os.path.join(HERE, os.environ.get("EXP_TASKS", "tasks"), task, "task.md"), encoding="utf8").read()
    return f"""You are taking part in a programming-language experiment. Solve the task below in the
requested language and write the complete program to this exact file path:

    {out_path}

Rules:
- After reading this instruction file, use only the Write tool, exactly once, to create that file.
  Do not run any command, do not compile, do not test, do not read any other file. This measures
  your first attempt, so think carefully before writing.
- The program reads standard input and writes standard output exactly as specified.
- After writing the file, reply with just: DONE

=== Language ===
{LANG_NOTES[lang]}

=== Task ===
{statement}
"""

def repair_prompt(lang, grade):
    if not grade["compiled"]:
        what = "The program did not compile. Compiler output:\n\n" + grade["compile_output"]
    else:
        fails = []
        for f in grade["failures"][:3]:
            fails.append(
                f"- test {f['test']}: exit code {f['exit_code']}; first different line {f['first_diff_line']}: "
                f"expected {f['expected_line']!r}, got {f['got_line']!r}; stderr: {f['stderr'][-300:]!r}; "
                f"input starts with: {f['input_preview'][:200]!r}")
        what = (f"The program compiled but passed {grade['passed']}/{grade['total']} hidden tests. Failures:\n"
                + "\n".join(fails))
    return f"""{what}

Fix the program. Again use only the Write tool once to overwrite the same file with the complete
corrected program, do not run anything, then reply with just: DONE"""
