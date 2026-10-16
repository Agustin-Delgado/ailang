"""Render the benchmark charts used by the top-level README.

Reads bench/results.json, bench/http/results*.json and bench/web/results.txt, and writes SVGs to
docs/charts/. Run from anywhere: python bench/charts.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "charts"

AIL = "#2f6fdf"
RUST = "#d9692b"
AXUM = "#8a8f98"
WIN = "#2e9e5b"
TIE = "#9aa3ad"
LOSE = "#d24b4b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def native():
    rows = json.loads((ROOT / "bench" / "results.json").read_text())
    rows.sort(key=lambda r: r["rust_s"] / r["ail_s"])
    names = [r["bench"] for r in rows]
    speedup = [r["rust_s"] / r["ail_s"] for r in rows]
    colors = [WIN if s >= 1.05 else LOSE if s <= 0.9 else TIE for s in speedup]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(names, speedup, color=colors, height=0.65)
    ax.axvline(1.0, color="#222", lw=1.2, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5x", "1x", "2x", "4x", "8x"])
    ax.set_xlim(0.5, 9)
    for y, s in enumerate(speedup):
        ax.text(max(s, 1.0) * 1.04, y, f"{s:.2f}x", va="center", fontsize=10)
    ax.set_xlabel("speed of ail relative to Rust (Rust time / ail time, log scale)")
    ax.set_title(f"ail vs idiomatic Rust, {len(rows)} native tasks\n"
                 "right of the line = ail is faster; identical output in every task",
                 fontsize=12, loc="left")
    save(fig, "native.svg")


def http_chart(results, out, title):
    data = json.loads((ROOT / "bench" / "http" / results).read_text())["results"]
    servers = [("ail-par", "ail http-serve", AIL), ("ail", "ail http-next loop", "#8fb3ef"),
               ("axum", "Rust axum + tokio", AXUM), ("std", "Rust std, 1 thread", RUST)]
    conc = sorted({r["concurrency"] for r in data})
    rps = {(r["server"], r["concurrency"]): r["rps"] for r in data}
    servers = [s for s in servers if (s[0], conc[0]) in rps]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.8 / len(servers)
    top = max(rps.values())
    for i, (key, label, color) in enumerate(servers):
        xs = [c + (i - (len(servers) - 1) / 2) * width for c in range(len(conc))]
        vals = [rps[(key, c)] for c in conc]
        bars = ax.bar(xs, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, f"{v:,.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(conc)))
    ax.set_xticklabels([f"{c} client{'s' if c > 1 else ''}" for c in conc])
    ax.set_ylabel("requests / second (higher is better)")
    ax.set_title(title, fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, out)


def http():
    http_chart("results.json", "http.svg",
               "Real HTTP server, I/O-bound: todo REST API over TCP\n30% POST, 70% GET of 20 todos")
    http_chart("results_search.json", "http_search.svg",
               "Real HTTP server, CPU-bound: every GET scans 100,000 todos\n10% POST, 90% GET /search")


def web():
    text = (ROOT / "bench" / "web" / "results.txt").read_text()
    parts = {}
    for lang in ("ail", "rust"):
        m = re.search(lang + r"\s+update ([\d.]+) ms/frame\s+view ([\d.]+)\s+host ([\d.]+)", text)
        parts[lang] = [float(x) for x in m.groups()]
    labels = ["ail", "Rust"]
    stages = [("update", "#4c78a8"), ("view", "#f2a541"), ("host decode", "#7a7a7a")]

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    for y, lang in enumerate(("ail", "rust")):
        left = 0.0
        for (stage, color), v in zip(stages, parts[lang]):
            ax.barh(y, v, left=left, color=color, height=0.55, label=stage if y == 0 else None)
            left += v
        ax.text(left + 0.02, y, f"{left:.2f} ms", va="center", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.7)
    ax.set_xlabel("milliseconds per frame (lower is better)")
    ax.set_title("Browser game (WebAssembly): 4,000 entities, same checksum", fontsize=12, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "web.svg")


def experiment():
    # From experiment/RESULTS.md: share of programs correct on the first attempt (hidden tests).
    groups = ["r2 Sonnet\n16 tasks", "r2 Haiku\n16 tasks", "r3 Sonnet\n8 harder tasks",
              "r3 Haiku\n8 harder tasks"]
    langs = [("ail", AIL, [15 / 16, 15 / 16, 7 / 8, 7 / 8]), ("Rust", RUST, [1, 1, 1, 1]),
             ("TypeScript", "#3a9d9d", [1, 1, 1, 7 / 8])]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    width = 0.26
    for i, (lang, color, vals) in enumerate(langs):
        xs = [g + (i - 1) * width for g in range(len(groups))]
        bars = ax.bar(xs, [v * 100 for v in vals], width, color=color, label=lang)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% correct on first attempt")
    ax.set_title("A language no model has seen: correct programs on the first try\n"
                 "(only SPEC.md in context; both ail misses in r3 were compiler bugs, fixed since)",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "experiment.svg")


if __name__ == "__main__":
    native()
    http()
    web()
    experiment()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))
