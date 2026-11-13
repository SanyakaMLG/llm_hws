import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import markdown

import matplotlib.pyplot as plt
import numpy as np

from experiments import EXPERIMENTS, METRIC

RESULTS = Path("results")
RUNS = Path("runs")
DATA = Path("data")
PLOTS = Path("plots")
GROUPS = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML = """<!doctype html><html><head><meta charset="utf-8">
<script>MathJax = {{tex: {{inlineMath: [["$", "$"]]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<style>
body {{font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45; max-width: 1000px; margin: auto;}}
table {{border-collapse: collapse; margin: 8px 0; font-size: 9.5pt;}}
th, td {{border: 1px solid #bbb; padding: 3px 6px; vertical-align: top;}}
th {{background: #f2f2f2;}}
img {{max-width: 100%;}}
code {{background: #f4f4f4; padding: 0 3px; font-size: 9.5pt;}}
pre {{background: #f4f4f4; padding: 8px;}}
h2 {{border-bottom: 1px solid #ddd; padding-bottom: 3px;}}
</style></head><body>{}</body></html>
"""


def load(path):
    return json.loads(Path(path).read_text())


def predictions(name):
    return [json.loads(line) for line in open(RESULTS / name / "predictions.jsonl")]


def error_kind(p):
    if p["check"]["valid"]:
        return "correct"
    error = str(p["check"].get("error"))
    if p["check"].get("error_type"):
        return p["check"]["error_type"].split(":")[-1]
    if "не требовался" in error:
        return "extra call"
    if "Ожидался" in error:
        return "missing call"
    return "malformed"


def smooth(values, window=10):
    values = np.array(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid") if len(values) >= window else values


def eda_plots():
    eda = load(DATA / "eda.json")
    lengths = load(DATA / "eda_lengths.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    bins = np.linspace(0, 3000, 61)
    axes[0].hist(lengths["toolace"], bins=bins, alpha=0.6, density=True, label="ToolACE")
    axes[0].hist(lengths["bfcl"], bins=bins, alpha=0.6, density=True, label="BFCL sample")
    axes[0].set_xlabel("prompt tokens")
    axes[0].legend()
    for axis, key, title in ((axes[1], "tools_per_prompt", "tools in prompt"), (axes[2], "calls_per_answer", "calls in answer")):
        ours, theirs = eda[key], eda["bfcl_" + key]
        xs = sorted({int(x) for x in ours} | {int(x) for x in theirs})
        total_ours, total_theirs = sum(ours.values()), sum(theirs.values())
        axis.bar([x - 0.2 for x in xs], [ours.get(str(x), 0) / total_ours for x in xs], width=0.4, label="ToolACE")
        axis.bar([x + 0.2 for x in xs], [theirs.get(str(x), 0) / total_theirs for x in xs], width=0.4, label="BFCL sample")
        axis.set_xlabel(title)
        axis.set_ylabel("share")
        axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "eda.png", dpi=120)
    plt.close(fig)


def training_plots(names):
    keys = [
        ("loss", "loss"),
        ("reward", "train reward (mean over 128 completions)"),
        ("frac_reward_zero_std", "share of groups with equal rewards"),
        ("completions/mean_length", "completion length, tokens"),
    ]
    fig, grid = plt.subplots(2, 3, figsize=(16, 8))
    axes = grid.flatten()[: len(keys) + 1]
    for name in names:
        history = load(RUNS / name / "log_history.json")
        train = [h for h in history if "loss" in h and "step" in h and "eval_loss" not in h and "train_runtime" not in h]
        for axis, (key, title) in zip(axes, keys):
            steps = [h["step"] for h in train]
            values = [float(h[key]) for h in train]
            line, = axis.plot(steps, values, alpha=0.2)
            smoothed = smooth(values)
            axis.plot(steps[len(steps) - len(smoothed):], smoothed, color=line.get_color(), label=name)
            axis.set_title(title)
            axis.set_xlabel("step")
        evals = [(h["step"], float(h[METRIC])) for h in history if METRIC in h]
        selected = int((RUNS / name / "selected_step.txt").read_text())
        axes[-1].plot(*zip(*evals), marker="o", label=name)
        axes[-1].scatter([selected], [dict(evals)[selected]], s=120, facecolors="none", edgecolors="black")
    axes[-1].set_title("val checker accuracy (circle = selected step)")
    axes[-1].set_xlabel("step")
    scatter = grid.flatten()[-1]
    for name in names:
        history = load(RUNS / name / "log_history.json")
        val = max(float(h[METRIC]) for h in history if METRIC in h)
        bfcl = load(RESULTS / name / "metrics.json")["overall"]
        scatter.scatter(val, bfcl, s=60)
        scatter.annotate(name, (val, bfcl), textcoords="offset points", xytext=(5, 5), fontsize=9)
    scatter.set_xlabel("best val checker accuracy")
    scatter.set_ylabel("BFCL overall")
    scatter.set_title("validation vs benchmark")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "training.png", dpi=120)
    plt.close(fig)


def tables(names):
    rows = ["| run | selected step | Overall | Simple | Multiple | Parallel | Parallel multiple | Irrelevance |", "|---" * 8 + "|"]
    for name in ["base"] + names:
        m = load(RESULTS / name / "metrics.json")
        step = (RUNS / name / "selected_step.txt").read_text().strip() if name != "base" else "-"
        rows.append(f"| {name} | {step} | {m['overall']:.3f} | " + " | ".join(f"{m['groups'][g]:.3f}" for g in GROUPS) + " |")
    kinds = Counter()
    per_run = {}
    for name in ["base"] + names:
        per_run[name] = Counter(error_kind(p) for p in predictions(name))
        kinds.update(per_run[name])
    order = [k for k, _ in kinds.most_common() if k != "correct"]
    rows += ["", "| run | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
    for name, counts in per_run.items():
        rows.append(f"| {name} | " + " | ".join(str(counts.get(k, 0)) for k in order) + " |")
    base = {p["id"]: p["check"]["valid"] for p in predictions("base")}
    rows += ["", "| run | fixed vs base | broken vs base | z |", "|---|---|---|---|"]
    for name in names:
        run = {p["id"]: p["check"]["valid"] for p in predictions(name)}
        fixed = sum(run[i] and not base[i] for i in base)
        broken = sum(base[i] and not run[i] for i in base)
        rows.append(f"| {name} | {fixed} | {broken} | {(fixed - broken) / max(fixed + broken, 1) ** 0.5:.2f} |")
    (RESULTS / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def pdf():
    text = Path("README.md").read_text()
    formulas = []

    def keep(match):
        formulas.append(match.group(0))
        return f"MATH{len(formulas) - 1}MATH"

    text = re.sub(r"\$\$.+?\$\$|\$[^$\n]+\$", keep, text, flags=re.S)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = re.sub(r"MATH(\d+)MATH", lambda m: formulas[int(m.group(1))], body)
    Path("report.html").write_text(HTML.format(body))
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
         "--print-to-pdf=report.pdf", Path("report.html").resolve().as_uri()],
        check=True,
        capture_output=True,
    )


def main():
    PLOTS.mkdir(exist_ok=True)
    names = [n for n in EXPERIMENTS if (RESULTS / n / "metrics.json").exists()]
    eda_plots()
    training_plots(names)
    tables(names)
    pdf()


if __name__ == "__main__":
    main()
