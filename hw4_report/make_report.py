import json
from pathlib import Path

import matplotlib.pyplot as plt

from sweep import EXPERIMENTS

RESULTS = Path("results")
PLOTS = Path("plots")
GROUPS = {
    "r": ["lora_r4", "lora_r8", "lora", "lora_r64", "lora_r256"],
    "alpha": ["lora_alpha8", "lora_alpha16", "lora", "lora_alpha64", "lora_alpha128"],
    "dropout": ["lora_dropout0", "lora", "lora_dropout0.1", "lora_dropout0.3"],
    "target_modules": ["lora_qv", "lora_attn", "lora_mlp", "lora"],
    "r, alpha = 2r": ["lora_r4_alpha8", "lora", "lora_r64_alpha128", "lora_r256_alpha512"],
}


def load():
    return {
        name: json.loads((RESULTS / name / "summary.json").read_text())
        for name in EXPERIMENTS
        if (RESULTS / name / "summary.json").exists()
    }


def predictions(name):
    with open(RESULTS / name / "predictions.jsonl") as f:
        return [json.loads(line)["correct"] for line in f]


def row(name, s):
    c = s["config"]
    lora = c["mode"] == "lora"
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} |".format(
        name,
        c["mode"],
        c["r"] if lora else "-",
        c["alpha"] if lora else "-",
        c["dropout"] if lora else "-",
        c["targets"] if lora else "-",
        c["lr"] if c["mode"] != "base" else "-",
        f"{s['trainable_params'] / 1e6:.2f}M ({100 * s['trainable_params'] / s['total_params']:.2f}%)" if "trainable_params" in s else "-",
        f"{s['train_seconds'] / 60:.1f}" if "train_seconds" in s else "-",
        f"{s['peak_memory_gib']:.1f}" if "peak_memory_gib" in s else "-",
        s["exec_accuracy"],
        s["mmlu"],
    )


def main():
    summaries = load()
    lines = [
        "| run | mode | r | alpha | dropout | targets | lr | trainable params | train, min | peak mem, GiB | exec acc | MMLU |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ] + [row(name, s) for name, s in summaries.items()]
    complexities = list(summaries["base"]["exec_by_complexity"])
    lines += ["", "| run | " + " | ".join(complexities) + " |", "|---" * (len(complexities) + 1) + "|"]
    for name in ("base", "full", "lora"):
        by = summaries[name]["exec_by_complexity"]
        lines.append(f"| {name} | " + " | ".join(f"{by[k]:.3f}" for k in complexities) + " |")
    lines += ["", "| pair | only first correct | only second correct | z |", "|---|---|---|---|"]
    for a, b in (("full", "lora"), ("lora", "lora_seed1"), ("lora", "base")):
        x, y = (predictions(n) for n in (a, b))
        only_a = sum(p and not q for p, q in zip(x, y))
        only_b = sum(q and not p for p, q in zip(x, y))
        lines.append(f"| {a} vs {b} | {only_a} | {only_b} | {(only_a - only_b) / (only_a + only_b) ** 0.5:.2f} |")
    (RESULTS / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    PLOTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(4 * len(GROUPS), 7))
    for column, (title, names) in enumerate(GROUPS.items()):
        names = [n for n in names if n in summaries]
        labels = [n.removeprefix("lora_") if n != "lora" else "base cfg" for n in names]
        for axis, metric in zip(axes[:, column], ("exec_accuracy", "mmlu")):
            values = [summaries[n][metric] for n in names]
            axis.bar(labels, values, color=["tab:orange" if n == "lora" else "tab:blue" for n in names])
            axis.axhline(summaries["full"][metric], color="tab:red", linestyle="--", label="full FT")
            axis.axhline(summaries["base"][metric], color="gray", linestyle=":", label="base")
            axis.set_ylim(min(values + [summaries["base"][metric]]) - 0.03, max(values + [summaries["full"][metric]]) + 0.02)
            axis.tick_params(axis="x", rotation=30)
            axis.set_title(f"{title}: {metric}")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "lora_params.png", dpi=120)

    plt.figure(figsize=(9, 4))
    for name in ("full", "lora", "lora_r4", "lora_r256"):
        if name in summaries:
            steps, losses = zip(*summaries[name]["loss_history"])
            plt.plot(steps, losses, label=name)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "train_loss.png", dpi=120)


if __name__ == "__main__":
    main()
