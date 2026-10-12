import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments import EXPERIMENTS

RESULTS = Path("results")
PLOTS = Path("plots")
BASELINE = "baseline_300"
FINAL = ["final_pf_compile_bs256_300", "final_pf_compile_bs128_300", "final_pf_compile_bs64_300"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, TEXT, TEXT_2, GRID, SURFACE = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "legend.frameon": False, "legend.labelcolor": TEXT, "lines.linewidth": 1.8, "font.size": 10,
})

LABELS = {
    "attn_implementation": lambda v: "FA2" if v == "flash_attention_2" else v,
    "padding_free": lambda v: "padding-free",
    "packing": lambda v: "packing",
    "torch_compile": lambda v: "compile",
    "use_liger_kernel": lambda v: "Liger",
    "gradient_checkpointing": lambda v: "act. checkpointing",
    "activation_offloading": lambda v: "act. offloading",
}


def describe(overrides):
    parts = [LABELS[k](v) for k, v in overrides.items() if k in LABELS and v]
    bs = overrides.get("per_device_train_batch_size", 32)
    ga = overrides.get("gradient_accumulation_steps", 2)
    return ", ".join(parts + [f"bs {bs} × ga {ga}"]) if parts or (bs, ga) != (32, 2) else "исходный конфиг (SDPA, bs 32 × ga 2)"


SINGLE = {"bs64_ga1", "grad_ckpt", "fa2", "padding_free_fa2", "packing_fa2", "compile", "compile_180",
          "liger", "offload", "offload_fa2"}


def group(name):
    if name.startswith("baseline"):
        return "бейзлайн"
    if name.startswith("final"):
        return "финальные 300 с"
    return "один механизм" if name in SINGLE else "сочетание"


GROUP_COLORS = {"бейзлайн": GRAY, "один механизм": BLUE, "сочетание": ORANGE, "финальные 300 с": AQUA}


def load(name):
    path = RESULTS / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def failure(name):
    log = RESULTS / f"{name}.log"
    if not log.exists():
        return "не запускался"
    lines = [line for line in log.read_text().splitlines() if "Error" in line]
    return lines[-1].strip()[:140] if lines else "упал без сообщения об ошибке"


def table():
    base = load(BASELINE)["tokens_per_second"]
    rows = ["| Эксперимент | Что включено | Длит., с | tok/s | S | Пик, ГиБ | val loss | Шагов |",
            "|---|---|---|---|---|---|---|---|"]
    failed = []
    for name, overrides in EXPERIMENTS.items():
        s = load(name)
        if s is None:
            failed.append(f"- `{name}` ({describe(overrides)}): `{failure(name)}`")
            continue
        tps = s["tokens_per_second"]
        tps_cell, speedup = (f"{tps:,.0f}".replace(",", " "), f"{tps / base:.2f}") if tps else ("n/a", "n/a")
        rows.append(f"| `{name}` | {describe(overrides)} | {s['train_seconds']:.0f} | {tps_cell} | {speedup} | "
                    f"{s['peak_allocated_gib']:.1f} | {s['validation_loss']:.3f} | {s['optimizer_steps']} |")
    return "\n".join(rows), "\n".join(failed)


def throughput_plot():
    runs = [(n, load(n)) for n in EXPERIMENTS]
    runs = [(n, s) for n, s in runs if s and s["tokens_per_second"]]
    runs.sort(key=lambda item: item[1]["tokens_per_second"])
    base = load(BASELINE)["tokens_per_second"]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(runs) + 1.4))
    values = [s["tokens_per_second"] / 1000 for _, s in runs]
    ax.barh([n for n, _ in runs], values, color=[GROUP_COLORS[group(n)] for n, _ in runs], height=0.7)
    for i, (v, (_, s)) in enumerate(zip(values, runs)):
        ax.text(v + 0.3, i, f"{s['tokens_per_second'] / base:.2f}×", va="center", fontsize=8, color=TEXT_2)
    ax.axvline(base / 1000, color=TEXT_2, linewidth=1, linestyle="--")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("полезных токенов в секунду, тыс. (подписи — ускорение к baseline_300)")
    ax.set_title("Пропускная способность всех успешных конфигураций", loc="left")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in GROUP_COLORS.items()], loc="lower right", fontsize=9)
    save(fig, "throughput.png")


def loss_curve(name):
    elapsed, losses = [], []
    for line in (RESULTS / f"{name}.log").read_text().splitlines():
        if line.startswith('{"step"'):
            record = json.loads(line)
            if record["step"] % 10 == 0:
                elapsed.append(record["elapsed_seconds"])
        elif line.startswith("{'loss'"):
            losses.append(ast.literal_eval(line)["loss"])
    n = min(len(elapsed), len(losses))
    return elapsed[:n], losses[:n]


def loss_plot():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, color in [(BASELINE, GRAY), (FINAL[2], BLUE), (FINAL[1], ORANGE), (FINAL[0], AQUA)]:
        s = load(name)
        if s is None:
            continue
        t, loss = loss_curve(name)
        ax.plot(t, loss, color=color, label=f"{name}: val loss {s['validation_loss']:.3f}")
    ax.set_xlabel("время с начала обучения, с (включая компиляцию)")
    ax.set_ylabel("train loss")
    ax.set_ylim(3.5, 11.5)
    ax.set_title("Train loss по реальному времени: бейзлайн и финальные запуски", loc="left")
    ax.legend(fontsize=9)
    save(fig, "loss_vs_time.png")


def save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)
    print("saved", PLOTS / name)


if __name__ == "__main__":
    rows, failed = table()
    print(rows)
    print()
    print(failed)
    throughput_plot()
    loss_plot()
