import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results")
LOGS = Path("logs")
PLOTS = Path("plots")
TABLES = Path("tables")
SLA_SECONDS = 15


def metrics(run):
    data = json.loads((run / "profile" / "profile_export_aiperf.json").read_text())
    return {
        "mean": data["request_latency"]["avg"] / 1000,
        "p50": data["request_latency"]["p50"] / 1000,
        "p90": data["request_latency"]["p90"] / 1000,
        "ttft": data["time_to_first_token"]["avg"],
        "ttft_p90": data["time_to_first_token"]["p90"],
        "itl": data["inter_token_latency"]["avg"],
        "osl": data["output_sequence_length"]["avg"],
        "rps": data["request_throughput"]["avg"],
        "tokens": data["output_token_throughput"]["avg"],
        "errors": data.get("error_summary") or [],
        "count": data["request_latency"]["count"],
        "cache_hit": cache_hit(run),
    }


def cache_hit(run):
    path = run / "profile" / "server_metrics_export.json"
    if not path.exists():
        return None
    server = json.loads(path.read_text())["metrics"]
    total = lambda name: sum(series["stats"]["total"] for series in server[name]["series"])
    return total("vllm:prefix_cache_hits") / total("vllm:prefix_cache_queries")


def acceptance(config, mode="latency"):
    values = [
        float(v)
        for log in LOGS.glob(f"{config}.{mode}.gpu*.server.log")
        for v in re.findall(r"Mean acceptance length: ([0-9.]+)", log.read_text())
    ]
    return sum(values) / len(values) if values else None


def responses(run):
    data = json.loads((run / "profile" / "outputs.json").read_text())["data"]
    return {r["session_num"]: r["response_text"] for r in data}


def common_prefix(run, reference):
    a, b = responses(run), responses(reference)
    shares = [len(os.path.commonprefix([a[k], b[k]])) / max(len(b[k]), 1) for k in a]
    return sum(shares) / len(shares)


def latency_table():
    runs = {run.name: metrics(run) for run in sorted((RESULTS / "latency").iterdir()) if (run / "profile" / "profile_export_aiperf.json").exists()}
    base = runs["baseline"]["mean"]
    rows = [
        "| конфигурация | mean, с | S | P50, с | P90, с | TTFT, мс | TTFT P90, мс | ITL, мс | токенов/с | принято за шаг | общий префикс с бейзлайном |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in sorted(runs.items(), key=lambda item: item[1]["mean"]):
        accepted = acceptance(name)
        rows.append(
            f"| `{name}` | {m['mean']:.3f} | {base / m['mean']:.2f} | {m['p50']:.3f} | {m['p90']:.3f} | {m['ttft']:.0f} | "
            f"{m['ttft_p90']:.0f} | {m['itl']:.2f} | {m['tokens']:.0f} | {f'{accepted:.2f}' if accepted else '-'} | "
            f"{common_prefix(RESULTS / 'latency' / name, RESULTS / 'latency' / 'baseline'):.0%} |"
        )
    return "\n".join(rows)


def load_runs(config):
    folder = RESULTS / "load" / config
    if not folder.exists():
        return {}
    return {int(run.name): metrics(run) for run in folder.iterdir() if (run / "profile" / "profile_export_aiperf.json").exists()}


def load_table(config):
    rows = [
        f"| RPM | P90, с | mean, с | P50, с | TTFT P90, мс | ITL, мс | выполнено RPM | из кэша префиксов | ошибки | P90 < {SLA_SECONDS} с |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rpm, m in sorted(load_runs(config).items()):
        ok = m["p90"] < SLA_SECONDS
        rows.append(
            f"| {rpm} | {m['p90']:.2f} | {m['mean']:.2f} | {m['p50']:.2f} | {m['ttft_p90']:.0f} | {m['itl']:.2f} | "
            f"{m['rps'] * 60:.0f} | {m['cache_hit']:.0%} | {sum(e.get('count', 1) for e in m['errors'])} | {'да' if ok else 'нет'} |"
        )
    return "\n".join(rows)


def spec_table(plain, spec):
    a, b = load_runs(plain), load_runs(spec)
    rows = [
        f"| RPM | P90 `{plain}`, с | P90 `{spec}`, с | P90 спекдек / без | mean `{plain}`, с | mean `{spec}`, с | выполнено RPM без / со спекдеком |",
        "|---|---|---|---|---|---|---|",
    ]
    for rpm in sorted(set(a) & set(b)):
        x, y = a[rpm], b[rpm]
        rows.append(
            f"| {rpm} | {x['p90']:.2f} | {y['p90']:.2f} | {y['p90'] / x['p90']:.2f} | {x['mean']:.2f} | {y['mean']:.2f} | "
            f"{x['rps'] * 60:.0f} / {y['rps'] * 60:.0f} |"
        )
    return "\n".join(rows)


def capacity_table(configs):
    rows = [
        f"| конфигурация | максимум с P90 < {SLA_SECONDS} с | P90 на максимуме, с | первое нарушение | P90 там, с |",
        "|---|---|---|---|---|",
    ]
    for config in configs:
        runs = sorted(load_runs(config).items())
        ok = [(r, m) for r, m in runs if m["p90"] < SLA_SECONDS]
        if not ok:
            continue
        best, m_best = max(ok)
        bad = [(r, m) for r, m in runs if r > best]
        rows.append(
            f"| `{config}` | {best} RPM | {m_best['p90']:.2f} | "
            + (f"{bad[0][0]} RPM | {bad[0][1]['p90']:.2f} |" if bad else "- | - |")
        )
    return "\n".join(rows)


def spec_plot(panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.5), sharey=True)
    for axis, (title, configs) in zip(axes, panels.items()):
        for config in configs:
            runs = sorted(load_runs(config).items())
            if runs:
                axis.plot([r for r, _ in runs], [m["p90"] for _, m in runs], marker="o", label=config)
        axis.axhline(SLA_SECONDS, color="red", linestyle="--", linewidth=1, label=f"{SLA_SECONDS} с")
        axis.set_title(title)
        axis.set_xlabel("нагрузка, RPM")
        axis.set_yscale("log")
        axis.grid(alpha=0.3, which="both")
        axis.legend()
    axes[0].set_ylabel("P90 latency, с")
    fig.tight_layout()
    fig.savefig(PLOTS / "spec_vs_load.png", dpi=120)
    plt.close(fig)


def load_plot(config, max_rpm=1200):
    runs = sorted((r, m) for r, m in load_runs(config).items() if r <= max_rpm)
    ok = [(r, m) for r, m in runs if m["p90"] < SLA_SECONDS]
    bad = [(r, m) for r, m in runs if (r, m) not in ok and r > max(ok)[0]] if ok else runs
    plt.figure(figsize=(9, 4.5))
    plt.plot([r for r, _ in runs], [m["p90"] for _, m in runs], marker="o", label="P90 E2E latency")
    plt.plot([r for r, _ in runs], [m["mean"] for _, m in runs], marker=".", alpha=0.6, label="mean latency")
    plt.axhline(SLA_SECONDS, color="red", linestyle="--", label=f"SLA: P90 < {SLA_SECONDS} с")
    if ok:
        rpm, m = max(ok)
        plt.scatter([rpm], [m["p90"]], s=160, color="green", zorder=3, label=f"максимум: {rpm} RPM")
    if bad:
        rpm, m = bad[0]
        plt.scatter([rpm], [m["p90"]], s=160, color="red", zorder=3, label=f"нарушение SLA: {rpm} RPM")
    plt.xlabel("нагрузка, RPM")
    plt.ylabel("latency, с")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "load_test.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    PLOTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    best = "fp8_dflash_k15_safe"
    tables = {
        "1_latency_120rpm.md": latency_table(),
        "2_spec_baseline_vs_eagle3.md": spec_table("baseline", "eagle3_k3"),
        "2_spec_baseline_vs_dflash.md": spec_table("baseline", "dflash_k15_safe"),
        "2_spec_fp8_vs_fp8_dflash.md": spec_table("fp8", best),
        "3_load_test_best.md": load_table(best),
        "3_max_rpm_by_config.md": capacity_table([best, "fp8_eagle3_k3_fulldecode", "fp8", "dflash_k15_safe", "eagle3_k3", "baseline"]),
    }
    for name, table in tables.items():
        (TABLES / name).write_text(table + "\n")
        print(name, table, sep="\n", end="\n\n")
    spec_plot({
        "bf16": ["baseline", "eagle3_k3", "dflash_k15_safe"],
        "FP8 веса": ["fp8", "fp8_eagle3_k3_fulldecode", best],
    })
    load_plot(best)
