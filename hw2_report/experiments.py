import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

GPUS = os.environ.get("GPUS", "0,1,2,3").split(",")
RESULTS = Path("results")
SHORT_SECONDS = 90
BASELINE = {"attn_implementation": "sdpa", "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2,
            "torch_compile": False, "padding_free": False}
FA2 = {"attn_implementation": "flash_attention_2"}
PF = {**FA2, "padding_free": True}
PACK = {**FA2, "packing": True}
BS64 = {"per_device_train_batch_size": 64, "gradient_accumulation_steps": 1}
COMPILE_180 = {"torch_compile": True, "train_seconds": 180}

EXPERIMENTS = {
    "baseline_300": {"train_seconds": 300},
    "baseline_90": {},
    "bs64_ga1": BS64,
    "grad_ckpt": {"gradient_checkpointing": True},
    "fa2": FA2,
    "padding_free_fa2": PF,
    "packing_fa2": PACK,
    "compile": {"torch_compile": True},
    "liger": {"use_liger_kernel": True},
    "offload": {"activation_offloading": True},
    "offload_fa2": {**FA2, "activation_offloading": True},
    "liger_bs64_ga1": {"use_liger_kernel": True, **BS64},
    "pf_fa2_bs64_ga1": {**PF, **BS64},
    "pf_fa2_liger": {**PF, "use_liger_kernel": True},
    "pf_fa2_liger_bs256_ga1": {**PF, "use_liger_kernel": True, "per_device_train_batch_size": 256, "gradient_accumulation_steps": 1},
    "pack_fa2_liger": {**PACK, "use_liger_kernel": True},
    "pack_fa2_liger_bs64_ga1": {**PACK, "use_liger_kernel": True, **BS64},
    "pack_fa2_liger_ckpt_bs128_ga1": {**PACK, "use_liger_kernel": True, "gradient_checkpointing": True, "per_device_train_batch_size": 128, "gradient_accumulation_steps": 1},
    "compile_180": {"torch_compile": True, "train_seconds": 180},
    "pf_fa2_compile_180": {**PF, "torch_compile": True, "train_seconds": 180},
    "pf_fa2_liger_compile_180": {**PF, "use_liger_kernel": True, "torch_compile": True, "train_seconds": 180},
    "pack_fa2_liger_compile_180": {**PACK, "use_liger_kernel": True, "torch_compile": True, "train_seconds": 180},
    "pf_fa2_compile_bs64_ga1_180": {**PF, **COMPILE_180, **BS64},
    "pf_fa2_compile_bs128_ga1_180": {**PF, **COMPILE_180, "per_device_train_batch_size": 128, "gradient_accumulation_steps": 1},
    "pf_fa2_compile_bs256_ga1_180": {**PF, **COMPILE_180, "per_device_train_batch_size": 256, "gradient_accumulation_steps": 1},
    "pack_fa2_compile_180": {**PACK, **COMPILE_180},
    "pack_fa2_compile_bs64_ga1_180": {**PACK, **COMPILE_180, **BS64},
    "final_pf_compile_bs256_300": {**PF, "torch_compile": True, "per_device_train_batch_size": 256, "gradient_accumulation_steps": 1, "train_seconds": 300},
    "final_pf_compile_bs128_300": {**PF, "torch_compile": True, "per_device_train_batch_size": 128, "gradient_accumulation_steps": 1, "train_seconds": 300},
    "final_pf_compile_bs64_300": {**PF, "torch_compile": True, **BS64, "train_seconds": 300},
    "baseline_300_rerun": {"train_seconds": 300},
}

free_gpus = Queue()
for gpu in GPUS:
    free_gpus.put(gpu)


def run(name, overrides):
    out = RESULTS / name
    if (out / "summary.json").exists() or (RESULTS / f"{name}.log").exists():
        return
    shutil.rmtree(out, ignore_errors=True)
    gpu = free_gpus.get()
    try:
        config = {"train_seconds": SHORT_SECONDS, **BASELINE, **overrides, "output_dir": str(out)}
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        print(f"start {name} on GPU {gpu}", flush=True)
        with open(RESULTS / f"{name}.log", "w") as log:
            code = subprocess.run(
                [sys.executable, "train.py", json.dumps(config)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        print(f"done  {name} on GPU {gpu}, exit code {code}", flush=True)
    finally:
        free_gpus.put(gpu)


def load(name):
    path = RESULTS / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def report():
    base = {90: load("baseline_90"), 300: load("baseline_300")}
    print(f"{'name':32} {'sec':>4} {'tok/s':>9} {'S':>6} {'peak GiB':>9} {'val loss':>9} {'steps':>6}")
    for name, overrides in EXPERIMENTS.items():
        summary = load(name)
        if summary is None:
            print(f"{name:32} FAILED, see {RESULTS / name}.log")
            continue
        seconds = overrides.get("train_seconds", SHORT_SECONDS)
        reference = base.get(seconds) or base[90]
        tps = summary["tokens_per_second"]
        if tps is None:
            print(f"{name:32} {seconds:4} {'n/a':>9} {'n/a':>6} {summary['peak_allocated_gib']:9.1f} "
                  f"{summary['validation_loss']:9.3f} {summary['optimizer_steps']:6}")
            continue
        speedup = tps / reference["tokens_per_second"] if reference else float("nan")
        print(f"{name:32} {seconds:4} {tps:9.0f} {speedup:6.2f} "
              f"{summary['peak_allocated_gib']:9.1f} {summary['validation_loss']:9.3f} {summary['optimizer_steps']:6}")


def main():
    RESULTS.mkdir(exist_ok=True)
    if not Path("data/dataset").exists():
        import train
        train.prepare_data()
    with ThreadPoolExecutor(len(GPUS)) as pool:
        list(pool.map(lambda item: run(*item), EXPERIMENTS.items()))
    report()


if __name__ == "__main__":
    main()
