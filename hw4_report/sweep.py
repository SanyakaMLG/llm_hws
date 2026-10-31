import os
import subprocess
import sys
import time

EXPERIMENTS = {
    "full": ["--mode", "full", "--lr", "1e-5"],
    "base": ["--mode", "base"],
    "lora": [],
    "lora_r4": ["--r", "4"],
    "lora_r8": ["--r", "8"],
    "lora_r64": ["--r", "64"],
    "lora_r256": ["--r", "256"],
    "lora_alpha8": ["--alpha", "8"],
    "lora_alpha16": ["--alpha", "16"],
    "lora_alpha64": ["--alpha", "64"],
    "lora_alpha128": ["--alpha", "128"],
    "lora_dropout0": ["--dropout", "0"],
    "lora_dropout0.1": ["--dropout", "0.1"],
    "lora_dropout0.3": ["--dropout", "0.3"],
    "lora_qv": ["--targets", "qv"],
    "lora_attn": ["--targets", "attn"],
    "lora_mlp": ["--targets", "mlp"],
    "lora_r4_alpha8": ["--r", "4", "--alpha", "8"],
    "lora_r64_alpha128": ["--r", "64", "--alpha", "128"],
    "lora_r256_alpha512": ["--r", "256", "--alpha", "512"],
    "lora_seed1": ["--seed", "1"],
}


def main(gpus=4):
    from run import DATA, RESULTS, prepare_data

    if not (DATA / "test.jsonl").exists():
        prepare_data()
    names = sys.argv[1:] or list(EXPERIMENTS)
    queue = [name for name in names if not (RESULTS / name / "summary.json").exists()]
    running = {}
    while queue or running:
        for gpu in range(gpus):
            if gpu in running and running[gpu][1].poll() is not None:
                name, process = running.pop(gpu)
                print(time.strftime("%H:%M:%S"), "done", name, "exit", process.returncode, flush=True)
            if gpu not in running and queue:
                name = queue.pop(0)
                (RESULTS / name).mkdir(parents=True, exist_ok=True)
                with open(RESULTS / name / "log.txt", "w") as log:
                    process = subprocess.Popen(
                        [sys.executable, "run.py", "--name", name, *EXPERIMENTS[name]],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    )
                running[gpu] = (name, process)
                print(time.strftime("%H:%M:%S"), "start", name, "gpu", gpu, flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
