import json
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINTS = Path("checkpoints")
RESULTS = Path("results")
LOGS = Path("logs")
METRIC = "eval_rewards/checker_reward/mean"
STALL_SECONDS = 900
EXPERIMENTS = {
    "grpo_checker": ["--loss", "grpo", "--reward", "checker"],
    "dr_grpo_checker": ["--loss", "dr_grpo", "--scale-rewards", "none", "--reward", "checker"],
    "grpo_partial": ["--loss", "grpo", "--reward", "partial"],
    "grpo_informative": ["--loss", "grpo", "--reward", "checker", "--informative", "--steps", "100"],
}


def best_step(name):
    history = json.loads((CHECKPOINTS / name / "log_history.json").read_text())
    evals = [(h[METRIC], h["step"]) for h in history if METRIC in h]
    saved = {int(p.name.split("-")[1]) for p in (CHECKPOINTS / name).glob("checkpoint-*")}
    return max((r, s) for r, s in evals if s in saved)[1]


def run(command, log):
    print(" ".join(command), flush=True)
    with open(log, "a") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)


def train(name):
    log = LOGS / f"{name}.log"
    while not (CHECKPOINTS / name / "log_history.json").exists():
        print("train", name, flush=True)
        log.touch()
        with open(log, "a") as f:
            process = subprocess.Popen([sys.executable, "train.py", "--name", name, *EXPERIMENTS[name]], stdout=f, stderr=subprocess.STDOUT)
        while process.poll() is None and time.time() - log.stat().st_mtime < STALL_SECONDS:
            time.sleep(30)
        if process.poll() is None:
            print("stalled, restarting from the last checkpoint", flush=True)
            process.kill()
            process.wait()
        elif process.returncode:
            raise RuntimeError(f"{name} failed")


def main():
    LOGS.mkdir(exist_ok=True)
    for name in sys.argv[1:] or EXPERIMENTS:
        if (RESULTS / name / "metrics.json").exists():
            continue
        train(name)
        step = best_step(name)
        (CHECKPOINTS / name / "selected_step.txt").write_text(f"{step}\n")
        final = CHECKPOINTS / name / "final"
        run([sys.executable, "export.py", str(final), "--adapter", str(CHECKPOINTS / name / f"checkpoint-{step}")], LOGS / f"{name}.log")
        run([sys.executable, "evaluate.py", str(final), "--output", str(RESULTS / name)], LOGS / f"{name}_eval.log")


if __name__ == "__main__":
    main()
