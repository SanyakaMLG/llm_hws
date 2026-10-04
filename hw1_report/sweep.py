import glob
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from train import MAX_LENGTH, NUM_SHARDS, OUTPUT_DIR, TRAINING_CONFIG, prepare_dataset

GPUS = os.environ.get('GPUS', '0,1,2,3').split(',')
SWEEP_DIR = f'{OUTPUT_DIR}/sweep'
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')

BASE = {
    'eval_strategy': 'no',
    'save_strategy': 'no',
    'load_best_model_at_end': False,
    'save_only_model': True,
    'per_device_eval_batch_size': 32,
    'lr_scheduler_type': 'constant_with_warmup',
    'per_device_train_batch_size': 32,
    'learning_rate': 6e-4,
}

MAX_STEPS = {16: 3210, 32: 1720, 64: 900}


def decayed(batch_size, lr, scheduler='cosine', **extra):
    return {'per_device_train_batch_size': batch_size, 'learning_rate': lr,
            'lr_scheduler_type': scheduler, 'max_steps': MAX_STEPS[batch_size], **extra}


BEST_WSD = decayed(32, 6e-4, 'warmup_stable_decay', lr_scheduler_kwargs={'num_decay_steps': MAX_STEPS[32] // 5})


EXPERIMENTS = {
    # batch size x learning rate
    'bs8_lr3e-4': {'per_device_train_batch_size': 8, 'learning_rate': 3e-4},
    'bs8_lr6e-4': {'per_device_train_batch_size': 8},
    'bs16_lr3e-4': {'per_device_train_batch_size': 16, 'learning_rate': 3e-4},
    'bs16_lr6e-4': {'per_device_train_batch_size': 16},
    'bs16_lr1e-3': {'per_device_train_batch_size': 16, 'learning_rate': 1e-3},
    'bs32_lr6e-4': {},
    'bs32_lr1e-3': {'learning_rate': 1e-3},
    'bs64_lr6e-4': {'per_device_train_batch_size': 64},
    'bs64_lr1e-3': {'per_device_train_batch_size': 64, 'learning_rate': 1e-3},
    # gradient accumulation: same effective batch 32 as bs32_lr6e-4
    'bs16_ga2': {'per_device_train_batch_size': 16, 'gradient_accumulation_steps': 2},
    'bs8_ga4': {'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 4},
    # one change at a time vs bs32_lr6e-4
    'no_compile': {'torch_compile': False},
    'adamw_torch': {'optim': 'adamw_torch'},
    'adafactor': {'optim': 'adafactor'},
    'no_bf16': {'bf16': False},
    'no_tf32': {'tf32': False},

    'bs32_lr6e-4_seed1': {'seed': 1},
    'bs32_lr6e-4_seed2': {'seed': 2},
    'bs32_lr3e-4': {'learning_rate': 3e-4},
    'bs64_lr3e-4': {'per_device_train_batch_size': 64, 'learning_rate': 3e-4},
    'lin_bs32_lr6e-4': decayed(32, 6e-4, 'linear'),
    'cos_bs32_lr6e-4': decayed(32, 6e-4),
    'wsd_bs32_lr6e-4': BEST_WSD,
    'cos_bs32_lr1e-3': decayed(32, 1e-3),
    'cos_bs16_lr3e-4': decayed(16, 3e-4),
    'cos_bs16_lr6e-4': decayed(16, 6e-4),
    'cos_bs64_lr6e-4': decayed(64, 6e-4),
    'cos_bs64_lr1e-3': decayed(64, 1e-3),

    'final_wsd_seed42': {**BEST_WSD, 'save_strategy': 'steps', 'save_steps': 344, 'save_total_limit': None},
    'final_wsd_seed1': {**BEST_WSD, 'seed': 1},
    'final_wsd_seed2': {**BEST_WSD, 'seed': 2},
    'final_wsd_seed3': {**BEST_WSD, 'seed': 3},
}

free_gpus = Queue()
for gpu in GPUS:
    free_gpus.put(gpu)


def run(name, overrides):
    out = f'{SWEEP_DIR}/{name}'
    if os.path.exists(f'{out}/trainer_state.json'):
        return
    gpu = free_gpus.get()
    try:
        os.makedirs(out, exist_ok=True)
        config = {**BASE, **overrides, 'output_dir': out, 'run_name': name}
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
        print(f'start {name} on GPU {gpu}', flush=True)
        with open(f'{out}/train.log', 'w') as log:
            code = subprocess.run(
                [sys.executable, TRAIN_SCRIPT, json.dumps(config)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        print(f'done  {name} on GPU {gpu}, exit code {code}', flush=True)
    finally:
        free_gpus.put(gpu)


def report():
    rows = []
    for name, overrides in EXPERIMENTS.items():
        path = f'{SWEEP_DIR}/{name}/trainer_state.json'
        if not os.path.exists(path):
            print(f'{name:16} FAILED, see {SWEEP_DIR}/{name}/train.log')
            continue
        with open(path) as f:
            state = json.load(f)
        eval_loss = [e['eval_loss'] for e in state['log_history'] if 'eval_loss' in e][-1]
        config = {**TRAINING_CONFIG, **BASE, **overrides}
        tokens = state['global_step'] * config['per_device_train_batch_size'] \
            * config['gradient_accumulation_steps'] * MAX_LENGTH
        rows.append((eval_loss, name, state['global_step'], tokens))
    for eval_loss, name, steps, tokens in sorted(rows):
        print(f'{name:16} eval_loss={eval_loss:.4f}  ppl={math.exp(eval_loss):8.1f}  '
              f'steps={steps:6}  tokens={tokens / 1e6:.1f}M')


def main():
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_PROJECT', 'qwen3-1b-ru-pretrain')
    if len(glob.glob(f'{OUTPUT_DIR}/*.parquet')) < NUM_SHARDS:
        prepare_dataset()
    with ThreadPoolExecutor(len(GPUS)) as pool:
        list(pool.map(lambda item: run(*item), EXPERIMENTS.items()))
    report()


if __name__ == '__main__':
    main()
