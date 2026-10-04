import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sweep import BASE, EXPERIMENTS
from train import MAX_LENGTH, TRAINING_CONFIG

RUNS = 'results/runs'
PLOTS = 'plots'
FINAL = 'final_wsd_seed42'

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, TEXT, TEXT_2, GRID, SURFACE = '#a3a29c', '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'xtick.color': TEXT_2, 'ytick.color': TEXT_2,
    'legend.frameon': False, 'legend.labelcolor': TEXT, 'lines.linewidth': 1.8, 'font.size': 10,
})


def config(name):
    return {**TRAINING_CONFIG, **BASE, **EXPERIMENTS[name]}


def load(name):
    with open(f'{RUNS}/{name}/trainer_state.json') as f:
        history = json.load(f)['log_history']
    cfg = config(name)
    train = [e for e in history if 'loss' in e]
    return {
        'step': np.array([e['step'] for e in train]),
        'loss': np.array([e['loss'] for e in train]),
        'lr': np.array([e['learning_rate'] for e in train]),
        'eval_loss': [e['eval_loss'] for e in history if 'eval_loss' in e][-1],
        'tokens_per_step': cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps'] * MAX_LENGTH,
    }


def smooth(run, tokens=0.5e6):
    window = max(1, round(tokens / run['tokens_per_step']))
    kernel = np.ones(window) / window
    return run['step'][window - 1:], np.convolve(run['loss'], kernel, mode='valid')


def tokens_m(run, steps):
    return steps * run['tokens_per_step'] / 1e6


def group(name):
    if name.startswith(('lin_', 'cos_', 'wsd_', 'final_')):
        return 'шедулер lr (с max_steps)'
    if name in ('no_compile', 'adamw_torch', 'adafactor', 'no_bf16', 'no_tf32'):
        return 'compile / оптимизатор / dtype'
    return 'batch size / lr / накопление'


GROUP_COLORS = {'batch size / lr / накопление': BLUE, 'compile / оптимизатор / dtype': ORANGE,
                'шедулер lr (с max_steps)': AQUA}


def ranking(runs):
    names = sorted(runs, key=lambda n: runs[n]['eval_loss'], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(names) + 1.4))
    seeds = [runs[n]['eval_loss'] for n in ('bs32_lr6e-4', 'bs32_lr6e-4_seed1', 'bs32_lr6e-4_seed2') if n in runs]
    if seeds:
        ax.axvspan(min(seeds), max(seeds), color=GRID, alpha=0.9, lw=0, label='разброс 3 сидов bs32_lr6e-4')
    for i, n in enumerate(names):
        ax.scatter(runs[n]['eval_loss'], i, s=42, color=GROUP_COLORS[group(n)], zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
    for g, c in GROUP_COLORS.items():
        ax.scatter([], [], s=42, color=c, label=g)
    ax.set_yticks(range(len(names)), names)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('финальный eval_loss (меньше — лучше)')
    ax.set_title('Все эксперименты: финальный eval_loss за 15 минут обучения', loc='left')
    losses = [runs[n]['eval_loss'] for n in names]
    ax.set_xlim(min(losses) - 0.02, max(losses) + 0.02)
    ax.legend(loc='upper right', fontsize=9)
    save(fig, 'ranking.png')


def batch_lr(runs):
    sizes = [8, 16, 32, 64]
    lrs = {3e-4: BLUE, 6e-4: ORANGE, 1e-3: AQUA}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, bs in zip(axes, sizes):
        for lr, color in lrs.items():
            name = f'bs{bs}_lr{lr:.0e}'.replace('e-0', 'e-')
            if name not in runs:
                continue
            run = runs[name]
            steps, loss = smooth(run)
            ax.plot(tokens_m(run, steps), loss, color=color, label=f'lr {lr:g}  →  eval {run["eval_loss"]:.3f}')
        ax.set_title(f'batch size {bs}', loc='left')
        ax.set_xlabel('токенов увидено, млн')
        ax.set_ylim(3.3, 5.2)
        ax.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel('train loss (сглажен)')
    fig.suptitle('Batch size × learning rate (constant lr после warmup)', x=0.01, ha='left', color=TEXT)
    save(fig, 'batch_lr.png')


def schedulers(runs):
    names = {'bs32_lr6e-4': ('constant', GRAY), 'lin_bs32_lr6e-4': ('linear', ORANGE),
             'cos_bs32_lr6e-4': ('cosine', BLUE), 'wsd_bs32_lr6e-4': ('WSD', AQUA)}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, height_ratios=[1, 2])
    for name, (label, color) in names.items():
        run = runs[name]
        top.plot(run['step'], run['lr'] * 1e4, color=color, label=label)
        steps, loss = smooth(run)
        bottom.plot(steps, loss, color=color, label=f'{label}  →  eval {run["eval_loss"]:.3f}')
    top.set_ylabel('lr, ×1e-4')
    top.set_title('Шедулеры lr при bs32, lr 6e-4: расписание и train loss', loc='left')
    top.legend(fontsize=8, ncol=4, loc='lower left')
    bottom.set_ylim(3.3, 5.2)
    bottom.set_ylabel('train loss (сглажен)')
    bottom.set_xlabel('шаг')
    bottom.legend(fontsize=9, loc='upper right')
    save(fig, 'schedulers.png')


def compile_optim_dtype(runs):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for name, label, color in [('bs32_lr6e-4', 'база: fused AdamW + compile', GRAY),
                               ('adamw_torch', 'AdamW без fused', BLUE),
                               ('adafactor', 'Adafactor', AQUA),
                               ('no_compile', 'без torch.compile', ORANGE)]:
        run = runs[name]
        steps, loss = smooth(run)
        left.plot(tokens_m(run, steps), loss, color=color, label=f'{label}  →  eval {run["eval_loss"]:.3f}')
    left.set_title('torch.compile и оптимизатор', loc='left')
    for name, label, color in [('bs32_lr6e-4', 'база, seed 42', GRAY), ('bs32_lr6e-4_seed1', 'база, seed 1', GRAY),
                               ('bs32_lr6e-4_seed2', 'база, seed 2', GRAY),
                               ('no_bf16', 'bf16=False', ORANGE), ('no_tf32', 'tf32=False', BLUE)]:
        run = runs[name]
        steps, loss = smooth(run)
        right.plot(tokens_m(run, steps), loss, color=color, label=f'{label}  →  eval {run["eval_loss"]:.3f}',
                   linewidth=1.2 if color == GRAY else 1.8)
    right.set_title('dtype-флаги на фоне шума между сидами', loc='left')
    for ax in (left, right):
        ax.set_ylim(3.3, 5.2)
        ax.set_xlabel('токенов увидено, млн')
        ax.legend(fontsize=8, loc='upper right')
    left.set_ylabel('train loss (сглажен)')
    save(fig, 'compile_optim_dtype.png')


def throughput(runs):
    names = [n for n in runs if 'max_steps' not in config(n) and not n.startswith('final_') and 'seed' not in n]
    names.sort(key=lambda n: tokens_m(runs[n], runs[n]['step'][-1]))
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(names) + 1.2))
    values = [tokens_m(runs[n], runs[n]['step'][-1]) for n in names]
    ax.barh(names, values, color=[GROUP_COLORS[group(n)] for n in names], height=0.7)
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=8, color=TEXT_2)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('токенов обработано за 15 минут, млн (с паддингом)')
    ax.set_title('Пропускная способность конфигураций', loc='left')
    save(fig, 'throughput.png')


def final_run(runs):
    curve_path = f'{RUNS}/{FINAL}/eval_curve.json'
    if FINAL not in runs or not os.path.exists(curve_path):
        return
    with open(curve_path) as f:
        curve = json.load(f)
    run = runs[FINAL]
    decay_start = config(FINAL)['max_steps'] - config(FINAL)['lr_scheduler_kwargs']['num_decay_steps']
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax in (left, right):
        ax.axvspan(decay_start, config(FINAL)['max_steps'], color=GRID, lw=0, label='фаза затухания WSD')
    steps, loss = smooth(run)
    left.plot(run['step'], run['loss'], color=GRAY, linewidth=0.6, alpha=0.6, label='train loss (сырой)')
    left.plot(steps, loss, color=BLUE, label='train loss (сглажен)')
    points = [p for p in curve if p['step'] > 0]
    left.plot([p['step'] for p in points], [p['eval_loss'] for p in points], color=ORANGE, marker='o',
              markersize=6, label='eval loss (чекпоинты)')
    last = curve[-1]
    left.annotate(f"{last['eval_loss']:.3f}", (last['step'], last['eval_loss']), textcoords='offset points',
                  xytext=(-10, 10), ha='right', color=TEXT, fontsize=9)
    left.set_ylim(3.2, 7)
    left.set_xlabel('шаг')
    left.set_ylabel('loss')
    left.set_title('Финальная модель: loss', loc='left')
    left.legend(fontsize=8, loc='upper right')
    right.plot([p['step'] for p in curve], [p['perplexity'] for p in curve], color=ORANGE, marker='o', markersize=6,
               label='eval perplexity')
    for p in curve:
        right.annotate(f"{p['perplexity']:,.0f}".replace(',', ' '), (p['step'], p['perplexity']),
                       textcoords='offset points', xytext=(6, 6), fontsize=8, color=TEXT_2)
    right.set_yscale('log')
    right.set_xlabel('шаг (0 = случайная инициализация)')
    right.set_ylabel('perplexity (лог. шкала)')
    right.set_title('Финальная модель: perplexity на валидации', loc='left')
    right.legend(fontsize=8, loc='upper right')
    save(fig, 'final_run.png')


def summary(runs):
    with open('results/summary.csv', 'w') as f:
        f.write('name,group,overrides,eval_loss,perplexity,steps,tokens_m\n')
        for n in sorted(runs, key=lambda n: runs[n]['eval_loss']):
            run = runs[n]
            overrides = json.dumps(EXPERIMENTS[n], ensure_ascii=False).replace('"', "'")
            f.write(f'{n},{group(n)},"{overrides}",{run["eval_loss"]:.4f},{math.exp(run["eval_loss"]):.1f},'
                    f'{run["step"][-1]},{tokens_m(run, run["step"][-1]):.1f}\n')
    print('saved results/summary.csv')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/{name}', dpi=150)
    plt.close(fig)
    print('saved', name)


if __name__ == '__main__':
    os.makedirs(PLOTS, exist_ok=True)
    runs = {name: load(name) for name in EXPERIMENTS if os.path.exists(f'{RUNS}/{name}/trainer_state.json')}
    summary(runs)
    ranking(runs)
    batch_lr(runs)
    schedulers(runs)
    compile_optim_dtype(runs)
    throughput(runs)
    final_run(runs)
