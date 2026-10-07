"""Reproducible K sensitivity study: 1,000 episodes, seeds 42/43/44 per K.

Run: python DQN_sensitivity.py
Each training process uses one CPU thread. Outputs remain separate from
DQN_results. Means and sample standard deviations describe three training
seeds, not confidence intervals or proof of optimality.
"""
import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def run_case(k, seed, episodes, output):
    destination = output / f'K{k}_seed{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'DQN_reviewed.py'), '--k-max', str(k),
               '--seed', str(seed), '--episodes', str(episodes), '--output', str(destination)]
    with (destination / 'training.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((destination / 'dqn_summary.json').read_text())
    cuts = json.loads((destination / 'dqn_cut_plan.json').read_text())
    # Verify complete interval coverage, action lengths, mass and COM limits.
    cursor = 0
    for cut in cuts:
        assert cut['start_idx'] == cursor
        assert cut['k_slices'] == cut['end_idx'] - cursor + 1
        assert 1 <= cut['k_slices'] <= k
        assert cut['mass_kg'] <= result['mass_limit_kg'] + 1e-6
        assert all(cut[f'dcom_{axis}_cm'] <= 50 + 1e-6 for axis in 'xyz')
        cursor = cut['end_idx'] + 1
    assert cursor == result['data']['slices'] and result['complete']
    assert np.isclose(sum(c['mass_kg'] for c in cuts), result['data']['modeled_mass_kg'])
    return dict(k_max=k, seed=seed, episodes=episodes, num_cuts=result['num_cuts'],
                total_area_m2=result['total_area_m2'], evaluation_return=result['evaluation_return'],
                max_block_mass_kg=result['max_block_mass_kg'],
                max_dcom_x_cm=result['max_dcom_cm']['x'],
                max_dcom_y_cm=result['max_dcom_cm']['y'],
                max_dcom_z_cm=result['max_dcom_cm']['z'],
                max_slices_used=max(c['k_slices'] for c in cuts))


def summarize(rows, output):
    rows.sort(key=lambda r: (r['k_max'], r['seed']))
    with (output / 'sensitivity_runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    metrics = ['num_cuts', 'total_area_m2', 'evaluation_return', 'max_block_mass_kg',
               'max_dcom_x_cm', 'max_dcom_y_cm', 'max_dcom_z_cm', 'max_slices_used']
    for k in sorted({r['k_max'] for r in rows}):
        cases = [r for r in rows if r['k_max'] == k]
        group = dict(k_max=k, num_seeds=len(cases), episodes=cases[0]['episodes'])
        for metric in metrics:
            values = [r[metric] for r in cases]
            group[metric + '_mean'] = float(np.mean(values))
            group[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        groups.append(group)
    with (output / 'sensitivity_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    (output / 'sensitivity_summary.json').write_text(json.dumps(groups, indent=2))
    plot(rows, groups, output)
    report = ['# Reviewed Double DQN sensitivity to maximum grouped slices K', '',
              'Results use DQN_reviewed.py; the original DQN_Sep.py is preserved. See DQN_REVIEW.md for corrections.',
              'Same dataset, 20,000 kg limit, 50 cm per-axis COM limit, reward and DQN hyperparameters.',
              'Each K uses the same seeds; training runs independently for the same episode budget.',
              'Final plans use greedy masked evaluation. Values below are mean ± sample standard deviation.', '',
              '| K | Cuts | Cutting area (m²) | Evaluation return |',
              '|---|---:|---:|---:|']
    for g in groups:
        report.append(f"| {g['k_max']} | {g['num_cuts_mean']:.2f} ± {g['num_cuts_std']:.2f} | "
                      f"{g['total_area_m2_mean']:.2f} ± {g['total_area_m2_std']:.2f} | "
                      f"{g['evaluation_return_mean']:.2f} ± {g['evaluation_return_std']:.2f} |")
    report += ['', 'All recorded plans passed coverage, action-length, mass and COM checks.',
               'K changes both the available block sizes and the observation/output dimensions (5+2K inputs, K actions).',
               'Hidden layers remain 256/256. Equal episode budgets may involve different transition counts.',
               'Three seeds give a limited estimate of training variability; this study does not prove optimality.',
               'Cut and area accounting includes final removal, following the original script.',
               'The same two out-of-range bodies (1,679.384 kg) remain excluded by the original data mapping.',
               'Each K/seed folder contains its full cut plan, checkpoint, training metrics, and plots.']
    from PPO_Sep import ShipCutEnv
    env = ShipCutEnv(ROOT / 'Data_New_15cm', k_max=max(20, max(g['k_max'] for g in groups)))
    reachable, edges = {0}, []
    for start in range(env.N):
        if start in reachable:
            for action in np.flatnonzero(env.masks[start]):
                k = int(action) + 1
                edges.append((start, k))
                reachable.add(start + k)
    area_means = [g['total_area_m2_mean'] for g in groups]
    spread = 100 * (max(area_means) - min(area_means)) / min(area_means)
    report += ['', '## Interpretation', '',
               f'The range of mean cutting areas is {spread:.3f}% of the smallest mean.',
               f'The largest feasible action from any reachable state at K={env.k_max} is '
               f'{max(k for _, k in edges)} slices.',
               'Feasible actions above the smallest tested K (zero-based start index, slices): '
               + str([(i, k) for i, k in edges if k > min(g['k_max'] for g in groups)]),
               'Thus the added feasible physical choices are limited to the tail of this dataset.',
               'Observed differences also reflect stochastic training and changed network input/output sizes.',
               'Do not interpret a larger action set as a guarantee of a better learned plan.']
    (output / 'sensitivity_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(groups, indent=2), flush=True)


def plot(rows, groups, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2375ad', '#df8620', '#15967e', '#a354a5', '#bb4444', '#786647']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for ax, metric, title, unit in zip(axes[0],
            ['num_cuts', 'total_area_m2', 'evaluation_return'],
            ['Final number of cuts', 'Final total cutting area', 'Final evaluation return'],
            ['Cuts', 'Area (m²)', 'Return (higher is better)']):
        for i, g in enumerate(groups):
            ax.errorbar(i, g[metric + '_mean'], yerr=g[metric + '_std'],
                        fmt='o', color=colors[i % len(colors)], capsize=6, ms=8)
            values = [r[metric] for r in rows if r['k_max'] == g['k_max']]
            ax.scatter(i + np.linspace(-0.10, 0.10, len(values)), values,
                       color=colors[i % len(colors)], s=20, alpha=0.5)
        ax.set(xticks=range(len(groups)), xticklabels=[f"K={g['k_max']}" for g in groups],
               title=title, ylabel=unit)
        ax.grid(alpha=0.2)
    for i, g in enumerate(groups):
        histories = []
        for row in rows:
            if row['k_max'] == g['k_max']:
                path = output / f"K{row['k_max']}_seed{row['seed']}" / 'dqn_training_metrics.csv'
                with path.open() as stream:
                    histories.append(list(csv.DictReader(stream)))
        for ax, key, title, ylabel in zip(axes[1, :2], ['episode_return', 'num_cuts'],
                ['Training reward (25-episode rolling mean)', 'Training cut count (25-episode rolling mean)'],
                ['Return', 'Cuts']):
            data = np.array([[float(r[key]) for r in h] for h in histories])
            window = min(25, data.shape[1])
            smooth = np.array([np.convolve(v, np.ones(window) / window, mode='valid') for v in data])
            x = np.arange(window, data.shape[1] + 1)
            mean = smooth.mean(axis=0)
            std = smooth.std(axis=0, ddof=1) if len(smooth) > 1 else np.zeros_like(mean)
            ax.plot(x, mean, color=colors[i % len(colors)], label=f"K={g['k_max']}")
            ax.fill_between(x, mean - std, mean + std, color=colors[i % len(colors)], alpha=0.13)
            ax.set(title=title, xlabel='Episode', ylabel=ylabel)
        ax = axes[1, 2]
        x = np.arange(3) + (i - (len(groups) - 1) / 2) * 0.13
        ax.bar(x, [g[f'max_dcom_{a}_cm_mean'] for a in 'xyz'], width=0.12,
               yerr=[g[f'max_dcom_{a}_cm_std'] for a in 'xyz'],
               color=colors[i % len(colors)], label=f"K={g['k_max']}", capsize=3)
    axes[1, 2].axhline(50, ls='--', color='#b33333', label='50 cm limit')
    axes[1, 2].set(xticks=[0, 1, 2], xticklabels=['X', 'Y', 'Z'],
                   ylabel='Maximum |ΔCOM| per plan (cm)', title='Remaining-body COM changes')
    for ax in axes[1]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"DQN sensitivity to K | {groups[0]['episodes']} episodes per run | "
                 f"{groups[0]['num_seeds']} seeds per K\n"
                 'Points/bars: mean ± sample SD; shaded bands: seed variability', fontsize=14)
    fig.savefig(output / 'dqn_k_sensitivity.png', dpi=180)
    fig.savefig(output / 'dqn_k_sensitivity.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ks', type=int, nargs='+', default=[7, 8, 9, 10, 15, 20])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', type=Path, default=ROOT / 'DQN_sensitivity_results')
    args = parser.parse_args()
    if min(args.ks) < 1 or args.episodes < 1 or args.workers < 1:
        parser.error('K, episodes and workers must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'study_config.json').write_text(json.dumps(
        dict(ks=args.ks, seeds=args.seeds, episodes=args.episodes,
             mass_limit_kg=20000, com_limit_cm=50), indent=2))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_case, k, seed, args.episodes, args.output): (k, seed)
                for seed in args.seeds for k in args.ks}
        for future in as_completed(jobs):
            row = future.result()
            results.append(row)
            print(f"Completed K={row['k_max']}, seed={row['seed']}: "
                  f"{row['num_cuts']} cuts, {row['total_area_m2']:.3f} m²", flush=True)
    summarize(results, args.output)


if __name__ == '__main__':
    main()
