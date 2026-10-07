"""Present the saved DQN run with the lowest final total cutting area."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DQN_Sep import DQN
from DQN_reviewed import evaluate
from PPO_Sep import ShipCutEnv


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'DQN_best_result'


def main():
    torch.set_num_threads(1)
    summaries = [(p, json.loads(p.read_text())) for p in
                 (ROOT / 'DQN_sensitivity_results').glob('K*_seed*/dqn_summary.json')]
    path, summary = min((item for item in summaries if item[1]['complete']),
                        key=lambda item: (item[1]['total_area_m2'], -item[1]['evaluation_return']))
    source = path.parent
    cuts = json.loads((source / 'dqn_cut_plan.json').read_text())
    env = ShipCutEnv(ROOT / 'Data_New_15cm', summary['k_max'])
    model = DQN(5 + 2*env.k_max, env.k_max)
    checkpoint = torch.load(source / 'dqn_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    replayed = evaluate(model, env)
    assert len(cuts) == len(replayed)
    for saved, actual in zip(cuts, replayed):
        for key in saved:
            np.testing.assert_allclose(saved[key], actual[key], atol=1e-6)
    assert env.index == env.N
    with (source / 'dqn_training_metrics.csv').open() as stream:
        history = list(csv.DictReader(stream))
    mass = np.array([c['mass_kg'] for c in cuts])
    delta = np.array([[c[f'dcom_{a}_cm'] for a in 'xyz'] for c in cuts])
    steps = np.arange(1, len(cuts)+1)
    ends = np.array([c['end_idx']+1 for c in cuts])
    # No remaining-body COM exists after final removal: use NaN, not an origin jump.
    remaining_com = env.com[np.r_[0, ends]].copy()
    remaining_mass = env.remaining_mass[np.r_[0, ends]]
    remaining_com[remaining_mass <= 1e-12] = np.nan
    cumulative = np.r_[0., np.cumsum([c['boundary_area_cm2'] for c in cuts])/1e4]
    assert np.isclose(cumulative[-1], summary['total_area_m2'])
    assert mass.max() <= 20000 + 1e-6 and delta.max() <= 50 + 1e-6
    metrics = [('Number of cross-section boundaries', len(env.x), ''),
               ('Number of modeled slices (N)', env.N, ''),
               ('Maximum look-ahead (K_max)', env.k_max, 'slices'),
               ('Initial modeled total mass M0', env.remaining_mass[0], 'kg')]
    metrics += [(f'Initial COM_{a}', env.com[0, i], 'cm') for i, a in enumerate('xyz')]
    metrics += [('Total number of cuts', len(cuts), ''),
                ('Maximum mass per cut', mass.max(), 'kg'),
                ('Average mass per cut', mass.mean(), 'kg'),
                ('Minimum mass per cut', mass.min(), 'kg')]
    for i, a in enumerate('xyz'):
        metrics.extend([(f'Maximum |Delta COM_{a}|', delta[:, i].max(), 'cm'),
                        (f'Average |Delta COM_{a}|', delta[:, i].mean(), 'cm'),
                        (f'Minimum |Delta COM_{a}|', delta[:, i].min(), 'cm')])
    metrics += [('Total cutting area', cumulative[-1], 'm²'),
                ('Final evaluation reward (undiscounted)', summary['evaluation_return'], ''),
                ('Training episodes', len(history), ''), ('Random seed', summary['seed'], '')]
    OUT.mkdir(exist_ok=True)
    for name in ('dqn_model.pt', 'dqn_cut_plan.json', 'dqn_cut_plan_metrics.csv',
                 'dqn_training_metrics.csv', 'dqn_summary.json'):
        shutil.copy2(source / name, OUT / name)
    with (OUT / 'best_dqn_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Metric', 'Value', 'Unit'])
        writer.writerows(metrics)
    with (OUT / 'remaining_com_trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cut_number', 'remaining_mass_kg', 'COM_x_cm', 'COM_y_cm', 'COM_z_cm', 'cumulative_area_m2'])
        for i in range(len(cuts)+1):
            writer.writerow([i, remaining_mass[i], *['' if not np.isfinite(v) else v for v in remaining_com[i]], cumulative[i]])
    formatted = [(name.replace('Delta', 'Δ'), f'{value:,.3f}' if isinstance(value, (float, np.floating))
                  else str(value), unit) for name, value, unit in metrics]
    plot_metrics(formatted, summary)
    plot_figures(env, cuts, history, mass, delta, remaining_com, cumulative, summary)
    report = ['# Best saved DQN result', '',
              f"Selected **K={env.k_max}, seed={summary['seed']}**, trained for {summary['episodes']:,} episodes.",
              'Selection criterion: lowest final total cutting area among the 18 complete saved DQN runs.',
              'This is the best observed area, not a claim of global optimality or the highest reward.',
              'The checkpoint was reloaded and its greedy cut sequence reproduced exactly within numerical tolerance.', '',
              '| Metric | Value | Unit |', '|---|---:|---|']
    report += [f'| {name} | {value} | {unit} |' for name, value, unit in formatted]
    report += ['', '## Figures', '', '![Best DQN figures](best_dqn_figures.png)', '',
               'Separate PNG and SVG files are provided for each panel; SVG files preserve vector detail.', '',
               '## Reward function', '',
               r'$$r_t=-1.9\frac{A_t}{A_{\mathrm{median}}+10^{-9}}-0.01|\Delta C_{y,t}|-0.001(|\Delta C_{x,t}|+|\Delta C_{z,t}|)-1.2(1-i_t/N)+1.5B(m_t).$$', '',
               'Here A is boundary area in cm², COM changes are in cm, i is the starting slice index, and N=259.',
               'B(m)=2 for 14,000 ≤ m ≤ 19,000 kg; B(m)=1.05 for 11,000 ≤ m < 14,000 kg or '
               '19,000 < m ≤ 20,000 kg; otherwise B(m)=0. Discount factor γ=0.995.',
               'The reward graph shows undiscounted training episode returns; the darker line is a 25-episode moving average.', '',
               '## Interpretation and conventions', '',
               '- There are 260 DXF cross-section boundaries and 259 intervals. The code uses N for intervals.',
               '- Initial mass and COM describe the modeled bodies within those intervals. Two out-of-range bodies '
               '(1,679.384 kg) remain excluded by the original COM-based assignment.',
               '- Each entire CSV body is assigned to one interval by its X COM; DXF filename areas are used without geometry parsing.',
               '- All blocks satisfy 20,000 kg and 50 cm per-axis change limits.',
               '- Absolute remaining-body COM is undefined after the final removal; the trajectory leaves that final point blank.',
               '- Per-cut COM-change statistics retain the original final-removal convention of zero change. '
               'Thus minima include the terminal zero and averages include all 106 cuts.',
               '- Final removal counts as a cut and its boundary area is included in total area, matching the previous studies.',
               '- The block-mass panel displays mass versus cut number; the separate cut-count graph shows counts per training episode.',
               '- DQN_reviewed.py was used, preserving the original network and hyperparameters while correcting action/constraint handling.']
    (OUT / 'best_dqn_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(dict(source=str(source), metrics=[dict(metric=n, value=float(v), unit=u) for n,v,u in metrics]), indent=2))


def plot_metrics(rows, summary):
    fig, ax = plt.subplots(figsize=(9, 11.5), layout='constrained')
    ax.axis('off')
    ax.set_title(f"Best DQN result | K={summary['k_max']}, seed={summary['seed']}\n"
                 'Selected by lowest total cutting area', fontsize=15, pad=18)
    table = ax.table(cellText=rows, colLabels=['Metric', 'Value', 'Unit'],
                     colWidths=[0.64, 0.23, 0.13], cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.85)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#153a58')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f5f8' if r % 2 else 'white')
            if c == 1:
                cell.set_text_props(ha='right')
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'best_dqn_metrics.{ext}', dpi=220)
    plt.close(fig)


def plot_figures(env, cuts, history, mass, delta, com, cumulative, summary):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    steps = np.arange(1, len(cuts)+1)
    colors = ['#2477ad', '#db8529', '#23836c']
    def absolute_com(axes):
        for i, ax in enumerate(axes):
            ax.plot(np.arange(len(com)), com[:, i], '.-', color=colors[i], ms=3, lw=1.2)
            ax.set_ylabel(f'COM {"XYZ"[i]} (cm)')
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.grid(alpha=.2)
        axes[0].set_title('(a) Remaining-body center of mass')
        axes[-1].set_xlabel('Completed cuts (0 = initial body)')
    def delta_com(axes):
        for i, ax in enumerate(axes):
            ax.plot(steps, delta[:, i], '.-', color=colors[i], ms=3, lw=1.2)
            ax.set_ylabel(f'|ΔCOM {"XYZ"[i]}| (cm)')
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.grid(alpha=.2)
        axes[0].set_title('(b) Per-cut COM changes (limit: 50 cm per axis)')
        axes[-1].set_xlabel('Cut number')
    def area(ax):
        ax.plot(np.r_[0, steps], cumulative, '.-', ms=3)
        ax.set(title=f'(c) Accumulated cutting area: {cumulative[-1]:.3f} m²',
               xlabel='Cut number', ylabel='Cumulative cutting area (m²)')
    def distribution(ax):
        ax.bar(env.x[:-1], env.mass / 1000, width=np.diff(env.x), align='edge', color='#168d88')
        ax.set(title='(d) Mass distribution along the ship', xlabel='X position (cm)',
               ylabel='Mass per cross-section interval (tonnes)')
    def blocks(ax):
        ax.bar(steps, mass/1000, color='#2477ad')
        ax.axhline(20, ls='--', color='#b84040', label='20-tonne limit')
        ax.set(title=f'(e) Number of cuts: {len(cuts)}', xlabel='Cut number', ylabel='Block mass (tonnes)')
        ax.legend(fontsize=9, loc='lower left')
    def reward(ax):
        episodes = np.arange(1,len(history)+1)
        values = np.array([float(h['episode_return']) for h in history])
        ax.plot(episodes, values, lw=.55, alpha=.5, label='Episode return')
        window = min(25, len(values))
        ax.plot(episodes[window-1:], np.convolve(values, np.ones(window)/window, mode='valid'),
                lw=1.5, color='#173c5a', label='25-episode mean')
        ax.set(title='(f) Reward evolution during DQN training', xlabel='Episode', ylabel='Undiscounted episode return')
        ax.legend(fontsize=9)
    panels=[('cumulative_cutting_area', area, (1,0)), ('longitudinal_mass_distribution', distribution, (1,1)),
            ('mass_per_cut', blocks, (2,0)), ('reward_evolution', reward, (2,1))]
    fig=plt.figure(figsize=(15,13), layout='constrained')
    grid=fig.add_gridspec(3,2)
    for column, draw in enumerate([absolute_com,delta_com]):
        sub=grid[0,column].subgridspec(3,1)
        draw([fig.add_subplot(sub[i]) for i in range(3)])
    for _,draw,pos in panels:
        ax=fig.add_subplot(grid[pos]); draw(ax); ax.grid(alpha=.2)
    fig.suptitle(f"Best saved DQN result by cutting area | K={env.k_max}, seed={summary['seed']}\n"
                 f"{len(cuts)} cuts  •  {cumulative[-1]:.3f} m²  •  1,000 training episodes", fontsize=16)
    for ext in ('png','svg'):
        fig.savefig(OUT / f'best_dqn_figures.{ext}', dpi=200)
    plt.close(fig)
    for name,draw,_ in panels:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        draw(ax); ax.grid(alpha=.2)
        for ext in ('png','svg'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=220)
        plt.close(fig)
    for name,draw in [('center_of_mass_trajectory',absolute_com),('center_of_mass_changes',delta_com)]:
        fig,axes=plt.subplots(3,1,figsize=(9,7),layout='constrained')
        draw(axes)
        for ext in ('png','svg'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=220)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.plot([int(h['episode']) for h in history],[int(h['num_cuts']) for h in history],lw=.7)
    ax.axhline(len(cuts),ls='--',color='#b84040',label=f'Final greedy plan: {len(cuts)} cuts')
    ax.set(title='Number of cuts during DQN training',xlabel='Episode',ylabel='Number of cuts')
    ax.grid(alpha=.2); ax.legend()
    for ext in ('png','svg'):
        fig.savefig(OUT / f'number_of_cuts_training.{ext}',dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
