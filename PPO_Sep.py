"""Masked PPO for sequential ship cutting (cm, kg).

Run: python PPO_Sep.py --episodes 1000
Data default: Data_New_15cm next to this script. DXF filenames supply X and
area; CSV bodies are assigned wholly by their X COM, as in DQN_Sep.py.
The reward matches DQN_Sep.py. The configured limits are 20,000 kg
and 50 cm per axis per cut. Final removal has zero COM shift by convention.
Unlike the DQN training shields, executed cuts always satisfy these limits.
Backward reachability excludes choices that cannot lead to a complete plan.
Dependencies: numpy, torch, matplotlib (no Gym or stable-baselines required).
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

MAX_BLOCK_MASS_KG = 20000.0
MAX_COM_SHIFT_CM = 50.0
K_MAX = 20
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
DXF_REGEX = re.compile(
    r"^cross_section_(-?\d+(?:\.\d+)?)_Area_(-?\d+(?:\.\d+)?)\.dxf$", re.I
)


class ShipCutEnv:
    def __init__(self, directory, k_max=K_MAX):
        if k_max < 1:
            raise ValueError("k_max must be positive")
        self.k_max = int(k_max)
        directory = Path(directory)
        boundaries = []
        for path in directory.iterdir():
            match = DXF_REGEX.match(path.name)
            if path.is_file() and match:
                boundaries.append(tuple(map(float, match.groups())))
        boundaries.sort()
        if len(boundaries) < 2:
            raise ValueError('At least two matching DXF filenames are required.')
        self.x, areas = np.asarray(boundaries, dtype=float).T
        if np.any(np.diff(self.x) <= 0) or np.any(areas < 0):
            raise ValueError('DXF boundaries must have unique X and nonnegative areas.')
        self.N = len(self.x) - 1
        self.area = areas[1:]
        self.area_scale = float(np.median(self.area)) or 1.0
        bodies = []
        with (directory / 'physical_properties_new.csv').open(
            encoding='utf-8-sig', newline=''
        ) as stream:
            reader = csv.DictReader(stream)
            columns = ['Mass', 'Center of Mass-x', 'Center of Mass-y', 'Center of Mass-z']
            if not set(columns).issubset(reader.fieldnames or []):
                raise ValueError(f'CSV requires columns: {columns}')
            for line, row in enumerate(reader, 2):
                try:
                    values = [float(row[key]) for key in columns]
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'Invalid CSV numbers at row {line}') from exc
                if not np.all(np.isfinite(values)) or values[0] < 0:
                    raise ValueError(f'Invalid mass/COM at CSV row {line}')
                bodies.append(values)
        if not bodies:
            raise ValueError('CSV contains no bodies.')
        bodies = np.asarray(bodies)
        indices = np.searchsorted(self.x, bodies[:, 1], side='right') - 1
        covered = (indices >= 0) & (indices < self.N)
        self.data_summary = {
            'dxf_boundaries': len(self.x), 'slices': self.N,
            'csv_bodies': len(bodies), 'excluded_bodies': int((~covered).sum()),
            'excluded_mass_kg': float(bodies[~covered, 0].sum()),
        }
        self.mass = np.bincount(indices[covered], weights=bodies[covered, 0], minlength=self.N)
        moments = np.column_stack([
            np.bincount(indices[covered], weights=bodies[covered, 0] * bodies[covered, axis],
                        minlength=self.N) for axis in (1, 2, 3)
        ])
        self.prefix = np.r_[0.0, np.cumsum(self.mass)]
        # Reverse sums avoid cancellation near the final interval.
        self.remaining_mass = np.r_[np.cumsum(self.mass[::-1])[::-1], 0.0]
        remaining_moments = np.vstack([np.cumsum(moments[::-1], axis=0)[::-1], np.zeros(3)])
        self.com = np.divide(remaining_moments, self.remaining_mass[:, None],
                             out=np.zeros_like(remaining_moments),
                             where=self.remaining_mass[:, None] > 1e-12)
        if self.remaining_mass[0] <= 0:
            raise ValueError('No positive mass inside the DXF intervals.')
        self.com_scale = max(1.0, float(np.max(np.abs(self.com[0]))))
        self.data_summary['modeled_mass_kg'] = float(self.remaining_mass[0])
        self.local_legal = np.zeros((self.N, self.k_max), dtype=bool)
        for start in range(self.N):
            for action in range(min(self.k_max, self.N - start)):
                mass, delta, _ = self.preview(start, action)
                self.local_legal[start, action] = (
                    mass <= MAX_BLOCK_MASS_KG + 1e-9
                    and np.all(delta <= MAX_COM_SHIFT_CM + 1e-9)
                )
        # The remaining body depends only on the current boundary index.
        reachable = np.zeros(self.N + 1, dtype=bool)
        reachable[self.N] = True
        self.masks = self.local_legal.copy()
        for start in range(self.N - 1, -1, -1):
            for action in np.flatnonzero(self.local_legal[start]):
                self.masks[start, action] = reachable[start + action + 1]
            reachable[start] = self.masks[start].any()
        if not reachable[0]:
            forward = {0}
            for start in range(self.N):
                if start in forward:
                    forward.update(start + int(a) + 1 for a in np.flatnonzero(self.local_legal[start]))
            last = max(forward)
            candidates = [self.preview(last, a) for a in range(min(self.k_max, self.N - last))]
            detail = {
                **self.data_summary, 'furthest_reachable_slice_index': last,
                'furthest_reachable_x_cm': float(self.x[last]),
                'smallest_next_block_mass_kg': min(item[0] for item in candidates),
                'next_single_slice_dcom_cm': candidates[0][1].tolist(),
            }
            raise ValueError(f'No complete feasible plan exists under the {MAX_BLOCK_MASS_KG:,.0f} kg / 50 cm '
                             f'limits and K_MAX={self.k_max}. Limits have not been relaxed.\n'
                             + json.dumps(detail, indent=2))
        self.reset()

    def preview(self, start, action):
        end = start + action + 1  # exclusive
        mass = float(self.prefix[end] - self.prefix[start])
        delta = (np.zeros(3) if self.remaining_mass[end] <= 1e-12
                 else np.abs(self.com[end] - self.com[start]))
        return mass, delta, end

    def state(self):
        masses, areas = np.zeros(self.k_max), np.zeros(self.k_max)
        for action in range(min(self.k_max, self.N - self.index)):
            mass, _, end = self.preview(self.index, action)
            masses[action] = mass / MAX_BLOCK_MASS_KG
            areas[action] = self.area[end - 1] / (self.area_scale + 1e-9)
        return np.r_[self.index / self.N,
                     self.remaining_mass[self.index] / self.remaining_mass[0],
                     self.com[self.index] / self.com_scale, masses, areas].astype(np.float32)

    def reset(self):
        self.index = 0
        return self.state()

    def step(self, action):
        if self.index >= self.N or not 0 <= action < self.k_max or not self.masks[self.index, action]:
            raise ValueError('Invalid or infeasible action; no action substitution is performed.')
        start = self.index
        mass, delta, end = self.preview(start, action)
        scaled_area = self.area[end - 1] / (self.area_scale + 1e-9)
        cut_penalty = 0.8 * (1.0 - start / self.N)
        # Preserve DQN reward formula; hard masking excludes masses over 20 t.
        bonus = 2.0 if 14000 <= mass <= 19000 else (
            1.05 if 11000 <= mass < 14000 or 19000 < mass <= MAX_BLOCK_MASS_KG else 0.0)
        reward = -(1.9 * scaled_area + 0.01 * delta[1]
                   + 0.001 * (delta[0] + delta[2]) + 1.5 * cut_penalty - 1.5 * bonus)
        self.index = end
        info = dict(start_idx=start, end_idx=end - 1, k_slices=action + 1,
                    x_start_cm=float(self.x[start]), x_end_cm=float(self.x[end]),
                    mass_kg=mass, dcom_x_cm=float(delta[0]), dcom_y_cm=float(delta[1]),
                    dcom_z_cm=float(delta[2]), boundary_area_cm2=float(self.area[end - 1]),
                    reward=float(reward))
        return self.state(), float(reward), end == self.N, info


class ActorCritic(nn.Module):
    """(5 + 2K) -> 256 ReLU -> 256 ReLU, with K policy logits and a value."""
    def __init__(self, k_max=K_MAX):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(5 + 2 * k_max, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.actor = nn.Linear(256, k_max)
        self.critic = nn.Linear(256, 1)

    def forward(self, states, masks):
        features = self.trunk(states)
        logits = self.actor(features).masked_fill(~masks, -torch.inf)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)


def train(env, episodes=100, seed=42, device='cpu', episodes_per_update=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ActorCritic(env.k_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    env.training_history = []
    for first in range(0, episodes, episodes_per_update):
        states, masks, actions, old_logs, advantages, returns = [], [], [], [], [], []
        for _ in range(min(episodes_per_update, episodes - first)):
            obs = env.reset()
            rewards, values = [], []
            episode_area = 0.0
            done = False
            while not done:
                state = torch.as_tensor(obs, device=device)
                mask = torch.as_tensor(env.masks[env.index], device=device)
                with torch.no_grad():
                    distribution, value = model(state, mask)
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action)
                states.append(state)
                masks.append(mask)
                actions.append(action)
                old_logs.append(log_prob)
                obs, reward, done, info = env.step(action.item())
                episode_area += info['boundary_area_cm2']
                rewards.append(reward)
                values.append(value.item())
            # Complete episodes: terminal bootstrap is zero; no cross-episode GAE.
            gae, next_value = 0.0, 0.0
            episode_advantages = []
            for reward, value in zip(reversed(rewards), reversed(values)):
                td_error = reward + GAMMA * next_value - value
                gae = td_error + GAMMA * GAE_LAMBDA * gae
                episode_advantages.append(gae)
                next_value = value
            episode_advantages.reverse()
            advantages.extend(episode_advantages)
            returns.extend(np.asarray(episode_advantages) + np.asarray(values))
            history.append(float(sum(rewards)))
            env.training_history.append(dict(episode=len(history),
                episode_return=history[-1], num_cuts=len(rewards),
                total_area_m2=episode_area / 1e4))
        states, masks = torch.stack(states), torch.stack(masks)
        actions, old_logs = torch.stack(actions), torch.stack(old_logs)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(UPDATE_EPOCHS):
            order = torch.randperm(len(actions), device=device)
            for batch in order.split(MINIBATCH_SIZE):
                distribution, values = model(states[batch], masks[batch])
                ratio = (distribution.log_prob(actions[batch]) - old_logs[batch]).exp()
                unclipped = ratio * advantages[batch]
                clipped = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (values - returns[batch]).square().mean()
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * distribution.entropy().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite PPO loss.')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        print(f'Episodes {len(history)}/{episodes} | recent mean return '
              f'{np.mean(history[first:]):.3f}', flush=True)
    return model, history


@torch.no_grad()
def evaluate(model, env):
    device = next(model.parameters()).device
    obs, cuts, done = env.reset(), [], False
    model.eval()
    while not done:
        distribution, _ = model(torch.as_tensor(obs, device=device),
                                torch.as_tensor(env.masks[env.index], device=device))
        action = distribution.logits.argmax().item()
        obs, _, done, info = env.step(action)
        cuts.append(info)
    return cuts


def plot_results(env, cuts, rewards, output):
    """Export a six-panel overview and separate, full-resolution plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    steps = np.arange(1, len(cuts) + 1)
    episodes = np.arange(1, len(rewards) + 1)
    masses = np.array([c['mass_kg'] for c in cuts])
    cumulative = np.cumsum([c['boundary_area_cm2'] for c in cuts]) / 1e4

    def com_plot(axes):
        for ax, axis in zip(axes, 'xyz'):
            ax.plot(steps, [c[f'dcom_{axis}_cm'] for c in cuts], '.-', lw=1, ms=3)
            ax.axhline(MAX_COM_SHIFT_CM, ls='--', color='#bb3333', lw=1)
            ax.set(ylabel=f'|ΔCOM {axis.upper()}| (cm)', ylim=(-2, 55))
            ax.grid(alpha=0.2)
        axes[0].set_title('(a) Remaining-body COM change per cut')
        axes[-1].set_xlabel('Cut number')

    def mass_plot(ax):
        ax.bar(env.x[:-1], env.mass / 1000, width=np.diff(env.x), align='edge',
               color='#008b8b', linewidth=0)
        ax.set(title='(b) Input mass distribution', xlabel='X position (cm)', ylabel='Slice mass (tonnes)')

    def blocks_plot(ax):
        ax.bar(steps, masses / 1000, color='#267bb2')
        ax.axhline(MAX_BLOCK_MASS_KG / 1000, ls='--', color='#bb3333', label='20 t limit')
        ax.set(title=f'(c) Block masses — {len(cuts)} cuts', xlabel='Cut number', ylabel='Block mass (tonnes)')
        ax.legend(loc='lower left', fontsize=8)

    def reward_plot(ax):
        ax.plot(episodes, rewards, lw=0.65, alpha=0.55, label='Episode return')
        window = min(25, len(rewards))
        ax.plot(episodes[window-1:], np.convolve(rewards, np.ones(window)/window, mode='valid'),
                color='#133d68', lw=1.6, label=f'{window}-episode mean')
        ax.set(title='(d) PPO reward evolution', xlabel='Episode', ylabel='Undiscounted episode return')
        ax.legend(fontsize=8)

    def area_plot(ax):
        ax.plot(np.r_[0, steps], np.r_[0, cumulative], '.-', ms=3)
        ax.set(title=f'(e) Total cutting area — {cumulative[-1]:.2f} m²',
               xlabel='Cut number', ylabel='Cumulative cutting area (m²)')

    def counts_plot(ax):
        ax.plot(episodes, [r['num_cuts'] for r in env.training_history], lw=0.7)
        ax.axhline(len(cuts), ls='--', color='#bb3333', label=f'Final greedy plan: {len(cuts)}')
        ax.set(title='(f) Number of cuts during training', xlabel='Episode', ylabel='Number of cuts')
        ax.legend(fontsize=8)

    fig = plt.figure(figsize=(14, 13), layout='constrained')
    grid = fig.add_gridspec(3, 2)
    com_grid = grid[0, 0].subgridspec(3, 1)
    com_plot([fig.add_subplot(com_grid[i]) for i in range(3)])
    panels = [('mass_distribution', mass_plot, (0, 1)),
              ('block_masses', blocks_plot, (1, 0)),
              ('reward_evolution', reward_plot, (1, 1)),
              ('total_cutting_area', area_plot, (2, 0)),
              ('number_of_cuts', counts_plot, (2, 1))]
    for _, draw, position in panels:
        ax = fig.add_subplot(grid[position])
        draw(ax)
        ax.grid(alpha=0.2)
    fig.suptitle('PPO ship cutting | Maximum block mass: 20 tonnes | COM limit: 50 cm per axis\n'
                 f'{len(cuts)} cuts  •  Total area: {cumulative[-1]:.2f} m²  •  {len(rewards)} training episodes',
                 fontsize=15)
    fig.savefig(output / 'ppo_results.png', dpi=200)
    fig.savefig(output / 'ppo_results.svg')
    plt.close(fig)
    for name, draw, _ in panels:
        fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
        draw(ax)
        ax.grid(alpha=0.2)
        fig.savefig(output / f'ppo_{name}.png', dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), layout='constrained')
    com_plot(axes)
    fig.savefig(output / 'ppo_com_changes.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'Data_New_15cm')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'PPO_results')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--k-max', type=int, default=K_MAX)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--check-data', action='store_true', help='Validate data and complete-plan feasibility only')
    args = parser.parse_args()
    if args.episodes < 1 or args.threads < 1:
        parser.error('--episodes and --threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        env = ShipCutEnv(args.data, args.k_max)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Dataset validation failed: {exc}\n')
    print(json.dumps(env.data_summary, indent=2))
    print('Complete feasible plan exists with the configured constraints.')
    if args.check_data:
        return
    model, rewards = train(env, args.episodes, args.seed, args.device)
    cuts = evaluate(model, env)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'ppo_cut_plan_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cut_number', *cuts[0]])
        writer.writeheader()
        writer.writerows(dict(cut_number=i, **cut) for i, cut in enumerate(cuts, 1))
    total_area = sum(cut['boundary_area_cm2'] for cut in cuts)
    summary = dict(data=env.data_summary, complete=env.index == env.N, num_cuts=len(cuts),
                   total_area_cm2=total_area, total_area_m2=total_area / 1e4,
                   max_block_mass_kg=max(cut['mass_kg'] for cut in cuts),
                   max_dcom_cm={axis: max(cut[f'dcom_{axis}_cm'] for cut in cuts) for axis in 'xyz'},
                   mean_dcom_cm={axis: float(np.mean([cut[f'dcom_{axis}_cm'] for cut in cuts])) for axis in 'xyz'},
                   evaluation_return=sum(cut['reward'] for cut in cuts),
                   episodes=args.episodes, seed=args.seed, k_max=env.k_max,
                   mass_limit_kg=MAX_BLOCK_MASS_KG, com_shift_limit_cm=MAX_COM_SHIFT_CM)
    (args.output / 'ppo_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output / 'ppo_cut_plan.json').write_text(json.dumps(cuts, indent=2), encoding='utf-8')
    with (args.output / 'ppo_training_rewards.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['episode', 'return'])
        writer.writerows(enumerate(rewards, 1))
    with (args.output / 'ppo_training_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(env.training_history[0]))
        writer.writeheader()
        writer.writerows(env.training_history)
    torch.save(dict(model_state_dict=model.state_dict(), architecture=f'{5+2*env.k_max}-256-256 actor{env.k_max} critic1',
                    summary=summary, area_scale=env.area_scale, com_scale=env.com_scale),
               args.output / 'ppo_model.pt')
    plot_results(env, cuts, rewards, args.output)
    print(json.dumps(summary, indent=2))
    print(f'Saved PPO outputs to {args.output.resolve()}')


if __name__ == '__main__':
    main()
