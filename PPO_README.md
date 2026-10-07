# PPO ship cutting

`PPO_Sep.py` is a standalone PyTorch implementation of clipped PPO with GAE,
an actor-critic network (45 inputs, two 256-unit ReLU layers, 20 policy logits,
one value output), and masked actions. It uses the adjacent `Data_New_15cm`
folder by default. It requires numpy, torch, and matplotlib.

```powershell
python PPO_Sep.py --check-data
python PPO_Sep.py --episodes 1000
python -m unittest test_ppo_sep.py
```

Optional arguments: `--data`, `--output`, `--seed`, `--device`, `--threads`.
Use `--k-max 7` (or 8/10) to change the maximum slices grouped in one block.
The observation and policy dimensions adjust to 5+2K and K automatically;
the hidden layers stay at 256/256. The default remains K=20.

For the controlled K=7,8,10 comparison, run `python PPO_sensitivity.py`.
This runs 1,000 episodes for each K with seeds 42,43,44, saves each model and
cut plan separately, and creates mean/standard-deviation comparison tables
and plots in `PPO_sensitivity_results`. The previous K=20 results are preserved.
Successful training writes a model checkpoint, cut plan JSON and CSV, summary
JSON, episode rewards and cut-count CSVs, a six-panel overview (PNG and SVG),
and separate PNG graphs to `PPO_results`. The graphs show per-cut COM shifts
in X/Y/Z, input slice mass distribution, block mass versus cut number,
episode reward evolution, cumulative cutting area, and cuts per episode.
Default training is 1,000 episodes; final metrics describe the deterministic
greedy policy evaluated after the final update, not the best sampled episode.

## Preserved model and objective

- DXF filenames supply boundaries and areas; DXF geometry is not parsed.
- CSV body masses are assigned wholly to the slice containing their X COM.
- Units: cm, cm², and kg, without conversion.
- Limits: 20,000 kg/block (updated at the user's request) and 50 cm COM movement
  per axis per cut. The original DQN file is unchanged.
- Actions group 1–20 consecutive slices, proceeding in increasing X order.
- Reward is the same weighted area, COM movement, cut penalty, and mass bonus
  expression as DQN_Sep.py; gamma remains 0.995.
- Final removal has zero COM shift by the original convention.

Unlike the original training environment's incomplete shields, this version
enforces limits on the executed action and never silently changes its size.
The same masks are used during collection, PPO updates, and evaluation.
Backward reachability filters out choices that would prevent completion;
it does not optimize the objective or prescribe the best cutting sequence.
No complete feasible solutions are removed by this filter because the
remaining body is fully determined by the current boundary index.

## Validation of the supplied dataset

The folder has 260 DXF boundaries, 259 intervals, and 1,066 CSV bodies.
The original COM assignment covers approximately 1,785,302.657 kg. Two bodies,
totaling approximately 1,679.384 kg, lie outside the intervals and are excluded
by that same assignment rule; the script reports this explicitly.

A complete feasible plan exists with the updated 20,000 kg limit. The earlier
15,000 kg configuration was infeasible: the slice starting at X = -2602 cm
has mass approximately 18,358.098 kg. The 50 cm COM constraint is unchanged.

"Number of cuts" follows the original script's convention: each block removal
counts as a cut, including final removal; the final boundary's filename area
is included in total cutting area. COM plots show absolute per-axis changes
of the remaining body between consecutive removals, not cumulative movement
from the initial COM. Final removal is plotted as zero by convention.

Tests use a separate synthetic feasible dataset to verify PPO parameter
updates, reward arithmetic, complete and legal cuts, output generation, and
rejection of an impossible mass case. No optimality guarantee is made for PPO.
