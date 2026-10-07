# Review of DQN_Sep.py and the corrected sensitivity study

The reviewed original currently has a 20,000 kg limit, K=10, 100 episodes,
and the correct local dataset path. It is preserved unchanged. The study
uses **DQN_reviewed.py**, not an unmodified run of the original script.

## Findings, ordered by importance

1. **High: hard constraints are not enforced in training (lines 249–327).**
   `block_mass_acc` adds only the endpoint slice, not the whole requested block.
   The mass overflow, lookahead and tail branches contain only `pass`.
   Reducing k once does not guarantee that either constraint is satisfied.
   Reproduction on the supplied dataset: at the initial state, request seven
   slices; the original step executes six weighing **21,362.195 kg**, above
   the 20,000 kg limit. This is an actual violation, not just a theoretical risk.

2. **High: exported actions do not necessarily match executed cuts
   (lines 254–327 and 536–550).** `step()` silently reduces the requested k,
   while evaluation computes `start_idx` and `k_slices` from the original
   action. For example, an initially legal two-slice request becomes one
   executed slice, but the recorded start would be -1 and length 2. This
   invalidates cut-list interpretation and can undo the pre-execution mask.

3. **High: constrained-action handling is inconsistent (lines 419–423,
   461–469, and 526–537).** Exploration, greedy training actions, and Double
   DQN target selection consider illegal actions. Only evaluation masks
   candidates, and even then step can modify them. For a constrained problem,
   selection and bootstrapping must use the same feasible action set.

4. **Medium: an all-terminal replay batch crashes (line 411).**
   `torch.cat([])` is called when every sampled transition is terminal.
   This is uncommon on this dataset, but a real edge case. Terminal samples
   should simply use the immediate reward as their target.

5. **Medium: incomplete plans may be summarized as ordinary results
   (lines 533–535 and 567 onward).** Evaluation breaks on no legal action and
   still reports the partial cut count/area without a completion flag.
   Such values cannot be compared directly with complete plans.

6. **Medium: loading and modeling exclusions are silent (lines 67–115).**
   Parsing exceptions skip CSV rows; X COM assignment excludes bodies outside
   the DXF intervals. In the supplied dataset, two bodies totaling
   **1,679.384 kg** are outside the intervals. Each entire body is assigned
   to the interval containing its COM; spanning bodies are not subdivided.
   DXF geometry is not read; filename areas are assumed to be cm².

## Corrections used for this study

- Reuse the already validated physical `ShipCutEnv` from PPO_Sep.py.
  This is shared physics/data/reward code; the learning algorithm is Double DQN.
- Mask exploration, greedy actions, and policy action selection in the Double
  DQN bootstrap. Target network evaluates the policy-selected legal action.
- Execute precisely the selected block; report its true boundaries and length.
- Use the same backward completion-feasibility filter as PPO. It removes
  choices leading to dead ends, not complete feasible plans.
- Handle terminal samples without evaluating an empty next-state tensor.
- Report completion, excluded mass, constraints, seeds, hyperparameters,
  training transitions, elapsed time, and the original source SHA-256.
- Use deterministic state-index replay as a compact equivalent of uniform
  transition replay: observations, rewards and next indices are fixed by
  current index and action. No dynamic-programming reward optimization or
  change to the learned Bellman objective is performed.

## Preserved algorithm and experiment settings

- Network: (5+2K) → 256 ReLU → 256 ReLU → K Q-values.
- Double DQN, Huber loss; AdamW(lr=0.0005, amsgrad=True), default weight decay 0.01.
- Batch 256; replay capacity 100,000; gamma 0.995; tau 0.005.
- Epsilon 0.95 → 0.02, exponential decay scale 3,000 transitions.
- Gradient component clipping at 100; one gradient update per transition
  after the replay buffer contains 256 entries; target update every step.
- Original weighted cutting-area/COM/cut-penalty/mass-bonus reward unchanged.
- Study: K=7,8,9,10,15,20; seeds 42,43,44; 1,000 episodes per run, matching
  the PPO sensitivity study's episode budget rather than the original 100 default.
- Same 20,000 kg and 50 cm per-axis limits; same final-removal convention
  (counted as a cut, its boundary area included, zero final COM shift).
- Final greedy model evaluation; no selection of the best training episode.

Equal episodes do not imply equal compute between DQN and PPO. K also changes
input/output dimensions and exploration probabilities. Three seeds are a
limited measure of variability and do not establish statistical superiority.

## Reproduce

```powershell
python -m unittest test_dqn_reviewed.py
python DQN_reviewed.py --k-max 7 --episodes 1000 --seed 42 --output DQN_results_K7
python DQN_sensitivity.py
```

The full study saves separate logs, checkpoints, cut plans, training metrics
and figures for every K/seed in `DQN_sensitivity_results`, plus aggregate
tables, report and comparison plot. PPO outputs are preserved.
