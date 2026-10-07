# -*- coding: utf-8 -*-
"""
DQN optimization for grouped ship cuts (centimeters; kg)
- Minimize boundary cutting area by grouping slices into blocks
- Constraints per cut:
    (1) Block mass ≤ 20,000 kg
    (2) Remaining-body COM shift ≤ 50 cm in X, Y, Z (Y is most critical)
Data assumptions:
- DXF filenames: cross_section_<X>_Area_<A>.dxf where X is in cm, A treated as cm^2
- CSV: physical_properties_new.csv with columns:
    Body Name, Mass, Center of Mass-x, Center of Mass-y, Center of Mass-z  (COM in cm, mass in kg)
"""

import os, re, csv, math, random
from collections import namedtuple, deque
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------- Config -------------
UNIT = "cm"
DXF_REGEX = re.compile(r"^cross_section_(-?\d+(?:\.\d+)?)_Area_(-?\d+(?:\.\d+)?)\.dxf$", re.IGNORECASE)

MAX_BLOCK_MASS_KG = 20000.0
MAX_COM_SHIFT_CM  = 50.0
K_MAX = 10  # maximum slices per cut action (action = 1..K_MAX)

# DQN hyperparams (tweak as needed)
BATCH_SIZE = 256
GAMMA = 0.995
EPS_START = 0.95
EPS_END = 0.02
EPS_DECAY = 3000
TAU = 0.005
LR = 5e-4
MEM_CAP = 100000
EPISODES = 100

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                      else "cpu")

# ------------- Data helpers (cm / kg) -------------
def find_dxfs_cm(directory: str) -> List[Dict]:
    dxfs = []
    for fn in os.listdir(directory):
        m = DXF_REGEX.match(fn)
        if not m:
            continue
        dxfs.append({
            "x_cm": float(m.group(1)),
            "area_label_cm2": float(m.group(2)),
            "name": fn,
            "file": os.path.join(directory, fn),
        })
    dxfs.sort(key=lambda d: d["x_cm"])
    return dxfs

def read_bodies_cm(csv_path: str) -> List[Dict]:
    bodies = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                bodies.append({
                    "name": (row.get("Body Name") or "").strip() or "Unnamed Body",
                    "mass": float(row["Mass"]),                            # kg
                    "cx": float(row["Center of Mass-x"]),                  # cm
                    "cy": float(row.get("Center of Mass-y", 0.0)),         # cm
                    "cz": float(row.get("Center of Mass-z", 0.0)),         # cm
                })
            except Exception:
                continue
    return bodies

def build_slice_arrays(dxfs, bodies):
    """
    Returns:
      m[i]  (kg)
      Wx/Wy/Wz (kg·cm)
      boundary_area[i] (cm^2) — boundary right after slice i (i = 0..N-1)
    Slices are [x_i, x_{i+1}) with N = len(dxfs) - 1
    """
    N = max(0, len(dxfs) - 1)
    m  = np.zeros(N, dtype=float)
    Wx = np.zeros(N, dtype=float)
    Wy = np.zeros(N, dtype=float)
    Wz = np.zeros(N, dtype=float)
    boundary_area = np.array([dxfs[i+1]["area_label_cm2"] for i in range(N)], dtype=float)

    if N == 0:
        return m, Wx, Wy, Wz, boundary_area

    bx = np.array([b["cx"] for b in bodies], dtype=float)
    by = np.array([b["cy"] for b in bodies], dtype=float)
    bz = np.array([b["cz"] for b in bodies], dtype=float)
    bm = np.array([b["mass"] for b in bodies], dtype=float)

    for i in range(N):
        x0, x1 = sorted((dxfs[i]["x_cm"], dxfs[i+1]["x_cm"]))
        mask = (bx >= x0) & (bx < x1)
        if not np.any(mask):
            continue
        m[i]  = bm[mask].sum()
        Wx[i] = (bm[mask] * bx[mask]).sum()
        Wy[i] = (bm[mask] * by[mask]).sum()
        Wz[i] = (bm[mask] * bz[mask]).sum()

    return m, Wx, Wy, Wz, boundary_area

# ------------- Environment -------------
class ShipCutEnv(gym.Env):
    """
    Observation:
      [ cur_idx/N,  M_rem/M0,  Cx_rem/scale, Cy_rem/scale, Cz_rem/scale,
        next_k1_mass/MAX_BLOCK, ..., next_kK_mass/MAX_BLOCK,
        next_k1_area_scaled, ..., next_kK_area_scaled ]
    Action: discrete {0..K_MAX-1} meaning cut k = action+1 slices (with shields)
    """

    def __init__(self, directory, k_max=K_MAX,
                 max_block_mass=MAX_BLOCK_MASS_KG, max_com_shift=MAX_COM_SHIFT_CM):
        super().__init__()
        self.directory = directory
        self.k_max = int(k_max)
        self.max_block_mass = float(max_block_mass)
        self.max_com_shift = float(max_com_shift)

        # Load data
        dxfs = find_dxfs_cm(directory)
        if len(dxfs) < 2:
            raise RuntimeError("Need at least two DXFs to form slice intervals.")
        csv_path = os.path.join(directory, "physical_properties_new.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        bodies = read_bodies_cm(csv_path)

        self.m, self.Wx, self.Wy, self.Wz, self.boundary_area = build_slice_arrays(dxfs, bodies)
        self.N = len(self.m)

        # Totals & prefix sums
        self.M0  = float(self.m.sum())
        self.Wx0 = float(self.Wx.sum())
        self.Wy0 = float(self.Wy.sum())
        self.Wz0 = float(self.Wz.sum())

        self.m_pref  = np.concatenate(([0.0], np.cumsum(self.m)))
        self.Wx_pref = np.concatenate(([0.0], np.cumsum(self.Wx)))
        self.Wy_pref = np.concatenate(([0.0], np.cumsum(self.Wy)))
        self.Wz_pref = np.concatenate(([0.0], np.cumsum(self.Wz)))

        # Area autoscale for O(1) rewards
        self.area_scale = float(np.median(self.boundary_area)) if np.median(self.boundary_area) > 0 else 1.0

        # COM scale for state features
        def _com_totals_to_tuple(M, Wx, Wy, Wz):
            if M <= 1e-12: return (0.0, 0.0, 0.0)
            return (Wx/M, Wy/M, Wz/M)
        C0 = _com_totals_to_tuple(self.M0, self.Wx0, self.Wy0, self.Wz0)
        self.com_scale = max(1.0, np.max(np.abs(C0)))

        # Gym spaces
        obs_dim = 5 + 2*self.k_max
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k_max)

        self.reset()

    # ---- helpers ----
    def remaining_after(self, cut_idx):
        M  = self.M0  - self.m_pref[cut_idx]
        Wx = self.Wx0 - self.Wx_pref[cut_idx]
        Wy = self.Wy0 - self.Wy_pref[cut_idx]
        Wz = self.Wz0 - self.Wz_pref[cut_idx]
        return M, Wx, Wy, Wz

    def com_from_totals(self, M, Wx, Wy, Wz):
        if M <= 1e-12:
            return (np.nan, np.nan, np.nan)
        return (Wx/M, Wy/M, Wz/M)

    def delta_com_if_cut(self, start_idx, end_idx, prev_C):
        """end_idx inclusive; returns |ΔC| for remaining body (cm)."""
        M_new  = self.M0  - self.m_pref[end_idx+1]
        Wx_new = self.Wx0 - self.Wx_pref[end_idx+1]
        Wy_new = self.Wy0 - self.Wy_pref[end_idx+1]
        Wz_new = self.Wz0 - self.Wz_pref[end_idx+1]
        if M_new <= 1e-12:
            return np.array([0.0, 0.0, 0.0], float)
        C_new = np.array([Wx_new/M_new, Wy_new/M_new, Wz_new/M_new], float)
        return np.abs(C_new - np.array(prev_C, float))

    def compute_state(self):
        M, Wx, Wy, Wz = self.remaining_after(self.cur_idx)
        Cx, Cy, Cz = self.com_from_totals(M, Wx, Wy, Wz)

        masses = []
        areas  = []
        for k in range(1, self.k_max+1):
            j = self.cur_idx + k - 1
            if j >= self.N:
                masses.append(0.0)
                areas.append(0.0)
            else:
                block_mass = float(self.m[self.cur_idx:j+1].sum())
                masses.append(block_mass / self.max_block_mass)
                areas.append(float(self.boundary_area[j]) / (self.area_scale + 1e-9))

        obs = np.array([
            self.cur_idx / max(1, self.N),
            (M / max(1e-9, self.M0)),
            (0.0 if not np.isfinite(Cx) else Cx / self.com_scale),
            (0.0 if not np.isfinite(Cy) else Cy / self.com_scale),
            (0.0 if not np.isfinite(Cz) else Cz / self.com_scale),
            *masses,
            *areas
        ], dtype=np.float32)
        return obs

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_idx = 0
        self.block_start_idx = 0
        self.block_mass_acc = 0.0
        self.total_area_cost = 0.0

        M, Wx, Wy, Wz = self.remaining_after(0)
        self.prev_C = np.array(self.com_from_totals(M, Wx, Wy, Wz), dtype=float)

        return self.compute_state(), {"whole_com_cm": self.prev_C.copy()}

    def step(self, action):
        # Convert to k slices; clamp to available
        k = int(action) + 1
        if self.cur_idx >= self.N:
            return self.compute_state(), 0.0, True, False, {}
        if self.cur_idx + k - 1 >= self.N:
            k = self.N - self.cur_idx  # cap so we can finish exactly

        j_end = self.cur_idx + k - 1
        mass_this = float(self.m[j_end])
        self.block_mass_acc += mass_this

        # ---- Shields ----
        # Zero-mass guard (don't close on zero unless last)
        if self.m[j_end] <= 1e-9 and j_end < self.N - 1:
            k = max(1, k-1); j_end = self.cur_idx + k - 1

        # Mass overflow guard (force cut if next would exceed)
        if j_end < self.N - 1:
            next_mass = float(self.m[j_end + 1])
            if self.block_mass_acc + next_mass > self.max_block_mass:
                # keep k; we'll close this block now
                pass

        # COM-now shield: if closing now breaches, postpone (unless last)
        d_now = self.delta_com_if_cut(self.block_start_idx, j_end, self.prev_C)
        if np.any(d_now > self.max_com_shift - 1e-9) and j_end < self.N - 1:
            # postpone: reduce k by 1 if possible
            if k > 1:
                k -= 1
                j_end = self.cur_idx + k - 1
                self.block_mass_acc -= mass_this  # undo last add
                mass_this = float(self.m[j_end])
                self.block_mass_acc += mass_this
                d_now = self.delta_com_if_cut(self.block_start_idx, j_end, self.prev_C)

        # COM-next shield: if postponing would make next breach, force cut now
        if j_end < self.N - 1:
            d_next = self.delta_com_if_cut(self.block_start_idx, j_end + 1, self.prev_C)
            if np.any(d_next > self.max_com_shift - 1e-9):
                # keep current j_end (force cut)
                pass

        # Short lookahead for feasibility
        SAFE_LOOK = 20
        if j_end < self.N - 1:
            end_scan = min(self.N - 1, j_end + SAFE_LOOK)
            safe_ahead = False
            for e in range(j_end, end_scan + 1):
                bm = float(self.m_pref[e+1] - self.m_pref[self.block_start_idx])
                if bm <= 0 or bm > self.max_block_mass: 
                    continue
                d_ahead = self.delta_com_if_cut(self.block_start_idx, e, self.prev_C)
                if np.all(d_ahead <= self.max_com_shift):
                    safe_ahead = True
                    break
            if not safe_ahead:
                # force cut at current j_end
                pass

        # Tail shield (ensure final removal safe)
        TAIL = 5
        if j_end >= self.N - TAIL:
            bm_tail = float(self.m_pref[self.N] - self.m_pref[self.block_start_idx])
            d_tail = self.delta_com_if_cut(self.block_start_idx, self.N - 1, self.prev_C)
            if bm_tail > self.max_block_mass or np.any(d_tail > self.max_com_shift):
                # force a cut now
                pass

        # Tiny-block avoidance unless forced
        goal_mass = 0.85 * self.max_block_mass
        tiny = (self.block_mass_acc < max(0.25 * goal_mass, 2000.0))
        if tiny and j_end < self.N - 1:
            # if not forced by overflow/feasibility, postpone (reduce k by 1)
            next_mass = float(self.m[j_end + 1])
            forced_over = (self.block_mass_acc + next_mass > self.max_block_mass)
            if not forced_over and k > 1:
                k -= 1
                j_end = self.cur_idx + k - 1
                self.block_mass_acc -= mass_this
                mass_this = float(self.m[j_end])
                self.block_mass_acc += mass_this
                d_now = self.delta_com_if_cut(self.block_start_idx, j_end, self.prev_C)

        # ---------- Execute this cut ----------
        # Area cost uses boundary at j_end
        # Area cost uses boundary at j_end
        area_cost_scaled = float(self.boundary_area[j_end]) / (self.area_scale + 1e-9)
        raw_area_cm2     = float(self.boundary_area[j_end])  # <-- keep raw for reporting

        # --- Mass bonus to encourage 15–18 t blocks ---
        # Current block mass (kg) for this cut
        bm = float(self.m_pref[j_end+1] - self.m_pref[self.block_start_idx])
        # Progressive cut penalty (encourages fewer, larger cuts)
        cut_penalty = 0.8 * (1.0 - (self.cur_idx / self.N))  # More penalty early
        
        mass_bonus = 0.0
        if 14000.0 <= bm <= 19000.0:
            mass_bonus += 2.0          # ideal window → best bonus
        elif 11000.0 <= bm < 14000.0 or 19000.0 < bm <= self.max_block_mass:
            mass_bonus += 1.05          # near-ideal window → smaller bonus

        # --- Reward: minimize area; prioritize Y; soften X/Z; small per-cut tax ---
        # (returns stay ~O(1) thanks to area autoscaling)
        reward = -(
            1.9*area_cost_scaled
            + 0.01 * d_now[1]                   # Y is critical
            + 0.001 * (d_now[0] + d_now[2])      # soften X, Z
            + 1.5*cut_penalty                       # small per-cut tax
            - 1.5*mass_bonus                         # subtracting a negative == bonus
        )

        # Update remaining-body COM
        M_new  = self.M0  - self.m_pref[j_end+1]
        Wx_new = self.Wx0 - self.Wx_pref[j_end+1]
        Wy_new = self.Wy0 - self.Wy_pref[j_end+1]
        Wz_new = self.Wz0 - self.Wz_pref[j_end+1]
        if M_new > 1e-12:
            self.prev_C = np.array([Wx_new/M_new, Wy_new/M_new, Wz_new/M_new], float)

        # Advance indices
        self.cur_idx = j_end + 1
        self.block_start_idx = self.cur_idx
        self.block_mass_acc = 0.0
        self.total_area_cost += raw_area_cm2  # accumulate real cm²

        done = (self.cur_idx >= self.N)
        obs = self.compute_state()
        info = {
            "block_mass": bm,
            "area_cost": area_cost_scaled,      # scaled (for reward sanity)
            "area_cm2": raw_area_cm2,           # <-- NEW: raw area for reporting
            "delta_com_cm": d_now,
            "remaining_mass": M_new
        }
        return obs, float(reward), done, False, info

    
###############################################################

# ------------- DQN -------------
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x): return self.net(x)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None], dim=0)
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)

    q_vals = policy_net(state_batch).gather(1, action_batch)

    # Double DQN
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
        next_q = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
        next_state_values[non_final_mask] = next_q
    target_q = reward_batch + GAMMA * next_state_values

    loss = F.smooth_l1_loss(q_vals.squeeze(1), target_q)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100.0)
    optimizer.step()

def soft_update(target, source, tau=TAU):
    tgt = target.state_dict(); src = source.state_dict()
    for k in tgt: tgt[k] = tau * src[k] + (1 - tau) * tgt[k]
    target.load_state_dict(tgt)

# ------------- Training loop -------------
def train(directory_path, episodes=EPISODES):
    env = ShipCutEnv(directory_path)

    n_actions = env.action_space.n
    obs0, _ = env.reset()
    n_obs = obs0.shape[0]

    policy_net = DQN(n_obs, n_actions).to(device)
    target_net = DQN(n_obs, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEM_CAP)

    steps_done = 0
    episode_rewards = []

    for ep in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_r = 0.0

        while True:
            eps_th = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
            steps_done += 1
            if random.random() > eps_th:
                with torch.no_grad():
                    action = policy_net(state).argmax(dim=1, keepdim=True)
            else:
                a = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
                action = a

            next_obs, reward, done, truncated, info = env.step(int(action.item()))
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            total_r += reward

            if done or truncated:
                next_state = None
            else:
                next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_t)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)
            soft_update(target_net, policy_net, TAU)

            if done or truncated:
                break

        episode_rewards.append(total_r)
        if (ep+1) % 20 == 0:
            print(f"Ep {ep+1}/{episodes} | Return: {total_r:.1f} | eps~{eps_th:.3f}")

    # Plot training return
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("DQN training (area-minimizing grouped cuts)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    return policy_net, env, episode_rewards

# ------------- Greedy evaluation with legal masking -------------
def preview_cut(env: ShipCutEnv, start_idx: int, k: int, prev_C) -> dict:
    N = env.N
    j_end = start_idx + k - 1
    if j_end >= N: return {"ok": False, "reason": "beyond_end"}
    block_mass = float(env.m_pref[j_end+1] - env.m_pref[start_idx])
    if block_mass > env.max_block_mass + 1e-9:
        return {"ok": False, "reason": "block_mass_limit", "block_mass": block_mass}
    d = env.delta_com_if_cut(start_idx, j_end, prev_C)
    if np.any(d > env.max_com_shift + 1e-9):
        return {"ok": False, "reason": "com_shift_limit", "delta_cm": d, "block_mass": block_mass}
    return {"ok": True, "block_mass": block_mass, "delta_cm": d, "j_end": j_end}

@torch.no_grad()
def evaluate_greedy_legal(policy_net: nn.Module, env: ShipCutEnv, verbose=True):
    obs, info = env.reset()
    dev = next(policy_net.parameters()).device
    state = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)

    history = []
    prev_C = info.get("whole_com_cm")

    if verbose:
        print(f"[Eval] N={env.N}, K_MAX={env.k_max}, M0={env.M0:.1f} kg, COM0(cm)={prev_C}")

    while True:
        q_all = policy_net(state).squeeze(0).cpu().numpy()
        legal, scores = [], []
        for a in range(env.k_max):
            k = a + 1
            chk = preview_cut(env, env.cur_idx, k, prev_C)
            if chk["ok"]:
                legal.append(a); scores.append(q_all[a])
        if not legal:
            if verbose: print(f"[Eval] No legal actions at idx={env.cur_idx}.")
            break
        best_a = legal[int(np.argmax(scores))]
        next_obs, reward, done, truncated, inf = env.step(best_a)
        new_idx = env.cur_idx
        d = inf.get("delta_com_cm", [np.nan]*3)
        bm = inf.get("block_mass", np.nan)
        area = inf.get("area_cost", np.nan)

        area_scaled = inf.get("area_cost", np.nan)
        area_cm2    = inf.get("area_cm2", np.nan)   # <-- new

        history.append({
            "type": "cut",
            "start_idx": new_idx - (best_a + 1),
            "end_idx": new_idx - 1,
            "k_slices": (best_a + 1),
            "mass": float(bm),
            "delta_cm": np.array(d, float),
            "area_cost": float(area_scaled),
            "area_cm2": float(area_cm2)             # <-- new
        })


        if done or truncated: break
        prev_C = env.prev_C.copy()
        state = torch.tensor(next_obs, dtype=torch.float32, device=dev).unsqueeze(0)

    if verbose:
        print(f"[Eval] Cuts recorded: {len([h for h in history if h['type']=='cut'])}")
    return history

# ------------- Metrics & Plots -------------
def summarize_and_plot(history, directory_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os, csv as _csv

    cuts = [h for h in history if h.get("type") == "cut"]
    if not cuts:
        print("No cuts recorded."); return {}

    masses = np.array([c["mass"] for c in cuts], dtype=float)
    deltas = np.vstack([np.asarray(c["delta_cm"], float) for c in cuts])
    areas_cm2 = np.array([c.get("area_cm2", np.nan) for c in cuts], dtype=float)

    # --- totals ---
    total_area_cm2 = float(np.nansum(areas_cm2))
    total_area_m2  = total_area_cm2 / 1e4  # 1 m² = 10,000 cm²

    metrics = {
        "num_cuts": len(cuts),
        "mass_max": float(np.max(masses)),
        "mass_avg": float(np.mean(masses)),
        "mass_min": float(np.min(masses)),
        "dcom_x_max": float(np.max(np.abs(deltas[:,0]))),
        "dcom_x_avg": float(np.mean(np.abs(deltas[:,0]))),
        "dcom_x_min": float(np.min(np.abs(deltas[:,0]))),
        "dcom_y_max": float(np.max(np.abs(deltas[:,1]))),
        "dcom_y_avg": float(np.mean(np.abs(deltas[:,1]))),
        "dcom_y_min": float(np.min(np.abs(deltas[:,1]))),
        "dcom_z_max": float(np.max(np.abs(deltas[:,2]))),
        "dcom_z_avg": float(np.mean(np.abs(deltas[:,2]))),
        "dcom_z_min": float(np.min(np.abs(deltas[:,2]))),
        "total_area_cm2": total_area_cm2,
        "total_area_m2": total_area_m2
    }

    # --- print summary ---
    print("\n===== Cut Metrics (cm / kg) =====")
    print(f"Number of cuts: {metrics['num_cuts']}")
    print(f"Mass per cut (kg): max={metrics['mass_max']:.2f}, "
          f"avg={metrics['mass_avg']:.2f}, min={metrics['mass_min']:.2f}")
    print(f"|ΔCOM_x| (cm): max={metrics['dcom_x_max']:.2f}, "
          f"avg={metrics['dcom_x_avg']:.2f}, min={metrics['dcom_x_min']:.2f}")
    print(f"|ΔCOM_y| (cm): max={metrics['dcom_y_max']:.2f}, "
          f"avg={metrics['dcom_y_avg']:.2f}, min={metrics['dcom_y_min']:.2f}")
    print(f"|ΔCOM_z| (cm): max={metrics['dcom_z_max']:.2f}, "
          f"avg={metrics['dcom_z_avg']:.2f}, min={metrics['dcom_z_min']:.2f}")
    print(f"Total cutting area: {total_area_cm2:,.0f} cm²  ({total_area_m2:,.2f} m²)")

    steps = np.arange(1, len(cuts)+1)

    # --- Mass per cut ---
    plt.figure(figsize=(10,4))
    plt.bar(steps, masses)
    plt.axhline(MAX_BLOCK_MASS_KG, linestyle="--", linewidth=1)
    plt.title("Mass per cut")
    plt.xlabel("Cut #"); plt.ylabel("Mass (kg)")
    plt.tight_layout()

    # --- ΔCOM (three subplots) ---
    fig, axes = plt.subplots(3,1, figsize=(10,8), sharex=True)
    axes[0].plot(steps, deltas[:,0], marker="o")
    axes[0].axhline(MAX_COM_SHIFT_CM, ls="--", lw=1); axes[0].axhline(-MAX_COM_SHIFT_CM, ls="--", lw=1)
    axes[0].set_ylabel("ΔCOM_x (cm)"); axes[0].grid(True, alpha=0.3); axes[0].set_title("COM change per cut")

    axes[1].plot(steps, deltas[:,1], marker="o")
    axes[1].axhline(MAX_COM_SHIFT_CM, ls="--", lw=1); axes[1].axhline(-MAX_COM_SHIFT_CM, ls="--", lw=1)
    axes[1].set_ylabel("ΔCOM_y (cm)"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, deltas[:,2], marker="o")
    axes[2].axhline(MAX_COM_SHIFT_CM, ls="--", lw=1); axes[2].axhline(-MAX_COM_SHIFT_CM, ls="--", lw=1)
    axes[2].set_ylabel("ΔCOM_z (cm)"); axes[2].set_xlabel("Cut #"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()

    # --- NEW: cumulative cutting area ---
    plt.figure(figsize=(10,4))
    cum_area_m2 = np.cumsum(np.nan_to_num(areas_cm2)) / 1e4
    plt.plot(steps, cum_area_m2, marker="o")
    plt.title("Cumulative cutting area")
    plt.xlabel("Cut #"); plt.ylabel("Area (m²)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Optional: compact cut list for other tooling
    import json
    cuts_json = [
        {"start_idx": int(h["start_idx"]), "end_idx": int(h["end_idx"]), "k": int(h["k_slices"])}
        for h in history if h.get("type") == "cut"
    ]
    cuts_json_path = os.path.join(directory_path, "cut_plan.json")
    with open(cuts_json_path, "w") as f:
        json.dump(cuts_json, f, indent=2)
    print(f"Saved cut list to: {cuts_json_path}")


    


    # --- CSV export with raw area ---
    if directory_path:
        out_csv = os.path.join(directory_path, "cut_plan_metrics.csv")
        with open(out_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["cut#", "start_idx", "end_idx", "k_slices", "mass_kg",
                        "dcom_x_cm", "dcom_y_cm", "dcom_z_cm",
                        "boundary_area_cm2", "area_scaled"])
            for i, c in enumerate(cuts, start=1):
                d = c["delta_cm"]
                w.writerow([i, c["start_idx"], c["end_idx"], c["k_slices"], c["mass"],
                            d[0], d[1], d[2], c.get("area_cm2", np.nan), c.get("area_cost", np.nan)])
        print(f"Saved CSV: {out_csv}")

    return metrics


# ------------- Run -------------
if __name__ == "__main__":
    # >>> SET YOUR DATA FOLDER HERE <<<
    directory_path = r"C:\Users\zu119713\OneDrive - LUT University\0 Phd LUT\2 Primary Cutting Plan\M\Final Sep Work\cuts 119\Example other algorithm\Data_New_15cm"

    policy, env, rewards = train(directory_path, episodes=EPISODES)

    # Greedy evaluation (legal-masked)
    history = evaluate_greedy_legal(policy, env, verbose=True)
    metrics = summarize_and_plot(history, directory_path)
