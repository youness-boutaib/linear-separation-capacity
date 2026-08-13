"""
Reservoir Computing: Implementation and Training on Lorenz System
=================================================================
Predicts the next 3D Lorenz step from the history of the first coordinate.
Inputs are standardised: X_t = (x_t - mean) / std for numerical stability.
The Lorenz trajectory is generated ONCE and shared across ALL 2p simulations
(p symmetric-W runs and p iid-W runs), so that only the randomness of W
varies between experiments.

Definitions
------------------------------
Input snapshot (zero-padded, right-aligned), t = 0, ..., T:
    X_t = (0, 0, ..., 0, X_0, X_1, ..., X_t)  in  R^{T+1}
    (T - t leading zeros, followed by X_0, ..., X_t)

Reservoir state h_t in R^N is produced by the usual recurrence
    h_0 = u * X_0,   h_t = u * X_t_scalar + W h_{t-1}.

Separation proxy, for s < t:
    r_{s,t} = |h_t - h_s| / |X_t - X_s|
where
    |h| = ||h||_2 / sqrt(N)         (h in R^N)
    |X| = ||X||_2 / sqrt(T+1)       (X in R^{T+1}, the padded snapshot above)

r_{s,t} compares how much the reservoir state changed against how much the (padded) input
history changed, both expressed as root-mean-square (normalised) norms.

Losses (train / test) are reported and optimised as MEAN squared error

User-configurable parameters are set in the CONFIG section below.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────
#  CONFIG  –  edit these values
# ──────────────────────────────────────────────
N         = 25       # reservoir dimension
T         = 150      # time-series length: states h_0 .. h_T, targets v_1 .. v_T
p         = 50        # number of simulations of the random connectivity matrix
alpha     = 0.6       # exponent for entry std = 1/N^alpha
TRAINABLE = "no"      # "yes" or "no". Whether to train the input mask u.
TRAJ_SEED = 10        # seed for the single fixed Lorenz initial condition


# ── Lorenz trajectory ─────────────────────────────────────────────────────────
def generate_lorenz(T_steps, sigma=10.0, rho=28.0, beta=8.0/3.0,
                    dt=0.01, x0=None, rng=None):
    """
    Integrate Lorenz ODE with RK4 for T_steps steps.
    Returns array of shape (T_steps+1, 3): states v_0, v_1, ..., v_{T_steps}.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if x0 is None:
        x0 = rng.uniform(-15, 15, size=3)
    traj = np.zeros((T_steps + 1, 3))
    traj[0] = x0

    def f(state):
        x, y, z = state
        return np.array([sigma * (y - x),
                         x * (rho - z) - y,
                         x * y - beta * z])

    state = x0.copy()
    for t in range(T_steps):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj[t + 1] = state
    return traj   # shape (T+1, 3)


# ── Neural network F ──────────────────────────────────────────────────────────
class ReservoirNet(nn.Module):
    """3-hidden-layer MLP: N -> 128 -> 64 -> 16 -> 3."""
    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
    def forward(self, x):
        return self.net(x)


# ── X_t snapshots (right-aligned, zero-padded convention) ────────────────────
def X_snapshot(X_seq, t, T):
    """
    Return X_t as a (T+1,)-vector with the RIGHT-ALIGNED, zero-padded
    convention:
        X_t[k] = 0                 for k < T - t
        X_t[k] = X_seq[k - (T-t)] for k >= T - t
    i.e. the last (t+1) entries are X_0, X_1, ..., X_t and the first
    (T - t) entries are zero.
    """
    snap = np.zeros(T + 1)
    start_col = T - t
    snap[start_col:] = X_seq[:t + 1]
    return snap


def build_all_X_snapshots(X_seq, T):
    """Precompute X_t for all t = 0, ..., T once (avoids O(T^2) rebuilds)."""
    return {t: X_snapshot(X_seq, t, T) for t in range(T + 1)}


# ── Separation ratio statistics ────────────────────────────────────────────────
def compute_r_stats(h_states, X_snaps, T, N):
    """
    Compute, for all s < t:
        r_{s,t} = |h_t - h_s| / |X_t - X_s|
    with
        |h| = ||h||_2 / sqrt(N)     (h_t, h_s in R^N)
        |X| = ||X||_2 / sqrt(T+1)   (X_t, X_s in R^{T+1}, zero-padded snapshots)

    h_states : dict  t -> np.array of shape (N,),  t in {0, ..., T}
    X_snaps  : dict  t -> np.array of shape (T+1,), from build_all_X_snapshots()
    T        : total length (= global T)
    N        : reservoir dimension

    Note: h_0 = u * X_0 is included in the ratio statistics. It carries
    no target (the target sequence starts at v_1) but is a valid reservoir
    state for measuring separation.

    Returns (r_mm, r_avg, r_med, r_dom).
    """
    times = sorted(h_states.keys())
    sqrt_N  = math.sqrt(N)
    sqrt_Tp = math.sqrt(T + 1)

    ratios = []
    for i, t in enumerate(times):
        for s in times[:i]:      # s < t
            diff_X = X_snaps[t] - X_snaps[s]
            norm_X = float(np.linalg.norm(diff_X)) / sqrt_Tp
            if norm_X < 1e-15:
                continue
            diff_h = h_states[t] - h_states[s]
            norm_h = float(np.linalg.norm(diff_h)) / sqrt_N
            r = norm_h / norm_X
            if not math.isfinite(r):
                continue
            ratios.append(r)

    if not ratios:
        return 0.0, 0.0, 0.0, 0.0

    r_min = min(ratios)
    r_max = max(ratios)
    r_sum = sum(ratios)
    r_mm  = r_min / r_max if r_max > 0 else 0.0
    r_avg = r_sum / len(ratios)
    r_med = float(np.median(ratios))
    r_dom = r_max / r_sum if r_sum > 0 else 0.0
    return r_mm, r_avg, r_med, r_dom


# ── One full experiment ───────────────────────────────────────────────────────
def run_experiment(W_sym: bool, trainable: bool,
                   X_seq, targets,
                   seed_offset: int = 0):
    """
    Run p simulations for a fixed input time series and return mean scores.

    Parameters
    ----------
    W_sym       : whether W is symmetric (True) or i.i.d. (False)
    trainable   : whether u is a learnable parameter
    X_seq       : standardised input sequence, shape (T+1,) — shared, fixed
    targets     : one-step-ahead targets v_1 .. v_T, shape (T, 3), original scale
                  targets[t] = v_{t+1}, so h_t predicts targets[t] (t = 0..T-1)
    seed_offset : added to per-simulation seed so sym and iid seeds differ

    Train/test split
    ----------------
    Valid prediction times are t = 0, ..., T-1 (state h_t -> target v_{t+1}).
    Train : t in {0, ..., n_train-1}   (first 75%)
    Test  : t in {n_train, ..., T-1}   (last 25%)
    h_T is computed for the ratio statistics but has no forecast target.

    Returns dict with keys: r_mm, r_avg, r_med, r_dom, loss_train, loss_test
    """
    # Train/test split — identical for every simulation
    n_valid = T                         # valid indices: 0 .. T-1
    n_train = int(0.75 * n_valid)      # 112 for T=150
    train_times = set(range(n_train))              # {0, ..., 111}
    test_times  = set(range(n_train, n_valid))     # {112, ..., 149}
    n_train_samples = len(train_times)
    n_test_samples  = len(test_times)

    # X_t snapshots depend only on X_seq and T -> precompute once, reused
    # for every simulation (they don't depend on u, W, or the seed).
    X_snaps = build_all_X_snapshots(X_seq, T)

    all_r_mm    = []
    all_r_avg   = []
    all_r_med   = []
    all_r_dom   = []
    all_loss_tr = []
    all_loss_te = []

    for i in range(p):
        # Seeds control W draw and F_net initialisation only.
        # X_seq and targets are fixed externally.
        rng_W = np.random.default_rng(42 + i + seed_offset)
        torch.manual_seed(42 + i + seed_offset)

        # ── Generate W ──────────────────────────────────────────────────────
        w_std = 1.0 / (N ** alpha)
        if W_sym:
            upper = rng_W.normal(0, w_std, (N, N))
            W_np  = np.triu(upper) + np.triu(upper, 1).T   # diagonal drawn once
        else:
            W_np  = rng_W.normal(0, w_std, (N, N))
        W = torch.tensor(W_np, dtype=torch.float32)

        # ── Input mask u ────────────────────────────────────────────────────
        u_init = np.ones(N) / math.sqrt(N)
        if trainable:
            u_param = nn.Parameter(torch.tensor(u_init, dtype=torch.float32))
        else:
            u_param = torch.tensor(u_init, dtype=torch.float32)

        # ── Readout network F ────────────────────────────────────────────────
        F_net   = ReservoirNet(N)
        loss_fn = nn.MSELoss()

        # ── Optimizer ────────────────────────────────────────────────────────
        params = list(F_net.parameters()) + ([u_param] if trainable else [])
        optimizer = optim.Adam(params, lr=1e-3)

        # ── Reservoir forward pass (returns states h_0 .. h_T) ──────────────
        def forward_all(u_vec):
            """
            h_0 = u * X_0
            h_t = u * X_t + W @ h_{t-1},  t = 1, ..., T
            Returns list of (t, h_t tensor),  t = 0 .. T.
            Closes over W and X_seq (both fixed for this simulation).
            """
            states = []
            h_prev = u_vec * float(X_seq[0])
            states.append((0, h_prev))
            for t_idx in range(1, T + 1):
                h_t = u_vec * float(X_seq[t_idx]) + W @ h_prev
                states.append((t_idx, h_t))
                h_prev = h_t
            return states

        # ── Training loop ────────────────────────────────────────────────────
        # objective is a MEAN squared error.
        N_EPOCHS = 300
        for _ in range(N_EPOCHS):
            optimizer.zero_grad()
            states = forward_all(u_param)
            total_loss = torch.zeros(1)
            for t_idx, h_t in states:
                if t_idx in train_times:
                    pred = F_net(h_t.unsqueeze(0))
                    tgt  = torch.tensor(targets[t_idx],
                                        dtype=torch.float32).unsqueeze(0)
                    total_loss = total_loss + loss_fn(pred, tgt)
            total_loss = total_loss / n_train_samples
            if total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()

        # ── Evaluation ───────────────────────────────────────────────────────
        F_net.eval()
        with torch.no_grad():
            states = forward_all(u_param)
            # Collect reservoir states as numpy for ratio computation
            h_states_np = {t_idx: h_t.detach().numpy().copy()
                           for t_idx, h_t in states}

            tr_loss_sum = te_loss_sum = 0.0
            tr_count    = te_count    = 0
            for t_idx, h_t in states:
                if t_idx in train_times:
                    pred = F_net(h_t.unsqueeze(0)).squeeze(0)
                    tgt  = torch.tensor(targets[t_idx], dtype=torch.float32)
                    tr_loss_sum += float(((pred - tgt) ** 2).mean())
                    tr_count    += 1
                elif t_idx in test_times:
                    pred = F_net(h_t.unsqueeze(0)).squeeze(0)
                    tgt  = torch.tensor(targets[t_idx], dtype=torch.float32)
                    te_loss_sum += float(((pred - tgt) ** 2).mean())
                    te_count    += 1

        loss_tr = tr_loss_sum / tr_count if tr_count > 0 else 0.0
        loss_te = te_loss_sum / te_count if te_count > 0 else 0.0

        # ── Ratio statistics ─────────────────────────────────────────────────
        # h_states_np contains t = 0 .. T; h_T has no forecast target but
        # is a valid state for separation measurement. X_snaps does not
        # depend on u/W/seed, so it is reused across all p simulations.
        r_mm, r_avg, r_med, r_dom = compute_r_stats(h_states_np, X_snaps, T, N)

        all_r_mm.append(r_mm)
        all_r_avg.append(r_avg)
        all_r_med.append(r_med)
        all_r_dom.append(r_dom)
        all_loss_tr.append(loss_tr)
        all_loss_te.append(loss_te)

        print(f"  sim {i+1}/{p}:  loss_train={loss_tr:.3e}  "
              f"loss_test={loss_te:.3e}  r_mm={r_mm:.3e}  "
              f"r_avg={r_avg:.3e}  r_med={r_med:.3e}  r_dom={r_dom:.3e}")

    return {
        "r_mm":       float(np.mean(all_r_mm)),
        "r_avg":      float(np.mean(all_r_avg)),
        "r_med":      float(np.mean(all_r_med)),
        "r_dom":      float(np.mean(all_r_dom)),
        "loss_train": float(np.mean(all_loss_tr)),
        "loss_test":  float(np.mean(all_loss_te)),
    }


# ── LaTeX table builder ───────────────────────────────────────────────────────
def sci(v):
    """Format a float as LaTeX scientific notation."""
    if not math.isfinite(v):
        return r"\infty" if v > 0 else r"-\infty"
    if v == 0:
        return "0"
    sign     = "-" if v < 0 else ""
    v        = abs(v)
    exp      = int(math.floor(math.log10(v)))
    mantissa = v / (10 ** exp)
    return rf"{sign}{mantissa:.3f} \times 10^{{{exp}}}"


def make_latex_row(w_type, scores):
    return (f"  & {w_type}"
            f" & ${sci(scores['r_mm'])}$"
            f" & ${sci(scores['r_avg'])}$"
            f" & ${sci(scores['r_med'])}$"
            f" & ${sci(scores['r_dom'])}$"
            f" & ${sci(scores['loss_train'])}$"
            f" & ${sci(scores['loss_test'])}$ \\\\")


def print_latex_table(alpha_val, scores_sym, scores_iid):
    print("\n% ── LaTeX Table ─────────────────────────────────────────")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{c l r r r r r r}")
    print(r"\hline")
    print(r"$\alpha$ & $W$ type & $r_{m\text{-}m}$ & $\overline{r}$ "
          r"& $r_{\mathrm{med}}$ & $r_{\mathrm{dom}}$ & $\mathcal{L}_{\mathrm{train}}$"
          r" & $\mathcal{L}_{\mathrm{test}}$ \\")
    print(r"\hline")
    print(f"  \\multirow{{2}}{{*}}{{${alpha_val}$}}"
          f"{make_latex_row('sym', scores_sym)}")
    print(f"  {make_latex_row('iid', scores_iid)}")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% ─────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    trainable = TRAINABLE.lower() == "yes"

    print("=" * 60)
    print("Reservoir Computing on Lorenz System")
    print(f"N={N}  T={T}  p={p}  alpha={alpha}  trainable={TRAINABLE}")
    print("=" * 60)

    # Generate ONE trajectory, shared across all 2p simulations.
    # Only the W draws differ between simulations.
    traj_rng = np.random.default_rng(TRAJ_SEED)
    traj     = generate_lorenz(T, rng=traj_rng)   # shape (T+1, 3)

    # Input: first coordinate, standardised over the full trajectory.
    x_seq  = traj[:, 0]
    x_mean = float(np.mean(x_seq))
    x_std  = float(np.std(x_seq))
    if x_std < 1e-12:
        x_std = 1.0
    X_seq = (x_seq - x_mean) / x_std   # shape (T+1,)

    # Targets: v_{t+1} for t = 0, ..., T-1.
    # targets[t] = traj[t+1] = (x_{t+1}, y_{t+1}, z_{t+1}).
    targets = traj[1:, :]               # shape (T, 3), raw Lorenz scale

    print(f"\nTrajectory: x0={traj[0]}  x_mean={x_mean:.4f}  x_std={x_std:.4f}")
    print(f"Train times: t = 0 .. {int(0.75*T)-1}  "
          f"({int(0.75*T)} samples)")
    print(f"Test  times: t = {int(0.75*T)} .. {T-1}  "
          f"({T - int(0.75*T)} samples)")

    # ── Symmetric W ─────────────────────────────────────────────────────────
    print("\n[W = SYMMETRIC]")
    scores_sym = run_experiment(W_sym=True,  trainable=trainable,
                                X_seq=X_seq, targets=targets,
                                seed_offset=0)

    # ── IID W ───────────────────────────────────────────────────────────────
    print("\n[W = IID]")
    scores_iid = run_experiment(W_sym=False, trainable=trainable,
                                X_seq=X_seq, targets=targets,
                                seed_offset=1000)

    # ── Summary table ────────────────────────────────────────────────────────
    header = f"{'Score':<14}  {'W=sym':>14}  {'W=iid':>14}"
    sep    = "-" * len(header)
    print("\n" + "=" * 60)
    print("FINAL AVERAGED SCORES")
    print("=" * 60)
    print(header)
    print(sep)
    for key in ["r_mm", "r_avg", "r_med", "r_dom", "loss_train", "loss_test"]:
        print(f"{key:<14}  {scores_sym[key]:>12.4e}  {scores_iid[key]:>12.4e}")
    print(sep)

    print_latex_table(alpha, scores_sym, scores_iid)


if __name__ == "__main__":
    main()