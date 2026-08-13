"""
Reservoir Computing: Implementation and Training
=================================================
Predicts digits of pi with a delay d using a reservoir (echo state) network.

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
N = 50  # reservoir dimension
T = 300  # max time-series length
p = 50  # number of simulations of the random connectivity matrix
alpha = 0.7  # exponent for std = 1/N^alpha
d = 20  # delay
TRAINABLE = "no"  # "yes" or "no". Whether to train the input mask u.

# ──────────────────────────────────────────────

# ── Pi digits ──────────────────────────────────────────────────────────────────
PI_DIGITS_STR = (
    "31415926535897932384626433832795028841971693993751"
    "05820974944592307816406286208998628034825342117067"
    "98214808651328230664709384460955058223172535940812"
    "84811174502841027019385211055596446229489549303819"
    "64428810975665933446128475648233786783165271201909"
    "14564856692346034861045432664821339360726024914127"
    "37245870066063155881748815209209628292540917153643"
    "67892590360011330530548820466521384146951941511609"
)

def get_pi_digits(n):
    """Return first n+1 decimal digits of pi as a list of ints."""
    digits = [int(c) for c in PI_DIGITS_STR if c.isdigit()]
    assert len(digits) >= n + 1, "Need more pre-computed pi digits."
    return digits[:n + 1]

PI = get_pi_digits(T)  # x_0, x_1, ..., x_T

# ── Neural network F ───────────────────────────────────────────────────────────
class ReservoirNet(nn.Module):
    """3-hidden-layer MLP: N -> 128 -> 64 -> 16 -> 10."""
    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)

# ── Helper: build X_t snapshot vectors (right-aligned, zero-padded) ───────────
def X_snapshot(x_seq, t, T_loc):
    """
    Builds the right-aligned, zero-padded snapshot vector
        X_t = (0, ..., 0, X_0, X_1, ..., X_t)  in R^{T_loc+1}
    i.e. the first (T_loc - t) entries are zero and the last (t+1) entries
    are the raw scalar sequence values x_seq[0..t].
    """
    X_t = np.zeros(T_loc + 1)
    X_t[T_loc - t:] = x_seq[:t + 1]
    return X_t

def precompute_X_snapshots(x_seq, T_loc):
    """
    Precompute all X_t for t = 0, ..., T_loc.
    """
    return {t: X_snapshot(x_seq, t, T_loc) for t in range(T_loc + 1)}

# ── Helper: compute r_{s,t} ratios ────────────────────────────────────────────
def compute_r_stats(z_states, X_snaps, T_loc):
    """
    z_states : dict  t -> np.array shape (N,)   (reservoir output z_t)
    X_snaps  : dict  t -> np.array shape (T_loc+1,)  (precomputed X_t)
    Returns min/max ratio, avg, med, dom, based on
        r_{s,t} = |z_t - z_s| / |X_t - X_s|
    with |z| = ||z||_2 / sqrt(N)  and  |X| = ||X||_2 / sqrt(T_loc + 1).
    """
    times = sorted(z_states.keys())
    N_dim = len(next(iter(z_states.values())))
    sqrt_N = math.sqrt(N_dim)
    sqrt_Tp1 = math.sqrt(T_loc + 1)

    ratios = []
    for i, t in enumerate(times):
        X_t = X_snaps[t]
        z_t = z_states[t]
        for s in times[:i]:  # s < t
            diff_X = X_t - X_snaps[s]
            norm_X = float(np.linalg.norm(diff_X)) / sqrt_Tp1
            if norm_X < 1e-15:
                continue
            diff_z = z_t - z_states[s]
            norm_z = float(np.linalg.norm(diff_z)) / sqrt_N
            if not (math.isfinite(norm_z) and math.isfinite(norm_X)):
                continue
            ratios.append(norm_z / norm_X)

    if len(ratios) == 0:
        return 0.0, 0.0, 0.0, 0.0

    r_min = min(ratios)
    r_max = max(ratios)
    r_sum = sum(ratios)
    r_mm = r_min / r_max if r_max > 0 else 0.0
    avg = r_sum / len(ratios)
    med = float(np.median(ratios))
    r_dom = r_max / r_sum if r_sum > 0 else 0.0

    return r_mm, avg, med, r_dom

# ── One full experiment ────────────────────────────────────────────────────────
def run_experiment(W_sym: bool, trainable: bool, X_snaps, seed_offset: int = 0):
    all_r_mm = []
    all_avg = []
    all_med = []
    all_r_dom = []
    all_acc_tr = []
    all_acc_te = []

    for i in range(p):
        rng = np.random.default_rng(42 + i + seed_offset)
        torch.manual_seed(42 + i + seed_offset)

        # ── Generate W ──────────────────────────────────────────────────────
        std = 1.0 / (N ** alpha)
        if W_sym:
            upper = rng.normal(0, std, (N, N))
            W_np = np.triu(upper) + np.triu(upper, 1).T
        else:
            W_np = rng.normal(0, std, (N, N))
        W = torch.tensor(W_np, dtype=torch.float32)

        # ── Mask u ─────────────────────────────────────────────────────────
        u_init = np.ones(N) / math.sqrt(N)
        if trainable:
            u_param = nn.Parameter(torch.tensor(u_init, dtype=torch.float32))
        else:
            u_param = torch.tensor(u_init, dtype=torch.float32)

        # ── Neural network F ────────────────────────────────────────────────
        F_net = ReservoirNet(N)

        # ── Optimizer ──────────────────────────────────────────────────────
        if trainable:
            optimizer = optim.Adam(list(F_net.parameters()) + [u_param], lr=1e-3)
        else:
            optimizer = optim.Adam(F_net.parameters(), lr=1e-3)

        loss_fn = nn.CrossEntropyLoss()

        # ── Training / test split ───────────────────────────────────────────
        valid_times = list(range(d, T + 1))
        n_train = int(0.75 * len(valid_times))
        train_times = set(valid_times[:n_train])
        test_times = set(valid_times[n_train:])

        # ── Forward pass helper ─────────────────────────────────────────────
        def forward_all(u_vec_t):
            states = []
            y_prev = u_vec_t * PI[0]
            states.append((0, y_prev))
            for t_idx in range(1, T + 1):
                y_t = u_vec_t * PI[t_idx] + W @ y_prev
                states.append((t_idx, y_t))
                y_prev = y_t
            return states

        # ── Training loop ───────────────────────────────────────────────────
        N_EPOCHS = 300
        for epoch in range(N_EPOCHS):
            optimizer.zero_grad()
            u_vec_t = u_param
            states = forward_all(u_vec_t)

            logits_list, labels_list = [], []
            for t_idx, y_t in states:
                if t_idx in train_times:
                    logits_list.append(F_net(y_t.unsqueeze(0)))
                    labels_list.append(PI[t_idx - d])

            if len(logits_list) == 0:
                continue

            logits = torch.cat(logits_list, dim=0)
            labels = torch.tensor(labels_list, dtype=torch.long)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        # ── Evaluation ──────────────────────────────────────────────────────
        F_net.eval()
        with torch.no_grad():
            u_vec_t = u_param
            states = forward_all(u_vec_t)

            z_states_np = {}
            for t_idx, y_t in states:
                z_states_np[t_idx] = y_t.numpy().copy()

            tr_correct, tr_total = 0, 0
            te_correct, te_total = 0, 0

            for t_idx, y_t in states:
                if t_idx in train_times or t_idx in test_times:
                    logit = F_net(y_t.unsqueeze(0))
                    pred = logit.argmax(dim=1).item()
                    true = PI[t_idx - d]
                    if t_idx in train_times:
                        tr_total += 1
                        tr_correct += (pred == true)
                    else:
                        te_total += 1
                        te_correct += (pred == true)

        acc_tr = tr_correct / tr_total if tr_total > 0 else 0.0
        acc_te = te_correct / te_total if te_total > 0 else 0.0

        # ── r stats  ───────────────────
        r_mm, avg_r, med_r, r_dom = compute_r_stats(z_states_np, X_snaps, T)

        all_r_mm.append(r_mm)
        all_avg.append(avg_r)
        all_med.append(med_r)
        all_r_dom.append(r_dom)
        all_acc_tr.append(acc_tr)
        all_acc_te.append(acc_te)

        print(f"  sim {i + 1}/{p}:  acc_train={acc_tr:.3e}  acc_test={acc_te:.3e}"
              f"  r_mm={r_mm:.3e}  avg={avg_r:.3e}  med={med_r:.3e}  r_dom={r_dom:.3e}")

    return {
        "r_mm": float(np.mean(all_r_mm)),
        "avg": float(np.mean(all_avg)),
        "med": float(np.mean(all_med)),
        "r_dom": float(np.mean(all_r_dom)),
        "acc_train": float(np.mean(all_acc_tr)),
        "acc_test": float(np.mean(all_acc_te)),
    }

# ── LaTeX table builder ────────────────────────────────────────────────────────
def make_latex_row(alpha_val, w_type, scores):
    def sci(v):
        if not math.isfinite(v):
            return r"\infty" if v > 0 else r"-\infty"
        if v == 0:
            return "0"
        exp = int(math.floor(math.log10(abs(v))))
        mantissa = v / (10 ** exp)
        return rf"{mantissa:.3f} \times 10^{{{exp}}}"

    return (f"  & {w_type}"
            f" & ${sci(scores['r_mm'])}$"
            f" & ${sci(scores['avg'])}$"
            f" & ${sci(scores['med'])}$"
            f" & ${sci(scores['r_dom'])}$"
            f" & ${sci(scores['acc_train'])}$"
            f" & ${sci(scores['acc_test'])}$ \\\\")

def print_latex_table(alpha_val, scores_sym, scores_iid):
    print("\n% ── LaTeX Table ─────────────────────────────────────────")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{c l r r r r r r}")
    print(r"\hline")
    print(r"$\alpha$ & $W$ type & $r_{m\text{-}m}$ & $\overline{r}$ "
          r"& $r_{\mathrm{med}}$ & $r_{\mathrm{dom}}$ & $\mathrm{acc}_{\mathrm{train}}$"
          r" & $\mathrm{acc}_{\mathrm{test}}$ \\")
    print(r"\hline")
    row_sym = make_latex_row(alpha_val, "sym", scores_sym)
    row_iid = make_latex_row(alpha_val, "iid", scores_iid)
    print(f"  \\multirow{{2}}{{*}}{{${alpha_val}$}}{row_sym}")
    print(f"  {row_iid}")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% ─────────────────────────────────────────────────────────\n")

def main():
    trainable = TRAINABLE.lower() == "yes"

    print("=" * 60)
    print(f"Reservoir Computing  |  N={N}  T={T}  p={p}"
          f"  alpha={alpha}  d={d}  trainable={TRAINABLE}")
    print("=" * 60)

    X_snaps = precompute_X_snapshots(PI[:T + 1], T)

    print("\n[W = SYMMETRIC]")
    scores_sym = run_experiment(W_sym=True, trainable=trainable, X_snaps=X_snaps, seed_offset=0)

    print("\n[W = IID]")
    scores_iid = run_experiment(W_sym=False, trainable=trainable, X_snaps=X_snaps, seed_offset=1000)

    header = (f"{'Score':<14}  {'W=sym':>14}  {'W=iid':>14}")
    sep = "-" * len(header)

    print("\n" + "=" * 60)
    print("FINAL AVERAGED SCORES")
    print("=" * 60)
    print(header)
    print(sep)
    for key in ["r_mm", "avg", "med", "r_dom", "acc_train", "acc_test"]:
        print(f"{key:<14}  {scores_sym[key]:>12.4e}  {scores_iid[key]:>12.4e}")
    print(sep)

    print_latex_table(alpha, scores_sym, scores_iid)

if __name__ == "__main__":
    main()