"""
Reservoir Computing: Time-Series Classification on ECG5000
============================================================
Classifies whole ECG heartbeat time series (5 classes) using a reservoir
(echo-state) network. Each series x = (x_0, ..., x_T) is fed through the
reservoir once; the FINAL reservoir state h_x in R^N is used both as the
input to the readout classifier F and as the representation compared across
different series for the separation statistics below.

Separation proxy:
--------------------------------------------------------------
For two DIFFERENT time series x, y in the training set:
    r_{x,y} = |h_x - h_y| / |X - Y|
where
    h_x, h_y in R^N   are the final reservoir outputs for x and y
    X = (x_0, ..., x_T), Y = (y_0, ..., y_T)  in R^{T+1}  (the raw/standardised
                                                             series themselves)
    |h| = ||h||_2 / sqrt(N)
    |X| = ||X||_2 / sqrt(T+1)
r_{m-m}, r_avg, r_med, r_dom are computed over all pairs of DISTINCT training
series.

Data
----
Expects the standard UCR-archive ECG5000 files (tab/space-separated, first
column = integer class label in {1,...,5}, remaining columns = the 140
time-steps of the series):
    ECG5000_TRAIN.txt   (500 series)
    ECG5000_TEST.txt    (4500 series)
".arff" versions of the same files are also supported (auto-detected from
the file extension). Set TRAIN_PATH / TEST_PATH below to point at your local
copies; the loader raises a clear error if the files are not found.

User-configurable parameters are set in the CONFIG section below.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# ──────────────────────────────────────────────
#  CONFIG  –  edit these values
# ──────────────────────────────────────────────
N          = 50      # reservoir dimension
p          = 50        # number of simulations of the random connectivity matrix
alpha      = 0.6       # exponent for entry std = 1/N^alpha
TRAINABLE  = "no"     # "yes" or "no". Whether to train the input mask u.
N_EPOCHS   = 300       # full-batch training epochs
TRAIN_PATH = "ECG5000_TRAIN.tsv"
TEST_PATH  = "ECG5000_TEST.tsv"
# T (series length - 1) is set automatically from the data once loaded.
# ── Data loading ───────────────────────────────────────────────────────────────
def _load_text(path):
    """
    Load .txt or .tsv UCR files.
    First column = class label
    Remaining columns = time-series values.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        delimiter = "\t"
    else:
        # np.loadtxt handles arbitrary whitespace (.txt)
        delimiter = None
    raw = np.loadtxt(path,
                     delimiter=delimiter,
                     dtype=np.float64)
    labels = raw[:, 0].astype(int)
    X = raw[:, 1:]
    return X, labels
def _load_arff(path):
    """UCR-style .arff file: last attribute is the class label."""
    from scipy.io import arff
    data, _meta = arff.loadarff(path)
    rows = [list(row) for row in data]
    arr = np.array(rows, dtype=object)
    raw_labels = arr[:, -1]
    def to_int(v):
        if isinstance(v, bytes):
            v = v.decode()
        return int(float(v))
    labels = np.array([to_int(v) for v in raw_labels])
    X = arr[:, :-1].astype(np.float64)
    return X, labels
def load_ecg5000_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Download the ECG5000 dataset from the "
            f"UCR Time Series Archive and set TRAIN_PATH / TEST_PATH to the "
            f"local ECG5000_TRAIN / ECG5000_TEST files (.txt or .arff)."
        )
    ext = os.path.splitext(path)[1].lower()
    if ext == ".arff":
        return _load_arff(path)
    elif ext in [".txt", ".tsv"]:
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
# ── Neural network F ───────────────────────────────────────────────────────────
class ReservoirNet(nn.Module):
    """3-hidden-layer MLP: N -> 128 -> 64 -> 16 -> 10."""
    def __init__(self, N, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
    def forward(self, x):
        return self.net(x)  # raw logits (softmax applied by loss)
# ── Batched reservoir forward pass (final state only) ──────────────────────────
def forward_final_batch(X_batch_t, u_vec, W, T_loc):
    """
    X_batch_t : tensor (batch, T_loc+1)  — standardised series x_0..x_T per row
    u_vec     : tensor (N,)
    W         : tensor (N, N)
    Recurrence:  h_0 = u * x_0,   h_t = u * x_t + W h_{t-1}
    Batched via  h_t = x_t[:,None]*u[None,:] + h_{t-1} @ W^T
    (since (W h)_j = sum_k W_jk h_k  <=>  batched row-vector form h @ W^T).
    Returns only the FINAL state h_T, shape (batch, N).
    """
    h_prev = X_batch_t[:, 0:1] * u_vec.unsqueeze(0)          # (batch, N)
    for t_idx in range(1, T_loc + 1):
        h_prev = X_batch_t[:, t_idx:t_idx + 1] * u_vec.unsqueeze(0) + h_prev @ W.t()
    return h_prev
# ── Pairwise separation ratio statistics (over distinct series) ────────────────
def compute_r_stats(H, X, N_dim):
    """
    H : np.array (n, N)      final reservoir states for n training series
    X : np.array (n, T+1)    the (standardised) series themselves
    N_dim : reservoir dimension N
    For every pair of DISTINCT series i != j:
        r_{i,j} = |h_i - h_j| / |X_i - X_j|
    with |h| = ||h||_2/sqrt(N), |X| = ||X||_2/sqrt(T+1).
    Vectorised via the squared-distance identity
        ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    rather than an O(n^2) Python double loop.
    Returns (r_mm, r_avg, r_med, r_dom).
    """
    n, T_plus_1 = X.shape[0], X.shape[1]
    sqrt_N  = math.sqrt(N_dim)
    sqrt_Tp = math.sqrt(T_plus_1)
    def pairwise_dist(M):
        sq = np.sum(M * M, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (M @ M.T)
        np.fill_diagonal(d2, 0.0)
        d2 = np.clip(d2, 0.0, None)   # guard tiny negative values from fp error
        return np.sqrt(d2)
    D_h = pairwise_dist(H) / sqrt_N
    D_X = pairwise_dist(X) / sqrt_Tp
    iu = np.triu_indices(n, k=1)     # i < j, distinct series only
    denom = D_X[iu]
    numer = D_h[iu]
    valid = denom > 1e-15
    ratios = numer[valid] / denom[valid]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    r_min = float(ratios.min())
    r_max = float(ratios.max())
    r_sum = float(ratios.sum())
    r_mm  = r_min / r_max if r_max > 0 else 0.0
    r_avg = r_sum / ratios.size
    r_med = float(np.median(ratios))
    r_dom = r_max / r_sum if r_sum > 0 else 0.0
    return r_mm, r_avg, r_med, r_dom
# ── One full experiment ─────────────────────────────────────────────────────────
def run_experiment(W_sym: bool, trainable: bool,
                   X_train_t, y_train_t, X_test_t, y_test_t,
                   N_dim, T_loc, num_classes, seed_offset: int = 0):
    """
    Run p simulations for the fixed (already standardised) ECG5000 split and
    return mean scores across simulations.
    """
    all_r_mm    = []
    all_r_avg   = []
    all_r_med   = []
    all_r_dom   = []
    all_acc_tr  = []
    all_acc_te  = []
    for i in range(p):
        rng = np.random.default_rng(42 + i + seed_offset)
        torch.manual_seed(42 + i + seed_offset)
        # ── Generate W ──────────────────────────────────────────────────────
        std = 1.0 / (N_dim ** alpha)
        if W_sym:
            upper = rng.normal(0, std, (N_dim, N_dim))
            W_np  = np.triu(upper) + np.triu(upper, 1).T
        else:
            W_np  = rng.normal(0, std, (N_dim, N_dim))
        W = torch.tensor(W_np, dtype=torch.float32)
        # ── Input mask u ────────────────────────────────────────────────────
        u_init = np.ones(N_dim) / math.sqrt(N_dim)
        if trainable:
            u_param = nn.Parameter(torch.tensor(u_init, dtype=torch.float32))
        else:
            u_param = torch.tensor(u_init, dtype=torch.float32)
        # ── Readout network F ────────────────────────────────────────────────
        F_net   = ReservoirNet(N_dim, num_classes)
        loss_fn = nn.CrossEntropyLoss()
        params = list(F_net.parameters()) + ([u_param] if trainable else [])
        optimizer = optim.Adam(params, lr=1e-3)
        # ── Training loop (full-batch; final state -> class logits) ─────────
        for _ in range(N_EPOCHS):
            optimizer.zero_grad()
            h_train_final = forward_final_batch(X_train_t, u_param, W, T_loc)
            logits = F_net(h_train_final)
            loss = loss_fn(logits, y_train_t)
            loss.backward()
            optimizer.step()
        # ── Evaluation ───────────────────────────────────────────────────────
        F_net.eval()
        with torch.no_grad():
            h_train_final = forward_final_batch(X_train_t, u_param, W, T_loc)
            h_test_final  = forward_final_batch(X_test_t,  u_param, W, T_loc)
            train_pred = F_net(h_train_final).argmax(dim=1)
            test_pred  = F_net(h_test_final).argmax(dim=1)
            acc_tr = float((train_pred == y_train_t).float().mean())
            acc_te = float((test_pred  == y_test_t).float().mean())
            H_train_np = h_train_final.numpy()
        # ── r stats: pairs of DISTINCT training series ──────────────────────
        X_train_np = X_train_t.numpy()
        r_mm, r_avg, r_med, r_dom = compute_r_stats(H_train_np, X_train_np, N_dim)
        all_r_mm.append(r_mm)
        all_r_avg.append(r_avg)
        all_r_med.append(r_med)
        all_r_dom.append(r_dom)
        all_acc_tr.append(acc_tr)
        all_acc_te.append(acc_te)
        print(f"  sim {i+1}/{p}:  acc_train={acc_tr:.3e}  acc_test={acc_te:.3e}"
              f"  r_mm={r_mm:.3e}  r_avg={r_avg:.3e}  r_med={r_med:.3e}  r_dom={r_dom:.3e}")
    return {
        "r_mm":       float(np.mean(all_r_mm)),
        "r_avg":      float(np.mean(all_r_avg)),
        "r_med":      float(np.mean(all_r_med)),
        "r_dom":      float(np.mean(all_r_dom)),
        "acc_train":  float(np.mean(all_acc_tr)),
        "acc_test":   float(np.mean(all_acc_te)),
    }
# ── LaTeX table builder ───────────────────────────────────────────────────────
def sci(v):
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
            f" & ${sci(scores['acc_train'])}$"
            f" & ${sci(scores['acc_test'])}$ \\\\")
def print_latex_table(alpha_val, N_dim, T_loc, n_train, n_test, num_classes,
                      scores_sym, scores_iid):
    print("\n% ── LaTeX Table ─────────────────────────────────────────")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{c l r r r r r r}")
    print(r"\hline")
    print(r"$\alpha$ & $W$ type & $r_{m\text{-}m}$ & $\overline{r}$ "
          r"& $r_{\mathrm{med}}$ & $r_{\mathrm{dom}}$ & $\mathrm{acc}_{\mathrm{train}}$"
          r" & $\mathrm{acc}_{\mathrm{test}}$ \\")
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
    print("Reservoir Computing — ECG5000 Classification")
    print(f"N={N}  p={p}  alpha={alpha}  trainable={TRAINABLE}")
    print("=" * 60)
    # ── Load data ───────────────────────────────────────────────────────────
    X_train_raw, y_train_raw = load_ecg5000_split(TRAIN_PATH)
    X_test_raw,  y_test_raw  = load_ecg5000_split(TEST_PATH)
    T_loc = X_train_raw.shape[1] - 1     # series: x_0, ..., x_T
    assert X_test_raw.shape[1] - 1 == T_loc, "Train/test series length mismatch."
    # ── Standardise using TRAIN statistics only (avoid test leakage) ────────
    x_mean = float(X_train_raw.mean())
    x_std  = float(X_train_raw.std())
    if x_std < 1e-12:
        x_std = 1.0
    X_train = (X_train_raw - x_mean) / x_std
    X_test  = (X_test_raw  - x_mean) / x_std
    # ── Map class labels to 0..num_classes-1 ─────────────────────────────────
    classes = sorted(set(np.unique(y_train_raw)).union(np.unique(y_test_raw)))
    label_map = {c: idx for idx, c in enumerate(classes)}
    num_classes = len(classes)
    y_train = np.array([label_map[c] for c in y_train_raw], dtype=np.int64)
    y_test  = np.array([label_map[c] for c in y_test_raw],  dtype=np.int64)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)
    print(f"\nSeries length: T+1 = {T_loc + 1}")
    print(f"Train series: {X_train_t.shape[0]}   Test series: {X_test_t.shape[0]}")
    print(f"Classes ({num_classes}): {classes}  ->  mapped to 0..{num_classes-1}")
    class_counts = {c: int((y_train_raw == c).sum()) for c in classes}
    print(f"Train class counts: {class_counts}")
    # ── Symmetric W ─────────────────────────────────────────────────────────
    print("\n[W = SYMMETRIC]")
    scores_sym = run_experiment(W_sym=True, trainable=trainable,
                                X_train_t=X_train_t, y_train_t=y_train_t,
                                X_test_t=X_test_t,   y_test_t=y_test_t,
                                N_dim=N, T_loc=T_loc, num_classes=num_classes,
                                seed_offset=0)
    # ── IID W ───────────────────────────────────────────────────────────────
    print("\n[W = IID]")
    scores_iid = run_experiment(W_sym=False, trainable=trainable,
                                X_train_t=X_train_t, y_train_t=y_train_t,
                                X_test_t=X_test_t,   y_test_t=y_test_t,
                                N_dim=N, T_loc=T_loc, num_classes=num_classes,
                                seed_offset=1000)
    # ── Summary table ────────────────────────────────────────────────────────
    header = f"{'Score':<14}  {'W=sym':>14}  {'W=iid':>14}"
    sep    = "-" * len(header)
    print("\n" + "=" * 60)
    print("FINAL AVERAGED SCORES")
    print("=" * 60)
    print(header)
    print(sep)
    for key in ["r_mm", "r_avg", "r_med", "r_dom", "acc_train", "acc_test"]:
        print(f"{key:<14}  {scores_sym[key]:>12.4e}  {scores_iid[key]:>12.4e}")
    print(sep)
    print_latex_table(alpha, N, T_loc, X_train_t.shape[0], X_test_t.shape[0],
                      num_classes, scores_sym, scores_iid)
if __name__ == "__main__":
    main()