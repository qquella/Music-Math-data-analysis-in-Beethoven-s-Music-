# DataFrame summaries + persistent homology + Betti curves + barcode plots.

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps
try:
    from ripser import ripser
except Exception:
    ripser = None

try:
    from persim import plot_diagrams  # nice PD plotting if available
except Exception:
    plot_diagrams = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def summarize_dataframe(df: pd.DataFrame, name: str, outdir: Path) -> None:
    """Write a compact human-readable report + CSV dump for a DataFrame."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / f"SUMMARY__{name}.txt"

    buff = io.StringIO()
    buff.write("=" * 80 + "\n")
    buff.write(f"DataFrame: {name}\n")
    buff.write("=" * 80 + "\n")
    buff.write(f"shape: {df.shape}\n\n")
    buff.write("-- dtypes --\n")
    buff.write(df.dtypes.to_string() + "\n\n")
    buff.write("-- head(10) --\n")
    buff.write(df.head(10).to_string(index=False) + "\n")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        buff.write("\n-- describe (numeric) --\n")
        buff.write(df[num_cols].describe().to_string() + "\n")

    obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if obj_cols:
        buff.write("\n-- describe (non-numeric) --\n")
        buff.write(df[obj_cols].describe(include=[object]).to_string() + "\n")

    # A few categorical value counts (small-cardinality columns)
    buff.write("\n-- value counts (selected) --\n")
    candidates = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if nunique <= max(15, int(0.02 * max(1, len(df)))):  # heuristic
            candidates.append(c)
    for c in candidates[:6]:
        buff.write(f"\nColumn: {c}\n")
        vc = df[c].value_counts(dropna=False).head(25)
        buff.write(vc.to_string() + "\n")

    _write_text(report_path, buff.getvalue())
    df.to_csv(outdir / f"{name}.csv", index=False)


def betti_curves_from_diagrams(
    dgms: List[np.ndarray], n_samples: int = 200
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Return (epsilon grid, {k: beta_k(epsilon)}) computed from PH intervals."""
    finite = []
    for D in dgms:
        if D is None or len(D) == 0:
            continue
        F = D[np.isfinite(D[:, 1])]
        if len(F):
            finite.append(F)
    if not finite:
        eps_min, eps_max = 0.0, 1.0
    else:
        all_births = (
            np.concatenate([F[:, 0] for F in finite]) if finite else np.array([0.0])
        )
        all_deaths = (
            np.concatenate([F[:, 1] for F in finite]) if finite else np.array([1.0])
        )
        eps_min = float(np.nanmin(all_births)) if len(all_births) else 0.0
        eps_max = float(np.nanmax(all_deaths)) if len(all_deaths) else 1.0
        if not np.isfinite(eps_min):
            eps_min = 0.0
        if not np.isfinite(eps_max) or eps_max <= eps_min:
            eps_max = max(1.0, eps_min + 1.0)

    eps_grid = np.linspace(eps_min, eps_max, n_samples)

    curves: Dict[int, np.ndarray] = {}
    for k, D in enumerate(dgms):
        if D is None or len(D) == 0:
            curves[k] = np.zeros_like(eps_grid)
            continue
        births = D[:, 0]
        deaths = D[:, 1]
        deaths_filled = deaths.copy()
        deaths_filled[~np.isfinite(deaths_filled)] = (
            eps_max + 1e-9
        )  # keep infinities alive
        beta = np.zeros_like(eps_grid, dtype=int)
        for i, eps in enumerate(eps_grid):
            beta[i] = int(np.count_nonzero((births <= eps) & (eps < deaths_filled)))
        curves[k] = beta
    return eps_grid, curves


def plot_barcodes(dgms: List[np.ndarray], outdir: Path):
    """Simple barcode plots (one PNG per homology degree)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    if plt is None:
        return written
    for k, D in enumerate(dgms):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if D is None or len(D) == 0:
            ax.set_title(f"Barcodes H{k} (no intervals)")
        else:
            F = D[np.isfinite(D[:, 1])]
            if len(F) == 0:
                ax.set_title(f"Barcodes H{k} (no finite intervals)")
            else:
                L = F[:, 1] - F[:, 0]
                order = np.argsort(-L)
                F = F[order]
                for y, (b, d) in enumerate(F):
                    ax.plot([b, d], [y, y], linewidth=2)
                ax.set_title(f"Barcodes H{k}")
                ax.set_xlabel("epsilon")
                ax.set_ylabel("interval index")
        png = outdir / f"barcodes_H{k}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        written.append(png)
    return written


def plot_persistence_diagrams(dgms: List[np.ndarray], outdir: Path):
    """Persistence diagram figure if persim+matplotlib available."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if plot_diagrams is None or plt is None:
        return None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_diagrams(dgms, show=False, ax=ax)
    ax.set_title("Persistence Diagrams")
    png = outdir / "persistence_diagrams.png"
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    return png


def compute_persistence_and_reports(
    X: np.ndarray,
    outdir: Path,
    maxdim: int = 1,
    n_betti_samples: int = 200,
    metric: str = "euclidean",
    *,
    max_points: int = 2000,  # cap point count for PH
    n_perm: Optional[int] = 800,  # sparse Rips landmarks; set None to disable
    thresh: Optional[float] = None,  # prune filtration at this distance
    seed: int = 0,
):
    """Run ripser on a (possibly subsampled) point cloud X and write:
    - persistence_dgms.npz, barcodes, PD figure, betti_curves.csv/png
    Returns dict with 'dgms', 'eps_grid', 'betti_curves', 'used_indices'.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if ripser is None:
        raise ImportError("ripser not found. `pip install ripser persim`")

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_points, n_features), got {X.shape}")

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_points:
        used_idx = rng.choice(n, size=max_points, replace=False)
        X_work = X[used_idx]
    else:
        used_idx = np.arange(n)
        X_work = X

    # Ripser kwargs
    rp = dict(maxdim=maxdim, metric=metric)
    if thresh is not None:
        rp["thresh"] = float(thresh)
    if n_perm is not None:
        rp["n_perm"] = int(min(n_perm, len(X_work)))  # landmark sparse Rips

    res = ripser(X_work, **rp)
    dgms = res["dgms"]

    np.savez(
        outdir / "persistence_dgms.npz", **{f"H{k}": d for k, d in enumerate(dgms)}
    )
    np.save(outdir / "ph_used_indices.npy", used_idx)

    # Figures
    png_pd = plot_persistence_diagrams(dgms, outdir)
    bar_pngs = plot_barcodes(dgms, outdir)

    # Betti curves
    eps_grid, curves = betti_curves_from_diagrams(dgms, n_samples=n_betti_samples)
    betti_df = pd.DataFrame({"epsilon": eps_grid})
    for k, beta in curves.items():
        betti_df[f"beta{k}"] = beta
    betti_df.to_csv(outdir / "betti_curves.csv", index=False)

    # Optional Betti figure
    if plt is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in sorted(curves.keys()):
            ax.plot(eps_grid, curves[k], label=f"beta{k}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Betti number")
        ax.set_title("Betti Curves")
        if len(curves) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "betti_curves.png", dpi=150)
        plt.close(fig)

    return {
        "dgms": dgms,
        "eps_grid": eps_grid,
        "betti_curves": curves,
        "used_indices": used_idx,
        "pd_figure": str(png_pd) if png_pd else None,
        "barcode_figures": [str(p) for p in bar_pngs],
    }


import re
from itertools import permutations

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

_NOTE_TOKEN = re.compile(r"[A-Ga-g][#bx\-]*\d*")

_PITCH2PC_LUT = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def _name_to_pc(name):
    name = name.strip().replace("♯", "#").replace("♭", "b")
    # strip octave digits
    head = re.match(r"^[A-Ga-g][#bx\-]*", name)
    if not head:
        raise ValueError
    base = head.group(0).replace("-", "b").upper()
    base = (
        base.replace("X", "#").replace("##", "#").replace("B", "#")
        if "X" in base
        else base
    )
    base = base.replace("##", "#")  # normalize any accidental duplication
    if base in _PITCH2PC_LUT:
        return _PITCH2PC_LUT[base]
    # fall back: count sharps/flats relative to natural
    letter = base[0]
    accs = base[1:]
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    for a in accs:
        if a == "#":
            pc += 1
        elif a in ("B", "b"):
            pc -= 1
        elif a == "X":
            pc += 2
    return pc % 12


def _circular_pc_distance(a, b):
    d = abs((a - b) % 12)
    return min(d, 12 - d)


def _vl_cost_for_transposition(A, B, k):
    A_k = [(a + k) % 12 for a in A]
    n = max(len(A_k), len(B))
    if len(A_k) < n:
        A_k = A_k + A_k[: (n - len(A_k))]
    if len(B) < n:
        B = B + B[: (n - len(B))]
    A_k = sorted(A_k)
    B = sorted(B)
    C = np.array(
        [[_circular_pc_distance(ai, bj) for bj in B] for ai in A_k], dtype=float
    )
    if linear_sum_assignment is not None and n <= 64:
        r, c = linear_sum_assignment(C)
        return float(C[r, c].sum()) / n
    best = float("inf")
    if n <= 8:
        for perm in permutations(range(n)):
            s = 0.0
            for i in range(n):
                s += C[i, perm[i]]
            if s < best:
                best = s
        return best / n
    # fallback: greedy
    used = set()
    s = 0.0
    for i in range(n):
        j = min((j for j in range(n) if j not in used), key=lambda j: C[i, j])
        used.add(j)
        s += C[i, j]
    return s / n


def vl_distance(A, B):
    A = [int(x) % 12 for x in A]
    B = [int(x) % 12 for x in B]
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    best = float("inf")
    for k in range(12):
        best = min(best, _vl_cost_for_transposition(A, B, k))
    return best


def ensure_pcs_column(meta: pd.DataFrame) -> pd.DataFrame:
    if (
        "pcs" in meta.columns
        and meta["pcs"]
        .apply(lambda x: isinstance(x, (list, tuple)) and len(x) >= 1)
        .all()
    ):
        return meta

    if "pc_hist" in meta.columns:

        def _from_hist(v):
            try:
                arr = np.asarray(v, dtype=float).ravel()
                return [i for i, x in enumerate(arr[:12]) if x > 0.0]
            except Exception:
                return []

        guess = meta["pc_hist"].apply(_from_hist)
        if guess.apply(len).gt(0).any():
            meta = meta.copy()
            meta["pcs"] = guess.apply(lambda xs: sorted(set([int(x) % 12 for x in xs])))
            return meta

    if "label" in meta.columns:

        def _from_label(s):
            toks = re.findall(_NOTE_TOKEN, str(s))
            pcs = []
            for t in toks:
                try:
                    pcs.append(_name_to_pc(t))
                except Exception:
                    pass
            return sorted(set(pcs))

        guess = meta["label"].apply(_from_label)
        if guess.apply(len).gt(0).any():
            meta = meta.copy()
            meta["pcs"] = guess
            return meta

    if "notes" in meta.columns:

        def _from_notes(v):
            xs = []
            for n in (v if isinstance(v, (list, tuple)) else []):
                try:
                    xs.append(int(n) % 12)
                except Exception:
                    pass
            return sorted(set(xs))

        guess = meta["notes"].apply(_from_notes)
        if guess.apply(len).gt(0).any():
            meta = meta.copy()
            meta["pcs"] = guess
            return meta

    raise ValueError(
        "No usable pitch-class column found. Add meta['pcs'] as list of ints per event."
    )


def build_vl_matrix_from_meta(meta: pd.DataFrame) -> np.ndarray:
    meta = ensure_pcs_column(meta)
    pcs_list = meta["pcs"].tolist()
    N = len(pcs_list)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Ai = pcs_list[i]
        for j in range(i + 1, N):
            d = vl_distance(Ai, pcs_list[j])
            D[i, j] = D[j, i] = d
    return D
