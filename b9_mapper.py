#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B9 4th Movement — Chord Relations via TDA Mapper
=================================================

Pipeline
--------
1) Load MIDI with music21.
2) Extract vertical events (chords) using chordify; keep onset/duration.
3) Encode each chord as a feature vector:
   - 12D pitch-class binary vector (pc12)
   - 6D interval vector (iv6)
   - 12D root class one-hot (root12)
   - 5D quality one-hot (maj, min, dim, aug, other)
   - 1D chord size (num distinct PCs)
   -> Feature dimension: 12 + 6 + 12 + 5 + 1 = 36
4) Lens for Mapper: UMAP(2D) if available else PCA(2D).
5) Mapper cover (n_cubes, overlap) and clustering (HDBSCAN -> DBSCAN -> KMeans).
6) Visualize graph with tooltips (chord labels) and colors (e.g., root PC or time).

Install (first time only):
--------------------------
pip install music21 kmapper umap-learn hdbscan scikit-learn pandas numpy

Usage:
------
python b9_mapper.py --midi /path/to/B9_4th.mid --outdir ./out --ncubes 15 --overlap 0.45
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# optional, if you added the heatmap util:
from make_figures import heatmap_voiceleading
from tda_extras import (
    build_vl_matrix_from_meta,
    compute_persistence_and_reports,
    ensure_pcs_column,
    summarize_dataframe,
)


# ------------- Robust imports with fallbacks -------------
def safe_imports():
    mods = {}
    # music21 is required
    import music21 as m21

    mods["m21"] = m21

    try:
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        mods["StandardScaler"] = StandardScaler
        mods["PCA"] = PCA
        mods["KMeans"] = KMeans
        mods["DBSCAN"] = DBSCAN
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from e

    # Optional: UMAP
    try:
        import umap

        mods["umap"] = umap
    except Exception:
        mods["umap"] = None

    # Optional: HDBSCAN
    try:
        import hdbscan

        mods["hdbscan"] = hdbscan
    except Exception:
        mods["hdbscan"] = None

    # KeplerMapper
    try:
        import kmapper as km

        mods["km"] = km
    except Exception as e:
        raise RuntimeError(
            "KeplerMapper is required. Install with: pip install kmapper"
        ) from e

    return mods


# ------------- Music21 helpers -------------


def chordify_and_extract(m21_score, min_chord_pcs=3, drop_rests=True):
    """
    Turn a score into a sequence of chord objects via chordify.
    Keep onset (offset), duration (quarterLength), and measure number if available.
    Filter out events with < min_chord_pcs distinct pitch classes.
    """
    ch_stream = m21_score.chordify()

    events = []
    for el in ch_stream.recurse():
        if "Chord" in getattr(el, "classes", []):
            chord_obj = el
            pcs = sorted(set([p.pitchClass for p in chord_obj.pitches]))
            if len(pcs) < min_chord_pcs:
                continue
            onset = float(chord_obj.offset)
            dur = float(chord_obj.quarterLength or 0.0)
            meas = getattr(chord_obj, "measureNumber", None)

            try:
                root = chord_obj.root()
                root_name = root.name if root else "N"
                root_pc = root.pitchClass if root else None
            except Exception:
                root_name, root_pc = "N", None

            try:
                quality = (
                    chord_obj.quality
                )  # 'major','minor','diminished','augmented', etc.
            except Exception:
                quality = "unknown"

            label = f"{root_name}:{quality} pcs={pcs}"
            events.append(
                {
                    "onset": onset,
                    "duration": dur,
                    "measure": meas,
                    "pcs": pcs,
                    "root_pc": root_pc,
                    "quality": quality,
                    "size": len(pcs),
                    "label": label,
                    "m21obj": chord_obj,
                }
            )
    events = sorted(events, key=lambda d: d["onset"])
    return events


def pc12_vector(pcs):
    v = np.zeros(12, dtype=float)
    for pc in pcs:
        v[int(pc) % 12] = 1.0
    return v


def interval_vector_6(chord_obj):
    try:
        iv = chord_obj.intervalVector
        if iv is None:
            return np.zeros(6, dtype=float)
        return np.array([int(x) for x in iv], dtype=float)
    except Exception:
        return np.zeros(6, dtype=float)


def root_one_hot(root_pc):
    v = np.zeros(12, dtype=float)
    if root_pc is not None:
        v[int(root_pc) % 12] = 1.0
    return v


QUALITIES = ["major", "minor", "diminished", "augmented"]


def quality_one_hot(q):
    v = np.zeros(5, dtype=float)  # 4 known + 1 other
    if q in QUALITIES:
        v[QUALITIES.index(q)] = 1.0
    else:
        v[-1] = 1.0
    return v


def build_feature_matrix(events):
    feats = []
    rows = []
    for i, ev in enumerate(events):
        ch = ev["m21obj"]
        pcs = ev["pcs"]
        v_pc = pc12_vector(pcs)
        v_iv = interval_vector_6(ch)
        v_root = root_one_hot(ev["root_pc"])
        v_qual = quality_one_hot(ev["quality"])
        v_size = np.array([float(ev["size"])], dtype=float)
        v = np.concatenate([v_pc, v_iv, v_root, v_qual, v_size], axis=0)
        feats.append(v)
        rows.append(
            {
                "idx": i,
                "onset": ev["onset"],
                "duration": ev["duration"],
                "measure": ev["measure"],
                "root_pc": ev["root_pc"],
                "quality": ev["quality"],
                "size": ev["size"],
                "label": ev["label"],
                "pcs": "-".join(str(p) for p in pcs),
            }
        )
    X = np.vstack(feats) if feats else np.zeros((0, 36))
    meta = pd.DataFrame(rows)
    return X, meta


def run_mapper(X, meta, outdir, n_cubes=15, overlap=0.45, color_mode="root_pc"):
    mods = safe_imports()
    StandardScaler, PCA = mods["StandardScaler"], mods["PCA"]
    KMeans, DBSCAN = mods["KMeans"], mods["DBSCAN"]
    umap, hdbscan, km = mods["umap"], mods["hdbscan"], mods["km"]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=0, metric="euclidean")
        lens = reducer.fit_transform(Xs)
        lens_name = "UMAP-2D"
    else:
        reducer = PCA(n_components=2, random_state=0)
        lens = reducer.fit_transform(Xs)
        lens_name = "PCA-2D"

    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=6)
        cluster_name = "HDBSCAN"
    else:
        clusterer = DBSCAN(eps=0.7, min_samples=6)
        cluster_name = "DBSCAN/KMeans fallback"

    mapper = km.KeplerMapper(verbose=1)
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    graph = mapper.map(lens, Xs, cover=cover, clusterer=clusterer)

    if cluster_name.startswith("DBSCAN") and len(graph["nodes"]) <= 1:
        print("DBSCAN produced degenerate cover; retrying with KMeans(k=8).")
        clusterer = KMeans(n_clusters=8, random_state=0)
        graph = mapper.map(lens, Xs, cover=cover, clusterer=clusterer)
        cluster_name = "KMeans(k=8)"

    # ----- color vector selection -----
    if color_mode == "root_pc":
        color_vec = meta["root_pc"].to_numpy()
        color_name = "Root PC (0=C ... 11=B)"
    elif color_mode == "measure":
        color_vec = meta["measure"].fillna(-1).astype(int).to_numpy()
        color_name = "Measure"
    elif color_mode == "time":
        color_vec = meta["onset_qL"].to_numpy()
        color_name = "Onset (qL)"
    elif color_mode == "size":
        color_vec = meta["size"].to_numpy()
        color_name = "Chord size (#PCs)"
    else:
        color_vec = meta["root_pc"].to_numpy()
        color_name = "Root PC (default)"

        # ----- tooltips: make sure it's a NumPy array -----
    tip_series = meta.assign(measure_f=meta["measure"].fillna(-1).astype(int)).apply(
        lambda r: f"m.{r['measure_f'] if r['measure_f']>=0 else '?'} — {r['label']}",
        axis=1,
    )
    custom_tooltips = tip_series.to_numpy(dtype=object)

    # ----- color vector selection (already in your code) -----
    # color_vec: shape (n_samples,)

    # Make it (n_samples, 1) so there's ONE color function (one column)
    color_values = color_vec.reshape(-1, 1)

    # (Optional but recommended) quick sanity checks
    assert color_values.shape[0] == X.shape[0]
    assert custom_tooltips.shape[0] == X.shape[0]

    html_path = outdir / "b9_mapper.html"

    mapper.visualize(
        graph,
        path_html=str(html_path),
        title="Beethoven 9 (IV) — Chord-space Mapper",
        custom_tooltips=custom_tooltips,
        color_values=color_values,  # <-- 2D array, not a list
        color_function_name=[color_name],  # <-- one name for that one column
        X=X,
        lens=lens,
    )

    meta.to_csv(outdir / "b9_chords_meta.csv", index=False)
    np.savetxt(outdir / "b9_features.csv", X, delimiter=",")
    np.savetxt(outdir / "b9_lens.csv", lens, delimiter=",")

    print(f"Saved: {html_path}")
    print(f"Also wrote: b9_chords_meta.csv, b9_features.csv, b9_lens.csv in {outdir}")


import ast
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import MDS


def _parse_pcs_cell(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) % 12 for v in x]
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [int(w) % 12 for w in v]
        except Exception:
            pass
        # fallback: try split on non-digits/commas like "C:E:G" -> unsupported here
    return None


def pcs_hist(pcs, normalize=True):
    h = np.zeros(12, dtype=float)
    for p in pcs:
        h[int(p) % 12] += 1.0
    if normalize and h.sum() > 0:
        h /= h.sum()
    return h


def _emd_line(h, g):
    # 1D Wasserstein-1 on a line with unit spacing = sum |cumsum(h-g)|
    diff = h - g
    c = np.cumsum(diff)
    return np.sum(np.abs(c))


def emd_circle(h, g):
    # Wasserstein-1 on the circle S^1 (12 points):
    # choose the best rotation/cut; compute 1D EMD on the line
    best = np.inf
    for s in range(12):
        hs = np.roll(h, s)
        w = _emd_line(hs, g)
        if w < best:
            best = w
    return float(best)


def build_vl_matrix_from_meta(
    meta, pcs_col_candidates=("pcs", "pcset", "pitch_classes", "pcset_str")
):
    pcs_lists = None
    for col in pcs_col_candidates:
        if col in meta.columns:
            parsed = meta[col].apply(_parse_pcs_cell)
            if parsed.notnull().all():
                pcs_lists = parsed.tolist()
                break
    if pcs_lists is None:
        raise ValueError(
            "No usable pitch-class column found. Add a column 'pcs' containing a list/tuple of pitch classes per event."
        )

    H = np.stack([pcs_hist(pcs, normalize=True) for pcs in pcs_lists], axis=0)
    n = H.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = emd_circle(H[i], H[j])
            D[i, j] = D[j, i] = d
    return D


def vl_successive_series(D, order_index=None):
    if order_index is None:
        idx = np.arange(D.shape[0])
    else:
        idx = np.asarray(order_index, dtype=int)
    s = []
    for k in range(len(idx) - 1):
        s.append(D[idx[k], idx[k + 1]])
    return np.array(s)


def embed_vl_mds(D, random_state=0):
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=200,
        eps=1e-3,
    )
    Y = mds.fit_transform(D)
    return Y


def build_vl_graph(D, threshold):
    n = D.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= threshold:
                G.add_edge(i, j, weight=float(D[i, j]))
    return G


def plot_vl_embedding(Y, color=None, title="Voice-leading MDS (circular Wasserstein)"):
    plt.figure(figsize=(6.0, 5.2))
    if color is None:
        plt.scatter(Y[:, 0], Y[:, 1], s=8)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], s=8, c=color)
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.title(title)
    plt.tight_layout()


def plot_vl_graph(G, Y=None, title="Parsimonious progression network"):
    plt.figure(figsize=(6.4, 5.6))
    pos = {i: Y[i] for i in G.nodes()} if Y is not None else nx.spring_layout(G, seed=0)
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()


def plot_vl_series(series, qlen=None, title="Instantaneous voice-leading cost"):
    plt.figure(figsize=(7.0, 2.8))
    plt.plot(series, linewidth=1.0)
    if qlen is not None:
        plt.xlabel("Event index (≈ ordered by onset; qL grid)")
    else:
        plt.xlabel("Event index")
    plt.ylabel("VL cost")
    plt.title(title)
    plt.tight_layout()


def run_voice_leading(meta, outdir, vl_thresh=0.35, color_key="root_pc"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    D = build_vl_matrix_from_meta(meta)
    np.savetxt(outdir / "b9_voiceleading_dist.csv", D, delimiter=",")

    if "onset_qL" in meta.columns:
        order_idx = np.argsort(meta["onset_qL"].values)
    else:
        order_idx = np.arange(len(meta))

    series = vl_successive_series(D, order_index=order_idx)
    np.savetxt(outdir / "b9_voiceleading_series.csv", series, delimiter=",")

    Y = embed_vl_mds(D, random_state=0)
    np.savetxt(outdir / "b9_voiceleading_mds2.csv", Y, delimiter=",")

    if color_key in meta.columns:
        color_vec = meta[color_key].to_numpy()
    else:
        color_vec = None

    plot_vl_embedding(Y, color=color_vec, title="VL map (MDS on circular EMD)")
    plt.savefig(outdir / "b9_vl_map.png", dpi=160)
    plt.close()

    G = build_vl_graph(D, threshold=vl_thresh)
    plot_vl_graph(G, Y=Y, title=f"Parsimonious VL network (≤ {vl_thresh:.2f})")
    plt.savefig(outdir / "b9_vl_graph.png", dpi=160)
    plt.close()

    plot_vl_series(series, qlen=True, title="Instantaneous VL cost across events")
    plt.savefig(outdir / "b9_vl_series.png", dpi=160)
    plt.close()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, help="Path to Beethoven 9 (4th mvt) MIDI")
    ap.add_argument("--outdir", default="./out", help="Output directory for HTML/CSVs")
    ap.add_argument(
        "--minpcs",
        type=int,
        default=3,
        help="Minimum distinct pitch classes to accept an event",
    )
    ap.add_argument("--ncubes", type=int, default=15, help="Mapper cover n_cubes")
    ap.add_argument(
        "--overlap", type=float, default=0.45, help="Mapper cover perc_overlap (0..1)"
    )
    ap.add_argument(
        "--color",
        default="root_pc",
        choices=["root_pc", "time", "measure", "size"],
        help="Node coloring scheme",
    )
    args = ap.parse_args()

    mods = safe_imports()
    m21 = mods["m21"]

    midi_path = Path(args.midi)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")

    score = m21.converter.parse(str(midi_path))
    events = chordify_and_extract(score, min_chord_pcs=args.minpcs)
    if not events:
        raise RuntimeError(
            "No chord events extracted. Try lowering --minpcs or check the MIDI."
        )

    X, meta = build_feature_matrix(events)
    meta = ensure_pcs_column(meta)
    meta["size"] = meta["pcs"].apply(lambda xs: len(xs))

    from tda_extras import build_vl_matrix_from_meta

    # right before run_voice_leading(...) or wherever you want the VL analysis:
    D = build_vl_matrix_from_meta(meta)
    np.savetxt(Path(outdir) / "b9_voiceleading_D.csv", D, delimiter=",")

    run_voice_leading(meta, outdir=args.outdir, vl_thresh=0.35, color_key="root_pc")

    outdir = Path(args.outdir)

    # 0) “Visualization” / insight dumps for inspection
    summarize_dataframe(meta, "chords_meta", outdir)

    # Compute PH quickly (scalable defaults)
    ph = compute_persistence_and_reports(
        X,
        outdir=outdir,
        maxdim=1,  # start with H0/H1
        n_betti_samples=300,
        metric="euclidean",
        max_points=2000,  # cap points
        n_perm=800,  # sparse Rips
        thresh=None,  # set a value to prune if you like
        seed=0,
    )
    print("PH done; results in", outdir)


if __name__ == "__main__":
    main()

# TODO:
# 1. trying different clusterings
# 2. find meaning for the data
# 3. muscial interpretation?
# 4. explore different overlap/args & see what change
# 5. add different composition/movements
