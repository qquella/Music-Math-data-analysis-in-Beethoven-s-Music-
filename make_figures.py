import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Silence some noisy warnings to keep output readable
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------- Utilities: safe imports ----------------------
def safe_imports():
    mods = {}
    try:
        import umap  # umap-learn
    except Exception:
        umap = None
    mods["umap"] = umap

    try:
        import hdbscan
    except Exception:
        hdbscan = None
    mods["hdbscan"] = hdbscan

    try:
        import kmapper as km
    except Exception as e:
        raise RuntimeError(
            "KeplerMapper (kmapper) is required. pip install kmapper"
        ) from e
    mods["km"] = km

    # sklearn
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    mods["StandardScaler"] = StandardScaler
    mods["PCA"] = PCA
    mods["DBSCAN"] = DBSCAN
    mods["KMeans"] = KMeans

    try:
        from ripser import ripser
    except Exception as e:
        ripser = None
    mods["ripser"] = ripser

    import matplotlib.pyplot as plt

    mods["plt"] = plt

    return mods


# ---------------------- Music parsing / features ----------------------
def pc_vector_from_chord(ch):
    """Return 12-dim presence vector for pitch classes in chord ch."""
    pcs = [p.pitchClass for p in ch.pitches]
    vec = np.zeros(12, dtype=float)
    for pc in pcs:
        vec[pc % 12] = 1.0
    return vec, pcs


def duration_weighted_hist(ch):
    """12-dim duration-weighted histogram. If ch has duration, use it uniformly for present pcs."""
    vec = np.zeros(12, dtype=float)
    dur = float(ch.duration.quarterLength) if ch.duration is not None else 1.0
    pcs = [p.pitchClass for p in ch.pitches]
    if len(pcs) == 0:
        return vec
    w = dur / max(1, len(pcs))
    for pc in pcs:
        vec[pc % 12] += w
    return vec


def directed_interval_counts_from_bass(ch):
    """12-dim directed interval counts above bass (mod 12)."""
    vec = np.zeros(12, dtype=float)
    if len(ch.pitches) == 0:
        return vec
    # Choose bass pitch (lowest in chord)
    bass = min(ch.pitches, key=lambda p: p.midi).pitchClass
    for p in ch.pitches:
        ic = (p.pitchClass - bass) % 12
        vec[ic] += 1.0
    return vec


def estimate_root_pc(ch):
    """Best-effort root estimation:
    1) use ch.root() if available; else
    2) use bass pc; else
    3) modal pc of the set.
    """
    try:
        r = ch.root()
        if r is not None:
            return int(r.pitchClass)
    except Exception:
        pass
    if len(ch.pitches) > 0:
        bass_pc = min(ch.pitches, key=lambda p: p.midi).pitchClass
        return int(bass_pc % 12)
    # Modal pc
    pcs = [p.pitchClass for p in ch.pitches]
    if pcs:
        counts = np.bincount(np.array(pcs) % 12, minlength=12)
        return int(np.argmax(counts))
    return -1


def chord_label(ch):
    """Attempt a human-readable label; fall back to pitch-class set."""
    try:
        rn = ch.root()
        qual = ch.commonName if hasattr(ch, "commonName") else ""
        if rn is not None and qual:
            return f"{rn.name} ({qual})"
    except Exception:
        pass
    pcs = sorted({p.pitchClass for p in ch.pitches})
    return "{" + ",".join(map(str, pcs)) + "}"


def extract_events(midi_path):
    """Return (features matrix X: n x 36, meta DataFrame)."""
    from music21 import converter

    s = converter.parse(midi_path)
    # Chordify to get verticalities
    s_chord = s.chordify()

    rows = []
    feats = []
    for el in s_chord.recurse().getElementsByClass("Chord"):
        if not hasattr(el, "pitches") or len(el.pitches) == 0:
            continue

        onset = float(el.offset)
        meas = el.measureNumber if hasattr(el, "measureNumber") else None
        lab = chord_label(el)
        root_pc = estimate_root_pc(el)

        b, pcs = pc_vector_from_chord(el)
        h = duration_weighted_hist(el)
        v = directed_interval_counts_from_bass(el)

        x = np.concatenate([h, b, v])  # 36 dims
        feats.append(x)

        rows.append(
            {
                "onset_qL": onset,
                "measure": int(meas) if meas is not None else np.nan,
                "label": lab,
                "root_pc": root_pc,
                "size": len(set(pcs)),
            }
        )

    X = np.vstack(feats) if len(feats) else np.zeros((0, 36))
    import pandas as pd  # local import to ensure availability

    meta = pd.DataFrame(rows)
    return X, meta


# ---------------------- Mapper pipeline ----------------------
def run_mapper(X, meta, outdir, n_cubes=15, overlap=0.45, use_umap=True):
    mods = safe_imports()
    StandardScaler, PCA = mods["StandardScaler"], mods["PCA"]
    KMeans, DBSCAN = mods["KMeans"], mods["DBSCAN"]
    umap, hdbscan, km = mods["umap"], mods["hdbscan"], mods["km"]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if use_umap and umap is not None:
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

    # Fallback if DBSCAN degenerates
    if cluster_name.startswith("DBSCAN") and len(graph.get("nodes", {})) <= 1:
        print("DBSCAN produced degenerate cover; retrying with KMeans(k=8).")
        clusterer = KMeans(n_clusters=8, random_state=0)
        graph = mapper.map(lens, Xs, cover=cover, clusterer=clusterer)

    # Tooltips
    tip_series = meta.assign(measure_f=meta["measure"].fillna(-1).astype(int)).apply(
        lambda r: f"m.{r['measure_f'] if r['measure_f']>=0 else '?'} — {r['label']}",
        axis=1,
    )
    custom_tooltips = tip_series.to_numpy(dtype=object)

    # Three colorings
    color_specs = [
        (
            "root_pc",
            meta["root_pc"].to_numpy(),
            "Root PC (0=C ... 11=B)",
            "mapper_rootpc.html",
        ),
        (
            "measure",
            meta["measure"].fillna(-1).astype(int).to_numpy(),
            "Measure",
            "mapper_measure.html",
        ),
        ("size", meta["size"].to_numpy(), "Chord size (#PCs)", "mapper_size.html"),
    ]

    for key, vec, cname, filename in color_specs:
        color_mat = np.asarray(vec).reshape(-1, 1)  # (n_samples,1)
        out_html = str(outdir / filename)
        mapper.visualize(
            graph,
            path_html=out_html,
            title=f"Beethoven 9 (IV) — Mapper ({lens_name}, color={key})",
            custom_tooltips=custom_tooltips,
            color_values=color_mat,  # shape (n,1)
            color_function_name=[cname],  # list length = n_colors (1)
            X=X,  # original features
            lens=lens,  # 2D embedding
        )
        print(f"[OK] Wrote {out_html}")

    # Save matrices for later inspection
    meta.to_csv(outdir / "b9_chords_meta.csv", index=False)
    np.savetxt(outdir / "b9_features.csv", X, delimiter=",")
    np.savetxt(outdir / "b9_lens.csv", lens, delimiter=",")

    return lens


# ---------------------- Tonnetz-like persistence diagram ----------------------
def tonnetz_point_cloud(meta):
    """Map each event to a 2D point by averaging unit vectors of present pcs.
    Here we use only the estimated root_pc for simplicity; if you prefer,
    average over the full set of pcs per event (requires passing them through)."""
    pcs = meta["root_pc"].to_numpy()
    # Replace -1 (unknown) with NaN and drop
    pcs = pcs.astype(float)
    mask = ~np.isnan(pcs) & (pcs >= 0)
    pcs = pcs[mask].astype(int)
    theta = 2.0 * np.pi * (pcs % 12) / 12.0
    X = np.column_stack([np.cos(theta), np.sin(theta)])  # n x 2
    return X


def plot_persistence_diagram(X, out_png, maxdim=1):
    mods = safe_imports()
    ripser = mods["ripser"]
    plt = mods["plt"]
    if ripser is None:
        print("ripser not installed; skipping persistence diagram.")
        return
    # Compute pairwise PH
    res = ripser(X, maxdim=maxdim)
    dgms = res["dgms"]  # list of arrays per homology dimension
    # Plot birth-death
    plt.figure(figsize=(5.2, 5.2), dpi=160)
    for dim, dgm in enumerate(dgms):
        if dgm.size == 0:
            continue
        bd = dgm[np.isfinite(dgm).all(axis=1)]
        if bd.size == 0:
            continue
        plt.scatter(bd[:, 0], bd[:, 1], s=12, label=f"H{dim}")
    mx = 0.0
    for dgm in dgms:
        if dgm.size:
            mx = max(mx, np.nanmax(dgm[np.isfinite(dgm)]))
    plt.plot([0, mx], [0, mx], lw=1, alpha=0.5)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence diagram (Tonnetz-like lens)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_png}")


import matplotlib.pyplot as plt
import numpy as np


def heatmap_voiceleading(D, out_png):
    plt.figure(figsize=(8, 6))
    plt.imshow(D, interpolation="nearest")
    plt.colorbar(label="VL distance (avg circular semitones)")
    plt.xlabel("Event index")
    plt.ylabel("Event index")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate Mapper-based figures and a Tonnetz persistence diagram for Beethoven 9 (IV)."
    )
    ap.add_argument(
        "--midi", required=True, help="Path to the B9 4th-movement MIDI file"
    )
    ap.add_argument(
        "--outdir", default="./figures", help="Output directory for figures"
    )
    ap.add_argument(
        "--ncubes", type=int, default=15, help="Number of cubes per axis for the cover"
    )
    ap.add_argument(
        "--overlap", type=float, default=0.45, help="Cover overlap in [0,1)"
    )
    ap.add_argument(
        "--no-umap", action="store_true", help="Disable UMAP; use PCA(2) lens"
    )
    args = ap.parse_args()

    X, meta = extract_events(args.midi)
    if X.shape[0] == 0:
        raise RuntimeError(
            "No chord events extracted from the MIDI. Check the file or parsing assumptions."
        )

    lens = run_mapper(
        X,
        meta,
        outdir=args.outdir,
        n_cubes=args.ncubes,
        overlap=args.overlap,
        use_umap=(not args.no_umap),
    )

    # Tonnetz-like persistence diagram
    tonnetz_X = tonnetz_point_cloud(meta)
    out_png = str(Path(args.outdir) / "tonnetz_persistence.png")
    plot_persistence_diagram(tonnetz_X, out_png)


if __name__ == "__main__":
    main()
