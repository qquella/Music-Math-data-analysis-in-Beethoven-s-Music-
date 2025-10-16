# README — B9 4th Movement Chord Relations via TDA Mapper

## What this does

• Parses your Beethoven 9 (4th movement) MIDI using music21.
• Extracts vertical events (chords) with onset/duration/measure.
• Encodes each chord as a 36D vector: [pc12, intervalVector(6), root12, quality(5), size].
• Builds a 2D lens with UMAP (or PCA fallback).
• Runs TDA Mapper (KeplerMapper) with a cover + clustering (HDBSCAN, else DBSCAN/KMeans).
• Exports an interactive HTML graph where nodes = chord clusters and edges = overlaps.
• Also writes CSVs with chord metadata and features.

## Install (first time only)

python -m pip install --upgrade pip
pip install music21 kmapper umap-learn hdbscan scikit-learn pandas numpy

## How to run

python b9_mapper.py --midi /path/to/B9_4th.mid --outdir ./out --ncubes 15 --overlap 0.45 --color root_pc

```
python b9_mapper.py --midi ./B9_4th.mid --outdir ./out --ncubes 15 --overlap 0.45 --color root_pc
```

## Parameters

--minpcs Minimum distinct pitch classes to accept an event (default 3).
--ncubes Number of cover cubes (more = finer, but heavier).
--overlap Percentage overlap between cubes (0..1).
--color Node coloring: root_pc (default), time, measure, or size.

## How to interpret the Mapper plot

• Each node is a cluster of chords that landed in the same cover bin and got grouped by the clusterer.
• Edges indicate overlap (some chords appear in both adjacent nodes), revealing the “shape” of harmonic neighborhoods.
• Try color=root_pc to see branches corresponding to root progressions; color=measure to see form-related organization.
• Loops can indicate cycles in harmonic space (e.g., dominant–tonic cycles or modulatory circuits).
• Increase ncubes or decrease overlap to get more detailed partitions; adjust clusterer (HDBSCAN/DBSCAN/KMeans) for granularity.

## Troubleshooting

• If no or few nodes appear, reduce --ncubes or increase --overlap; or switch the clustering fallback.
• If you get no events: increase density of chordify by lowering --minpcs (e.g., 2), but beware many dyads.
• Ensure the MIDI has pitched parts (not only percussion).

## Outputs

out/b9_mapper.html — interactive Mapper graph
out/b9_chords_meta.csv — onset, duration, measure, label, pcs, root_pc, quality, size
out/b9_features.csv — the 36D feature matrix (rows=events)
out/b9_lens.csv — the 2D lens coordinates used for Mapper

Enjoy!
