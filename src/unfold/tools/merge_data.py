"""Merge per-era data pickles into a single ``data_all.pkl``.

The unfolder consumes ``data_all.pkl`` as a single pre-merged file
(see ``unfolder_core._load_data`` -> ``_resolve_input_path(... "data_all.pkl")``).
Unlike the MC side, there is no automatic era merge for data
(``_merge_eras`` only regenerates ``pythia_all.pkl``), so this script
provides the missing step: sum the per-era data files over the
``dataset`` axis into one accumulator with the same structure the
unfolder already expects.

Era files are expected to be ``coffea`` ``dict_accumulator`` objects with
the reco keys ``ptjet_rhojet_u_reco`` / ``ptjet_rhojet_g_reco`` (axes
``dataset, ptreco, mpt_reco, systematic``) plus the scalar accumulators
``sumw``, ``nev``, ``cutflow`` -- exactly the layout of the existing
``data_all.pkl``. Per-era files have disjoint ``dataset`` categories;
the merged ``dataset`` axis is their union.

Usage (from repo root, with the notebook venv):

    .venv/bin/python -m unfold.tools.merge_data \
        --input-dir inputs/zjet/rho/fixed_jec \
        --era-files minimal_rho_data_2016.pkl minimal_rho_data_2017.pkl minimal_rho_data_2018.pkl
"""

from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

import hist
from coffea.processor.accumulator import dict_accumulator, defaultdict_accumulator

# Keys carried into the merged file -- the structure the unfolder expects.
HIST_KEYS = ("ptjet_rhojet_u_reco", "ptjet_rhojet_g_reco")
SCALAR_KEYS = ("sumw", "nev", "cutflow")


def _merge_hist(per_era):
    """Sum a list of same-structured Hists over a disjoint ``dataset`` axis.

    All non-``dataset`` axes (ptreco, mpt_reco, systematic) are identical
    across era files, so each era's per-dataset view is copied verbatim
    into a fresh Hist whose ``dataset`` axis is the union of categories.
    """
    template = per_era[0]
    non_dataset_axes = [ax for ax in template.axes if ax.name != "dataset"]

    datasets = []
    for h in per_era:
        for ds in h.axes["dataset"]:
            if ds in datasets:
                raise ValueError(f"Duplicate dataset across era files: {ds!r}")
            datasets.append(ds)

    merged = hist.Hist(
        hist.axis.StrCategory(datasets, name="dataset", growth=False),
        *non_dataset_axes,
        storage=template.storage_type(),
    )
    for h in per_era:
        for ds in h.axes["dataset"]:
            merged.view(flow=True)[merged.axes["dataset"].index(ds)] = (
                h[{"dataset": ds}].view(flow=True)
            )
    return merged


def _merge_scalars(per_era_dicts, key):
    out = defaultdict_accumulator(float)
    for d in per_era_dicts:
        if key not in d:
            continue
        for name, val in d[key].items():
            out[name] += val
    return out


def merge_data(input_dir, era_files, out_name="data_all.pkl"):
    input_dir = Path(input_dir)
    paths = [input_dir / f for f in era_files]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    per_era = [pkl.load(open(p, "rb")) for p in paths]

    merged = dict_accumulator({})
    for key in HIST_KEYS:
        missing = [str(p) for p, d in zip(paths, per_era) if key not in d]
        if missing:
            raise KeyError(f"{key!r} missing from: {', '.join(missing)}")
        merged[key] = _merge_hist([d[key] for d in per_era])
    for key in SCALAR_KEYS:
        merged[key] = _merge_scalars(per_era, key)

    out_path = input_dir / out_name
    with open(out_path, "wb") as f:
        pkl.dump(merged, f, protocol=4)

    ref = merged[HIST_KEYS[0]]
    print(f"Wrote {out_path}")
    print(f"  keys     : {list(merged.keys())}")
    print(f"  datasets : {list(ref.axes['dataset'])}")
    print(f"  total ({HIST_KEYS[0]}) : {float(ref.values().sum()):.0f}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Merge per-era data pickles into data_all.pkl")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--era-files", nargs="+", required=True)
    ap.add_argument("--out-name", default="data_all.pkl")
    args = ap.parse_args()
    merge_data(args.input_dir, args.era_files, args.out_name)


if __name__ == "__main__":
    main()
