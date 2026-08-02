#!/usr/bin/env python3
"""Stage the four per-era JMS/JMR-unity all_syst pkls into inputs/zjet/rho/jmsjmr_unity/.

Unlike stage_jmsjmr_unity_inputs.py (which took ONE all-era pkl with a dataset
axis and split it), this reads the four separate per-era productions from the
casa ``jmsjmr-unity`` all_syst run:

    minimal_rho_pythia_r2_{2016,2016APV,2017,2018}.pkl

each carrying a single-dataset ``dataset`` axis and rho axes exactly 2x finer
than arc_r2 (pairwise-nested edges). For every file it:

  1. rebins mpt_reco/mpt_gen by a factor 2 (pairwise merge -> arc_r2 edges);
  2. reads the REAL dataset name inside the pkl and writes it to the arc_r2
     input filename that holds that dataset -- this reproduces the arc_r2
     2016<->2016APV label swap automatically (pythia_2016.pkl holds the APV
     dataset), because the mapping is keyed on the dataset, not the filename;
  3. accumulates the dataset-summed pythia_all.pkl;
  4. copies data_all.pkl / herwig_all.pkl over from arc_r2 (data carries no
     JMS/JMR; herwig is overlay/bias-test only).

Run from the repository root once the four pkls are downloaded locally:

    .venv/bin/python scripts/staging/stage_jmsjmr_unity_perera.py --source-dir ~/Downloads
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARC_DIR = REPO_ROOT / "inputs/zjet/rho/arc_r2"
OUT_DIR = REPO_ROOT / "inputs/zjet/rho/jmsjmr_unity"

# dataset held by each pkl -> arc_r2 input filename it must be written to.
# This is the inverse of stage_jmsjmr_unity_inputs.ERA_FILE_TO_DATASET; the
# 2016/2016APV labels are deliberately swapped (matches arc_r2 staging).
DATASET_TO_INPUTFILE = {
    "pythia_UL16NanoAODAPVv9": "pythia_2016.pkl",
    "pythia_UL16NanoAODv9": "pythia_2016APV.pkl",
    "pythia_UL17NanoAODv9": "pythia_2017.pkl",
    "pythia_UL18NanoAODv9": "pythia_2018.pkl",
}
KEEP_KEYS = [
    "response_matrix_rho_u", "response_matrix_rho_g",
    "ptjet_rhojet_u_reco", "ptjet_rhojet_g_reco",
    "ptjet_rhojet_u_gen", "ptjet_rhojet_g_gen",
]
REBIN_AXES = ("mpt_reco", "mpt_gen")
SOURCE_GLOB = "minimal_rho_pythia_r2_*.pkl"


def rebin2(h):
    sel = {ax.name: slice(None, None, 2j) for ax in h.axes if ax.name in REBIN_AXES}
    return h[sel] if sel else h


def dataset_of(h):
    """The single dataset name on a per-era hist's dataset axis."""
    ds_axis = [a for a in h.axes if a.name == "dataset"][0]
    entries = list(ds_axis)
    if len(entries) != 1:
        raise ValueError(f"expected one dataset on the axis, got {entries}")
    return entries[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory holding the four per-era pkls")
    args = parser.parse_args()

    src_dir = args.source_dir.expanduser()
    sources = sorted(src_dir.glob(SOURCE_GLOB))
    # the all-era / TEST merges share the prefix -- keep only the four eras
    sources = [p for p in sources
               if p.stem.split("_")[-1] in ("2016", "2016APV", "2017", "2018")]
    if len(sources) != 4:
        raise SystemExit(
            f"expected 4 per-era pkls in {src_dir}, found {[p.name for p in sources]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # arc_r2 reference for the exact-edge assertion
    with open(ARC_DIR / "pythia_2018.pkl", "rb") as f:
        ref = pickle.load(f)

    merged: dict = {}
    for src in sources:
        with open(src, "rb") as f:
            raw = pickle.load(f)
        rebinned = {k: rebin2(raw[k]) for k in KEEP_KEYS}

        dataset = dataset_of(rebinned[KEEP_KEYS[0]])
        if dataset not in DATASET_TO_INPUTFILE:
            raise SystemExit(f"unknown dataset {dataset!r} in {src.name}")
        fname = DATASET_TO_INPUTFILE[dataset]

        # edge sanity vs arc_r2
        for k in KEEP_KEYS:
            for ax in rebinned[k].axes:
                if ax.name in REBIN_AXES:
                    ref_ax = [a for a in ref[k].axes if a.name == ax.name][0]
                    assert list(ax.edges) == list(ref_ax.edges), (src.name, k, ax.name)

        with open(OUT_DIR / fname, "wb") as f:
            pickle.dump(rebinned, f)
        print(f"wrote {OUT_DIR / fname}  <- {src.name}  ({dataset})")

        for k in KEEP_KEYS:
            contrib = rebinned[k][{"dataset": sum}]
            merged[k] = contrib if k not in merged else merged[k] + contrib

    with open(OUT_DIR / "pythia_all.pkl", "wb") as f:
        pickle.dump(merged, f)
    print(f"wrote {OUT_DIR / 'pythia_all.pkl'}  (dataset-summed over 4 eras)")

    import shutil
    for aux in ("data_all.pkl", "herwig_all.pkl"):
        shutil.copy2(ARC_DIR / aux, OUT_DIR / aux)
        print(f"copied {aux} from arc_r2 (unaffected by JMS/JMR for this comparison)")


if __name__ == "__main__":
    main()
