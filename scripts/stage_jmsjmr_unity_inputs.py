#!/usr/bin/env python3
"""Stage the JMS/JMR-unity reskim pkl into inputs/zjet/rho/jmsjmr_unity/.

The casa reskim (smp_jetmass_run2 branch ``jmsjmr-unity``) produced ONE pkl
with all four eras on a dataset axis and rho axes exactly 2x finer than the
arc_r2 production (pairwise-nested edges). This script:

  1. rebins every mpt_reco/mpt_gen axis by a factor 2 (pairwise merge), which
     reproduces the arc_r2 edges exactly;
  2. splits the dataset axis into the four per-era pkls the Unfolder expects
     (era_mc_files), using the same filename convention as arc_r2 -- NOTE the
     arc_r2 staging swapped the 2016/2016APV labels (pythia_2016.pkl holds the
     APV dataset); the swap is label-only (the loader keeps the true dataset
     axis) and is reproduced here so the two input sets stay comparable;
  3. writes the dataset-summed pythia_all.pkl (no dataset axis, matching the
     arc_r2 merged file);
  4. copies data_all.pkl and herwig_all.pkl over from arc_r2 (data carries no
     JMS/JMR; herwig is overlay/bias-test only for this comparison).

Run from the repository root:

    .venv/bin/python scripts/stage_jmsjmr_unity_inputs.py \
        --source ~/Downloads/minimal_rho_pythia_r2_all.pkl
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARC_DIR = REPO_ROOT / "inputs/zjet/rho/arc_r2"
OUT_DIR = REPO_ROOT / "inputs/zjet/rho/jmsjmr_unity"

# arc_r2 filename -> dataset key it holds (labels 2016/2016APV deliberately
# swapped, mirroring the arc_r2 staging).
ERA_FILE_TO_DATASET = {
    "pythia_2016.pkl": "pythia_UL16NanoAODAPVv9",
    "pythia_2016APV.pkl": "pythia_UL16NanoAODv9",
    "pythia_2017.pkl": "pythia_UL17NanoAODv9",
    "pythia_2018.pkl": "pythia_UL18NanoAODv9",
}
# Only the hists the Unfolder reads (matches the arc_r2 per-era/merged pkls).
KEEP_KEYS = [
    "response_matrix_rho_u", "response_matrix_rho_g",
    "ptjet_rhojet_u_reco", "ptjet_rhojet_g_reco",
    "ptjet_rhojet_u_gen", "ptjet_rhojet_g_gen",
]
REBIN_AXES = ("mpt_reco", "mpt_gen")


def rebin2(h):
    """Pairwise-merge every rho axis (2x finer than arc_r2 -> exact match)."""
    sel = {}
    for ax in h.axes:
        if ax.name in REBIN_AXES:
            sel[ax.name] = slice(None, None, 2j)
    return h[sel] if sel else h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="all-era minimal_rho pkl from the jmsjmr-unity reskim")
    args = parser.parse_args()

    with open(args.source.expanduser(), "rb") as f:
        src = pickle.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rebinned = {k: rebin2(src[k]) for k in KEEP_KEYS}

    # sanity: rebinned axes must match the arc_r2 axes exactly
    with open(ARC_DIR / "pythia_2018.pkl", "rb") as f:
        ref = pickle.load(f)
    for k in KEEP_KEYS:
        for ax in rebinned[k].axes:
            if ax.name in REBIN_AXES:
                ref_ax = [a for a in ref[k].axes if a.name == ax.name][0]
                assert list(ax.edges) == list(ref_ax.edges), (k, ax.name)
    print("rebinned axes match arc_r2 edges exactly")

    for fname, dataset in ERA_FILE_TO_DATASET.items():
        out = {k: rebinned[k][{"dataset": [dataset]}] for k in KEEP_KEYS}
        with open(OUT_DIR / fname, "wb") as f:
            pickle.dump(out, f)
        print(f"wrote {OUT_DIR / fname}  ({dataset})")

    merged = {k: rebinned[k][{"dataset": sum}] for k in KEEP_KEYS}
    with open(OUT_DIR / "pythia_all.pkl", "wb") as f:
        pickle.dump(merged, f)
    print(f"wrote {OUT_DIR / 'pythia_all.pkl'}  (dataset-summed)")

    for aux in ("data_all.pkl", "herwig_all.pkl"):
        shutil.copy2(ARC_DIR / aux, OUT_DIR / aux)
        print(f"copied {aux} from arc_r2 (unaffected by JMS/JMR for this comparison)")


if __name__ == "__main__":
    main()
