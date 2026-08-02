#!/usr/bin/env python
"""Print the nonfiducial ("fake") event fraction per detector-level jet-pT bin.

Reproduces exactly the fake arithmetic of unfolder_core._make_inputs_numpy
(fakes = reco MC - response summed over IN-RANGE gen-pT bins, so cross-boundary
185-GeV migrations count as fakes/misses), then projects over the mass axis:

    fake_fraction(pT bin) = fakes(pT) / reco(pT)

Made for the SMP-25-010 approval question L175 ("What's the fraction of
nonfiducial events?"). Diagnostic only -- reads the arc_r2 input pickles,
runs no unfolding.

    python scripts/print_fake_fractions.py
    python scripts/print_fake_fractions.py --input inputs/zjet/rho/arc_r2/pythia_all.pkl --label PYTHIA
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

KEYS = {
    "groomed": ("ptjet_rhojet_g_reco", "response_matrix_rho_g"),
    "ungroomed": ("ptjet_rhojet_u_reco", "response_matrix_rho_u"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Input coffea pickle(s); default: arc_r2 pythia_all.pkl and herwig_all.pkl",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label per --input (same order).",
    )
    return parser.parse_args()


def _nominal(h):
    if "systematic" in h.axes.name:
        h = h[{"systematic": "nominal"}]
    return h


def _sum_inrange_pt(h, pt_axis):
    # identical to unfolder_core._make_inputs_numpy: in-range pt bins only,
    # so cross-185-GeV migrations stay in fakes/misses
    return h[{pt_axis: slice(0, len(h.axes[pt_axis]), sum)}]


def fake_fractions(out, reco_key, resp_key):
    reco = _nominal(out[reco_key]).project("ptreco")
    matched = _nominal(_sum_inrange_pt(out[resp_key], "ptgen")).project("ptreco")
    reco_vals = reco.values(flow=False)
    fake_vals = reco_vals - matched.values(flow=False)
    edges = reco.axes["ptreco"].edges
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(reco_vals > 0, fake_vals / reco_vals, np.nan)
    return edges, np.clip(frac, 0.0, 1.0), reco_vals


def main() -> int:
    args = parse_args()
    inputs = args.input or [
        REPO_ROOT / "inputs/zjet/rho/arc_r2/pythia_all.pkl",
        REPO_ROOT / "inputs/zjet/rho/arc_r2/herwig_all.pkl",
    ]
    labels = args.label or [Path(p).stem.replace("_all", "").upper() for p in inputs]
    if len(labels) != len(inputs):
        raise SystemExit("--label count must match --input count")

    for path, label in zip(inputs, labels):
        with open(path, "rb") as handle:
            out = pickle.load(handle)
        print(f"\n=== {label} ({path}) ===")
        for grooming, (reco_key, resp_key) in KEYS.items():
            edges, frac, reco_vals = fake_fractions(out, reco_key, resp_key)
            print(f"  {grooming}:")
            for i in range(len(edges) - 1):
                hi = "inf" if edges[i + 1] > 10000 else f"{edges[i + 1]:.0f}"
                sink = "  (auxiliary sink bin, not reported)" if edges[i + 1] <= 200 else ""
                print(
                    f"    pT {edges[i]:.0f}-{hi:<5}  fake fraction = {100.0 * frac[i]:5.1f} %"
                    f"   (reco MC events: {reco_vals[i]:,.0f}){sink}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
