#!/usr/bin/env python
"""Print the result-plot legend chi2/ndf for all generators x pT bins x groomings.

Reproduces exactly the `_mc_chi2_label` closure of plot_unfolded_fancy
(unfolder_core.py): a DIAGONAL chi2 over the shown bins,
sigma = max(total_up, total_down) per bin (stat + syst + model envelope),
ndf = (# shown usable bins) - 1 for the normalization constraint. This is the
number quoted in the legends of the unfolded-result figures (paper Fig. 4),
NOT the full-covariance bottom-line chi2.

Data + Pythia + Herwig come from outputs/zjet/rho/arc_r2/data/
uncertainty_summary_2d_{groomed,ungroomed}.pkl; the Vincia curve is
re-histogrammed from inputs/zjet/rho/model_weighters/vincia_gen_cache.npz.
Made for the SMP-25-010 approval question on MC discrimination (chi2 table).

    python scripts/diagnostics/print_result_chi2_table.py [--tag arc_r2]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# _bl_shown_floors(): spec.bl_shown_floors_groomed for groomed; ungroomed falls
# back to the uniform xlim_lower = -2.5
SHOWN_FLOORS = {
    "groomed": (-2.5, -3.0, -3.0, -3.5),
    "ungroomed": (-2.5, -2.5, -2.5, -2.5),
}
VINCIA_RHO_KEY = {"groomed": "v_rho_g", "ungroomed": "v_rho_u"}
REPORTED_PT = [1, 2, 3]  # 200-290, 290-400, >400; index 0 is the 185-200 sink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="arc_r2", help="Output tag (default arc_r2)")
    return parser.parse_args()


def shown_mask(rho_edges, floor):
    return rho_edges[:-1] >= floor - 1e-9


def shown_normalized(values, mask):
    total = values[mask].sum()
    return values / total if total > 0 else values


def vincia_ptnorm(cache, rho_key, pt_edges, rho_edges):
    """Histogram the standalone Vincia gen cache into the (pt, rho) layout."""
    rho = np.asarray(cache[rho_key], dtype=float)
    pt = np.asarray(cache["v_jet_pt"], dtype=float)
    w = np.asarray(cache["v_weight"], dtype=float)
    rows = []
    for i in range(len(pt_edges) - 1):
        sel = (pt >= pt_edges[i]) & (pt < pt_edges[i + 1])
        counts, _ = np.histogram(rho[sel], bins=rho_edges, weights=w[sel])
        rows.append(counts)
    return np.array(rows)


def main() -> int:
    args = parse_args()
    data_dir = REPO_ROOT / "outputs/zjet/rho" / args.tag / "data"
    cache = np.load(REPO_ROOT / "inputs/zjet/rho/model_weighters/vincia_gen_cache.npz")

    for suffix in ("groomed", "ungroomed"):
        with open(data_dir / f"uncertainty_summary_2d_{suffix}.pkl", "rb") as handle:
            payload = pickle.load(handle)
        unfolded_h = payload["unfolded"]
        rho_edges = unfolded_h.axes[1].edges
        pt_edges = np.asarray(payload["pt_edges"], dtype=float)
        data_2d = unfolded_h.values()
        sigma_2d = np.maximum(
            np.asarray(payload["unfolded_total_up"], dtype=float),
            np.asarray(payload["unfolded_total_down"], dtype=float),
        )
        preds_2d = {
            "PYTHIA8": payload["pythia_gen_ptnorm"].values(),
            "HERWIG7": payload["herwig_gen_ptnorm"].values(),
            "VINCIA": vincia_ptnorm(cache, VINCIA_RHO_KEY[suffix], pt_edges, rho_edges),
        }

        print(f"\n=== {suffix} (tag {args.tag}) ===")
        header = f"{'pT [GeV]':<12}" + "".join(f"{name:>22}" for name in preds_2d)
        print(header + "   (chi2/ndf)")
        print("-" * len(header))
        for i in REPORTED_PT:
            mask = shown_mask(rho_edges, SHOWN_FLOORS[suffix][i])
            norm = data_2d[i][mask].sum()
            d = data_2d[i] / norm
            sigma = sigma_2d[i] / norm
            hi = "inf" if pt_edges[i + 1] > 10000 else f"{pt_edges[i + 1]:.0f}"
            row = f"{pt_edges[i]:.0f}-{hi:<7}"
            for name, pred_2d in preds_2d.items():
                pred = shown_normalized(pred_2d[i], mask)
                good = (sigma > 0) & np.isfinite(pred) & mask
                ndf = max(int(good.sum()) - 1, 1)
                chi2 = float(np.sum(((d[good] - pred[good]) / sigma[good]) ** 2))
                row += f"{chi2:>14.1f} / {ndf:<5d}"
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
