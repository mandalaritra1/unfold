#!/usr/bin/env python3
r"""50-50 independent-half bias test across the 'ratio' regularization family.

Extends study_5050_bias.py (which covered ratio_curv vs tau=0) to ratio_deriv
and ratio_size, so the family can be compared on the decisive non-trivial test:
unfold one statistically independent PYTHIA half through the response built from
the other and check it recovers the half's truth.

For each mode the L-curve tau is scanned on the FULL data (as in production) and
frozen, then half-B's reco is unfolded through half-A's response at that tau.
Self-closure is exact (= trivial) for ratio_curv/ratio_deriv by construction, so
this independent test is what actually discriminates them; ratio_size has no
null-space for the prior and is expected to bias.

Usage: source scripts/setup_root.sh && python scripts/study_5050_ratio_modes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")

from unfold.tools.unfolder_core import Unfolder, get_spec, _declare_open_l
from unfold.utils.merge_helpers import unflatten_gen_by_pt
from study_regularization_rho import normalize
from study_regularization_grid import run  # mode-aware build + scan/unfold + cleanup
from study_5050_bias import reconstruct_halves

_declare_open_l()
MODES = ["none", "ratio_curv", "ratio_deriv", "ratio_size"]


def main():
    u = Unfolder(get_spec("zjet", "rho", "original"), groomed=True, do_syst=False)
    var = u.corrected_measured_variances
    resp_full = u.mosaic_dict["nominal"]
    misses_full = u.misses_2d
    true_full = resp_full.sum(axis=0) + misses_full
    data_full = u._apply_fake_correction(np.array(u.mosaic_2d, float), "nominal", False, False)

    R_A, R_B = reconstruct_halves(u)
    true_A = R_A.sum(axis=0)
    meas_B = R_B.sum(axis=1)
    truth_B = R_B.sum(axis=0)
    zero = np.zeros_like(true_A)
    truth_Bn = normalize(u, truth_B)

    print(f"\n{'mode':>12} {'tau':>8} | per-pT |closure| median (max)   [vs half-sample stat]")
    print("-" * 84)
    for mode in MODES:
        if mode == "none":
            tau = 0.0
        else:
            _, _, tau = run(u, resp_full, misses_full, true_full, data_full,
                            mode=mode, name=f"sc_{mode}", scan="lcurve", variances=var)
        yB, covB, _ = run(u, R_A, zero, true_A, meas_B, mode=mode, name=f"hb_{mode}", tau=tau)
        clo = np.abs(normalize(u, yB) / np.where(truth_Bn != 0, truth_Bn, 1.0) - 1.0)
        err = np.sqrt(np.clip(np.diag(covB), 0, None)) / np.where(yB != 0, np.abs(yB), 1.0)
        clo_pt = unflatten_gen_by_pt(clo, u.gen_edges_by_pt)
        err_pt = unflatten_gen_by_pt(err, u.gen_edges_by_pt)
        cells = []
        for i in u._reported_pt_indices():
            if u.pt_edges[i] < 200:
                continue
            sl = slice(1, None)  # drop the [-10,-4.5] underflow sink
            cells.append(f"{100*np.median(clo_pt[i][sl]):4.1f}%({100*np.max(clo_pt[i][sl]):4.1f})"
                         f"[{100*np.median(err_pt[i][sl]):3.0f}]")
        print(f"{mode:>12} {tau:>8.2g} | " + "  ".join(cells))
    print("  cell = median%(max%)[reg stat median%]; pT slices 200-290, 290-400, 400-inf")
    print("  ratio_curv/ratio_deriv share the prior null-space (safe); ratio_size does not.")


if __name__ == "__main__":
    main()
