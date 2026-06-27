#!/usr/bin/env python3
r"""50-50 independent-half PYTHIA bias test for the regularized rho unfolding.

The trivial self-closure (unfold the response's own matched reco) is exact at
the ratio-curvature tau by construction, so it cannot reveal a regularization
bias driven by statistical structure that differs from the MC prior. This test
splits PYTHIA into two statistically independent halves and unfolds one through
the response built from the other -- the standard way to expose such a bias.

No re-skim is needed: the delete-one-tenth jackknife response mosaics already
encode the per-tenth pieces. With T_i = mosaic - mosaic_jk_list[i] (tenth i),

    R_A = sum_{i in 0..4} T_i ,   R_B = sum_{i in 5..9} T_i ,   R_A + R_B = mosaic

are independent half-samples (matched response). Half B's matched reco is the
pseudo-data; half B's matched gen is the truth; the response/prior come from
half A. The production tau is scanned on the full data (as in production) and
frozen, then half-B reco is unfolded through half-A at tau=0 and that tau.

Outputs: per-pT-slice closure (unfolded_B / truth_B - 1) medians/max for tau=0
vs the production tau, and a per-slice closure plot with the half-sample
regularized stat band, to outputs/zjet/validation/bias_5050_<mode>.{pdf,png}.

NB matched-only (efficiency/fakes factor out of the regularization-bias test).

Usage:
    source scripts/setup_root.sh
    python scripts/study_5050_bias.py
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
import matplotlib.pyplot as plt
import mplhep as hep

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.utils.merge_helpers import unflatten_gen_by_pt
from study_regularization_rho import run_setup, normalize  # reuse the unfold helpers

OUT_DIR = REPO_ROOT / "outputs" / "zjet" / "validation"


def reconstruct_halves(u):
    """Two independent half-sample matched-response mosaics from the jackknife.

    With true disjoint tenths t_i, the delete-one-tenth replica is
    jk_i = sum_{j!=i} t_j, so sum_i jk_i = 9 * full and full = (sum_i jk_i)/9.
    Then t_i = full - jk_i *exactly* (self-consistent, independent of the
    separately-merged nominal mosaic, which differs by ~0.9%).
    """
    jk = [np.asarray(m, dtype=float) for m in u.mosaic_jk_list]
    assert len(jk) == 10, f"expected 10 jackknife mosaics, got {len(jk)}"
    full_jk = np.sum(jk, axis=0) / 9.0
    tenths = [full_jk - j for j in jk]                    # exact independent tenths t_i
    R_A = np.sum(tenths[0:5], axis=0)
    R_B = np.sum(tenths[5:10], axis=0)
    assert np.allclose(R_A + R_B, full_jk, rtol=1e-6, atol=1e-3), "half reconstruction failed"
    return R_A, R_B


def run_mode(groomed):
    spec = get_spec("zjet", "rho", "original")
    u = Unfolder(spec, groomed, do_syst=False)

    R_A, R_B = reconstruct_halves(u)
    true_A = R_A.sum(axis=0)            # half-A matched gen = regularization prior
    meas_B = R_B.sum(axis=1)           # half-B matched reco = independent pseudo-data
    truth_B = R_B.sum(axis=0)          # half-B matched gen   = the truth to recover
    zero_misses = np.zeros_like(true_A)

    # Production tau: scan the ratio-curvature L-curve on the FULL data unfold
    # (exactly as production does) and freeze it for the half-sample unfold.
    full_misses = u.misses_2d
    full_true = u.mosaic_dict["nominal"].sum(axis=0) + full_misses
    full_data = u._apply_fake_correction(np.array(u.mosaic_2d, float), "nominal", False, False)
    _, _, tau_prod, _ = run_setup(
        u, u.mosaic_dict["nominal"], full_misses, full_true, full_data,
        setup="trick", scan="lcurve", name=f"prodscan_{groomed}",
    )

    # Unfold half-B reco through half-A response at tau=0 and the production tau.
    y0, cov0, _, _ = run_setup(u, R_A, zero_misses, true_A, meas_B,
                               setup="curvature", tau=0.0, name=f"h0_{groomed}")
    yr, covr, _, _ = run_setup(u, R_A, zero_misses, true_A, meas_B,
                               setup="trick", tau=tau_prod, name=f"hr_{groomed}")

    truth_n = normalize(u, truth_B)
    clo0 = normalize(u, y0) / np.where(truth_n != 0, truth_n, 1.0) - 1.0
    clor = normalize(u, yr) / np.where(truth_n != 0, truth_n, 1.0) - 1.0
    # half-sample regularized stat error (fractional, on the unfolded result)
    err_r = np.sqrt(np.clip(np.diag(covr), 0, None)) / np.where(yr != 0, np.abs(yr), 1.0)

    mode = "groomed" if groomed else "ungroomed"
    print(f"\n=== 50-50 PYTHIA bias test [{mode}]   production tau = {tau_prod:.3g} ===")
    print(f"{'pt slice':>12} | {'|closure| tau=0':>18} | {'|closure| reg':>18} | {'reg stat (half)':>16}")
    print("-" * 74)
    c0_pt = unflatten_gen_by_pt(np.abs(clo0), u.gen_edges_by_pt)
    cr_pt = unflatten_gen_by_pt(np.abs(clor), u.gen_edges_by_pt)
    er_pt = unflatten_gen_by_pt(err_r, u.gen_edges_by_pt)
    for i in u._reported_pt_indices():
        if u.pt_edges[i] < 200:
            continue
        hi = int(u.pt_edges[i + 1]) if i + 1 < len(u.pt_edges) - 1 else None
        lab = f"{int(u.pt_edges[i])}-{hi}" if hi else f"{int(u.pt_edges[i])}-inf"
        # drop the [-10,-4.5] underflow sink bin (index 0) from the summary
        sl = slice(1, None)
        print(f"{lab+' GeV':>12} | "
              f"{f'{100*np.median(c0_pt[i][sl]):.2f}% (max {100*np.max(c0_pt[i][sl]):.1f}%)':>18} | "
              f"{f'{100*np.median(cr_pt[i][sl]):.2f}% (max {100*np.max(cr_pt[i][sl]):.1f}%)':>18} | "
              f"{f'{100*np.median(er_pt[i][sl]):.1f}%':>16}")
    print("  (closure = unfolded_B / truth_B - 1, normalized per slice; matched-only)")

    # ---- plot: per-slice closure, tau=0 vs reg, with half-sample reg stat band ----
    reported = [i for i in u._reported_pt_indices() if u.pt_edges[i] >= 200]
    fig, axes = plt.subplots(1, len(reported), figsize=(6 * len(reported), 6), sharey=True)
    if len(reported) == 1:
        axes = [axes]
    for ax, i in zip(axes, reported):
        edges = np.array(u.gen_edges_by_pt[i], dtype=float)
        ax.axhline(0.0, color="gray", ls="--")
        ax.stairs(100 * er_pt[i], edges, baseline=-100 * er_pt[i], fill=True,
                  color="0.8", label="half-sample reg stat")
        ax.stairs(100 * unflatten_gen_by_pt(clo0, u.gen_edges_by_pt)[i], edges,
                  color="k", ls="--", label=r"$\tau=0$")
        ax.stairs(100 * unflatten_gen_by_pt(clor, u.gen_edges_by_pt)[i], edges,
                  color="#e42536", label=rf"reg $\tau$={tau_prod:.2g}")
        ax.set_xlim(u._observable_xlim(i))
        ax.set_ylim(-20, 20)
        hi = int(u.pt_edges[i + 1]) if i + 1 < len(u.pt_edges) - 1 else None
        ax.set_title(f"{int(u.pt_edges[i])}-{hi if hi else '∞'} GeV", fontsize=13)
        ax.set_xlabel(u._observable_label())
    axes[0].set_ylabel("unfolded / truth - 1 (%)")
    axes[0].legend(title=f"50-50 PYTHIA closure ({mode})", fontsize=10)
    hep.cms.label("Simulation Internal", data=False, lumi="138", com="13", ax=axes[0], fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / f"bias_5050_{mode}"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out}.pdf / .png")


def main():
    for groomed in (True, False):
        run_mode(groomed)


if __name__ == "__main__":
    main()
