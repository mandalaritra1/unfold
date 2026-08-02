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
vs the production tau, and one closure plot per pT slice overlaying the tau=0
and regularized curves, each with its own half-sample stat band (so their
overlap shows statistical consistency), to
outputs/zjet/validation/bias_5050_<mode>_<i>.{pdf,png}.

NB matched-only (efficiency/fakes factor out of the regularization-bias test).

Usage:
    source scripts/setup_root.sh
    python scripts/studies/study_5050_bias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    # half-sample stat error (fractional, on the unfolded result) for both the
    # unregularized (tau=0) and the regularized unfold
    err_0 = np.sqrt(np.clip(np.diag(cov0), 0, None)) / np.where(y0 != 0, np.abs(y0), 1.0)
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

    # ---- one CMS-style 3:1 ratio figure per pT slice ----
    #   top  : the normalized rho spectra (half-B truth vs tau=0 vs regularized)
    #   bottom: the closure (unfolded/truth - 1) as a fraction -- regularized as a
    #           stat band, tau=0 as points with stat error bars, so one can read
    #           off that the two agree with each other and with zero.
    hep.style.use("CMS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_n = unflatten_gen_by_pt(truth_n, u.gen_edges_by_pt)
    u0_n = unflatten_gen_by_pt(normalize(u, y0), u.gen_edges_by_pt)
    ur_n = unflatten_gen_by_pt(normalize(u, yr), u.gen_edges_by_pt)
    c0s = unflatten_gen_by_pt(clo0, u.gen_edges_by_pt)
    crs = unflatten_gen_by_pt(clor, u.gen_edges_by_pt)
    e0_pt = unflatten_gen_by_pt(err_0, u.gen_edges_by_pt)
    reported = [i for i in u._reported_pt_indices() if u.pt_edges[i] >= 200]
    for i in reported:
        edges = np.array(u.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hi = int(u.pt_edges[i + 1]) if i + 1 < len(u.pt_edges) - 1 else None
        slice_lab = f"{int(u.pt_edges[i])}-{hi if hi else '∞'} GeV"

        fig, (a, r) = plt.subplots(
            2, 1, sharex=True, figsize=(8.5, 8.5),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        # --- top: normalized spectra (truth solid; unfolded dashed & dotted,
        #     the dashed tau=0 listed/drawn on top) ---
        a.stairs(u0_n[i], edges, color="k", ls="--", lw=2.0, zorder=3,
                 label=r"unfolded, $\tau=0$")
        a.stairs(ur_n[i], edges, color="#e42536", ls=":", lw=2.2, zorder=2,
                 label="unfolded, regularized")
        a.stairs(t_n[i], edges, color="#5790fc", ls="-", lw=2.0, zorder=1,
                 label="truth (half B)")
        # nominal (regularized) half-sample stat band on the spectrum itself
        # (identified by the ratio-panel legend, so it is left out of this one)
        nb = ur_n[i] * er_pt[i]
        a.fill_between(edges, np.append(ur_n[i] - nb, (ur_n[i] - nb)[-1]),
                       np.append(ur_n[i] + nb, (ur_n[i] + nb)[-1]), step="post",
                       facecolor="0.85", edgecolor="0.55", hatch="///",
                       linewidth=0.0, zorder=0)
        a.set_ylabel(u._normalized_ylabel())
        # headroom so the upper-left legend clears the spectrum (no overlap)
        vis = edges[:-1] >= u._observable_xlim(i)[0]
        ymax = max(np.max(t_n[i][vis]), np.max(u0_n[i][vis]),
                   np.max((ur_n[i] + nb)[vis]))
        a.set_ylim(0, 1.9 * ymax)
        a.legend(title=slice_lab, fontsize=13, title_fontsize=13, loc="upper left")
        a.text(0.5, 0.04, f"50-50 PYTHIA closure ({mode})", transform=a.transAxes,
               va="bottom", ha="center", fontsize=13)

        # --- bottom: closure. Grey hatched band = nominal (regularized)
        # half-sample stat unc. across perfect closure (0); both unfolds are
        # drawn as points with per-bin stick errors at the same bin centre
        # (regularized = red squares with solid sticks, tau=0 = black circles
        # with dashed sticks) so one can read them off against each other. ---
        r.axhline(0.0, color="gray", ls="--", lw=1.2)
        band = np.append(er_pt[i], er_pt[i][-1])
        r.fill_between(edges, -band, band, step="post", facecolor="0.85",
                       edgecolor="0.55", hatch="///", linewidth=0.0,
                       label="nominal stat. unc.")
        r.errorbar(centers, crs[i], yerr=er_pt[i], fmt="s", color="#e42536",
                   ms=4, elinewidth=1.3, capsize=2)
        eb0 = r.errorbar(centers, c0s[i], yerr=e0_pt[i], fmt="o", color="k",
                         ms=4, elinewidth=1.3, capsize=2)
        for barline in eb0[2]:            # dashed sticks for the tau=0 variation
            barline.set_linestyle((0, (4, 2)))
        r.set_ylabel(r"$\frac{\mathrm{unfolded}}{\mathrm{truth}} - 1$")
        r.set_ylim(-0.2, 0.2)
        r.set_xlim(u._observable_xlim(i))
        r.set_xlabel(u._observable_label())
        r.legend(fontsize=10, loc="lower left", frameon=False)

        # PYTHIA-only closure test -> CMS Simulation label, no integrated lumi.
        # data=False prepends "Simulation"; rlabel sets the right-hand text.
        hep.cms.label("Preliminary", data=False, rlabel="(13 TeV)", ax=a, fontsize=20)
        fig.tight_layout()
        out = OUT_DIR / f"bias_5050_{mode}_{i}"
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}.pdf / .png")


def main():
    for groomed in (True, False):
        run_mode(groomed)


if __name__ == "__main__":
    main()
