#!/usr/bin/env python3
"""ARC round-3 OP1/WS2: column-scaled model closure for AN section 5.7.

The test the ARC requested, in the form their own caveat prescribes: for each
modelling variation of the current workflow (Vincia; CR modes 1/2; Lund frag
hard/soft), pseudo-data are built as the reco-level projection of the
column-scaled varied response, unfolded through the w = 1 (nominal) response
of the same machinery, and compared with that variation's own generator-level
truth.  Everything lives in one consistent world (the fine 2018 response the
modelling envelope itself is derived from), so no cross-era baseline is
needed and the w = 1 case is an exact self-closure.

Because the variations are gen-reweighted PYTHIA, pseudo-data and response
share the underlying statistical dataset: following the reviewers' caveat the
residuals are compared with the MODELLING uncertainty band alone
(``normalized_results[i]['model_unc_frac']`` of the production unfolder).
Agreement is expected by construction -- these variations define the envelope
-- and the test verifies that the full unfolding chain (nominal prior,
efficiency, fake/miss treatment) recovers each alternative truth.

No statistical error bars are drawn on the residual curves: the pseudo-data
are deterministic projections of the reweighted response, and the shared-MC
fluctuation cancels between the unfolded result and the truth (demonstrated by
the exact w = 1 self-closure); the non-cancelling remainder is
O((w-1) x MC stat) and negligible.  Poisson errors of the projection would
overstate the uncertainty and contradict the reviewers' caveat.

Companion studies: ``study_model_closure_arc_r2.py`` / ``_arc_r3.py`` run the
same closure with the standalone model reskims instead of the column-scaled
variations; those mix in cross-era and weighter-approximation effects and are
kept as an internal follow-up.

Usage:
  source scripts/setup_root.sh
  .venv/bin/python scripts/studies/study_model_closure_colscaled.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.tools.model_envelope import (
    MODEL_SOURCES,
    _coarsened_nominal,
    _mosaic_inputs,
    _weighter_w,
)
from unfold.utils.merge_helpers import unflatten_gen_by_pt
from study_model_closure_arc_r3 import (
    COLORS,
    LABELS,
    _norm_shape,
    _pt_bin_label,
    _shown_mask,
    unfold_input,
)

OUT_DIR = REPO / "outputs" / "zjet" / "validation"


def run_mode(groomed, tag):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n========== {mode} ==========")
    spec = get_spec("zjet", "rho", tag)
    uf = Unfolder(spec, groomed, do_syst=True, compute_jackknife_stat=False,
                  cms_label="Internal")

    band_frac = [
        np.asarray(uf.normalized_results[i]["model_unc_frac"], float)
        for i in range(len(uf.normalized_results))
    ]

    # One consistent world: the fine 2018 nominal (w = 1) response in the
    # analysis binning is both the unfolding response and the source of every
    # varied pseudo-data set.
    resp_nom, gen_nom, e24, e48 = _coarsened_nominal(groomed)
    nom_mosaic, nom_misses, _ = _mosaic_inputs(uf, resp_nom, gen_nom, e24, e48)
    nom_truth = nom_mosaic.sum(axis=0) + nom_misses
    centers = 0.5 * (e24[:-1] + e24[1:])

    # w = 1 sanity: unfolding the nominal projection through the nominal
    # response must be an exact self-closure.
    y0, _ = unfold_input(uf, "colscaled_self", mosaic=nom_mosaic,
                         misses=nom_misses, meas=nom_mosaic.sum(axis=1),
                         meas_var=np.clip(nom_mosaic.sum(axis=1), 0.0, None),
                         true_flat=nom_truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        self_rel = np.where(nom_truth != 0, y0 / nom_truth - 1.0, 0.0)
    print(f"  w=1 self-closure |max| = {np.max(np.abs(self_rel)):.2e}")

    residuals, errors, shapes = {}, {}, {}
    nom_shapes = [
        _norm_shape(pt, np.zeros_like(pt), uf.gen_edges_by_pt[i])[0]
        for i, pt in enumerate(unflatten_gen_by_pt(nom_truth, uf.gen_edges_by_pt))
    ]
    for source in MODEL_SOURCES:
        w = _weighter_w(source, groomed, centers)          # (ptgen, 24)
        resp_var = resp_nom * w[None, None, :, :]
        gen_var = gen_nom * w
        var_mosaic, var_misses, _ = _mosaic_inputs(uf, resp_var, gen_var, e24, e48)

        meas = var_mosaic.sum(axis=1)                      # varied pseudo-data
        truth = var_mosaic.sum(axis=0) + var_misses        # varied own truth

        # Unfold through the NOMINAL (w = 1) response with the nominal prior:
        # exactly what the analysis does to data.
        y, ye = unfold_input(uf, f"colscaled_{source}", mosaic=nom_mosaic,
                             misses=nom_misses, meas=meas,
                             meas_var=np.clip(meas, 0.0, None),
                             true_flat=nom_truth)

        res_blocks, err_blocks = [], []
        y_pt = unflatten_gen_by_pt(y, uf.gen_edges_by_pt)
        ye_pt = unflatten_gen_by_pt(ye, uf.gen_edges_by_pt)
        t_pt = unflatten_gen_by_pt(truth, uf.gen_edges_by_pt)
        for i, edges in enumerate(uf.gen_edges_by_pt):
            u_n, u_e = _norm_shape(y_pt[i], ye_pt[i], edges)
            t_n, _ = _norm_shape(t_pt[i], np.zeros_like(t_pt[i]), edges)
            with np.errstate(divide="ignore", invalid="ignore"):
                ok = t_n != 0
                res_blocks.append(np.where(
                    ok, np.divide(u_n, t_n, out=np.zeros_like(u_n), where=ok) - 1.0,
                    0.0))
                err_blocks.append(np.where(
                    ok, np.abs(np.divide(u_e, np.where(u_n != 0, u_n, 1.0),
                                         out=np.zeros_like(u_e), where=ok)), 0.0))
        residuals[source] = res_blocks
        errors[source] = err_blocks
        shapes[source] = [
            (_norm_shape(y_pt[i], np.zeros_like(y_pt[i]), uf.gen_edges_by_pt[i])[0],
             _norm_shape(t_pt[i], np.zeros_like(t_pt[i]), uf.gen_edges_by_pt[i])[0])
            for i in range(len(uf.gen_edges_by_pt))
        ]

        published = range(max(1, getattr(uf, "first_reported_pt_bin", 0)),
                          len(uf.pt_edges) - 1)
        shown = np.concatenate([res_blocks[i][_shown_mask(uf, i)]
                                for i in published])
        shown_band = np.concatenate([band_frac[i][_shown_mask(uf, i)]
                                     for i in published])
        with np.errstate(divide="ignore", invalid="ignore"):
            over = np.max(np.abs(shown) / np.where(shown_band > 0, shown_band,
                                                   np.inf))
        print(f"  {LABELS[source]:18s} max |unf/truth-1| = {np.max(np.abs(shown)):.4f}"
              f"   max residual/band = {over:.2f}")

    _plot(uf, shapes, nom_shapes, band_frac, mode)


def _plot(uf, shapes, nom_shapes, band_frac, mode):
    """Main panel: each variation's truth (line) and unfolded pseudo-data
    (markers), so the distributions themselves are visible; ratio panel:
    Unfolded / Truth against the modelling band. No stat error bars (see
    module docstring)."""
    hep.style.use("CMS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    published = range(max(1, getattr(uf, "first_reported_pt_bin", 0)),
                      len(uf.pt_edges) - 1)
    for i in published:
        edges = np.asarray(uf.gen_edges_by_pt[i], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        visible = _shown_mask(uf, i)

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07},
        )

        # ---- main panel: normalized shapes --------------------------------
        ax_main.stairs(nom_shapes[i], edges, color="black", lw=2.5,
                       baseline=None, label="Nominal ($w=1$)")
        main_max = float(np.max(nom_shapes[i][visible])) if visible.any() else 1.0
        for source in shapes:
            u_n, t_n = shapes[source][i]
            ax_main.stairs(t_n, edges, color=COLORS[source], lw=2.0,
                           baseline=None, label=LABELS[source])
            ax_main.plot(centers[visible], u_n[visible], "o",
                         color=COLORS[source], ms=5)
            if visible.any():
                main_max = max(main_max, float(np.max(t_n[visible])),
                               float(np.max(u_n[visible])))
        ax_main.plot([], [], "o", color="0.3", ms=5, label="Unfolded (markers)")
        ax_main.set_ylim(0, main_max * 1.55)
        ax_main.set_ylabel(r"$1/N\; dN/d\log_{10}(\rho^{2})$")
        ax_main.legend(title=rf"$p_{{T}}$  {_pt_bin_label(uf, i)} GeV",
                       fontsize=13, ncol=2)
        hep.cms.label(uf.cms_label, data=False, lumi=uf._lumi_label(),
                      com=uf._com_label(), fontsize=20, ax=ax_main)

        # ---- ratio panel: closure vs modelling band -----------------------
        band = np.asarray(band_frac[i], float)
        ax_ratio.fill_between(edges, np.append(1 - band, 1 - band[-1]),
                              np.append(1 + band, 1 + band[-1]), step="post",
                              color="0.8", alpha=0.6, lw=0)
        rmax = float(np.max(band[visible])) if visible.any() else 0.01
        for source in shapes:
            u_n, t_n = shapes[source][i]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(t_n != 0, u_n / t_n, np.nan)
            ax_ratio.stairs(ratio, edges, color=COLORS[source], lw=2.0,
                            baseline=None)
            if visible.any():
                rmax = max(rmax, float(np.nanmax(np.abs(ratio[visible] - 1.0))))
        ax_ratio.axhline(1.0, color="gray", lw=1)
        lim = max(0.01, 1.25 * rmax)
        ax_ratio.set_ylim(1 - lim, 1 + lim)
        ax_ratio.set_ylabel("Unfolded / Truth", fontsize=18)
        ax_ratio.set_xlim(*uf._observable_xlim(i))
        ax_ratio.set_xlabel(uf._observable_label())
        for ax in (ax_main, ax_ratio):
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)

        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"model_closure_colscaled_{mode}_{i}.{ext}",
                        bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    print(f"  saved per-pT figures -> {OUT_DIR}/model_closure_colscaled_{mode}_*.pdf")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "jmsjmr_unity_groomed400_floor3"
    run_mode(groomed=True, tag=tag)
    run_mode(groomed=False, tag=tag)


if __name__ == "__main__":
    main()
