#!/usr/bin/env python3
"""ARC U2.3 redone without HERWIG: closure against the PS/HAD model variations.

Reviewer request (SMP-25-010, U#2.3) was: unfold the HERWIG sample with the
*reweighted* PYTHIA sample; the unfolded spectrum should agree with the original
gen-level HERWIG spectrum within statistical uncertainties.

HERWIG is no longer a modelling leg. The round-2 modelling uncertainty is the
two-leg PS/HAD envelope built from gen-reweighted PYTHIA: PS = Vincia, HAD =
envelope over CR1/CR2 and Lund frag hard/soft. Those samples are full reskims in
``inputs/zjet/rho/model_reco/<source>_all.pkl`` (own response, reco and gen) on
the 24-gen x 48-reco axes.

WHY THE REQUEST CANNOT BE TRANSCRIBED LITERALLY. In round 1 the two ingredients
were genuinely different objects: the input was HERWIG (its own shower AND its
own detector-level sample) while the response was PYTHIA reweighted to look like
HERWIG, so what remained under test was whether HERWIG's residual detector-level
difference mattered. The round-2 model variations ARE gen-reweighted PYTHIA, so
"the Vincia sample" and "PYTHIA reweighted to Vincia" are the same object:
unfolding <source> reco through the <source> response is an exact self-closure
that tests nothing. It is run below (``matched``) only to confirm it is exact.

The test that carries content is the unmatched one (``nominal``): the model
sample's matched reco is unfolded through the NOMINAL PYTHIA response -- the
response actually applied to the data -- and compared with that model's own gen
truth. This asks whether the unfolding recovers an alternate shower /
hadronization truth while carrying the PYTHIA prior, which is what the modelling
uncertainty has to cover. Same construction as the round-1 U2.4 model-closure
test, with Vincia/CR/frag in place of HERWIG.

CAVEAT (inherited from the production, not introduced here): the model samples
carry the old [0, 200] low-pT bin, while arc_r2 uses a [185, 200] migration
buffer. ``model_envelope`` already makes this approximation for the production
modelling uncertainty (see the ``RHO_ARC_R2_SPEC`` comment). The lowest pT bin
is not reported and is excluded from every number quoted here.

Usage:
  source scripts/setup_root.sh
  .venv/bin/python scripts/studies/study_model_closure_arc_r2.py [tag]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.tools.model_envelope import _mosaic_inputs, _coarsened_nominal
from unfold.utils.merge_helpers import unflatten_gen_by_pt

MODEL_DIR = REPO / "inputs" / "zjet" / "rho" / "model_reco"
OUT_DIR = REPO / "outputs" / "zjet" / "validation"

SOURCES = ["vincia", "cr1", "cr2", "fraghard", "fragsoft"]
LABELS = {
    "vincia": "Vincia (PS)",
    "cr1": "CR1 (HAD)",
    "cr2": "CR2 (HAD)",
    "fraghard": "Frag. hard (HAD)",
    "fragsoft": "Frag. soft (HAD)",
}
COLORS = {
    "vincia": "#e42536",
    "cr1": "#5790fc",
    "cr2": "#f89c20",
    "fraghard": "#964a8b",
    "fragsoft": "#7a21dd",
}


def load_model(source, groomed):
    """(response, gen, gen_edges24, reco_edges48) for one model sample."""
    tag = "g" if groomed else "u"
    with open(MODEL_DIR / f"{source}_all.pkl", "rb") as fh:
        d = pickle.load(fh)
    h = d[f"response_matrix_rho_{tag}"]
    resp = h.view(flow=False)["value"].sum(axis=0)[..., 0]
    gh = d[f"ptjet_rhojet_{tag}_gen"]
    gen = gh.view(flow=False)["value"].sum(axis=0)[..., 0]
    e24 = np.asarray(h.axes["mpt_gen"].edges, float)
    e48 = np.asarray(h.axes["mpt_reco"].edges, float)
    return resp, gen, e24, e48


def unfold_input(uf, key, *, mosaic, misses, meas, meas_var, true_flat=None):
    """Unfold a supplied measured vector through a supplied response.

    ``closure=True``: the measured vector here is the *matched* reco projection
    of a response, so it carries no fakes and must not have the fake correction
    applied to it (that would strip real signal and break closure).
    """
    uf.misses_2d_dict = getattr(uf, "misses_2d_dict", {})
    uf.misses_2d_dict[key] = misses
    saved_stat = uf.response_matrix_stat_available
    uf.response_matrix_stat_available = False   # band = model input stat only
    try:
        uf._perform_unfold(systematic=key, closure=True, meas_flat=meas,
                           meas_var=meas_var, resp_np=mosaic,
                           true_flat_override=true_flat)
    finally:
        uf.response_matrix_stat_available = saved_stat
    return (np.asarray(uf.y_unf_dict[key], float),
            np.asarray(uf.ye_unf_dict[key], float))


def _norm_shape(vals, err, edges):
    vals = np.asarray(vals, float)
    total = vals.sum()
    bw = np.diff(np.asarray(edges, float))
    if total <= 0:
        return np.zeros_like(vals), np.zeros_like(vals)
    return vals / bw / total, np.asarray(err, float) / bw / total


def _shown_mask(uf, i):
    edges = np.asarray(uf.gen_edges_by_pt[i], float)
    xlo, xhi = uf._observable_xlim(i)
    return (edges[:-1] >= xlo) & (edges[1:] <= xhi + 1e-9)


def run_mode(groomed, tag):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n========== {mode} ==========")
    spec = get_spec("zjet", "rho", tag)
    uf = Unfolder(spec, groomed, do_syst=False, compute_jackknife_stat=False,
                  cms_label="Preliminary")

    stat_frac = [
        np.asarray(uf.normalized_results[i]["stat_unc_frac"], float)
        for i in range(len(uf.normalized_results))
    ]

    nom_mosaic = np.asarray(uf.mosaic, float)
    nom_misses = np.asarray(uf.misses_2d, float)

    # OFFLINE-NOMINAL BASELINE (w = 1) on the same 24x48 grid as the model
    # samples. The model files carry the old [0, 200] low-pT bin while the
    # analysis response uses a [185, 200] buffer; with ~28% cross-pT migration
    # that mismatch alone produces a large apparent non-closure. It is a
    # property of the grid, not of the model, so it cancels when every source is
    # referred to this baseline -- exactly how ``model_envelope`` handles it.
    b_resp, b_gen, be24, be48 = _coarsened_nominal(groomed)
    b_mosaic, b_misses, _ = _mosaic_inputs(uf, b_resp, b_gen, be24, be48)
    b_meas = b_mosaic.sum(axis=1)
    b_truth = b_mosaic.sum(axis=0) + b_misses
    b_unf, _ = unfold_input(uf, "model_base", mosaic=nom_mosaic,
                            misses=nom_misses, meas=b_meas,
                            meas_var=np.clip(b_meas, 0.0, None))
    b_unf = np.asarray(b_unf, float)

    residuals, errors = {}, {}
    for source in SOURCES:
        resp, gen, e24, e48 = load_model(source, groomed)
        m_mosaic, m_misses, m_ff = _mosaic_inputs(uf, resp, gen, e24, e48)

        # pseudo-data = matched reco of the model response (no fakes mismatch)
        meas = m_mosaic.sum(axis=1)
        meas_var = np.clip(meas, 0.0, None)          # MC stat of the model sample
        truth = m_mosaic.sum(axis=0) + m_misses      # that model's own gen truth

        # (a) matched response (+ its own prior) -> exact self-closure, no content
        y_m, _ = unfold_input(uf, f"model_matched_{source}", mosaic=m_mosaic,
                              misses=m_misses, meas=meas, meas_var=meas_var,
                              true_flat=truth)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_m = np.where(truth != 0, y_m / truth - 1.0, 0.0)

        # (b) NOMINAL response + nominal prior -> the test that carries content
        y, ye = unfold_input(uf, f"model_nominal_{source}", mosaic=nom_mosaic,
                             misses=nom_misses, meas=meas, meas_var=meas_var)

        # Residual on the model EFFECT, referred to the w=1 baseline:
        #   recovered effect (unf_src / unf_base)  vs  true effect (truth_src / truth_base)
        # The grid/pT-buffer artifact is common to both and cancels.
        res_blocks, err_blocks = [], []
        y_pt = unflatten_gen_by_pt(y, uf.gen_edges_by_pt)
        ye_pt = unflatten_gen_by_pt(ye, uf.gen_edges_by_pt)
        t_pt = unflatten_gen_by_pt(truth, uf.gen_edges_by_pt)
        bu_pt = unflatten_gen_by_pt(b_unf, uf.gen_edges_by_pt)
        bt_pt = unflatten_gen_by_pt(b_truth, uf.gen_edges_by_pt)
        for i, edges in enumerate(uf.gen_edges_by_pt):
            u_n, u_e = _norm_shape(y_pt[i], ye_pt[i], edges)
            t_n, _ = _norm_shape(t_pt[i], np.zeros_like(t_pt[i]), edges)
            bu_n, _ = _norm_shape(bu_pt[i], np.zeros_like(bu_pt[i]), edges)
            bt_n, _ = _norm_shape(bt_pt[i], np.zeros_like(bt_pt[i]), edges)
            with np.errstate(divide="ignore", invalid="ignore"):
                ok = (t_n != 0) & (bu_n != 0) & (bt_n != 0)
                recovered = np.divide(u_n, bu_n, out=np.zeros_like(u_n), where=ok)
                true_eff = np.divide(t_n, bt_n, out=np.ones_like(t_n), where=ok)
                res_blocks.append(np.where(
                    ok, np.divide(recovered, true_eff,
                                  out=np.zeros_like(u_n), where=ok) - 1.0, 0.0))
                err_blocks.append(np.where(
                    ok, np.divide(u_e, np.where(u_n != 0, u_n, 1.0),
                                  out=np.zeros_like(u_e), where=ok), 0.0))
        residuals[source] = res_blocks
        errors[source] = err_blocks

        shown = np.concatenate([res_blocks[i][_shown_mask(uf, i)]
                                for i in uf._reported_pt_indices()])
        shown_e = np.concatenate([err_blocks[i][_shown_mask(uf, i)]
                                  for i in uf._reported_pt_indices()])
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = np.max(np.abs(shown) / np.where(shown_e > 0, shown_e, np.inf))
        print(f"  {LABELS[source]:18s} matched |max| = {np.max(np.abs(rel_m)):.2e}"
              f"   nominal-response max |unf/truth-1| = {np.max(np.abs(shown)):.4f}"
              f"   max pull = {pull:.2f}")

    _plot(uf, residuals, errors, stat_frac, mode)


def _pt_bin_label(uf, i):
    lo = uf.pt_edges[i]
    hi = uf.pt_edges[i + 1] if i + 2 < len(uf.pt_edges) else float("inf")
    hi_s = "∞" if not np.isfinite(hi) else f"{hi:.0f}"
    return f"{lo:.0f}–{hi_s}"


def _plot(uf, residuals, errors, stat_frac, mode):
    hep.style.use("CMS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in uf._reported_pt_indices():
        edges = np.asarray(uf.gen_edges_by_pt[i], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        visible = _shown_mask(uf, i)
        ymax = 0.0
        fig = plt.figure()

        band = np.asarray(stat_frac[i], float)
        plt.fill_between(edges, -np.append(band, band[-1]),
                         np.append(band, band[-1]), step="post",
                         color="0.8", alpha=0.6, lw=0,
                         label="Stat. unc. (data result)")
        if visible.any():
            ymax = max(ymax, float(np.max(band[visible])))

        for source in residuals:
            vals = residuals[source][i]
            errs = errors[source][i]
            if visible.any():
                ymax = max(ymax, float(np.max(np.abs(vals[visible]) + errs[visible])))
            hep.histplot(vals, edges, label=LABELS[source],
                         color=COLORS[source], lw=2.0)
            plt.errorbar(centers, vals, yerr=errs, fmt="none",
                         ecolor=COLORS[source], elinewidth=1.4, capsize=2, alpha=0.9)

        plt.axhline(0, color="gray", lw=1)
        lim = max(0.01, 1.25 * ymax)
        plt.ylim(-lim, lim)
        plt.xlim(*uf._observable_xlim(i))
        plt.xlabel(uf._observable_label())
        plt.ylabel("Unfolded / Truth $-$ 1")
        plt.legend(title=rf"$p_{{T}}$  {_pt_bin_label(uf, i)} GeV", fontsize=13)
        hep.cms.label(uf.cms_label, data=False, lumi=uf._lumi_label(),
                      com=uf._com_label(), fontsize=20)
        ax = plt.gca()
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)
        plt.subplots_adjust(left=0.16, bottom=0.15)
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"model_closure_{mode}_{i}.{ext}",
                        bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    print(f"  saved per-pT figures -> {OUT_DIR}/model_closure_{mode}_*.pdf")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "arc_r2"
    run_mode(groomed=True, tag=tag)
    run_mode(groomed=False, tag=tag)


if __name__ == "__main__":
    main()
