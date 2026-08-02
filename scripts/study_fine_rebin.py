#!/usr/bin/env python3
r"""Fine-then-rebin regularization study for the Z+jet rho unfolding.

Tests the proposal (Aritra's, via his advisor): unfold at a 4x-finer gen
binning with the production ratio-curvature regularization, let the L-curve
pick tau in that fine space, then SUM the fine unfolded result back down to the
reported coarse bins. The hypothesis was that regularizing in a fine space lets
the L-curve choose a tau that actually smooths structure *inside* a reported
bin, buying smaller statistical errors after the rebin.

It is compared head-to-head against the production-equivalent DIRECT coarse
unfolding, built from the SAME events (the fine pkls rebinned down first), so
the only difference is the granularity at which the unfold+regularization
happens. Three things are measured on the final reported coarse bins:

  * central value        -- normalized density per pT slice
  * statistical error    -- TUnfold GetEmatrixTotal, propagated through the rebin
  * HERWIG model bias    -- unfold HERWIG matched-reco through the PYTHIA response
  * PYTHIA self-closure  -- unfold PYTHIA's own matched-reco (sanity, must close)

The TUnfold setup (area constraint, the exact (1/m0,-2/m1,1/m2) ratio-curvature
L rows, L-curve scan, EMatrixTotal) mirrors Unfolder._perform_unfold exactly.

Usage:
    source scripts/setup_root.sh && python scripts/study_fine_rebin.py
"""
from __future__ import annotations

import pickle
import sys
from array import array
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT

from unfold.tools import binning as binmod
from unfold.tools.unfolder_core import _declare_open_l
from unfold.utils.merge_helpers import (
    merge_mass_flat,
    mosaic_no_padding,
    reorder_to_expected,
)

FINE_DIR = REPO_ROOT / "inputs" / "zjet" / "rho" / "finebins"
OUT = REPO_ROOT / "outputs" / "zjet" / "validation"
OUT.mkdir(parents=True, exist_ok=True)
PT_EDGES = [0, 200, 290, 400, 13000]
# Below these log10(rho^2) thresholds the fine binning collapses to a single
# merged tail bin: the low-rho fine bins are empty in data, so a uniformly-fine
# 48-bin gen unfold is rank-deficient there (TUnfold leaves those output bins
# unconstrained). 4x refinement is kept only in the populated region above the
# threshold, which is exactly where the reported coarse bins live anyway.
FINE_MERGE_BELOW = {False: -2.5, True: -5.0}  # ungroomed, groomed
_declare_open_l()


def _merge_below(edges, threshold, tol=1e-9):
    """Keep edges[0] and every edge >= threshold; drop the interior tail edges."""
    edges = [float(e) for e in edges]
    return [edges[0]] + [e for e in edges[1:] if e >= threshold - tol]


# Regularization methods, matching scripts/study_regularization_grid.py. The
# ratio_* modes build the L matrix by hand (kRegModeNone in the ctor); the
# built-in modes are driven by the ctor regmode over the full truth binning.
REGMODE = {
    "none": ROOT.TUnfold.kRegModeNone,
    "ratio_curv": ROOT.TUnfold.kRegModeNone,
    "ratio_curvature": ROOT.TUnfold.kRegModeNone,  # alias used by run()/tradeoff()
    "ratio_deriv": ROOT.TUnfold.kRegModeNone,
    "ratio_size": ROOT.TUnfold.kRegModeNone,
    "size": ROOT.TUnfold.kRegModeSize,
    "derivative": ROOT.TUnfold.kRegModeDerivative,
    "curvature": ROOT.TUnfold.kRegModeCurvature,
}
# Methods to scan for the model-uncertainty grid (L-curve operating point).
ALL_METHODS = ["none", "ratio_curv", "ratio_deriv", "ratio_size",
               "size", "derivative", "curvature"]


def _add_ratio_conditions(unfold, prior_flat, gen_by_pt, kind):
    """Add the hand-built L rows for a ratio_* mode (per pT slice, no crossing)."""
    offset = 0
    for e in gen_by_pt:
        nb = len(e) - 1
        if kind == "ratio_curv":
            for k in range(1, nb - 1):
                j0, j1, j2 = offset + k - 1, offset + k, offset + k + 1
                m0, m1, m2 = prior_flat[j0], prior_flat[j1], prior_flat[j2]
                if min(m0, m1, m2) > 0:
                    unfold.AddRegularisationCondition(
                        j0 + 1, 1.0 / m0, j1 + 1, -2.0 / m1, j2 + 1, 1.0 / m2)
        elif kind == "ratio_deriv":
            for k in range(0, nb - 1):
                j0, j1 = offset + k, offset + k + 1
                m0, m1 = prior_flat[j0], prior_flat[j1]
                if min(m0, m1) > 0:
                    unfold.AddRegularisationCondition(j0 + 1, 1.0 / m0, j1 + 1, -1.0 / m1)
        elif kind == "ratio_size":
            for k in range(0, nb):
                j = offset + k
                if prior_flat[j] > 0:
                    unfold.AddRegularisationCondition(j + 1, 1.0 / prior_flat[j])
        offset += nb


# --------------------------------------------------------------------------- #
# pkl access
# --------------------------------------------------------------------------- #
def _load(tag):
    with open(FINE_DIR / f"minimal_rho_fine_{tag}_2018.pkl", "rb") as f:
        return pickle.load(f)


def _keys(groomed):
    g = "g" if groomed else "u"
    return (
        f"response_matrix_rho_{g}",
        f"ptjet_rhojet_{g}_reco",
        f"ptjet_rhojet_{g}_gen",
    )


def _pick_nominal(h):
    """Project a hist to its values, selecting the single nominal systematic."""
    vals = h.values()
    if vals.shape[-1] == 1:  # trailing systematic axis
        return vals[..., 0]
    return vals


def matched_4d(d, groomed):
    """(gen_pt, gen_mass, reco_pt, reco_mass) matched migration counts."""
    key = _keys(groomed)[0]
    h = d[key].project("ptgen", "mpt_gen", "ptreco", "mpt_reco", "systematic")
    return _pick_nominal(h)


def gen_2d(d, groomed):
    """(gen_pt, gen_mass) total gen counts."""
    key = _keys(groomed)[2]
    return _pick_nominal(d[key].project("ptgen", "mpt_gen", "systematic"))


def reco_2d(d, groomed):
    """(reco_pt, reco_mass) total reco counts."""
    key = _keys(groomed)[1]
    return _pick_nominal(d[key].project("ptreco", "mpt_reco", "systematic"))


# --------------------------------------------------------------------------- #
# flatten the fine 4D into the production mosaic at a chosen reported binning
# --------------------------------------------------------------------------- #
def build_inputs(m4d, g2d, r2d, edges_reco_fine, edges_gen_fine,
                 reco_by_pt, gen_by_pt):
    """Return the flattened unfolding inputs at the reported (reco/gen)_by_pt.

    mosaic        (n_reco, n_gen)  matched migration matrix
    misses_flat   (n_gen,)         gen events not reconstructed (efficiency)
    matched_reco  (n_reco,)        mosaic.sum(axis=1), the closure measurement
    reco_tot_flat (n_reco,)        total reco (for the data fake fraction)
    truth_flat    (n_gen,)         mosaic.sum(0) + misses, the prior
    """
    H, _ = reorder_to_expected(m4d, edges_reco_fine, PT_EDGES, edges_gen_fine)
    mosaic, _ = mosaic_no_padding(
        H, edges_reco_fine, edges_gen_fine, reco_by_pt, gen_by_pt
    )

    matched_gen = m4d.sum(axis=(2, 3))            # (gen_pt, gen_mass)
    misses = np.clip(g2d - matched_gen, 0.0, None)
    misses_flat = merge_mass_flat(misses.T, edges_gen_fine, gen_by_pt)

    reco_tot_flat = merge_mass_flat(r2d.T, edges_reco_fine, reco_by_pt)

    truth_flat = mosaic.sum(axis=0) + misses_flat
    return mosaic, misses_flat, mosaic.sum(axis=1), reco_tot_flat, truth_flat


# --------------------------------------------------------------------------- #
# one TUnfold solve (mirrors Unfolder._perform_unfold)
# --------------------------------------------------------------------------- #
_SOLVE_N = [0]


def unfold_one(mosaic, misses_flat, meas_flat, prior_flat,
               reco_by_pt, gen_by_pt, *, tau=None, area=True, meas_var=None,
               reg="ratio_curvature"):
    """Solve one unfold; return (y, cov, tau).

    reg: "ratio_curvature" -> the production (1/m0,-2/m1,1/m2) L rows, with the
    L-curve scan (tau=None) or a frozen tau. "none" -> DoUnfold(0.0), the plain
    least-squares projection (no regularization) -- this is the bottom-line-test
    configuration, with no tau to choose.

    meas_var: per-reco-bin variance of the measured spectrum. When given, the
    TUnfold input errors are set to sqrt(var) (the data Poisson error, scaled by
    the fake survival) so GetEmatrixTotal propagates the real data statistics.
    When None, TUnfold falls back to sqrt(content) (used for the closure /
    HERWIG bias unfolds, whose covariance is not consumed).
    """
    _SOLVE_N[0] += 1
    tag = _SOLVE_N[0]
    n_reco, n_gen = mosaic.shape

    truth_root = ROOT.TUnfoldBinning(f"truth{tag}")
    reco_root = ROOT.TUnfoldBinning(f"reco{tag}")
    ts = truth_root.AddBinning("signal")
    rp = reco_root.AddBinning("primary")
    for i, e in enumerate(gen_by_pt):
        ts.AddBinning(f"pt{i}").AddAxis("mass", len(e) - 1, array("d", e), False, False)
    for i, e in enumerate(reco_by_pt):
        rp.AddBinning(f"pt{i}").AddAxis("mass", len(e) - 1, array("d", e), False, False)

    h_meas = reco_root.CreateHistogram(f"hMeas{tag}")
    h_true = truth_root.CreateHistogram(f"hTrue{tag}")
    h_resp = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(
        truth_root, reco_root, f"hResp{tag}"
    )
    for ir in range(n_reco):
        for jt in range(n_gen):
            h_resp.SetBinContent(jt + 1, ir + 1, float(mosaic[ir, jt]))
    for jt in range(n_gen):
        h_resp.SetBinContent(jt + 1, 0, float(misses_flat[jt]))
    for idx, v in enumerate(meas_flat, 1):
        h_meas.SetBinContent(idx, float(v))
        if meas_var is not None:
            h_meas.SetBinError(idx, float(np.sqrt(max(meas_var[idx - 1], 0.0))))
    for idx, v in enumerate(prior_flat, 1):
        h_true.SetBinContent(idx, float(v))

    e_con = ROOT.TUnfold.kEConstraintArea if area else ROOT.TUnfold.kEConstraintNone
    unfold = ROOT.TUnfoldDensityOpenL(
        h_resp, ROOT.TUnfold.kHistMapOutputHoriz, REGMODE[reg],
        e_con, ROOT.TUnfoldDensity.kDensityModeBinWidth, truth_root, reco_root,
    )
    if reg in ("ratio_curv", "ratio_curvature", "ratio_deriv", "ratio_size"):
        kind = "ratio_curv" if reg == "ratio_curvature" else reg
        _add_ratio_conditions(unfold, prior_flat, gen_by_pt, kind)

    status = unfold.SetInput(h_meas)
    if status >= 10000:
        raise RuntimeError("TUnfold input had overflow/underflow")
    if reg == "none":
        unfold.DoUnfold(0.0)            # plain least-squares, no regularization
        tau = 0.0
    elif tau is None:
        ROOT.RunUnfoldLcurveScan(unfold, 40)
        tau = float(unfold.GetTau())
    else:
        unfold.DoUnfold(tau)

    h_out = unfold.GetOutput(f"out{tag}")
    em = unfold.GetEmatrixTotal(f"em{tag}", "")
    y = np.array([h_out.GetBinContent(i) for i in range(1, n_gen + 1)])
    cov = np.array(
        [[em.GetBinContent(i, j) for j in range(1, n_gen + 1)]
         for i in range(1, n_gen + 1)]
    )
    return y, cov, tau


# --------------------------------------------------------------------------- #
# rebin a fine gen vector/covariance down to the coarse reported bins
# --------------------------------------------------------------------------- #
def rebin_matrix(fine_by_pt, coarse_by_pt, tol=1e-9):
    """Binary (n_coarse x n_fine) summation matrix, block-diagonal over pT."""
    A = np.zeros((sum(len(e) - 1 for e in coarse_by_pt),
                  sum(len(e) - 1 for e in fine_by_pt)))
    r0 = c0 = 0
    for fe, ce in zip(fine_by_pt, coarse_by_pt):
        fe = np.asarray(fe, float)
        ce = np.asarray(ce, float)
        # position of each coarse edge in the fine edge list
        pos = []
        for edge in ce:
            m = np.isclose(fe, edge, atol=tol)
            if not m.any():
                raise ValueError(f"coarse edge {edge} not in fine edges")
            pos.append(int(np.flatnonzero(m)[0]))
        for k in range(len(ce) - 1):
            for j in range(pos[k], pos[k + 1]):
                A[r0 + k, c0 + j] = 1.0
        r0 += len(ce) - 1
        c0 += len(fe) - 1
    return A


# --------------------------------------------------------------------------- #
# per-pT normalization to a density (matches plot_herwig_bias_test / unfolded)
# --------------------------------------------------------------------------- #
def densify(flat, by_pt, cov=None):
    """Normalize each pT slice to 1/binwidth/sum; propagate cov if given."""
    out_val, out_err = [], []
    offset = 0
    Jblocks = []
    for e in by_pt:
        e = np.asarray(e, float)
        n = len(e) - 1
        w = np.diff(e)
        seg = flat[offset:offset + n]
        s = seg.sum()
        dens = seg / w / s if s > 0 else np.zeros(n)
        out_val.append(dens)
        if cov is not None:
            # d(seg_i/s)/d(seg_j) = (delta_ij - seg_i/s)/s ; then /w_i
            J = (np.eye(n) - np.outer(seg, np.ones(n)) / s) / s
            J = J / w[:, None]
            Jblocks.append(J)
        offset += n
    val = np.concatenate(out_val)
    if cov is None:
        return val, None
    from scipy.linalg import block_diag
    J = block_diag(*Jblocks)
    cov_d = J @ cov @ J.T
    err = np.sqrt(np.clip(np.diag(cov_d), 0.0, None))
    return val, err


def unflatten(flat, by_pt):
    out, off = [], 0
    for e in by_pt:
        n = len(e) - 1
        out.append(np.asarray(flat[off:off + n], float))
        off += n
    return out


# --------------------------------------------------------------------------- #
# assemble the coarse + fine unfolding inputs from the fine pkls (shared)
# --------------------------------------------------------------------------- #
def prepare(groomed):
    py, hw, da = _load("pythia"), _load("herwig"), _load("data")
    rkey = _keys(groomed)[0]
    edges_reco_fine = list(py[rkey].axes["mpt_reco"].edges)
    edges_gen_fine = list(py[rkey].axes["mpt_gen"].edges)

    bins = binmod.bin_edges(groomed)
    reco_coarse = [list(e) for e in bins.reco_rho_edges_by_pt]
    gen_coarse = [list(e) for e in bins.gen_rho_edges_by_pt]
    thr = FINE_MERGE_BELOW[groomed]
    reco_fine = [_merge_below(edges_reco_fine, thr) for _ in PT_EDGES[:-1]]
    gen_fine = [_merge_below(edges_gen_fine, thr) for _ in PT_EDGES[:-1]]

    py4, pyg, pyr = matched_4d(py, groomed), gen_2d(py, groomed), reco_2d(py, groomed)
    hw4, hwg, hwr = matched_4d(hw, groomed), gen_2d(hw, groomed), reco_2d(hw, groomed)
    dar = reco_2d(da, groomed)

    def assemble(reco_by_pt, gen_by_pt):
        p_mos, p_mis, p_mreco, p_rtot, p_truth = build_inputs(
            py4, pyg, pyr, edges_reco_fine, edges_gen_fine, reco_by_pt, gen_by_pt)
        h_mos, h_mis, h_mreco, h_rtot, h_truth = build_inputs(
            hw4, hwg, hwr, edges_reco_fine, edges_gen_fine, reco_by_pt, gen_by_pt)
        data_flat = merge_mass_flat(dar.T, edges_reco_fine, reco_by_pt)

        def corrected(rtot, mreco):
            surv = 1.0 - np.divide(rtot - mreco, rtot,
                                   out=np.zeros_like(rtot), where=rtot > 0)
            return np.clip(data_flat * surv, 0.0, None), data_flat * surv**2

        # same data, fake-corrected with each generator's own fake fraction so
        # the PYTHIA- and HERWIG-matrix unfolds are each self-consistent.
        data_meas_p, data_var_p = corrected(p_rtot, p_mreco)
        data_meas_h, data_var_h = corrected(h_rtot, h_mreco)
        return dict(
            # PYTHIA response (nominal)
            p_mos=p_mos, p_mis=p_mis, p_mreco=p_mreco, p_truth=p_truth,
            data_meas=data_meas_p, data_var=data_var_p,
            # HERWIG response (model variation)
            h_mos=h_mos, h_mis=h_mis, h_mreco=h_mreco, h_truth=h_truth,
            data_meas_h=data_meas_h, data_var_h=data_var_h,
        )

    return dict(
        groomed=groomed, reco_coarse=reco_coarse, gen_coarse=gen_coarse,
        reco_fine=reco_fine, gen_fine=gen_fine,
        coarse=assemble(reco_coarse, gen_coarse),
        fine=assemble(reco_fine, gen_fine),
    )


# --------------------------------------------------------------------------- #
# main study for one grooming flag
# --------------------------------------------------------------------------- #
def run(groomed):
    gtag = "groomed" if groomed else "ungroomed"
    print(f"\n================ {gtag.upper()} ================")
    ctx = prepare(groomed)
    reco_coarse, gen_coarse = ctx["reco_coarse"], ctx["gen_coarse"]
    reco_fine, gen_fine = ctx["reco_fine"], ctx["gen_fine"]
    coarse, fine = ctx["coarse"], ctx["fine"]
    print(f"  fine binning (merge<{FINE_MERGE_BELOW[groomed]}): "
          f"n_reco={sum(len(e)-1 for e in reco_fine)}"
          f"  n_gen={sum(len(e)-1 for e in gen_fine)}"
          f"   coarse: n_reco={sum(len(e)-1 for e in reco_coarse)}"
          f"  n_gen={sum(len(e)-1 for e in gen_coarse)}")

    # ---- DIRECT COARSE (production-equivalent baseline) -------------------- #
    y_c, cov_c, tau_c = unfold_one(
        coarse["p_mos"], coarse["p_mis"], coarse["data_meas"], coarse["p_truth"],
        reco_coarse, gen_coarse, meas_var=coarse["data_var"])
    print(f"[coarse] tau = {tau_c:.4g}  (n_gen={len(y_c)})")
    cl_c, _, _ = unfold_one(
        coarse["p_mos"], coarse["p_mis"], coarse["p_mreco"], coarse["p_truth"],
        reco_coarse, gen_coarse, tau=tau_c)
    hb_c, _, _ = unfold_one(
        coarse["p_mos"], coarse["p_mis"], coarse["h_mreco"], coarse["p_truth"],
        reco_coarse, gen_coarse, tau=tau_c)

    # ---- FINE then REBIN -------------------------------------------------- #
    y_f, cov_f, tau_f = unfold_one(
        fine["p_mos"], fine["p_mis"], fine["data_meas"], fine["p_truth"],
        reco_fine, gen_fine, meas_var=fine["data_var"])
    print(f"[fine]   tau = {tau_f:.4g}  (n_gen={len(y_f)})")
    cl_f, _, _ = unfold_one(
        fine["p_mos"], fine["p_mis"], fine["p_mreco"], fine["p_truth"],
        reco_fine, gen_fine, tau=tau_f)
    hb_f, _, _ = unfold_one(
        fine["p_mos"], fine["p_mis"], fine["h_mreco"], fine["p_truth"],
        reco_fine, gen_fine, tau=tau_f)

    A = rebin_matrix(gen_fine, gen_coarse)
    y_fr = A @ y_f
    cov_fr = A @ cov_f @ A.T
    cl_fr = A @ cl_f
    hb_fr = A @ hb_f
    truth_coarse = coarse["p_truth"]
    htruth_coarse = coarse["h_truth"]

    # ---- normalize everything to densities -------------------------------- #
    d_yc, e_yc = densify(y_c, gen_coarse, cov_c)
    d_yfr, e_yfr = densify(y_fr, gen_coarse, cov_fr)
    d_truth, _ = densify(truth_coarse, gen_coarse)
    d_htruth, _ = densify(htruth_coarse, gen_coarse)
    d_clc, _ = densify(cl_c, gen_coarse)
    d_clfr, _ = densify(cl_fr, gen_coarse)
    d_hbc, _ = densify(hb_c, gen_coarse)
    d_hbfr, _ = densify(hb_fr, gen_coarse)

    res = dict(
        gtag=gtag, gen_coarse=gen_coarse, tau_c=tau_c, tau_f=tau_f,
        d_yc=d_yc, e_yc=e_yc, d_yfr=d_yfr, e_yfr=e_yfr,
        d_truth=d_truth, d_htruth=d_htruth,
        d_clc=d_clc, d_clfr=d_clfr, d_hbc=d_hbc, d_hbfr=d_hbfr,
    )
    _report(res)
    _plot(res)
    return res


def _report(r):
    """Averages over the *physics* bins (all but the merged low-rho tail).

    The first bin of each pT slice is the merged [-10, threshold] tail, which is
    IDENTICAL in both paths (both collapse below the fine threshold) and is a
    noisy catch-all whose central value can go negative -- so the fine-vs-coarse
    difference lives entirely in the bulk bins, and we summarize those.
    """
    gen = r["gen_coarse"]
    print(f"  {'pT slice':>14}  {'<relerr> coarse':>16}  {'<relerr> fine-rb':>16}"
          f"  {'err ratio':>10}  {'closure coarse':>15}  {'closure fine-rb':>15}"
          f"  {'herwig bias C':>14}  {'herwig bias F':>14}")
    yc = unflatten(r["d_yc"], gen); efc = unflatten(r["e_yc"], gen)
    yfr = unflatten(r["d_yfr"], gen); effr = unflatten(r["e_yfr"], gen)
    tr = unflatten(r["d_truth"], gen); htr = unflatten(r["d_htruth"], gen)
    clc = unflatten(r["d_clc"], gen); clfr = unflatten(r["d_clfr"], gen)
    hbc = unflatten(r["d_hbc"], gen); hbfr = unflatten(r["d_hbfr"], gen)

    def _avg(num, den):  # mean over bulk bins, |den| guard
        sl = slice(1, None)  # drop the merged tail bin
        d = np.abs(den[sl])
        return np.mean(np.abs(num[sl]) / np.where(d > 0, d, np.nan))

    r["bulk"] = {}
    for i in range(len(gen)):
        rel_c = _avg(efc[i], yc[i])
        rel_f = _avg(effr[i], yfr[i])
        cl_c = _avg(clc[i] - tr[i], tr[i])
        cl_f = _avg(clfr[i] - tr[i], tr[i])
        hb_c = _avg(hbc[i] - htr[i], htr[i])
        hb_f = _avg(hbfr[i] - htr[i], htr[i])
        r["bulk"][i] = dict(rel_c=rel_c, rel_f=rel_f, cl_c=cl_c, cl_f=cl_f,
                            hb_c=hb_c, hb_f=hb_f)
        lab = f"{int(PT_EDGES[i])}-{int(PT_EDGES[i+1]) if i+1 < len(PT_EDGES)-1 else 999}"
        print(f"  {lab:>14}  {rel_c:>16.4f}  {rel_f:>16.4f}  {rel_f/rel_c:>10.3f}"
              f"  {cl_c:>15.4f}  {cl_f:>15.4f}  {hb_c:>14.4f}  {hb_f:>14.4f}")


def _plot(r):
    hep.style.use("CMS")
    gen = r["gen_coarse"]
    n = len(gen)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12), squeeze=False)
    yc = unflatten(r["d_yc"], gen); efc = unflatten(r["e_yc"], gen)
    yfr = unflatten(r["d_yfr"], gen); effr = unflatten(r["e_yfr"], gen)
    tr = unflatten(r["d_truth"], gen); htr = unflatten(r["d_htruth"], gen)
    hbc = unflatten(r["d_hbc"], gen); hbfr = unflatten(r["d_hbfr"], gen)
    for i in range(n):
        e = np.asarray(gen[i], float)
        c = 0.5 * (e[:-1] + e[1:])
        # row 0: unfolded data densities + stat error bars
        ax = axes[0][i]
        hep.histplot(tr[i], e, ax=ax, color="C2", ls=":", label="PYTHIA gen")
        ax.errorbar(c, yc[i], yerr=efc[i], fmt="o", color="k", ms=4,
                    label=f"coarse (tau={r['tau_c']:.2g})")
        ax.errorbar(c, yfr[i], yerr=effr[i], fmt="s", color="C3", ms=4,
                    mfc="none", label=f"fine->rebin (tau={r['tau_f']:.2g})")
        ax.set_yscale("log")
        ax.set_title(f"{int(PT_EDGES[i])}-"
                     f"{int(PT_EDGES[i+1]) if i+1 < len(PT_EDGES)-1 else 999} GeV")
        if i == 0:
            ax.legend(fontsize=10)
            ax.set_ylabel("1/N dN/dx  (data)")
        # row 1: relative stat error
        ax = axes[1][i]
        rc = np.abs(efc[i] / np.where(yc[i] > 0, yc[i], np.nan))
        rf = np.abs(effr[i] / np.where(yfr[i] > 0, yfr[i], np.nan))
        hep.histplot(rc, e, ax=ax, color="k", label="coarse")
        hep.histplot(rf, e, ax=ax, color="C3", label="fine->rebin")
        if i == 0:
            ax.legend(fontsize=10)
            ax.set_ylabel("relative stat. error")
        # row 2: HERWIG model bias |unf - herwig|/herwig
        ax = axes[2][i]
        bc = np.abs(hbc[i] - htr[i]) / np.where(htr[i] > 0, htr[i], np.nan)
        bf = np.abs(hbfr[i] - htr[i]) / np.where(htr[i] > 0, htr[i], np.nan)
        hep.histplot(bc, e, ax=ax, color="k", label="coarse")
        hep.histplot(bf, e, ax=ax, color="C3", label="fine->rebin")
        if i == 0:
            ax.legend(fontsize=10)
            ax.set_ylabel("HERWIG bias |unf-hw|/hw")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
    fig.suptitle(f"Fine-then-rebin vs direct coarse  ({r['gtag']})", fontsize=18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fine_rebin_{r['gtag']}.{ext}", dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT}/fine_rebin_{r['gtag']}.png")


def _bulk_metrics(y, cov, hb, gen_coarse, htruth_coarse):
    """Mean bulk (non-tail) stat relerr and HERWIG bias for a coarse result."""
    d_y, e_y = densify(y, gen_coarse, cov)
    d_hb, _ = densify(hb, gen_coarse)
    d_htr, _ = densify(htruth_coarse, gen_coarse)
    yy = unflatten(d_y, gen_coarse); ee = unflatten(e_y, gen_coarse)
    hh = unflatten(d_hb, gen_coarse); ht = unflatten(d_htr, gen_coarse)
    rels, bias = [], []
    for i in range(len(gen_coarse)):
        sl = slice(1, None)
        dy = np.abs(yy[i][sl])
        rels.append(np.abs(ee[i][sl]) / np.where(dy > 0, dy, np.nan))
        dh = np.abs(ht[i][sl])
        bias.append(np.abs(hh[i][sl] - ht[i][sl]) / np.where(dh > 0, dh, np.nan))
    return np.nanmean(np.concatenate(rels)), np.nanmean(np.concatenate(bias))


def tradeoff(groomed):
    """Error-bias trade-off curves: coarse tau-sweep vs fine-then-rebin tau-sweep.

    Both paths are swept over the same tau grid and reported on the same coarse
    bins (the fine result rebinned). At strong tau both converge to the prior
    shape (low error, the intrinsic PYTHIA-vs-HERWIG bias); they diverge toward
    weak tau, where the fine path's variance explodes far faster. Whichever
    curve sits lower-and-left dominates: it reaches a given model bias at smaller
    statistical error. Each path's own L-curve operating point is marked.
    """
    gtag = "groomed" if groomed else "ungroomed"
    print(f"\n---- trade-off sweep ({gtag}) ----")
    ctx = prepare(groomed)
    rc, gc = ctx["reco_coarse"], ctx["gen_coarse"]
    rf, gf = ctx["reco_fine"], ctx["gen_fine"]
    co, fi = ctx["coarse"], ctx["fine"]
    A = rebin_matrix(gf, gc)

    def curve(inp, reco_by, gen_by, rebin=None):
        rels, biases = [], []
        for t in taus:
            y, cov, _ = unfold_one(inp["p_mos"], inp["p_mis"], inp["data_meas"],
                                   inp["p_truth"], reco_by, gen_by, tau=t,
                                   meas_var=inp["data_var"])
            hb, _, _ = unfold_one(inp["p_mos"], inp["p_mis"], inp["h_mreco"],
                                  inp["p_truth"], reco_by, gen_by, tau=t)
            if rebin is not None:
                y, cov, hb = rebin @ y, rebin @ cov @ rebin.T, rebin @ hb
            rel, bias = _bulk_metrics(y, cov, hb, gc, co["h_truth"])
            rels.append(rel); biases.append(bias)
        return np.array(rels), np.array(biases)

    # L-curve operating points for each path
    yC, covC, tauC = unfold_one(co["p_mos"], co["p_mis"], co["data_meas"],
                                co["p_truth"], rc, gc, meas_var=co["data_var"])
    hbC, _, _ = unfold_one(co["p_mos"], co["p_mis"], co["h_mreco"], co["p_truth"],
                           rc, gc, tau=tauC)
    relC, biasC = _bulk_metrics(yC, covC, hbC, gc, co["h_truth"])
    yF, covF, tauF = unfold_one(fi["p_mos"], fi["p_mis"], fi["data_meas"],
                                fi["p_truth"], rf, gf, meas_var=fi["data_var"])
    hbF, _, _ = unfold_one(fi["p_mos"], fi["p_mis"], fi["h_mreco"], fi["p_truth"],
                           rf, gf, tau=tauF)
    relF, biasF = _bulk_metrics(A @ yF, A @ covF @ A.T, A @ hbF, gc, co["h_truth"])

    taus = np.logspace(-4, 1.2, 16)
    cr, cb = curve(co, rc, gc)
    frr, frb = curve(fi, rf, gf, rebin=A)
    for t, a, b, c, d in zip(taus, cr, cb, frr, frb):
        print(f"  tau={t:9.4g}  coarse(err={a:6.3f},bias={b:5.3f})  "
              f"fine-rb(err={c:7.3f},bias={d:5.3f})")
    print(f"  L-curve points: coarse(err={relC:.3f},bias={biasC:.3f},tau={tauC:.2g})  "
          f"fine(err={relF:.3f},bias={biasF:.3f},tau={tauF:.2g})")

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(cr, cb, "-o", color="k", label="direct coarse (tau sweep)")
    ax.plot(frr, frb, "-s", color="C3", label="fine→rebin (tau sweep)")
    ax.plot([relC], [biasC], "*", ms=22, color="k", mec="gold", mew=1.5,
            label=f"coarse L-curve (tau={tauC:.2g})", zorder=5)
    ax.plot([relF], [biasF], "*", ms=22, color="C3", mec="gold", mew=1.5,
            label=f"fine L-curve (tau={tauF:.2g})", zorder=5)
    ax.set_xlabel("mean bulk statistical rel. error  →  worse")
    ax.set_ylabel("mean bulk HERWIG model bias  →  worse")
    ax.set_xscale("log")
    ax.set_title(f"Error–bias trade-off ({gtag})")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fine_rebin_tradeoff_{gtag}.{ext}", dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT}/fine_rebin_tradeoff_{gtag}.png")


def model_uncertainty(groomed, reg="none"):
    """Model (response) uncertainty on DATA: PYTHIA matrix vs HERWIG matrix.

    Unfolds the *same* measured data through the nominal PYTHIA response and
    through the HERWIG response, and takes the per-bin relative difference as the
    model uncertainty band -- the production herwigUp/Down systematic. Done at
    direct coarse vs fine-then-rebin, with regularization off (reg="none", the
    bottom-line-test configuration) so nothing smooths the comparison.
    """
    gtag = "groomed" if groomed else "ungroomed"
    print(f"\n======= MODEL UNCERTAINTY ({gtag}, reg={reg}) =======")
    ctx = prepare(groomed)
    rc, gc = ctx["reco_coarse"], ctx["gen_coarse"]
    rf, gf = ctx["reco_fine"], ctx["gen_fine"]
    co, fi = ctx["coarse"], ctx["fine"]
    A = rebin_matrix(gf, gc)

    def unf_pair(inp, reco_by, gen_by):
        # tau is scanned on the data×PYTHIA unfold and FROZEN for data×HERWIG, so
        # the band isolates the response change (production herwigUp/Down rule).
        yp, covp, tau = unfold_one(inp["p_mos"], inp["p_mis"], inp["data_meas"],
                                   inp["p_truth"], reco_by, gen_by, reg=reg,
                                   tau=None, meas_var=inp["data_var"])
        yh, _, _ = unfold_one(inp["h_mos"], inp["h_mis"], inp["data_meas_h"],
                              inp["h_truth"], reco_by, gen_by, reg=reg,
                              tau=tau, meas_var=inp["data_var_h"])
        return yp, covp, yh, tau

    ycp, covcp, ych, tau_c = unf_pair(co, rc, gc)
    yfp, covfp, yfh, tau_f = unf_pair(fi, rf, gf)
    yfp, covfp, yfh = A @ yfp, A @ covfp @ A.T, A @ yfh
    print(f"  tau: coarse={tau_c:.3g}  fine={tau_f:.3g}")

    d_cp, e_cp = densify(ycp, gc, covcp)
    d_ch, _ = densify(ych, gc)
    d_fp, e_fp = densify(yfp, gc, covfp)
    d_fh, _ = densify(yfh, gc)

    def munc(p, h):
        return np.abs(p - h) / np.where(np.abs(p) > 0, np.abs(p), np.nan)
    mu_c, mu_f = munc(d_cp, d_ch), munc(d_fp, d_fh)

    cp = unflatten(d_cp, gc); ecp = unflatten(e_cp, gc)
    fp = unflatten(d_fp, gc); efp = unflatten(e_fp, gc)
    muc = unflatten(mu_c, gc); muf = unflatten(mu_f, gc)
    print(f"  {'pT slice':>12}  {'statRelErr C':>12}  {'statRelErr F':>12}"
          f"  {'modelUnc C':>11}  {'modelUnc F':>11}")
    for i in range(len(gc)):
        sl = slice(1, None)
        se_c = np.nanmean(np.abs(ecp[i][sl]) / np.where(np.abs(cp[i][sl]) > 0, np.abs(cp[i][sl]), np.nan))
        se_f = np.nanmean(np.abs(efp[i][sl]) / np.where(np.abs(fp[i][sl]) > 0, np.abs(fp[i][sl]), np.nan))
        lab = f"{int(PT_EDGES[i])}-{int(PT_EDGES[i+1]) if i+1 < len(PT_EDGES)-1 else 999}"
        print(f"  {lab:>12}  {se_c:>12.3f}  {se_f:>12.3f}"
              f"  {np.nanmean(muc[i][sl]):>11.3f}  {np.nanmean(muf[i][sl]):>11.3f}")

    hep.style.use("CMS")
    n = len(gc)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)
    for i in range(n):
        e = np.asarray(gc[i], float)
        c = 0.5 * (e[:-1] + e[1:])
        ax = axes[0][i]
        ax.errorbar(c, cp[i], yerr=ecp[i], fmt="o", color="k", ms=4,
                    label="coarse: data×PYTHIA")
        hep.histplot(unflatten(d_ch, gc)[i], e, ax=ax, color="C0", ls="--",
                     label="coarse: data×HERWIG")
        ax.errorbar(c, fp[i], yerr=efp[i], fmt="s", color="C3", ms=4, mfc="none",
                    label="fine→rb: data×PYTHIA")
        hep.histplot(unflatten(d_fh, gc)[i], e, ax=ax, color="C1", ls=":",
                     label="fine→rb: data×HERWIG")
        ax.set_yscale("log")
        ax.set_title(f"{int(PT_EDGES[i])}-"
                     f"{int(PT_EDGES[i+1]) if i+1 < len(PT_EDGES)-1 else 999} GeV")
        if i == 0:
            ax.legend(fontsize=9)
            ax.set_ylabel("unfolded data density")
        ax = axes[1][i]
        hep.histplot(muc[i], e, ax=ax, color="k", label="coarse")
        hep.histplot(muf[i], e, ax=ax, color="C3", label="fine→rebin")
        if i == 0:
            ax.legend(fontsize=10)
            ax.set_ylabel("model unc. |PYTHIA−HERWIG|/PYTHIA")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
    fig.suptitle(f"Model uncertainty: data through PYTHIA vs HERWIG matrix  "
                 f"({gtag}, reg={reg}, tau_c={tau_c:.2g}/tau_f={tau_f:.2g})",
                 fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fine_rebin_modelunc_{gtag}_{reg}.{ext}", dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT}/fine_rebin_modelunc_{gtag}_{reg}.png")
    return {i: (np.nanmean(muc[i][1:]), np.nanmean(muf[i][1:]))
            for i in range(len(gc))}


def model_uncertainty_grid():
    """Run the data PYTHIA-vs-HERWIG model uncertainty for every reg method."""
    summary = {}
    for groomed in (False, True):
        gtag = "groomed" if groomed else "ungroomed"
        for method in ALL_METHODS:
            try:
                summary[(gtag, method)] = model_uncertainty(groomed, reg=method)
            except Exception as exc:  # a scan can fail on the fine ill-conditioning
                print(f"  !! {gtag}/{method} failed: {exc}")
                summary[(gtag, method)] = None
    # overview table: mean bulk model uncertainty (coarse vs fine→rebin), pT-averaged
    print("\n================ MODEL-UNC SUMMARY (mean over pT & bulk bins) ================")
    print(f"  {'channel':>10}  {'method':>13}  {'modelUnc coarse':>16}  "
          f"{'modelUnc fine→rb':>17}  {'fine/coarse':>11}")
    for (gtag, method), res in summary.items():
        if res is None:
            print(f"  {gtag:>10}  {method:>13}  {'(failed)':>16}")
            continue
        mc = np.nanmean([res[i][0] for i in res])
        mf = np.nanmean([res[i][1] for i in res])
        print(f"  {gtag:>10}  {method:>13}  {mc:>16.3f}  {mf:>17.3f}  {mf/mc:>11.2f}")
    _plot_modelunc_summary(summary)
    return summary


def _plot_modelunc_summary(summary):
    """Grouped bar chart: mean model uncertainty coarse vs fine→rebin per method."""
    hep.style.use("CMS")
    for gtag in ("groomed", "ungroomed"):
        methods = [m for m in ALL_METHODS if summary.get((gtag, m)) is not None]
        mc = [np.nanmean([summary[(gtag, m)][i][0] for i in summary[(gtag, m)]])
              for m in methods]
        mf = [np.nanmean([summary[(gtag, m)][i][1] for i in summary[(gtag, m)]])
              for m in methods]
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(x - 0.2, mc, 0.4, label="direct coarse", color="k")
        ax.bar(x + 0.2, mf, 0.4, label="fine→rebin", color="C3")
        for xi, (a, b) in enumerate(zip(mc, mf)):
            ax.annotate(f"{b/a:.1f}×", (xi, max(a, b)), ha="center",
                        va="bottom", fontsize=10, color="C3")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("mean |PYTHIA−HERWIG|/PYTHIA")
        ax.set_title(f"Model uncertainty per reg. method — {gtag}",
                     fontsize=15)
        ax.text(0.5, 0.92, "×labels = fine/coarse ratio", transform=ax.transAxes,
                ha="center", fontsize=10, color="C3")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fine_rebin_modelunc_summary_{gtag}.{ext}", dpi=120)
        plt.close(fig)
        print(f"  wrote {OUT}/fine_rebin_modelunc_summary_{gtag}.png")


if __name__ == "__main__":
    model_uncertainty_grid()
