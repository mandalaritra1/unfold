#!/usr/bin/env python
"""Dijet rho TUnfold on the native-binning (2026-07-24) producer outputs.

First unfolding pass for the smp_jetmass_run2 dijet inputs produced AFTER the
hadronic binning alignment (185-200 pt sink kept as a buffer bin, zjet-style
groomed-rho buffer axes, 2:1 reco:gen nesting) and the full-spectrum fill
semantics. Statistical uncertainties are fully analytic:

  - input stat        : TUnfold GetEmatrixInput, with the event-clustered
                        two-jet covariance (reco_cov_rho_u/g accumulated in
                        production, V_ij = sum_e w_e^2 n_ei n_ej) fed through
                        SetInput(hist_vyy). A diagonal-V unfold is run
                        alongside to quantify what ignoring the jet pairing
                        would do to the errors.
  - response MC stat  : GetEmatrixSysUncorr.

No model / experimental systematics yet. Without a data file on the new
binning this runs the MC self-closure (input = MC reco, must reproduce MC gen
to machine precision at tau=0); pass --data when the reprocessed JetHT pickle
exists and the same machinery unfolds data unchanged.

Phase-space conventions (the pt-flow caveat):
  the gen spectrum is the FULL truth (84% of it sits below the 185 GeV pt
  floor, in pt-underflow), so fakes/misses are derived by subtraction using
  in-range sums only:
      fakes_i  = reco_i - sum_{j in-range} A_ij   (includes matched-to-
                                                   out-of-range-gen jets)
      misses_j = gen_j  - sum_{i in-range} A_ij   (includes matched-to-
                                                   out-of-range-reco jets)

Run (needs ROOT):
    source scripts/setup_root.sh
    .venv/bin/python scripts/studies/run_dijet_rho_native.py \
        --mc inputs/dijet/rho/native2018/mg_pythia8_2018.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

HIST_KEYS = {
    "u": dict(reco="ptjet_rhojet_u_reco", gen="ptjet_rhojet_u_gen",
              matrix="response_matrix_rho_u", cov="reco_cov_rho_u"),
    "g": dict(reco="ptjet_rhojet_g_reco", gen="ptjet_rhojet_g_gen",
              matrix="response_matrix_rho_g", cov="reco_cov_rho_g"),
}
GROOM_LABEL = {"u": "ungroomed", "g": "groomed"}

# Reported window: pt >= 200 (the 185-200 sink is a buffer bin) and rho above
# the shown-region floor. Bins below the floor are hidden buffer bins whose
# role is to absorb migrations; they oscillate on a prior-free tau=0 data
# unfold BY DESIGN and are never reported.
SHOWN_RHO_FLOOR = {"u": -2.5, "g": -3.5}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc", type=Path,
                    default=REPO_ROOT / "inputs/dijet/rho/native2018/mg_pythia8_2018.pkl")
    ap.add_argument("--data", type=Path, default=None,
                    help="Reprocessed JetHT pickle on the native binning. "
                         "Omitted -> MC self-closure.")
    ap.add_argument("--outdir", type=Path,
                    default=REPO_ROOT / "outputs/dijet/2018/rho/native2018")
    ap.add_argument("--year", default="2018")
    return ap.parse_args()


def _nominal_summed(out, key):
    """hist -> (values, variances) with flow, nominal, summed over datasets."""
    h = out[key][{"systematic": "nominal"}][{"dataset": sum}]
    return h.values(flow=True), h.variances(flow=True), h


def load_channel(out, groom):
    """Extract in-range numpy inputs for one grooming.

    Flow-array layout (axis: pt, rho): index 0 = underflow, -1 = overflow;
    in-range = [1:-1]. The 185-200 sink is a REAL bin (index 1 in-range 0).
    """
    k = HIST_KEYS[groom]
    reco_v, reco_w2, h_reco = _nominal_summed(out, k["reco"])
    gen_v, gen_w2, h_gen = _nominal_summed(out, k["gen"])
    mat_v, mat_w2, _ = _nominal_summed(out, k["matrix"])   # (ptreco, ptgen, rho_reco, rho_gen)
    cov_v, _, _ = _nominal_summed(out, k["cov"])           # (pt_i, rho_i, pt_j, rho_j)

    s = slice(1, -1)
    reco2d, reco2d_w2 = reco_v[s, s], reco_w2[s, s]
    gen2d, gen2d_w2 = gen_v[s, s], gen_w2[s, s]
    # matrix in-range on all four axes, reorder to (ptreco, rho_reco, ptgen, rho_gen)
    A4 = np.transpose(mat_v[s, s, s, s], (0, 2, 1, 3))
    A4_w2 = np.transpose(mat_w2[s, s, s, s], (0, 2, 1, 3))
    cov4 = cov_v[s, s, s, s]                               # already (pt_i, rho_i, pt_j, rho_j)

    n_pt, n_rr = reco2d.shape
    _, n_rg = gen2d.shape
    Nr, Ng = n_pt * n_rr, n_pt * n_rg
    A = A4.reshape(Nr, Ng)
    A_w2 = A4_w2.reshape(Nr, Ng)
    reco_flat, reco_flat_w2 = reco2d.reshape(Nr), reco2d_w2.reshape(Nr)
    gen_flat, gen_flat_w2 = gen2d.reshape(Ng), gen2d_w2.reshape(Ng)
    V = cov4.reshape(Nr, Nr)

    fakes = reco_flat - A.sum(axis=1)
    misses = gen_flat - A.sum(axis=0)
    # Disjoint event sets (a jet is matched or not) -> variances subtract.
    fakes_w2 = np.maximum(reco_flat_w2 - A_w2.sum(axis=1), 0.0)
    misses_w2 = np.maximum(gen_flat_w2 - A_w2.sum(axis=0), 0.0)

    pt_edges = np.asarray(h_reco.axes["ptreco"].edges)
    rho_edges_reco = np.asarray(h_reco.axes["mpt_reco"].edges)
    rho_edges_gen = np.asarray(h_gen.axes["mpt_gen"].edges)

    return dict(
        A=A, A_w2=A_w2, V=V,
        reco=reco_flat, reco_w2=reco_flat_w2,
        gen=gen_flat, gen_w2=gen_flat_w2,
        fakes=fakes, fakes_w2=fakes_w2,
        misses=misses, misses_w2=misses_w2,
        n_pt=n_pt, n_rr=n_rr, n_rg=n_rg,
        pt_edges=pt_edges, rho_edges_reco=rho_edges_reco,
        rho_edges_gen=rho_edges_gen,
    )


def load_data_channel(out, groom):
    """Data pickle carries only reco spectra + covariance (no gen/matrix)."""
    k = HIST_KEYS[groom]
    reco_v, reco_w2, _ = _nominal_summed(out, k["reco"])
    cov_v, _, _ = _nominal_summed(out, k["cov"])
    s = slice(1, -1)
    reco2d, reco2d_w2 = reco_v[s, s], reco_w2[s, s]
    n_pt, n_rr = reco2d.shape
    Nr = n_pt * n_rr
    return dict(
        reco=reco2d.reshape(Nr), reco_w2=reco2d_w2.reshape(Nr),
        V=cov_v[s, s, s, s].reshape(Nr, Nr),
    )


def tunfold_run(ch, input_flat, input_V, label):
    """One TUnfoldDensity pass at tau=0. Returns dict of numpy results."""
    import ROOT

    ROOT.TH1.AddDirectory(False)
    Nr, Ng = ch["A"].shape

    # Response: gen on X (kHistMapOutputHoriz), reco on Y; misses in Y-underflow.
    hA = ROOT.TH2D(f"A_{label}", "", Ng, 0.0, float(Ng), Nr, 0.0, float(Nr))
    for j in range(Ng):
        for i in range(Nr):
            hA.SetBinContent(j + 1, i + 1, ch["A"][i, j])
            hA.SetBinError(j + 1, i + 1, np.sqrt(ch["A_w2"][i, j]))
        hA.SetBinContent(j + 1, 0, ch["misses"][j])
        hA.SetBinError(j + 1, 0, np.sqrt(ch["misses_w2"][j]))

    hIn = ROOT.TH1D(f"in_{label}", "", Nr, 0.0, float(Nr))
    for i in range(Nr):
        hIn.SetBinContent(i + 1, input_flat[i])
        hIn.SetBinError(i + 1, np.sqrt(max(input_V[i, i], 0.0)))

    hV = ROOT.TH2D(f"V_{label}", "", Nr, 0.0, float(Nr), Nr, 0.0, float(Nr))
    for i in range(Nr):
        for j in range(Nr):
            hV.SetBinContent(i + 1, j + 1, input_V[i, j])

    unf = ROOT.TUnfoldDensity(
        hA,
        ROOT.TUnfold.kHistMapOutputHoriz,
        ROOT.TUnfold.kRegModeCurvature,
        ROOT.TUnfold.kEConstraintArea,
        ROOT.TUnfoldDensity.kDensityModeNone,
    )
    status = unf.SetInput(hIn, 0.0, 0.0, hV)
    unf.DoUnfold(0.0)

    def _th1_to_np(h, n):
        return np.array([h.GetBinContent(i + 1) for i in range(n)])

    def _th2_to_np(h, n):
        return np.array([[h.GetBinContent(i + 1, j + 1) for j in range(n)]
                         for i in range(n)])

    hOut = unf.GetOutput(f"out_{label}")
    hEin = unf.GetEmatrixInput(f"ein_{label}")
    hEa = unf.GetEmatrixSysUncorr(f"ea_{label}")
    res = dict(
        status=int(status),
        unfolded=_th1_to_np(hOut, Ng),
        ematrix_input=_th2_to_np(hEin, Ng),
        ematrix_response=_th2_to_np(hEa, Ng),
        chi2A=float(unf.GetChi2A()),
        ndf=int(unf.GetNdf()),
    )
    return res


def run_grooming(out_mc, groom, args, data_out=None):
    ch = load_channel(out_mc, groom)
    Nr, Ng = ch["A"].shape
    gl = GROOM_LABEL[groom]
    print(f"\n===== {gl}: Nr={Nr} ({ch['n_pt']}pt x {ch['n_rr']}rho reco), "
          f"Ng={Ng} ({ch['n_pt']}pt x {ch['n_rg']}rho gen) =====")

    neg_fakes = ch["fakes"][ch["fakes"] < -1e-6 * np.maximum(ch["reco"], 1)]
    print(f"fakes: total {ch['fakes'].sum():.4g} "
          f"({ch['fakes'].sum() / max(ch['reco'].sum(), 1e-30):.2%} of reco); "
          f"negative bins: {neg_fakes.size}")
    print(f"misses: total {ch['misses'].sum():.4g} "
          f"({ch['misses'].sum() / max(ch['gen'].sum(), 1e-30):.2%} of in-range gen)")
    # covariance sanity
    dV = np.diag(ch["V"])
    with np.errstate(divide="ignore", invalid="ignore"):
        infl = np.where(ch["reco_w2"] > 0, dV / ch["reco_w2"], 1.0)
    asym = np.abs(ch["V"] - ch["V"].T).max()
    print(f"cov: max|V-V^T| = {asym:.3g}; diag/sumw2 in "
          f"[{infl[ch['reco_w2'] > 0].min():.3f}, {infl[ch['reco_w2'] > 0].max():.3f}]")

    is_data = data_out is not None
    if is_data:
        # Multiplicative per-bin fake correction (zjet convention,
        # unfolder_core line ~2007): input = data * (1 - f), with the fake
        # fraction f = fakes/reco from MC, and the input covariance scaled
        # by the same survival factor on both indices.
        dch = load_data_channel(data_out, groom)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(ch["reco"] > 0, ch["fakes"] / ch["reco"], 0.0)
        surv = 1.0 - np.clip(f, 0.0, 1.0)
        input_flat = dch["reco"] * surv
        input_V = dch["V"] * np.outer(surv, surv)
        d_w2 = dch["reco_w2"]
        with np.errstate(divide="ignore", invalid="ignore"):
            d_infl = np.where(d_w2 > 0, np.diag(dch["V"]) / d_w2, 1.0)
        print(f"data: reco total {dch['reco'].sum():.4g}; cov diag/sumw2 in "
              f"[{d_infl[d_w2 > 0].min():.3f}, {d_infl[d_w2 > 0].max():.3f}] "
              f"(prescale weights)")
    else:
        input_flat = ch["reco"] - ch["fakes"]        # MC self-closure
        input_V = ch["V"]

    res_full = tunfold_run(ch, input_flat, input_V, f"{groom}_full")
    res_diag = tunfold_run(ch, input_flat, np.diag(np.diag(input_V)), f"{groom}_diag")

    gen = ch["gen"]
    # A gen bin fed only by misses (zero response column) is invisible to the
    # unfold; TUnfold returns 0 there at tau=0. Judge closure on constrained
    # bins and report the unconstrained ones separately.
    constrained = ch["A"].sum(axis=0) > 0
    okg = (gen > 0) & constrained
    n_uncon = int(((gen > 0) & ~constrained).sum())
    frac_uncon = gen[(gen > 0) & ~constrained].sum() / max(gen.sum(), 1e-30)
    ratio = np.where(okg, res_full["unfolded"] / np.where(okg, gen, 1.0), np.nan)
    if is_data:
        # Judge the result inside the reported window only; buffer bins and
        # the pt sink oscillate by design on a tau=0 data unfold.
        edges = ch["rho_edges_gen"]
        shown2 = np.zeros((ch["n_pt"], ch["n_rg"]), dtype=bool)
        shown2[1:, :] = True
        shown2[:, edges[:-1] < SHOWN_RHO_FLOOR[groom]] = False
        shown = shown2.reshape(-1) & okg
        sratio = res_full["unfolded"][shown] / gen[shown]
        n_neg = int((res_full["unfolded"][shown] < 0).sum())
        print(f"unfolded data / MC gen, SHOWN window "
              f"(pt>=200, rho>={SHOWN_RHO_FLOOR[groom]}, {shown.sum()} bins): "
              f"median {np.median(sratio):.3f}, "
              f"range [{sratio.min():.3f}, {sratio.max():.3f}], "
              f"negative bins {n_neg}; SetInput status={res_full['status']}")
        print(f"  (all constrained bins incl. buffers: median "
              f"{np.nanmedian(ratio):.3f} -- buffer bins oscillate by design)")
        # Shape comparison: normalize data and MC per pT slice over the shown
        # window (zjet convention), removing the per-slice normalization
        # difference; residual = pure shape.
        unf2 = res_full["unfolded"].reshape(ch["n_pt"], ch["n_rg"])
        gen2m = gen.reshape(ch["n_pt"], ch["n_rg"])
        for ipt in range(1, ch["n_pt"]):
            m = shown2[ipt]
            if not m.any() or unf2[ipt][m].sum() <= 0:
                continue
            scale = gen2m[ipt][m].sum() / unf2[ipt][m].sum()
            rr = (unf2[ipt][m] * scale) / gen2m[ipt][m]
            lo_, hi_ = ch["pt_edges"][ipt], ch["pt_edges"][ipt + 1]
            print(f"  shape (per-pT normalized) pt[{lo_:.0f},{hi_:.0f}]: "
                  f"norm {1/scale:.3f}, ratio range "
                  f"[{rr.min():.3f}, {rr.max():.3f}]")
    else:
        print(f"closure (full V): max|unf/gen-1| = "
              f"{np.nanmax(np.abs(ratio - 1)):.3e} "
              f"over {okg.sum()} constrained bins "
              f"({n_uncon} unconstrained pure-miss bins holding {frac_uncon:.2e} "
              f"of gen); SetInput status={res_full['status']}")

    err_full = np.sqrt(np.maximum(np.diag(res_full["ematrix_input"]), 0.0))
    err_diag = np.sqrt(np.maximum(np.diag(res_diag["ematrix_input"]), 0.0))
    err_resp = np.sqrt(np.maximum(np.diag(res_full["ematrix_response"]), 0.0))
    denom = np.where(okg, res_full["unfolded"], np.nan) if is_data else gen
    with np.errstate(divide="ignore", invalid="ignore"):
        err_ratio = np.where(err_diag > 0, err_full / err_diag, np.nan)
        rel_full = np.where(okg, err_full / denom, np.nan)
        rel_resp = np.where(okg, err_resp / denom, np.nan)
    print(f"input-stat err, full-V/diag-V: median {np.nanmedian(err_ratio):.3f}, "
          f"max {np.nanmax(err_ratio):.3f}")
    print(f"rel. input stat (full V): median {np.nanmedian(rel_full):.3%}; "
          f"rel. response MC-stat: median {np.nanmedian(rel_resp):.3%}")

    plot_closure(ch, res_full, err_full, err_diag, groom, args,
                 constrained=constrained, is_data=is_data)
    if is_data:
        plot_shapes(ch, res_full, err_full, groom, args)

    return dict(
        groom=groom, Nr=Nr, Ng=Ng, constrained=constrained,
        pt_edges=ch["pt_edges"], rho_edges_gen=ch["rho_edges_gen"],
        gen=gen, unfolded=res_full["unfolded"],
        ematrix_input_fullV=res_full["ematrix_input"],
        ematrix_input_diagV=res_diag["ematrix_input"],
        ematrix_response=res_full["ematrix_response"],
        fakes=ch["fakes"], misses=ch["misses"],
        cov_diag_inflation=infl,
        closure_max_dev=float(np.nanmax(np.abs(ratio - 1))),
        chi2A=res_full["chi2A"], ndf=res_full["ndf"],
    )


def plot_shapes(ch, res, err_full, groom, args):
    """Overlaid normalized shapes: MC gen vs unfolded data, per pT slice.

    Each slice's gen and unfolded-data densities (1/N dN/drho) are normalized
    over the SHOWN rho window (zjet measurement-window convention), so the
    panels compare pure shape; the per-slice normalization factors are
    reported separately in the text output.
    """
    gl = GROOM_LABEL[groom]
    n_pt, n_rg = ch["n_pt"], ch["n_rg"]
    gen2 = ch["gen"].reshape(n_pt, n_rg)
    unf2 = res["unfolded"].reshape(n_pt, n_rg)
    ef2 = err_full.reshape(n_pt, n_rg)
    edges = ch["rho_edges_gen"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    floor = SHOWN_RHO_FLOOR[groom]
    shown_rho = edges[:-1] >= floor

    ncol = 3
    nrow = int(np.ceil(n_pt / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                            sharex=True)
    axs = np.atleast_2d(axs)
    for ipt in range(n_pt):
        ax = axs[ipt // ncol, ipt % ncol]
        # normalize over the shown window (whole slice for the sink panel)
        m = shown_rho if ipt > 0 else np.ones(n_rg, dtype=bool)
        gnorm = gen2[ipt][m].sum()
        unorm = unf2[ipt][m].sum()
        if gnorm <= 0 or unorm <= 0:
            ax.axis("off")
            continue
        gden = gen2[ipt] / gnorm / widths
        uden = unf2[ipt] / unorm / widths
        uerr = ef2[ipt] / unorm / widths
        if ipt == 0:
            ax.axvspan(edges[0], edges[-1], color="0.92", zorder=0)
        else:
            ax.axvspan(edges[0], floor, color="0.92", zorder=0)
        ax.stairs(gden, edges, color="C1", lw=1.5, label="MC gen (pythia8)")
        ax.errorbar(centers, uden, yerr=uerr, fmt="o", ms=3.5, capsize=2,
                    color="k", label="unfolded JetHT 2018")
        lo, hi = ch["pt_edges"][ipt], ch["pt_edges"][ipt + 1]
        sink = " (buffer sink)" if ipt == 0 else ""
        ax.set_title(f"$p_T$ [{lo:.0f}, {hi:.0f}] GeV{sink}", fontsize=10)
        ymax = 1.25 * max(gden[m].max(), (uden + uerr)[m].max())
        ax.set_ylim(0, ymax)
        if ipt // ncol == nrow - 1:
            ax.set_xlabel(r"$\rho$")
        if ipt % ncol == 0:
            ax.set_ylabel(r"$1/N\;dN/d\rho$ (shown-window norm)")
    for k in range(n_pt, nrow * ncol):
        axs[k // ncol, k % ncol].axis("off")
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, frameon=False)
    fig.suptitle(f"Dijet {gl} rho shapes: unfolded JetHT 2018 vs MC gen, "
                 f"each pT slice normalized over the shown window",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    outdir = args.outdir / gl
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"shapes_{gl}.png", dpi=140)
    plt.close(fig)


def plot_closure(ch, res, err_full, err_diag, groom, args, constrained=None,
                 is_data=False):
    gl = GROOM_LABEL[groom]
    n_pt, n_rg = ch["n_pt"], ch["n_rg"]
    if constrained is None:
        constrained = np.ones(n_pt * n_rg, dtype=bool)
    con2 = constrained.reshape(n_pt, n_rg)
    gen2 = ch["gen"].reshape(n_pt, n_rg)
    unf2 = res["unfolded"].reshape(n_pt, n_rg)
    ef2 = err_full.reshape(n_pt, n_rg)
    ed2 = err_diag.reshape(n_pt, n_rg)
    edges = ch["rho_edges_gen"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    ncol = 3
    nrow = int(np.ceil(n_pt / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                            sharex=True)
    axs = np.atleast_2d(axs)
    for ipt in range(n_pt):
        ax = axs[ipt // ncol, ipt % ncol]
        g, u = gen2[ipt], unf2[ipt]
        ok = (g > 0) & con2[ipt]
        gsafe = np.where(ok, g, 1.0)
        rel = np.where(ok, ef2[ipt] / gsafe, np.inf)
        rel_cut = 0.5 if is_data else 0.1
        good = ok & (rel <= rel_cut)
        low = ok & (rel > rel_cut)
        ylo, yhi = (0.0, 2.0) if is_data else (0.9, 1.1)
        main_label = ("unfolded data / MC gen (full V)" if is_data
                      else "unfolded/gen (full V)")
        # shade the hidden buffer region (and the whole 185-200 sink panel):
        # bins there absorb migrations and oscillate by design, never reported
        floor = SHOWN_RHO_FLOOR[groom]
        if ipt == 0:
            ax.axvspan(edges[0], edges[-1], color="0.92", zorder=0)
        else:
            ax.axvspan(edges[0], floor, color="0.92", zorder=0)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.errorbar(centers[good], (u / gsafe)[good], yerr=(ef2[ipt] / gsafe)[good],
                    fmt="o", ms=3, capsize=2, color="C0", label=main_label)
        diag_y = (u / gsafe)[good] if is_data else np.ones(good.sum())
        ax.errorbar(centers[good], diag_y, yerr=(ed2[ipt] / gsafe)[good],
                    fmt="none", ecolor="0.6", capsize=2, alpha=0.8,
                    label="diag-V errors")
        if low.any():
            ax.plot(centers[low], np.clip((u / gsafe)[low], ylo, yhi), "o",
                    ms=4, mfc="none", mec="0.6",
                    label=f"low-stat (rel err > {rel_cut:.0%})")
        lo, hi = ch["pt_edges"][ipt], ch["pt_edges"][ipt + 1]
        sink = " (buffer sink)" if ipt == 0 else ""
        ax.set_title(f"$p_T$ [{lo:.0f}, {hi:.0f}] GeV{sink}", fontsize=10)
        ax.set_ylim(ylo, yhi)
        if ipt // ncol == nrow - 1:
            ax.set_xlabel(r"$\rho$")
        if ipt % ncol == 0:
            ax.set_ylabel("unfolded data / MC gen" if is_data
                          else "unfolded / gen")
    for k in range(n_pt, nrow * ncol):
        axs[k // ncol, k % ncol].axis("off")
    # figure-level legend below the panels (an in-axes legend overlaps points
    # in the busy low-pT panels); dedupe labels across all axes
    handles, labels = [], []
    for a in axs.flat:
        for h, lab in zip(*a.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, frameon=False)
    if is_data:
        fig.suptitle(f"Dijet {gl} rho: unfolded JetHT 2018 vs MC gen, tau=0, "
                     f"analytic stat (event-clustered input covariance)",
                     fontsize=12)
    else:
        fig.suptitle(f"Dijet {gl} rho: MC self-closure, tau=0, analytic stat "
                     f"(event-clustered input covariance)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    outdir = args.outdir / gl
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "data_result" if is_data else "closure"
    fig.savefig(outdir / f"{stem}_{gl}.png", dpi=140)
    plt.close(fig)

    # error-inflation summary: full-V vs diag-V analytic input stat
    fig, ax = plt.subplots(figsize=(7.5, 4))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ed2 > 0, ef2 / ed2, np.nan)
    for ipt in range(n_pt):
        lo, hi = ch["pt_edges"][ipt], ch["pt_edges"][ipt + 1]
        ax.plot(centers, r[ipt], marker="o", ms=3,
                label=f"[{lo:.0f}, {hi:.0f}]")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("unfolded stat err: full V / diagonal V")
    ax.set_title(f"Dijet {gl}: two-jet covariance effect on stat errors"
                 + (" (JetHT 2018)" if is_data else ""))
    ax.legend(fontsize=7, ncol=3, title="$p_T$ [GeV]")
    fig.tight_layout()
    fig.savefig(outdir / f"stat_inflation_{'data_' if is_data else ''}{gl}.png",
                dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    out_mc = pickle.load(open(args.mc, "rb"))
    data_out = pickle.load(open(args.data, "rb")) if args.data else None

    results = {}
    for groom in ("u", "g"):
        results[GROOM_LABEL[groom]] = run_grooming(out_mc, groom, args, data_out)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = "native_data_results" if args.data else "native_closure_results"
    with open(args.outdir / f"{stem}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nwrote {args.outdir}/{stem}.pkl")


if __name__ == "__main__":
    main()
