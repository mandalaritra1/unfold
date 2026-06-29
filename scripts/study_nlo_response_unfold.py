#!/usr/bin/env python3
"""Unfold the SAME 2018+Run2 data through the LO (Pythia) vs NLO (amcatnloFXFX,
PtZ-stitched) response and compare. Tests the response/prior-generator dependence
of the unfolded rho spectrum. Jacobian stat propagation (official AN stat mode);
no detector systematics.

LO result: the full Unfolder init on the original_jacobian spec.
NLO result: the SAME unfolder, re-unfolding the same data through the NLO response
matrix (built from ~/Downloads/minimal_rho_nlo_ptz_all.pkl, dataset axis summed),
then _compute_input_stat_unc_from_covariance + _normalize_result (jacobian stat).
"""
import os
import pickle
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, RHO_ORIGINAL_SPEC, _with_jacobian_stat

hep.style.use(hep.style.CMS)
OUT = REPO / "outputs/zjet/rho/nlo_vs_lo_response"
OUT.mkdir(parents=True, exist_ok=True)
NLO_PKL = Path.home() / "Downloads" / "minimal_rho_nlo_ptz_all.pkl"


def nlo_hists_dataset_summed():
    o = pickle.load(open(NLO_PKL, "rb"))
    keys = ["response_matrix_rho_u", "response_matrix_rho_g",
            "ptjet_rhojet_u_reco", "ptjet_rhojet_g_reco",
            "ptjet_rhojet_u_gen", "ptjet_rhojet_g_gen"]
    return {k: (o[k][{"dataset": sum}] if "dataset" in [a.name for a in o[k].axes] else o[k])
            for k in keys}


def data_measured(uf, hist_key):
    d = pickle.load(open(uf.spec.input_dir + uf.spec.data_file, "rb"))
    return uf._flatten_prepared_2d(
        uf._select_nominal_histogram(d[hist_key["reco"]]),
        uf.edges, uf.reco_edges_by_pt, ("ptreco", uf.reco_axis))


def build_nlo_inputs(uf, dnlo, hist_key):
    _, mosaic = uf._prepared_response_mosaic(dnlo[hist_key["response"]], "nominal")
    matched_reco = mosaic.sum(axis=1)
    matched_gen = mosaic.sum(axis=0)
    gen_flat, _ = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(dnlo[hist_key["gen"]]),
        uf.edges_gen, uf.gen_edges_by_pt, ("ptgen", uf.gen_axis))
    reco_flat, _ = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(dnlo[hist_key["reco"]]),
        uf.edges, uf.reco_edges_by_pt, ("ptreco", uf.reco_axis))
    misses = gen_flat - matched_gen
    fakes = reco_flat - matched_reco
    fake_fraction = uf._compute_fake_fraction(fakes, matched_reco)
    truth = matched_gen + misses
    return mosaic, misses, fake_fraction, truth


def extract(nr, reco_edges_by_pt):
    out = []
    for i, r in enumerate(nr):
        pt = np.asarray(r["pt_bin"], float)
        out.append(dict(
            pt_lo=pt[0], pt_hi=pt[1],
            gen_edges=np.asarray(r["mgen_edges"], float),
            reco_edges=np.asarray(reco_edges_by_pt[i], float),
            unfolded=np.asarray(r["unfolded"], float),
            stat=np.asarray(r["stat_unc"], float),
            true=np.asarray(r["true"], float),
            measured=np.asarray(r["measured"], float),
            reco_mc=np.asarray(r["reco_mc"], float),
        ))
    return out


BL = OUT / "bottomline"


def _stash_bottomline(uf, resp, mode):
    """Move the freshly-written bottom_line/ + summary/ PDFs to BL/<resp>-tagged names."""
    odir = Path(uf.spec.output_dir)
    BL.mkdir(parents=True, exist_ok=True)
    for p in sorted((odir / "bottom_line").glob(f"bottom_line_{mode}_*.pdf")):
        shutil.move(str(p), BL / f"bl_{resp}_{p.name.replace('bottom_line_', '')}")
    summ = odir / "summary" / f"bottom_line_chi2_summary_{mode}.pdf"
    if summ.exists():
        shutil.move(str(summ), BL / f"bl_chi2_{resp}_{mode}.pdf")


def run_mode(groomed):
    mode = "groomed" if groomed else "ungroomed"
    spec = replace(_with_jacobian_stat(RHO_ORIGINAL_SPEC),
                   output_dir="outputs/zjet/rho/nlo_compare/")
    uf = Unfolder(spec, groomed, do_syst=False, compute_jackknife_stat=False,
                  cms_label="Internal")
    reco_edges = uf.reco_edges_by_pt
    lo = extract(deepcopy(uf.normalized_results), reco_edges)   # LO (official init)
    uf.plot_bottom_line(show=False)                          # LO bottom-line test
    uf.plot_bottom_line_chi2_summary(show=False)
    lo_bl = uf.bottom_line_test_by_pt()[0]
    _stash_bottomline(uf, "lo", mode)

    hist_key = spec.hist_keys_groomed if groomed else spec.hist_keys_ungroomed
    dnlo = nlo_hists_dataset_summed()
    mosaic, misses, fake_fraction, truth = build_nlo_inputs(uf, dnlo, hist_key)
    meas, meas_var = data_measured(uf, hist_key)            # SAME data
    uf.misses_2d_dict = getattr(uf, "misses_2d_dict", {})
    uf.misses_2d_dict["nominal"] = misses
    uf.fake_fraction_2d = fake_fraction
    uf._perform_unfold(systematic="nominal", resp_np=mosaic, meas_flat=meas,
                       meas_var=meas_var, true_flat_override=truth)
    uf._compute_input_stat_unc_from_covariance()
    uf._normalize_result()
    uf._compute_total_systematic()                          # attaches stat_unc keys
    nlo = extract(uf.normalized_results, reco_edges)        # NLO response
    uf.plot_bottom_line(show=False)                          # NLO bottom-line test
    uf.plot_bottom_line_chi2_summary(show=False)
    nlo_bl = uf.bottom_line_test_by_pt()[0]
    _stash_bottomline(uf, "nlo", mode)

    for resp, rows in (("LO", lo_bl), ("NLO", nlo_bl)):
        chis = [(round(r["smeared"]["chi2"], 1), round(r["unfolded"]["chi2"], 1)) for r in rows]
        print(f"  [{mode} {resp}] chi2 (smeared, unfold) per pT: {chis}")
    return mode, lo, nlo, lo_bl, nlo_bl


def bl_chart(mode, lo_bl, nlo_bl):
    """Clean LO-vs-NLO bottom-line bar chart (raw chi2, data-stat only)."""
    rows, nrow = lo_bl, nlo_bl                              # already the reported pT bins
    labels = [f"{r['pt_lo']:.0f}-{r['pt_hi']:.0f}" if r["pt_hi"] else f">{r['pt_lo']:.0f}"
              for r in rows]
    x = np.arange(len(rows)); w = 0.2
    series = [
        (-1.5*w, [r["smeared"]["chi2"] for r in rows], "#f4a6ad", r"LO resp: $\chi^2_{\rm smeared}$"),
        (-0.5*w, [r["unfolded"]["chi2"] for r in rows], "#e42536", r"LO resp: $\chi^2_{\rm unfold}$"),
        ( 0.5*w, [r["smeared"]["chi2"] for r in nrow], "#a6c8ec", r"NLO resp: $\chi^2_{\rm smeared}$"),
        ( 1.5*w, [r["unfolded"]["chi2"] for r in nrow], "#3f90da", r"NLO resp: $\chi^2_{\rm unfold}$"),
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    for off, vals, col, lab in series:
        ax.bar(x + off, vals, w, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel(r"$p_T$ slice (GeV)")
    ax.set_ylabel(r"$\chi^2$ vs gen truth (data-stat only)")
    ax.legend(loc="upper right", fontsize=14, frameon=False, ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    # PASS/FAIL marker per response per pT (unfold <= smeared)
    for off_u, off_s, src in ((-0.5*w, -1.5*w, rows), (1.5*w, 0.5*w, nrow)):
        for j, r in enumerate(src):
            ok = r["unfolded"]["chi2"] <= r["smeared"]["chi2"]
            ax.text(x[j] + off_u, r["unfolded"]["chi2"] * 1.3,
                    "PASS" if ok else "FAIL", ha="center", fontsize=10,
                    color=("#2a7" if ok else "#c00"), weight="bold", rotation=90)
    hep.cms.label("Preliminary", data=True, loc=0, rlabel="138 fb$^{-1}$ (13 TeV)", ax=ax)
    ax.text(0.02, 0.95, f"{mode.capitalize()} — bottom-line  ($\\chi^2_{{\\rm unfold}}\\leq\\chi^2_{{\\rm smeared}}$)",
            transform=ax.transAxes, fontsize=15, va="top")
    out = OUT / f"bl_summary_lo_vs_nlo_{mode}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def plot(mode, lo, nlo, groomed):
    xlo = -4.5 if groomed else -2.5
    for i in range(len(lo)):
        L, N = lo[i], nlo[i]
        pt_lo, pt_hi, edges = L["pt_lo"], L["pt_hi"], L["gen_edges"]
        u_lo, s_lo, t_lo = L["unfolded"], L["stat"], L["true"]
        u_nlo, s_nlo, t_nlo = N["unfolded"], N["stat"], N["true"]
        if pt_hi <= 200:
            continue
        cx = 0.5 * (edges[:-1] + edges[1:])
        ptlab = (f"{pt_lo:.0f} < $p_T$ < {pt_hi:.0f} GeV" if pt_hi < 9000
                 else f"$p_T$ > {pt_lo:.0f} GeV")

        fig = plt.figure(figsize=(10, 11))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
        ax = fig.add_subplot(gs[0]); rax = fig.add_subplot(gs[1], sharex=ax)
        ax.step(edges, np.append(t_lo, t_lo[-1]), where="post", color="#e42536",
                lw=1.1, ls=":", alpha=0.6, label="LO gen (prior)")
        ax.step(edges, np.append(t_nlo, t_nlo[-1]), where="post", color="#3f90da",
                lw=1.1, ls=":", alpha=0.6, label="NLO gen (prior)")
        ax.errorbar(cx, u_lo, yerr=s_lo, fmt="o", color="#e42536", ms=6,
                    label="Data unfolded / LO resp.")
        ax.errorbar(cx, u_nlo, yerr=s_nlo, fmt="s", color="#3f90da", ms=6,
                    mfc="none", label="Data unfolded / NLO resp.")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(u_lo > 0, u_nlo / u_lo, np.nan)
            rerr = np.abs(ratio) * np.sqrt((s_nlo / np.where(u_nlo>0,u_nlo,np.nan))**2
                                           + (s_lo / np.where(u_lo>0,u_lo,np.nan))**2)
        rax.errorbar(cx, ratio, yerr=rerr, fmt="o", color="black", ms=5)
        rax.axhline(1.0, color="gray", ls="--")
        ax.set_ylim(0, np.nanmax(np.concatenate([u_lo, u_nlo])) * 1.5)
        ax.set_xlim(xlo, 0)
        ax.set_ylabel(r"$(1/N)\,dN/d\rho$")
        ax.legend(loc="upper left", fontsize=15, frameon=False,
                  title=mode.capitalize(), title_fontsize=16)
        hep.cms.label("Preliminary", data=True, loc=0,
                      rlabel="138 fb$^{-1}$ (13 TeV)", ax=ax)
        ax.text(0.97, 0.93, ptlab, ha="right", va="top", transform=ax.transAxes, fontsize=17)
        plt.setp(ax.get_xticklabels(), visible=False)
        rax.set_ylim(0.85, 1.15); rax.set_xlim(xlo, 0)
        rax.set_ylabel("NLO / LO", fontsize=17)
        rax.set_xlabel(r"$\rho = 2\log_{10}(m/(p_T R))$")
        out = OUT / f"nlo_vs_lo_{mode}_pt{i}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


def bl_ratio(mode, lo, nlo, groomed):
    """Bottom-line ratio plots: reco data/MC vs gen unfolded/truth, LO and NLO."""
    xlo = -4.5 if groomed else -2.5
    for i in range(len(lo)):
        L, N = lo[i], nlo[i]
        if L["pt_hi"] <= 200:
            continue
        ptlab = (f"{L['pt_lo']:.0f} < $p_T$ < {L['pt_hi']:.0f} GeV" if L["pt_hi"] < 9000
                 else f"$p_T$ > {L['pt_lo']:.0f} GeV")
        ge = L["gen_edges"]; gx = 0.5 * (ge[:-1] + ge[1:])
        re = L["reco_edges"]; rx = 0.5 * (re[:-1] + re[1:])

        def safe(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan, float), where=b != 0)

        def reco_ratio_on_gen(meas, reco_mc):
            # densities -> per-bin fractions, sum reco pairs into gen bins, take ratio
            wr = np.diff(re)
            idx = np.searchsorted(np.round(re, 4), np.round(ge, 4))
            mf = np.add.reduceat(meas * wr, idx[:-1])
            gf = np.add.reduceat(reco_mc * wr, idx[:-1])
            return safe(mf, gf)

        fig, ax = plt.subplots(figsize=(10, 8.5))
        # reco-level data/MC (the "smeared" comparison), rebinned onto the gen grid
        rr_lo = reco_ratio_on_gen(L["measured"], L["reco_mc"])
        rr_nlo = reco_ratio_on_gen(N["measured"], N["reco_mc"])
        ax.step(ge, np.append(rr_lo, rr_lo[-1]), where="post",
                color="#e42536", lw=1.6, alpha=0.45, label="reco data/MC (LO)")
        ax.step(ge, np.append(rr_nlo, rr_nlo[-1]), where="post",
                color="#3f90da", lw=1.6, alpha=0.45, label="reco data/MC (NLO)")
        # gen-level unfolded/truth (the bottom-line quantity) as points + stat
        r_lo = safe(L["unfolded"], L["true"]); e_lo = safe(L["stat"], L["true"])
        r_nlo = safe(N["unfolded"], N["true"]); e_nlo = safe(N["stat"], N["true"])
        ax.errorbar(gx, r_lo, yerr=e_lo, fmt="o", color="#e42536", ms=6,
                    label="unfolded/truth (LO resp.)")
        ax.errorbar(gx, r_nlo, yerr=e_nlo, fmt="s", color="#3f90da", ms=6, mfc="none",
                    label="unfolded/truth (NLO resp.)")
        ax.axhline(1.0, color="gray", ls="--")
        ax.set_xlim(xlo, 0); ax.set_ylim(0.5, 1.7)
        ax.set_xlabel(r"$\rho = 2\log_{10}(m/(p_T R))$")
        ax.set_ylabel("data / MC  (reco)   ·   unfolded / truth  (gen)")
        ax.legend(loc="upper left", fontsize=13, frameon=False,
                  title=mode.capitalize(), title_fontsize=15)
        hep.cms.label("Preliminary", data=True, loc=0, rlabel="138 fb$^{-1}$ (13 TeV)", ax=ax)
        ax.text(0.97, 0.95, ptlab, ha="right", va="top", transform=ax.transAxes, fontsize=16)
        out = OUT / f"bl_ratio_{mode}_pt{i}.png"
        fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    for groomed in (False, True):
        mode, lo, nlo, lo_bl, nlo_bl = run_mode(groomed)
        plot(mode, lo, nlo, groomed)
        bl_chart(mode, lo_bl, nlo_bl)
        bl_ratio(mode, lo, nlo, groomed)
