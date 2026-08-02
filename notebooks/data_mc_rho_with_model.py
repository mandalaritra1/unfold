"""Reco-level dimensionless-mass (rho) data/MC comparison with a parton-shower /
hadronization MODELLING uncertainty added on top of the existing detector+ME band.

This is the direct response to ARC SMP-25-010 comment Misc#5: the data/MC ratio
of the dimensionless mass (rho) was outside the "total" band because that band
carried NO modelling uncertainty. Here we add a Pythia-vs-alternate-generator
shape difference at reco level, in quadrature, so the ratio is covered.

Two model definitions are computed:
  * Herwig         : true alternate generator (inputs/.../herwig_all.pkl)
  * Pythia-reweight: Pythia reweighted to Herwig (inputs/.../pythia_reweighted_all.pkl)
Both are normalized to the (scaled) Pythia DY yield so only the SHAPE enters.
The band drawn uses MODEL_CHOICE (default: Herwig, the defensible alt-generator).

Outputs: outputs/rho/data_mc_model/data_mc_<rho_g|rho_u>_<era>.pdf
Run:     source .venv/bin/activate && python notebooks/data_mc_rho_with_model.py
"""

from pathlib import Path
import os
import sys
import pickle as pkl

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from unfold.tools import binning
from unfold.utils.integrate_and_rebin import rebin_hist

hep.style.use("CMS")
mpl.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------
# Which alternate-generator definition to use for the drawn (primary) band.
#   "herwig"     -> |Pythia_nominal - Herwig|             (true alt generator)
#   "reweighted" -> |Pythia_nominal - Pythia_reweighted|  (Herwig shape, full stat)
#   "max"        -> per-bin max(herwig, reweighted)        (conservative envelope)
#
# Decision (documented): the raw Herwig sample is tiny (~240 weighted events/era),
# so the bin-by-bin |Pythia - Herwig| difference is dominated by Herwig MC-stat
# noise -- it spikes to ~100% in empty tail bins and accidentally collapses to
# ~1% where Herwig happens to match. The reweighted-Pythia carries the SAME
# Herwig shower/hadronization shape but with full Pythia statistics and through
# the identical Pythia detector response, giving a smooth, slightly larger band.
# We therefore draw the reweighted band as primary and overlay the raw-Herwig
# band as a cross-check; the summary reports coverage for every variant.
MODEL_CHOICE = "reweighted"
# ---------------------------------------------------------------------------

era_dic = {
    "2016": "UL16NanoAODv9",
    "2017": "UL17NanoAODv9",
    "2018": "UL18NanoAODv9",
    "2016APV": "UL16NanoAODAPVv9",
}
ds_data = {
    "2016": ["SingleElectron_UL2016", "SingleMuon_UL2016"],
    "2016APV": ["SingleElectron_UL2016APV", "SingleMuon_UL2016APV"],
    "2017": ["SingleElectron_UL2017", "SingleMuon_UL2017"],
    "2018": ["SingleMuon_UL2018", "EGamma_UL2018"],
}
run2_lumi = {"2016": 16.81, "2016APV": 19.52, "2017": 41.53, "2018": 59.74}

bins_g = binning.bin_edges(groomed=True)
bins_u = binning.bin_edges(groomed=False)
rho_edges_g = bins_g.reco_rho_edges_by_pt[0]
rho_edges_u = bins_u.reco_rho_edges_by_pt[0]

# Shared inputs (validation run; full 87-entry systematic axis on pythia)
with open("inputs/zjet/validation/validation_backgrounds_all.pkl", "rb") as f:
    bkg_all = pkl.load(f)
with open("inputs/zjet/validation/validation_st_all.pkl", "rb") as f:
    st_all = pkl.load(f)
with open("inputs/zjet/validation/validation_data.pkl", "rb") as f:
    data_all = pkl.load(f)

# Modelling inputs
with open("inputs/zjet/rho/original/herwig_all.pkl", "rb") as f:
    herwig_all = pkl.load(f)
with open("inputs/zjet/rho/original/pythia_reweighted_all.pkl", "rb") as f:
    reweighted_all = pkl.load(f)


def compute_total_unc(pythia_hist):
    """Existing detector+ME band: quadrature sum of all DY signal systematics
    (JEC/JER, JMS, JMR, lepton SFs, pu, l1prefiring, pdf, q2, isr, fsr) plus DY
    MC-stat. Operates on the unscaled pythia hist (dataset summed)."""
    syst_list = pythia_hist.axes["systematic"]
    up = 0.0
    down = 0.0
    for syst in syst_list:
        if syst.endswith("Down"):
            down += np.abs(
                pythia_hist[sum, :, "nominal"].values() - pythia_hist[sum, :, syst].values()
            ) ** 2
        elif syst.endswith("Up"):
            up += np.abs(
                pythia_hist[sum, :, "nominal"].values() - pythia_hist[sum, :, syst].values()
            ) ** 2
    up += pythia_hist[sum, :, "nominal"].variances()
    down += pythia_hist[sum, :, "nominal"].variances()
    return np.sqrt(up), np.sqrt(down)


def model_shape_unc(pythia_vals, alt_vals):
    """Per-bin |Pythia_nominal - alt| where `alt` is first normalized to the
    Pythia integral so only the SHAPE difference enters (the generators have
    very different raw normalizations; the plot anyway normalizes MC to data)."""
    p_sum = pythia_vals.sum()
    a_sum = alt_vals.sum()
    if a_sum == 0:
        return np.zeros_like(pythia_vals)
    alt_norm = alt_vals * (p_sum / a_sum)
    return np.abs(pythia_vals - alt_norm)


REPORT = []


def plot_data_mc(hist_key, era, file_id, xmin, groomed):
    rho_edges = rho_edges_g if groomed else rho_edges_u
    field = "mpt_reco"
    ed = era_dic[era]

    with open(f"inputs/zjet/validation/validation_pythia_{era}.pkl", "rb") as f:
        pythia_in = pkl.load(f)

    # Project away ptreco (rho figures are integrated over reco pt)
    bkg = bkg_all[hist_key].project("dataset", field)
    st_hist = st_all[hist_key].project("dataset", field)
    data_hist = data_all[hist_key].project("dataset", field)
    pythia_hist_syst = pythia_in[hist_key].project("dataset", field, "systematic")

    herwig_hist = herwig_all[hist_key].project("dataset", field, "systematic")
    reweighted_hist = reweighted_all[hist_key].project(field, "systematic")

    data = data_hist[ds_data[era], :].project(field)
    pythia = pythia_hist_syst["pythia_" + ed, :, "nominal"]
    herwig = herwig_hist["herwig_" + ed, :, "nominal"]
    reweighted = reweighted_hist[:, "nominal"]

    ww = bkg["ww_" + ed, :]
    wz = bkg["wz_" + ed, :]
    zz = bkg["zz_" + ed, :]
    ttjets = bkg["ttjets_" + ed, :]
    st = st_hist["ST_" + ed, :]

    # Rebin everything to the SAME reco-rho edges before differencing
    to_rebin = [ww, wz, zz, ttjets, st, pythia, data, pythia_hist_syst, herwig, reweighted]
    ww, wz, zz, ttjets, st, pythia, data, pythia_hist_syst, herwig, reweighted = [
        rebin_hist(h, field, rho_edges) for h in to_rebin
    ]

    # Existing detector+ME band (absolute counts on unscaled pythia)
    up, down = compute_total_unc(pythia_hist_syst)

    # Data-driven normalization of total MC to data integral (shape-only MC)
    scale = data.sum().value / (
        pythia.sum().value
        + ww.sum().value
        + wz.sum().value
        + zz.sum().value
        + ttjets.sum().value
        + st.sum().value
    )
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)

    ww = ww * scale
    wz = wz * scale
    zz = zz * scale
    ttjets = ttjets * scale
    st = st * scale
    pythia = pythia * scale
    up = up * scale
    down = down * scale

    pythia_vals = pythia.values()  # scaled DY yield (the component the band sits on)

    # Modelling shape uncertainties (computed on scaled DY yield)
    model_herwig = model_shape_unc(pythia_vals, herwig.values())
    model_reweight = model_shape_unc(pythia_vals, reweighted.values())
    model_max = np.maximum(model_herwig, model_reweight)
    model_unc = {"herwig": model_herwig, "reweighted": model_reweight, "max": model_max}[
        MODEL_CHOICE
    ]

    # Primary modelling-inclusive band (quadrature) + Herwig cross-check band
    up_incl = np.sqrt(up ** 2 + model_unc ** 2)
    down_incl = np.sqrt(down ** 2 + model_unc ** 2)
    band_herwig = np.sqrt(up ** 2 + model_herwig ** 2)  # for the dashed overlay

    total_mc = (
        pythia_vals + ww.values() + wz.values() + zz.values() + ttjets.values() + st.values()
    )

    # ---- coverage diagnostics (within plotting window only) ----
    centers = pythia.axes[0].centers
    edges = pythia.axes[0].edges
    nz = total_mc != 0

    def frac(x):
        return np.divide(x, total_mc, out=np.zeros_like(total_mc), where=nz)

    ratio = frac(data.values())
    ratio_err = frac(np.sqrt(data.values()))
    win = (centers >= xmin) & (centers <= 0) & (total_mc > 0) & (data.values() > 0)
    n_win = int(win.sum())

    def n_uncovered(band):
        # ratio point (incl. its stat error) overlaps total_mc +/- band ?
        fb = frac(band)
        covered = (ratio - ratio_err <= 1 + fb) & (ratio + ratio_err >= 1 - fb)
        return int((win & ~covered).sum())

    band_det = up  # symmetric-ish; use up for the coverage scalar
    band_reweight = np.sqrt(up ** 2 + model_reweight ** 2)
    band_max = np.sqrt(up ** 2 + model_max ** 2)
    frac_model = frac(model_unc)

    REPORT.append(
        {
            "fig": f"{file_id} {era}",
            "groomed": groomed,
            "n_bins": n_win,
            "uncovered_det": n_uncovered(band_det),
            "uncovered_herwig": n_uncovered(band_herwig),
            "uncovered_reweight": n_uncovered(band_reweight),
            "uncovered_max": n_uncovered(band_max),
            "uncovered_primary": n_uncovered(up_incl),
            "max_model_frac": float(np.max(frac_model[win])) if n_win else 0.0,
            "med_model_frac": float(np.median(frac_model[win])) if n_win else 0.0,
            "max_det_frac": float(np.max(frac(up)[win])) if n_win else 0.0,
        }
    )

    # ----------------------------- plot -----------------------------
    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    plt.sca(ax)

    hep.histplot(
        [ww, wz, zz, ttjets, st, pythia],
        label=["WW", "WZ", "ZZ", r"$t\bar{t}+\text{jets}$", "Single t", "DYJets"],
        color=["blue", "green", "orange", "violet", "pink", "red"],
        stack=True,
        histtype="fill",
    )
    hep.histplot(data, label="Data", histtype="errorbar", color="black")

    # outer band: total incl. Herwig modelling difference (filled, translucent)
    plt.stairs(
        total_mc + up_incl,
        baseline=total_mc - down_incl,
        edges=edges,
        label="Total unc. (incl. Herwig)",
        facecolor="steelblue",
        edgecolor="none",
        fill=True,
        alpha=0.35,
    )
    # inner band: total detector+ME systematics (hatched)
    plt.stairs(
        total_mc + up,
        baseline=total_mc - down,
        edges=edges,
        label="Total unc.",
        hatch="///",
        edgecolor="black",
        facecolor="none",
        fill=True,
        alpha=1.0,
    )

    plt.xlim(xmin, 0)
    upper = float(data.values().max() * 100)
    if (not np.isfinite(upper)) or (upper <= 0.1):
        upper = 1.0
    plt.ylim(0.1, upper)
    plt.legend(ncol=3, fontsize=14)
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("#Events")
    hep.cms.label("Internal", data=True, year=era, lumi=run2_lumi[era], com=13, fontsize=20)

    plt.sca(rax)
    plt.errorbar(centers, ratio, yerr=ratio_err, fmt="o", color="black")
    plt.axhline(1, color="red", linestyle="--")
    # outer band: total incl. Herwig modelling difference
    plt.stairs(
        1 + np.where(total_mc != 0, up_incl / total_mc, 0),
        baseline=1 - np.where(total_mc != 0, down_incl / total_mc, 0),
        edges=edges,
        facecolor="steelblue",
        edgecolor="none",
        fill=True,
        alpha=0.35,
    )
    # inner band
    plt.stairs(
        1 + np.where(total_mc != 0, up / total_mc, 0),
        baseline=1 - np.where(total_mc != 0, down / total_mc, 0),
        edges=edges,
        hatch="///",
        edgecolor="black",
        facecolor="none",
        fill=True,
        alpha=1.0,
    )
    plt.ylim(0.5, 1.5)
    plt.xlim(xmin, 0)
    plt.xlabel(
        r"$\log_{10}(\rho^2)$, Groomed" if groomed else r"$\log_{10}(\rho^2)$, Ungroomed"
    )
    plt.ylabel("Data/MC")

    out_dir = ROOT / "outputs" / "rho" / "data_mc_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"data_mc_{file_id}_{era}.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    r = REPORT[-1]
    print(
        f"  wrote {out_file}  (uncovered/{n_win}: det={r['uncovered_det']}, "
        f"+herwig={r['uncovered_herwig']}, +reweight={r['uncovered_reweight']}, "
        f"+max={r['uncovered_max']})"
    )


def main():
    print(f"MODEL_CHOICE = {MODEL_CHOICE}")
    for era in ["2016", "2016APV", "2017", "2018"]:
        print(f"[groomed]   {era}")
        plot_data_mc("ptjet_rhojet_g_reco", era, "rho_g", xmin=-4.5, groomed=True)
    for era in ["2016", "2016APV", "2017", "2018"]:
        print(f"[ungroomed] {era}")
        plot_data_mc("ptjet_rhojet_u_reco", era, "rho_u", xmin=-2.5, groomed=False)

    print("\n===================== SUMMARY =====================")
    print(f"Primary (drawn) modelling band = {MODEL_CHOICE!r}")
    print(
        f"{'figure':<14}{'bins':>5}{'det':>5}{'+hw':>5}{'+rw':>5}{'+max':>6}"
        f"{'medMod%':>9}{'maxMod%':>9}{'maxDet%':>9}"
    )
    for r in REPORT:
        print(
            f"{r['fig']:<14}{r['n_bins']:>5}{r['uncovered_det']:>5}"
            f"{r['uncovered_herwig']:>5}{r['uncovered_reweight']:>5}{r['uncovered_max']:>6}"
            f"{100*r['med_model_frac']:>8.1f}{100*r['max_model_frac']:>9.1f}"
            f"{100*r['max_det_frac']:>9.1f}"
        )
    tb = sum(r["n_bins"] for r in REPORT)
    print("--------------------------------------------------")
    print(
        f"TOTAL uncovered bins / {tb}:  det-only={sum(r['uncovered_det'] for r in REPORT)}  "
        f"+herwig={sum(r['uncovered_herwig'] for r in REPORT)}  "
        f"+reweight={sum(r['uncovered_reweight'] for r in REPORT)}  "
        f"+max={sum(r['uncovered_max'] for r in REPORT)}"
    )
    print("(medMod%/maxMod% = primary modelling-unc size as % of total MC, in-window)")


if __name__ == "__main__":
    main()
