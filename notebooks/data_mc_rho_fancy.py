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


with open("inputs/rho/backgrounds_all.pkl", "rb") as f:
    bkg = pkl.load(f)
with open("inputs/rho/st_all.pkl", "rb") as f:
    st = pkl.load(f)


bins_g = binning.bin_edges(groomed=True)
bins_u = binning.bin_edges(groomed=False)

rho_edges_g = bins_g.reco_rho_edges_by_pt[0]
rho_edges_u = bins_u.reco_rho_edges_by_pt[0]


def compute_total_unc(pythia_hist):
    syst_list = pythia_hist.axes["systematic"]

    up = 0
    down = 0
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
    up = np.sqrt(up)
    down = np.sqrt(down)

    return up, down


def plot_data_mc(bkg, st, hist, field, era, file_id="", xmin=-4.5, groomed=True):
    with open(f"inputs/rho/jms_pythia_{era}_syst.pkl", "rb") as f:
        pythia = pkl.load(f)
    with open(f"inputs/rho/rho_data_{era}.pkl", "rb") as f:
        data = pkl.load(f)

    era_dic = {
        "2016": "UL16NanoAODv9",
        "2017": "UL17NanoAODv9",
        "2018": "UL18NanoAODv9",
        "2016APV": "UL16NanoAODAPVv9",
    }
    bkg = bkg[hist].project("dataset", field)
    st_hist = st[hist].project("dataset", field)
    pythia_hist_syst = pythia[hist].project("dataset", field, "systematic")
    data_hist = data[hist].project("dataset", field)

    data = data_hist.project(field)
    pythia = pythia_hist_syst["pythia_" + era_dic[era], :, "nominal"]

    ww = bkg["ww_" + era_dic[era], :]
    wz = bkg["wz_" + era_dic[era], :]
    zz = bkg["zz_" + era_dic[era], :]
    ttjets = bkg["ttjets_" + era_dic[era], :]
    st = st_hist["ST_" + era_dic[era], :]

    rho_edges = rho_edges_g if groomed else rho_edges_u
    samples = [ww, wz, zz, ttjets, st, pythia, data, pythia_hist_syst]
    samples = [rebin_hist(h, field, rho_edges) for h in samples]
    ww, wz, zz, ttjets, st, pythia, data, pythia_hist_syst = samples

    up, down = compute_total_unc(pythia_hist_syst)
    print("pythia sum:", pythia.sum().value)
    scale = data.sum().value / (
        pythia.sum().value
        + ww.sum().value
        + wz.sum().value
        + zz.sum().value
        + ttjets.sum().value
        + st.sum().value
    )
    print("scale:", scale)

    ww = ww * scale
    wz = wz * scale
    zz = zz * scale
    ttjets = ttjets * scale
    st = st * scale
    pythia = pythia * scale
    up = up * scale
    down = down * scale

    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    plt.sca(ax)

    hep.histplot(
        [ww, wz, zz, ttjets, st, pythia],
        label=["WW", "WZ", "ZZ", r"$t\bar{t}+\text{jets}$", "Single t", "DYJets"],
        color=["blue", "green", "orange", "violet", "pink", "red"],
        stack=True,
        histtype="fill",
    )
    total_mc = pythia.values() + ww.values() + wz.values() + zz.values() + ttjets.values() + st.values()

    hep.histplot(data, label="Data", histtype="errorbar", color="black")
    plt.stairs(
        total_mc + up,
        baseline=total_mc - down,
        edges=pythia.axes[0].edges,
        label="Total Unc.",
        hatch="///",
        edgecolor="black",
        facecolor="none",
        fill=True,
        alpha=1.0,
    )

    plt.xlim(xmin, 0)
    # compute a safe y-limit: ensure upper > 0 and finite
    upper = float(data.values().max() * 100)
    lower = 0.1
    if (not np.isfinite(upper)) or (upper <= lower):
        upper = max(1.0, lower * 10)
    plt.ylim(lower, upper)
    plt.legend(ncol=4, fontsize=17)
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("#Events")
    hep.cms.label("Internal", data=True, rlabel=str(era))

    plt.sca(rax)
    ratio = data.values() / (pythia.values() + ww.values() + wz.values() + zz.values() + ttjets.values() + st.values())
    ratio_err = np.sqrt(data.values()) / (
        pythia.values() + ww.values() + wz.values() + zz.values() + ttjets.values() + st.values()
    )
    plt.errorbar(data.axes[0].centers, ratio, yerr=ratio_err, fmt="o", color="black")
    plt.axhline(1, color="red", linestyle="--")
    print("1+up/total_mc:", 1 + (up / total_mc))
    print("1-down/total_mc:", 1 - (down / total_mc))
    plt.stairs(
        1 + np.where(total_mc != 0, up / total_mc, 0),
        baseline=1 - np.where(total_mc != 0, down / total_mc, 0),
        edges=pythia.axes[0].edges,
        label="Total Unc.",
        hatch="///",
        edgecolor="black",
        facecolor="none",
        fill=True,
        alpha=1.0,
    )
    plt.xlim(xmin, 0)
    plt.ylim(0.5, 1.5)
    plt.xlabel(r"$\log_{10}(\rho^2)$, Groomed" if groomed else r"$\log_{10}(\rho^2)$, Ungroomed")
    plt.ylabel("Data/MC")

    output_dir = ROOT / "outputs" / "rho" / "data_mc"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"data_mc_{file_id}_{era}.pdf"
    fig.savefig(output_file)
    plt.close(fig)


def main():
    for era in ["2016", "2016APV", "2017", "2018"]:
        plot_data_mc(bkg, st, "ptjet_rhojet_g_reco", "mpt_reco", era, file_id="rho_g", groomed=True)

    for era in ["2016", "2016APV", "2017", "2018"]:
        plot_data_mc(
            bkg,
            st,
            "ptjet_rhojet_u_reco",
            "mpt_reco",
            era,
            file_id="rho_u",
            groomed=False,
            xmin=-2.5,
        )


if __name__ == "__main__":
    main()
