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

ds_data = {
    "2016": [
        "SingleElectron_UL2016",
        "SingleMuon_UL2016",
    ],
    "2016APV": [
        "SingleElectron_UL2016APV",
        "SingleMuon_UL2016APV",
    ],
    "2017": [
        "SingleElectron_UL2017",
        "SingleMuon_UL2017",
    ],
    "2018": [
        "SingleMuon_UL2018",
        "EGamma_UL2018",
    ],
}

with open("inputs/validation/validation_backgrounds_all.pkl", "rb") as f:
    bkg = pkl.load(f)
with open("inputs/validation/validation_st_all.pkl", "rb") as f:
    st = pkl.load(f)
with open("inputs/validation/validation_data.pkl", "rb") as f:
    data = pkl.load(f)

print("Backgrounds:", bkg.keys())

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


def plot_data_mc(bkg, st, hist, field, era, data, file_id="", xmin=None, xmax = None, groomed=True, xlabel = "Undefined"):
    with open(f"inputs/validation/validation_pythia_{era}.pkl", "rb") as f:
        pythia = pkl.load(f)
    # with open(f"inputs/validation/validation_data_data_{era}.pkl", "rb") as f:
    #     data = pkl.load(f)

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

    data = data_hist[ds_data[era],...].project(field)
    pythia = pythia_hist_syst["pythia_" + era_dic[era], :, "nominal"]

    ww = bkg["ww_" + era_dic[era], :]
    wz = bkg["wz_" + era_dic[era], :]
    zz = bkg["zz_" + era_dic[era], :]
    ttjets = bkg["ttjets_" + era_dic[era], :]
    st = st_hist["ST_" + era_dic[era], :]

    #rho_edges = rho_edges_g if groomed else rho_edges_u
    samples = [ww, wz, zz, ttjets, st, pythia, data, pythia_hist_syst]
    #samples = [rebin_hist(h, field, rho_edges) for h in samples]
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
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    print("scale:", scale)
    ## When not normalizing
    scale = 1.0

    ####
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

    if xmin is not None:
        plt.xlim(xmin, xmax)
    #compute a safe y-limit: ensure upper > 0 and finite
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
        # print("1+up/total_mc:", 1 + (up / total_mc))
        # print("1-down/total_mc:", 1 - (down / total_mc))
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
    # ratio x-limits: same safeguard as above

    plt.ylim(0.5, 1.5)
    plt.xlabel(xlabel)
    plt.ylabel("Data/MC")

    output_dir = ROOT / "outputs" / "rho" / "data_mc"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"data_mc_{file_id}_{era}.pdf"
    # try:
    #     fig.tight_layout()
    # except Exception:
    #     pass
    fig.savefig(output_file,  bbox_inches = 'tight')

    plt.close(fig)

def main():
    for era in ["2016",  "2017", "2018"]:
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


def main():
    info_dict = {
        # Muons
        "eta_mupos": {
            "field": "eta",
            "xmin": -2.5,
            "xmax": 2.5,
            "xlabel": r"$\eta$ of $\mu^+$",
        },

        "pt_mupos": {
            "field": "pt",
            "xmin": 0,
            "xmax": 1000,
            "xlabel": r"$p_T$ of $\mu^+$ [GeV]",
        },
        "phi_mupos": {
            "field": "phi",
            "xmin": -3.14,
            "xmax": 3.14,
            "xlabel": r"$\phi$ of $\mu^+$",
        },

        "eta_muneg": {
            "field": "eta",
            "xmin": -2.5,
            "xmax": 2.5,
            "xlabel": r"$\eta$ of $\mu^-$",
        },
        "pt_muneg": {
            "field": "pt",
            "xmin": 0,
            "xmax": 1000,
            "xlabel": r"$p_T$ of $\mu^-$ [GeV]",
        },
        "phi_muneg": {
            "field": "phi",
            "xmin": -3.14,
            "xmax": 3.14,
            "xlabel": r"$\phi$ of $\mu^-$",
        },
        # Electrons
        "eta_elpos": {
            "field": "eta",
            "xmin": -2.5,
            "xmax": 2.5,
            "xlabel": r"$\eta$ of $e^+$",
        },
        "pt_elpos": {
            "field": "pt",
            "xmin": 0,
            "xmax": 1000,
            "xlabel": r"$p_T$ of $e^+$ [GeV]",
        },
        "phi_elpos": {
            "field": "phi",
            "xmin": -3.14,
            "xmax": 3.14,
            "xlabel": r"$\phi$ of $e^+$",
        },
        "eta_elneg": {
            "field": "eta",
            "xmin": -2.5,
            "xmax": 2.5,
            "xlabel": r"$\eta$ of $e^-$",
        },
        "pt_elneg": {
            "field": "pt",
            "xmin": 0,
            "xmax": 1000,
            "xlabel": r"$p_T$ of $e^-$ [GeV]",
        },
        "phi_elneg": {
            "field": "phi",
            "xmin": -3.14,
            "xmax": 3.14,
            "xlabel": r"$\phi$ of $e^-$",
        },
        # Jets
        "pt_jet0": {
            "field": "pt",
            "xmin": 200,
            "xmax": 1000,
            "xlabel": r"$p_T$ of leading jet [GeV]",
        },
        "y_jet0": {
            "field": "y",
            "xmin": -2.5,
            "xmax": 2.5,
            "xlabel": r"$y$ of leading jet",
        },
        "phi_jet0": {
            "field": "phi",
            "xmin": -3.14,
            "xmax": 3.14,
            "xlabel": r"$\phi$ of leading jet",
        },
        "mass_jet0": {
            "field": "mass",
            "xmin": 0,
            "xmax": 500,
            "xlabel": r"$m$ of leading jet",
        },
        "nJets": {
            "field": "n",
            "xmin": 0,
            "xmax": 5,
            "xlabel": r"Number of jets",
        },
        # Z candidate
        "pt_Z": {
            "field": "pt",
            "xmin": 90,
            "xmax": 1000,
            "xlabel": r"$p_T$ of Z candidate [GeV]",
        },
        "eta_Z": {
            "field": "eta",
            "xmin": -2.5,
            "xmax": 2.5,
            "xlabel": r"$\eta$ of Z candidate",
        },
        "mass_Z": {
            "field": "mass",
            "xmin": 70,
            "xmax": 110,
            "xlabel": r"$m$ of Z candidate",
        },
        "phi_Z": {
            "field": "phi",
            "xmin": -3.14,
            "xmax": 3.14,
            "xlabel": r"$\phi$ of Z candidate",
        },
        "ptasym": {
            "field": "frac",
            "xmin": 0,
            "xmax": 0.3,
            "xlabel": r"$p_T$ asymmetry of Z candidate and leading jet",
        },
        "dr": {
            "field": "dr",
            "xmin": 1.5,
            "xmax": 5,
            "xlabel": r"$\Delta R_{Z, Jet}$ ",
        },
        "dphi": {
            "field": "dphi",
            "xmin": 1.5,
            "xmax": 3.14,
            "xlabel": r"$|\Delta \phi_{Z, Jet}|$ ",
        },

        
    }
    # plot_data_mc(
    #     bkg,
    #     st,
    #     "eta_mupos",
    #     "eta",
    #     "2016",
    #     data,
    #     file_id="eta_mupos",
    #     xmin = -2.5,
    #     xmax = 2.5,
    #     xlabel = r"$\eta$ of leading muon"
    # )
    for era in ["2016", "2016APV", "2017", "2018"]:   #"2016APV",
        for hist, info in info_dict.items():
            plot_data_mc(
                bkg,
                st,
                hist,
                info["field"],
                era,
                data,
                file_id=hist,
                xmin=info["xmin"],
                xmax=info["xmax"],
                xlabel=info["xlabel"],
            )
    # plot_data_mc(
    #     bkg,
    #     st,
    #     "eta_mu0",
    #     "eta",
    #     "2016",
    #     file_id="eta_mu0",
    #     xmin = -2.5,
    #     xmax = 2.5,
    #     xlabel = r"$\eta$ of leading muon"
    # )
    # plot_data_mc(
    #     bkg,
    #     st,
    #     "pt_jet0",
    #     "pt",
    #     "2018",
    #     file_id="pt_jet0",
    #     xmin = 200,
    #     xmax = 1000,
    #     xlabel = r"$p_T$ of leading jet"
    # )
    # plot_data_mc(
    #     bkg,
    #     st,
    #     "pt_Z",
    #     "pt",
    #     "2018",
    #     file_id="pt_Z",
    #     xmin = 90,
    #     xmax = 1000,
    #     xlabel = r"$p_T$ of Z candidate"
    # )
    
    # plot_data_mc(
    #     bkg,
    #     st,
    #     "mass_Z",
    #     "mass",
    #     "2018",
    #     file_id="mass_Z",
    #     xmin = 70,
    #     xmax = 110,
    #     xlabel = r"$m$ of Z candidate"
    # )

    # plot_data_mc(
    #     bkg,
    #     st,
    #     "pt_mu0",
    #     "pt",
    #     "2018",
    #     file_id="pt_mu0",
    #     xmin = 0,
    #     xmax = 1000,
    #     xlabel = r"$p_T$ of leading muon"
    # )

    # plot_data_mc(
    #     bkg,
    #     st,
    #     "pt_el1",
    #     "pt",
    #     "2017",
    #     file_id="pt_el1",
    #     xmin = 0,
    #     xmax = 1000,
    #     xlabel = r"$p_T$ of subleading electron"
    # )
    # plot_data_mc(
    #     bkg,
    #     st,
    #     "eta_el1",
    #     "eta",
    #     "2016",
    #     file_id="eta_el1",
    #     xmin = -2.5,
    #     xmax = 2.5,
    #     xlabel = r"$\eta$ of subleading electron"
    # )

    # plot_data_mc(
    #     bkg,
    #     st,
    #     "ptasym",
    #     "frac",
    #     "2018",
    #     file_id="ptasym",
    #     xmin = 0,
    #     xmax = 0.3,
    #     xlabel = r"$p_T$ asymmetry of Z candidate"
    # )

    # plot_data_mc(
    #     bkg,
    #     st,
    #     "nJet",
    #     "n",
    #     "2018",
    #     file_id="nJet",
    #     xmin = 0,
    #     xmax = 5,
    #     xlabel = r"Number of jets"
    # )
    # plot_data_mc(
    #     bkg,
    #     st,
    #     ""
    # )

if __name__ == "__main__":
    main()