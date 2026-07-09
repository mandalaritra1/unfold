"""Publication-style plots for the dijet rho method bake-off.

Produces, per mode (groomed/ungroomed):
  - per-pt-bin overlay: unfolded data (D'Agostini) vs PYTHIA gen vs HERWIG gen,
    normalized per pt slice (shape), with stat band.
  - a method-comparison panel (D'Agostini / Tikhonov / SVD / NNLS).
  - a closure/bottom-line summary (self-closure + HERWIG-as-data).
Outputs to outputs/dijet/2018/rho/method_study/.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.dijet_methods.loader import Binning, load_native, rebin, build_problem
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.result import (
    prepare, unfold_with_unc, systematic_band, channel_paths,
)

N_ITER = 4
OUT_BY_CHANNEL = {
    "dijet": "outputs/dijet/2018/rho/method_study",
    "trijet": "outputs/trijet/2018/rho/method_study",
    "zjet": "outputs/zjet/rho/method_study",
}
PT_EDGES_BY_CHANNEL = {
    "dijet": np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0]),
    "trijet": np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0]),
    "zjet": np.array([0.0, 200.0, 290.0, 400.0, 13000.0]),
}
PT_LABELS_BY_CHANNEL = {
    "dijet": ["0-200", "200-290", "290-400", "400-570", "570-760", "760+"],
    "trijet": ["0-200", "200-290", "290-400", "400-570", "570-760", "760+"],
    "zjet": ["0-200", "200-290", "290-400", "400+"],
}

OUT = "outputs/dijet/2018/rho/method_study"
PT6 = np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0])
PT_LABELS = ["0-200", "200-290", "290-400", "400-570", "570-760", "760+"]


def native_binning(mode, channel="dijet"):
    mc_path, _, _ = channel_paths(channel)
    nat = load_native(mc_path, mode)
    return Binning(PT_EDGES_BY_CHANNEL[channel], nat.rho_edges_reco.copy(),
                   nat.rho_edges_gen.copy())


def merged_tail_binning(mode, channel="dijet", eff_min=0.5):
    """Adaptive low-rho merge: collapse the lowest gen rho bins into one until
    every reported (pT>=200) gen bin has efficiency >= eff_min. Reco keeps the
    native (finer) edges above the merge boundary. Channel/mode-agnostic.
    """
    pt = PT_EDGES_BY_CHANNEL[channel]
    mc_path, _, _ = channel_paths(channel)
    nat = load_native(mc_path, mode)
    # efficiency at native-pt + native rho
    rb = rebin(nat, Binning(pt, nat.rho_edges_reco.copy(),
                            nat.rho_edges_gen.copy()))
    prob = build_problem(rb)
    n_rho_gen = len(nat.rho_edges_gen) - 1
    eff = prob.eff.reshape(len(pt) - 1, n_rho_gen)
    # min efficiency over reported pt bins (index >= 1) per gen rho bin
    min_eff = eff[1:].min(axis=0)
    # lowest gen edge index such that ALL higher bins satisfy eff_min
    boundary = 0
    for j in range(n_rho_gen):
        if np.all(min_eff[j:] >= eff_min):
            boundary = j
            break
    gen_edges = np.concatenate([[nat.rho_edges_gen[0]],
                                nat.rho_edges_gen[boundary + 1:]])
    low = gen_edges[1]   # first edge above the merged tail
    reco_edges = np.concatenate([[nat.rho_edges_reco[0]],
                                 nat.rho_edges_reco[nat.rho_edges_reco >= low]])
    return Binning(pt, reco_edges, gen_edges)


def _slice(vec, prob, ptg):
    m = prob.gen_pt_idx == ptg
    return vec[m]


def _gen_centers(binning):
    e = binning.rho_edges_gen
    return 0.5 * (e[:-1] + e[1:]), e


def _norm_shape(y, edges):
    w = np.diff(edges)
    dens = y / w
    area = np.sum(y)
    return dens / area if area > 0 else dens, w


def plot_overlay(mode, binning, tag, n_iter=N_ITER, channel="dijet", linear=False):
    OUT = OUT_BY_CHANNEL[channel]
    prep = prepare(mode, binning, channel)
    prob = prep.prob
    pyt = prob.gen
    her = prep.her_truth

    central, stat, cov, toys = unfold_with_unc(
        prep, M.dagostini, n_iter=n_iter, n_toys=200, response_toys=True)
    _, syst, _ = systematic_band(prep, M.dagostini, n_iter=n_iter)
    std = np.sqrt(stat**2 + syst**2)   # total stat (+) syst

    centers, edges = _gen_centers(binning)
    nptg = len(binning.pt_edges) - 1
    report = list(range(1, nptg))
    fig, axes = plt.subplots(2, len(report), figsize=(4 * len(report), 7),
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for col, ptg in enumerate(report):
        ax = axes[0][col]
        rax = axes[1][col]
        u = _slice(central, prob, ptg)
        us = _slice(std, prob, ptg)
        py = _slice(pyt, prob, ptg)
        # per-slice shape normalization
        un, w = _norm_shape(u, edges)
        usn = us / w / max(np.sum(u), 1e-30)
        pyn, _ = _norm_shape(py, edges)

        ax.errorbar(centers, un, yerr=usn, fmt="o", ms=4, color="k",
                    label="Unfolded data", zorder=5)
        ax.step(centers, pyn, where="mid", color="tab:blue", lw=1.6,
                label="PYTHIA8 gen")
        hwn = None
        if her is not None:
            hwn, _ = _norm_shape(_slice(her, prob, ptg), edges)
            ax.step(centers, hwn, where="mid", color="tab:red", lw=1.2, ls="--",
                    label="HERWIG gen")
        if linear:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_yscale("log")
        ax.set_title(f"{PT_LABELS_BY_CHANNEL[channel][ptg]} GeV")
        ax.set_xlim(edges[0], edges[-1])
        if col == 0:
            ax.set_ylabel("(1/N) dN/dρ")
            ax.legend(fontsize=8)
        # ratio to pythia
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(pyn > 0, un / pyn, np.nan)
            rrs = np.where(pyn > 0, usn / pyn, np.nan)
        rax.errorbar(centers, rr, yerr=rrs, fmt="o", ms=3, color="k")
        if hwn is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rh = np.where(pyn > 0, hwn / pyn, np.nan)
            rax.step(centers, rh, where="mid", color="tab:red", ls="--", lw=1)
        rax.axhline(1.0, color="tab:blue", lw=1)
        rax.set_ylim(0.5, 1.5)
        rax.set_xlim(edges[0], edges[-1])
        rax.set_xlabel(r"$\log_{10}(\rho^2)$")
        if col == 0:
            rax.set_ylabel("ratio / PYTHIA")
    scale = "linear" if linear else "log"
    fig.suptitle(f"{channel} 2018 {mode} rho — D'Agostini (n_iter={n_iter}) — "
                 f"{tag} [{scale} y]")
    fig.tight_layout()
    suffix = f"_{tag}_linear" if linear else f"_{tag}"
    path = os.path.join(OUT, f"overlay_{mode}{suffix}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_method_comparison(mode, binning, tag, channel="dijet"):
    OUT = OUT_BY_CHANNEL[channel]
    prep = prepare(mode, binning, channel)
    prob = prep.prob
    dm = prep.data_matched
    methods = {
        "D'Agostini n=3": lambda: M.dagostini(prob, dm, n_iter=3),
        "D'Agostini n=6": lambda: M.dagostini(prob, dm, n_iter=6),
        "Tikhonov": lambda: M.tikhonov(prob, dm, tau=5e-4),
        "SVD (tau)": lambda: M.svd_unfold(prob, dm, tau=1e-3),
        "NNLS": lambda: M.nnls_unfold(prob, dm, tau=1e-2),
    }
    results = {}
    for name, fn in methods.items():
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = None
            print(f"  [method {name} failed: {e}]")
    pyt = prob.gen
    centers, edges = _gen_centers(binning)
    nptg = len(binning.pt_edges) - 1
    report = list(range(1, nptg))
    fig, axes = plt.subplots(1, len(report), figsize=(4 * len(report), 3.5),
                             squeeze=False)
    for col, ptg in enumerate(report):
        ax = axes[0][col]
        py = _slice(pyt, prob, ptg)
        pyn, _ = _norm_shape(py, edges)
        for name, res in results.items():
            if res is None:
                continue
            u = _slice(res, prob, ptg)
            un, _ = _norm_shape(np.clip(u, 0, None), edges)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.where(pyn > 0, un / pyn, np.nan)
            ax.step(centers, rr, where="mid", lw=1, label=name)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_ylim(0.0, 2.0)
        ax.set_title(f"{PT_LABELS_BY_CHANNEL[channel][ptg]} GeV")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        if col == 0:
            ax.set_ylabel("unfolded/PYTHIA")
            ax.legend(fontsize=7)
    fig.suptitle(f"{channel} 2018 {mode} rho — method comparison (data) — {tag}")
    fig.tight_layout()
    path = os.path.join(OUT, f"methods_{mode}_{tag}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_closure(mode, binning, tag, n_iter=N_ITER, channel="dijet"):
    """Three honest closure tests:
      (1) self-closure: unfold MC reco with MC prior -> must be flat at 1.
      (2) reweighted-MC: distort truth by +/-15% rho tilt, fold, unfold with the
          *nominal* prior -> tests prior independence (must track the tilt).
      (3) HERWIG-as-data: unfold HERWIG reco with PYTHIA response -> model bias
          (skipped if the channel has no HERWIG sample).
    """
    OUT = OUT_BY_CHANNEL[channel]
    prep = prepare(mode, binning, channel)
    prob = prep.prob
    gen = prob.gen

    self_est = M.dagostini(prob, M.fold(prob, gen), n_iter=n_iter, prior=gen)

    tilt = np.ones_like(gen)
    for ptg in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        tilt[idx] = 1.0 + 0.15 * np.linspace(-1, 1, len(idx))
    gen_rw = gen * tilt
    rw_est = M.dagostini(prob, M.fold(prob, gen_rw), n_iter=n_iter, prior=gen)

    her_est = (M.dagostini(prob, prep.her_matched, n_iter=n_iter)
               if prep.her_matched is not None else None)

    centers, edges = _gen_centers(binning)
    nptg = len(binning.pt_edges) - 1
    report = list(range(1, nptg))
    fig, axes = plt.subplots(1, len(report), figsize=(4 * len(report), 3.6),
                             squeeze=False)
    for col, ptg in enumerate(report):
        ax = axes[0][col]
        g = _slice(gen, prob, ptg)
        grw = _slice(gen_rw, prob, ptg)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_self = np.where(g > 0, _slice(self_est, prob, ptg) / g, np.nan)
            r_rw = np.where(grw > 0, _slice(rw_est, prob, ptg) / grw, np.nan)
        ax.step(centers, r_self, where="mid", color="tab:green", lw=1.4,
                label="self-closure (MC prior)")
        ax.step(centers, r_rw, where="mid", color="tab:purple", lw=1.2,
                label="reweighted-MC (±15% tilt)")
        if her_est is not None:
            her = _slice(prep.her_truth, prob, ptg)
            with np.errstate(divide="ignore", invalid="ignore"):
                r_her = np.where(her > 0, _slice(her_est, prob, ptg) / her, np.nan)
            ax.step(centers, r_her, where="mid", color="tab:red", lw=1.2, ls="--",
                    label="HERWIG-as-data")
        ax.axhspan(0.95, 1.05, color="0.85", zorder=0)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_ylim(0.7, 1.3)
        ax.set_title(f"{PT_LABELS_BY_CHANNEL[channel][ptg]} GeV")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        if col == 0:
            ax.set_ylabel("unfolded/truth")
            ax.legend(fontsize=7)
    fig.suptitle(f"{channel} 2018 {mode} rho — closure tests (n_iter={n_iter}) — {tag}"
                 "  [grey band = ±5%]")
    fig.tight_layout()
    path = os.path.join(OUT, f"closure_{mode}_{tag}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(channels=("dijet", "trijet")):
    import sys
    if len(sys.argv) > 1:
        channels = tuple(sys.argv[1:])
    for channel in channels:
        os.makedirs(OUT_BY_CHANNEL[channel], exist_ok=True)
        for mode in ("groomed", "ungroomed"):
            for tag, bfn in (("native", native_binning),
                             ("merged", merged_tail_binning)):
                binning = bfn(mode, channel)
                print(f"[{channel}] overlay :",
                      plot_overlay(mode, binning, tag, channel=channel))
                print(f"[{channel}] methods :",
                      plot_method_comparison(mode, binning, tag, channel=channel))
                print(f"[{channel}] closure :",
                      plot_closure(mode, binning, tag, channel=channel))


if __name__ == "__main__":
    main()
