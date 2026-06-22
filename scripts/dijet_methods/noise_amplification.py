"""Noise-amplification comparison: push identical statistically-fluctuated toys
through D'Agostini vs matrix-inversion unfolders and plot the per-bin spread.

Shows that the inversion methods blow up the SAME input noise by 1-3 orders of
magnitude in the low-rho tail, while D'Agostini stays at the input data-stat
level. Directly demonstrates why iterative Bayes is trustworthy here.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.dijet_methods.plots import (
    merged_tail_binning, OUT_BY_CHANNEL, PT_LABELS_BY_CHANNEL, _gen_centers, _slice, N_ITER,
)
from scripts.dijet_methods.result import prepare
from scripts.dijet_methods import methods as M


def spreads(prep, n_toys=300, seed=0):
    prob = prep.prob
    truth = prob.gen
    reco_exp = prob.R.sum(1) + prob.fakes
    sigma = np.sqrt(np.clip(prep.data_var, 0, None))
    rng = np.random.default_rng(seed)
    methods = {
        "D'Agostini n=4": lambda m: M.dagostini(prob, m, n_iter=N_ITER, prior=truth),
        "Tikhonov": lambda m: M.tikhonov(prob, m, tau=5e-4),
        "SVD": lambda m: M.svd_unfold(prob, m, tau=1e-3),
        "near-inverse": lambda m: M.tikhonov(prob, m, tau=1e-6),
    }
    acc = {k: [] for k in methods}
    for _ in range(n_toys):
        noisy = np.clip(reco_exp + rng.normal(size=reco_exp.shape) * sigma, 0, None)
        matched = M.subtract_fakes(noisy, prob)
        for k, fn in methods.items():
            try:
                acc[k].append(fn(matched))
            except Exception:
                acc[k].append(np.full_like(truth, np.nan))
    out = {}
    for k, A in acc.items():
        A = np.array(A)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k] = np.nanstd(A, 0) / np.where(truth > 0, truth, np.nan)
    return truth, out


def plot(prep, truth, out, channel, mode):
    prob = prep.prob
    binning = prep.binning
    centers, edges = _gen_centers(binning)
    nptg = len(binning.pt_edges) - 1
    report = list(range(1, nptg))
    colors = {"D'Agostini n=4": "tab:blue", "Tikhonov": "tab:green",
              "SVD": "tab:red", "near-inverse": "tab:orange"}
    fig, axes = plt.subplots(1, len(report), figsize=(4 * len(report), 3.6),
                             squeeze=False)
    for col, ptg in enumerate(report):
        ax = axes[0][col]
        for k, sp in out.items():
            ax.step(centers, _slice(sp, prob, ptg), where="mid", lw=1.2,
                    color=colors[k], label=k)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e2)
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_title(f"{PT_LABELS_BY_CHANNEL[channel][ptg]} GeV")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        if col == 0:
            ax.set_ylabel("stat spread (std/truth)")
            ax.legend(fontsize=7)
    fig.suptitle(f"{channel} 2018 {mode} rho — noise amplification: identical "
                 f"data-stat toys, per method  (dotted = 100% rel. spread)")
    fig.tight_layout()
    path = os.path.join(OUT_BY_CHANNEL[channel], f"noise_amplification_{mode}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(channels=("dijet", "trijet")):
    import sys
    if len(sys.argv) > 1:
        channels = tuple(sys.argv[1:])
    for channel in channels:
        for mode in ("groomed", "ungroomed"):
            b = merged_tail_binning(mode, channel)
            prep = prepare(mode, b, channel)
            truth, out = spreads(prep)
            print(f"[{channel} {mode}] amplification plot:",
                  plot(prep, truth, out, channel, mode))


if __name__ == "__main__":
    main()
