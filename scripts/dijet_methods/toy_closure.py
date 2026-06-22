"""Statistical toy self-closure.

The noiseless self-closure (unfold P@truth) is exact by construction and tests
only bookkeeping. This is the honest version: take the MC pseudo-data
(reco = matched + fakes, whose truth = MC gen is known), inject realistic
statistical noise, fold/subtract-fakes/unfold each toy exactly like the real
data, and look at the toy distribution of unfolded/truth per bin:

  - bias  = mean(unfolded)/truth - 1     (should be ~0 -> unbiased)
  - spread= std(unfolded)/truth          (the real statistical uncertainty)

If |bias| << spread the method is unbiased and the spread is the honest stat
error. If bias grows, that is regularization bias.

Noise models:
  - "data": each reco bin fluctuated by sqrt(data variance) -> the precision the
            REAL measurement has.
  - "mc":   fluctuated by sqrt(MC reco variance) -> response/MC-stat level.
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


def run_toys(prep, n_iter=N_ITER, n_toys=500, noise="data", seed=0):
    prob = prep.prob
    truth = prob.gen
    reco_expected = prob.R.sum(axis=1) + prob.fakes      # = MC reco total
    if noise == "data":
        var = prep.data_var
    else:
        var = prep.rb_mc.reco.reshape(-1) if prep.rb_mc.reco_var is None \
            else prep.rb_mc.reco_var.reshape(-1)
    sigma = np.sqrt(np.clip(var, 0, None))
    rng = np.random.default_rng(seed)
    toys = np.empty((n_toys, len(truth)))
    for t in range(n_toys):
        noisy = np.clip(reco_expected + rng.normal(size=reco_expected.shape) * sigma, 0, None)
        matched = M.subtract_fakes(noisy, prob)
        toys[t] = M.dagostini(prob, matched, n_iter=n_iter, prior=truth)
    return truth, toys


def summarize(prep, truth, toys, label):
    prob = prep.prob
    rep = prob.gen_pt_idx >= 1
    keep = rep & (truth > truth.max() * 1e-4)       # ignore near-empty tail
    mean = toys.mean(axis=0)
    std = toys.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = (mean / truth - 1)[keep]
        spread = (std / truth)[keep]
    print(f"  {label:28s} bias: med={np.median(bias):+.4f} "
          f"p90|bias|={np.percentile(np.abs(bias),90):.4f} max|bias|={np.max(np.abs(bias)):.4f}"
          f"   spread: med={np.median(spread):.4f} p90={np.percentile(spread,90):.4f}")
    return mean, std


def plot(prep, truth, toys, channel, mode, noise):
    prob = prep.prob
    binning = prep.binning
    centers, edges = _gen_centers(binning)
    mean = toys.mean(axis=0)
    std = toys.std(axis=0)
    nptg = len(binning.pt_edges) - 1
    report = list(range(1, nptg))
    fig, axes = plt.subplots(1, len(report), figsize=(4 * len(report), 3.6),
                             squeeze=False)
    for col, ptg in enumerate(report):
        ax = axes[0][col]
        t = _slice(truth, prob, ptg)
        m = _slice(mean, prob, ptg)
        s = _slice(std, prob, ptg)
        toy_sl = toys[:, prob.gen_pt_idx == ptg]
        with np.errstate(divide="ignore", invalid="ignore"):
            # thin grey lines: a few individual toys
            for k in range(0, toy_sl.shape[0], max(1, toy_sl.shape[0] // 40)):
                ax.step(centers, np.where(t > 0, toy_sl[k] / t, np.nan),
                        where="mid", color="0.8", lw=0.4, zorder=0)
            rm = np.where(t > 0, m / t, np.nan)
            rs = np.where(t > 0, s / t, np.nan)
        ax.errorbar(centers, rm, yerr=rs, fmt="o", ms=3, color="tab:blue",
                    label="mean ± std of toys", zorder=5)
        ax.axhspan(0.95, 1.05, color="0.88", zorder=0)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_ylim(0.6, 1.4)
        ax.set_title(f"{PT_LABELS_BY_CHANNEL[channel][ptg]} GeV")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        if col == 0:
            ax.set_ylabel("unfolded / truth")
            ax.legend(fontsize=7)
    fig.suptitle(f"{channel} 2018 {mode} rho — TOY self-closure, {noise}-stat noise "
                 f"(n_iter={N_ITER}, {toys.shape[0]} toys)  [grey band ±5%]")
    fig.tight_layout()
    out = OUT_BY_CHANNEL[channel]
    path = os.path.join(out, f"toy_closure_{mode}_{noise}.png")
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
            print(f"\n#### {channel} {mode} (merged, n_iter={N_ITER}) ####")
            for noise in ("data", "mc"):
                truth, toys = run_toys(prep, noise=noise, n_toys=500)
                summarize(prep, truth, toys, f"{noise}-stat noise")
                print("   plot:", plot(prep, truth, toys, channel, mode, noise))


if __name__ == "__main__":
    main()
