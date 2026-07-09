"""Compare D'Agostini vs the locked-in TUnfold result for Z+jet rho.

Runs D'Agostini on the *exact* locked-in reporting binning
(bin_edges.gen_rho_edges_by_pt) and overlays it with the committed TUnfold
unfolded data (outputs/zjet/rho/original/unfold/unfolded_2d_<mode>.pkl) and the
PYTHIA8 truth, per pT slice (shape-normalized). Z+jet backgrounds (~1%) ignored.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unfold.tools.binning import bin_edges
from scripts.dijet_methods.loader import Binning, load_native
from scripts.dijet_methods.result import prepare, unfold_with_unc, systematic_band, channel_paths
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.plots import (
    _slice, _gen_centers, _norm_shape, N_ITER, OUT_BY_CHANNEL, PT_LABELS_BY_CHANNEL,
)

LOCKED = "outputs/zjet/rho/original/unfold/unfolded_2d_{mode}.pkl"


def locked_binning(mode):
    b = bin_edges(mode == "groomed")
    gen = np.array(b.gen_rho_edges_by_pt[1], dtype=float)
    nat = load_native(channel_paths("zjet")[0], mode)
    return Binning(np.array(b.pt_edges, dtype=float),
                   nat.rho_edges_reco.copy(), gen)


def shape_norm_row(values, edges):
    w = np.diff(edges)
    dens = values / w
    area = values.sum()
    return (dens / area if area > 0 else dens), w


def main(modes=("groomed", "ungroomed"), linear=False):
    channel = "zjet"
    out = OUT_BY_CHANNEL[channel]
    for mode in modes:
        binning = locked_binning(mode)
        prep = prepare(mode, binning, channel)
        prob = prep.prob
        # D'Agostini unfolded data + total (stat (+) syst) uncertainty
        central, stat, cov, toys = unfold_with_unc(
            prep, M.dagostini, n_iter=N_ITER, n_toys=200, response_toys=True)
        _, syst, _ = systematic_band(prep, M.dagostini, n_iter=N_ITER)
        tot = np.sqrt(stat**2 + syst**2)

        locked = pickle.load(open(LOCKED.format(mode=mode), "rb"))
        lk_unf = np.asarray(locked["unfolded_2dnorm"])       # (4, nrho) struct
        lk_val = lk_unf["value"]
        lk_err = np.sqrt(np.clip(lk_unf["variance"], 0, None))
        lk_true = np.asarray(locked["true_2dnorm"])          # (4, nrho) plain
        nptg = len(binning.pt_edges) - 1
        edges = binning.rho_edges_gen
        centers = 0.5 * (edges[:-1] + edges[1:])
        report = list(range(1, nptg))

        fig, axes = plt.subplots(2, len(report), figsize=(4.4 * len(report), 7),
                                 gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
        for col, ptg in enumerate(report):
            ax, rax = axes[0][col], axes[1][col]
            # mine (gen-binned vectors -> slice for this pt)
            u = _slice(central, prob, ptg); ue = _slice(tot, prob, ptg)
            py = _slice(prob.gen, prob, ptg)
            un, w = shape_norm_row(u, edges); uen = ue / w / max(u.sum(), 1e-30)
            pyn, _ = shape_norm_row(py, edges)
            # locked (row ptg)
            lkn, _ = shape_norm_row(lk_val[ptg], edges)
            lken = lk_err[ptg] / w / max(lk_val[ptg].sum(), 1e-30)
            lktn, _ = shape_norm_row(lk_true[ptg], edges)

            ax.step(centers, pyn, where="mid", color="tab:blue", lw=1.6,
                    label="PYTHIA8 gen", zorder=1)
            ax.errorbar(centers - 0.04, un, yerr=uen, fmt="o", ms=4.5, color="k",
                        label="D'Agostini (this work)", zorder=5)
            ax.errorbar(centers + 0.04, lkn, yerr=lken, fmt="s", ms=4.5,
                        color="tab:red", mfc="none", label="TUnfold (locked-in)",
                        zorder=4)
            if linear:
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_yscale("log")
            ax.set_title(f"{PT_LABELS_BY_CHANNEL[channel][ptg]} GeV")
            ax.set_xlim(edges[0], edges[-1])
            if col == 0:
                ax.set_ylabel("(1/N) dN/dρ")
                ax.legend(fontsize=8)
            with np.errstate(divide="ignore", invalid="ignore"):
                ax_r = lambda y: np.where(pyn > 0, y / pyn, np.nan)
                rax.errorbar(centers - 0.04, ax_r(un), yerr=ax_r(uen), fmt="o",
                             ms=3.5, color="k")
                rax.errorbar(centers + 0.04, ax_r(lkn), yerr=ax_r(lken), fmt="s",
                             ms=3.5, color="tab:red", mfc="none")
            rax.axhline(1.0, color="tab:blue", lw=1)
            rax.set_ylim(0.6, 1.4)
            rax.set_xlim(edges[0], edges[-1])
            rax.set_xlabel(r"$\log_{10}(\rho^2)$")
            if col == 0:
                rax.set_ylabel("ratio / PYTHIA")
        fig.suptitle(f"Z+jet 2018 {mode} rho — D'Agostini vs locked-in TUnfold "
                     f"(same binning, per-slice shape norm) "
                     f"[{'linear' if linear else 'log'} y]")
        fig.tight_layout()
        path = os.path.join(out, f"compare_locked_{mode}"
                            + ("_linear" if linear else "") + ".png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
