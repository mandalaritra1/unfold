#!/usr/bin/env python3
r"""Figure: how the regularization L-matrix is modified (curvature of x/x_MC).

Builds the real regularized zjet rho unfolder, extracts its L-matrix, and draws
a two-panel explanatory figure for docs/ratio_curvature_regularization.md:

  Left  : structure of L -- block-diagonal (one block per pT slice, no condition
          crosses a slice boundary); each row is a 3-point curvature stencil.
          Row-normalised so the (+1, -2, +1) pattern is visible regardless of
          the per-bin 1/m scaling.
  Right : why -- the row weights are (1/m0, -2/m1, 1/m2) with m the MC prior, so
          the penalised quantity is the curvature of x_j/m_j and any x proportional
          to the prior carries exactly zero penalty.

Usage: source scripts/setup_root.sh && python scripts/plotting/make_lmatrix_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.colors import TwoSlopeNorm

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.utils.cms_plot import (
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    stamp_figure,
)

hep.style.use(hep.style.CMS)

OUT = REPO_ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    u = Unfolder(get_spec("zjet", "rho", "original_jacobian_reg"), groomed=True,
                 do_syst=False, compute_jackknife_stat=False)
    L = u.L
    nx, ny = L.GetNbinsX(), L.GetNbinsY()
    Lnp = np.array([[L.GetBinContent(i + 1, j + 1) for j in range(ny)] for i in range(nx)])
    Lmat = Lnp if Lnp.shape[0] < Lnp.shape[1] else Lnp.T  # rows=conditions, cols=gen bins
    prior = u.mosaic_dict["nominal"].sum(axis=0) + u.misses_2d
    counts = [len(e) - 1 for e in u.gen_edges_by_pt]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # row-normalise so every condition shows the same (+0.5, -1, +0.5) stencil
    rownorm = Lmat / np.where(np.abs(Lmat).max(axis=1, keepdims=True) > 0,
                              np.abs(Lmat).max(axis=1, keepdims=True), 1.0)

    # 1x2 grid -> uniform ~10x10-per-axes CMS footprint (skill rule 2).
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(21, 10.6), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # ---- Panel A: structure of L ----
    im = axL.imshow(rownorm, aspect="auto", cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    for s in starts[1:]:
        axL.axvline(s - 0.5, color="k", lw=1.3)
    mid = [starts[i] + counts[i] / 2 - 0.5 for i in range(len(counts))]
    labels = ["0-200", "200-290", "290-400", "400-∞"]
    # L is block-diagonal, so each column block's own rows are occupied; drop the
    # label into whichever half of that column is empty (above or below the block).
    row_counts = [max(c - 2, 0) for c in counts]
    row_starts = np.concatenate([[0], np.cumsum(row_counts)[:-1]])
    n_rows = rownorm.shape[0]
    for i, (x, lab) in enumerate(zip(mid, labels)):
        r0, r1 = row_starts[i], row_starts[i] + row_counts[i]
        y = (r0 / 2.0 - 0.5) if r0 > (n_rows - r1) else ((r1 + n_rows) / 2.0 - 0.5)
        axL.text(x, y, f"{lab} GeV", ha="center", va="center",
                 fontsize=PUB_ANNOTATION_FONTSIZE - 4, color="0.25")
    axL.set_xlabel("gen bin index  (4 pT-slice blocks)", fontsize=PUB_LABEL_FONTSIZE)
    axL.set_ylabel("regularisation condition (row of $L$)", fontsize=PUB_LABEL_FONTSIZE)
    axL.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
    cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.02)
    cb.set_label("row of $L$ (normalised)", fontsize=PUB_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)

    # ---- Panel B: the 1/m reweighting and zero-penalty property ----
    i_pt = 1
    sl = slice(starts[i_pt], starts[i_pt] + counts[i_pt])
    m = prior[sl]
    edges = np.array(u.gen_edges_by_pt[i_pt], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axR.plot(centers, m / m.max(), "o-", color="#5790fc", lw=2,
             label=r"MC prior $m_j$ (PYTHIA gen)")
    k = 3
    for kk, w, dy in ((k - 1, r"$+1/m_{j-1}$", 22), (k, r"$-2/m_j$", -34),
                      (k + 1, r"$+1/m_{j+1}$", 22)):
        axR.annotate(w, (centers[kk], m[kk] / m.max()), textcoords="offset points",
                     xytext=(0, dy), ha="center", color="#e42536",
                     fontsize=PUB_ANNOTATION_FONTSIZE - 4, fontweight="bold")
    axR.plot(centers[k - 1:k + 2], m[k - 1:k + 2] / m.max(), "s", ms=11, mfc="none",
             mec="#e42536", mew=2)
    axR.set_yscale("log")
    axR.set_xlim(-4.5, 0)
    # Open an empty band below the curve for the equation box, and headroom above
    # so the "+1/m" annotations clear the top spine.
    norm_prior = m / m.max()
    axR.set_ylim(float(norm_prior[norm_prior > 0].min()) / 60.0, 3.0)
    axR.set_xlabel(r"$\log_{10}(\rho^2)$  (200-290 GeV slice)", fontsize=PUB_LABEL_FONTSIZE)
    axR.set_ylabel(r"prior $m_j$ (normalised)", fontsize=PUB_LABEL_FONTSIZE)
    axR.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
    axR.text(0.035, 0.045,
             r"$(Lx)_r=\dfrac{x_{j-1}}{m_{j-1}}-2\dfrac{x_j}{m_j}+\dfrac{x_{j+1}}{m_{j+1}}$"
             "\n\n"
             r"$x_j\propto m_j\ \Rightarrow\ (Lx)_r=c-2c+c=0$",
             transform=axR.transAxes, fontsize=PUB_ANNOTATION_FONTSIZE - 4, va="bottom",
             bbox=dict(boxstyle="round", fc="#fff6e6", ec="0.6"))
    axR.legend(loc="upper right", fontsize=PUB_LEGEND_FONTSIZE)

    # loc=0 (above the frame) on both panels: the left panel is a heatmap with a
    # colorbar, so there is no clear in-frame corner, and a split placement
    # across the two panels of one figure reads as a mistake.
    hep.cms.label("Internal", data=False, loc=0, ax=axL, rlabel="block-diagonal $L$")
    hep.cms.label("Internal", data=False, loc=0, ax=axR, rlabel=r"row weights $1/m_j$")

    stamp_figure(fig, inputs="zjet rho original_jacobian_reg")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ratio_curvature_lmatrix.{ext}", dpi=140)
    print(f"wrote {OUT}/ratio_curvature_lmatrix.png / .pdf  "
          f"(L shape {Lmat.shape}, {len(counts)} slices)")


if __name__ == "__main__":
    main()
