#!/usr/bin/env python3
"""Analytic LL rho curves redrawn in this analysis' convention and binning.

Takes the standalone quark/gluon Sudakov + soft-drop curves and puts them on the
axes the unfolded result uses, so the two can be compared (or overlaid) directly:

  * x is the analysis ``log10(rho^2)`` with rho = m/(pT R) -- a factor of two
    away from the theory-script axis (see ``unfold.tools.theory_ll``), which
    moves the soft-drop transition from -2 to -1;
  * each curve is normalized to unit area over the *shown* per-pT window, the
    same window the unfolded spectra are normalized over;
  * the coupling runs, so every pT bin gets its own curve;
  * quark (C_F) and gluon (C_A) are drawn as a bracket, since Z+jet is
    quark-enriched but mixed.

One square panel per pT bin per grooming, for the document grid to lay out.

Usage
-----
    python notebooks/plot_theory_rho_ll.py
    python notebooks/plot_theory_rho_ll.py --fixed-alphas 0.12   # original scripts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unfold.tools.theory_ll import (  # noqa: E402
    CA,
    CF,
    JET_R,
    ZCUT,
    flavor_bracket,
    softdrop_transition,
)


# pT binning of RHO_ARC_R2_SPEC. Bin 0 (185-200) is the migration sink and is
# not reported, so it is not drawn.
PT_EDGES = [185, 200, 290, 400, 13000]
REPORTED_PT_BINS = [1, 2, 3]

# Representative pT per bin, used to set mu = pT * R for the running coupling.
# The last bin is open-ended; the spectrum falls steeply, so ~500 GeV carries
# most of its weight.
REPRESENTATIVE_PT = {1: 240.0, 2: 340.0, 3: 500.0}

# Shown windows = normalization windows (normalize_over_shown=True). Groomed
# floors are the publication-space 5 GeV mass floors; ungroomed uses the uniform
# -2.5 xlim.
GROOMED_FLOORS = [-2.5, -3.0, -3.0, -3.5]
UNGROOMED_FLOOR = -2.5

QUARK_COLOR = "#5790fc"
GLUON_COLOR = "#e42536"


def pt_label(i):
    lo, hi = PT_EDGES[i], PT_EDGES[i + 1]
    if i == len(PT_EDGES) - 2:
        return rf"$p_{{\mathrm{{T}}}} > {lo:g}$ GeV"
    return rf"${lo:g} < p_{{\mathrm{{T}}}} < {hi:g}$ GeV"


def window_for(pt_bin, groomed):
    lo = GROOMED_FLOORS[pt_bin] if groomed else UNGROOMED_FLOOR
    return (lo, 0.0)


def make_panel(pt_bin, groomed, outdir, fixed_alphas=None):
    window = window_for(pt_bin, groomed)
    pt = REPRESENTATIVE_PT[pt_bin]
    quark, gluon = flavor_bracket(
        window, groomed, pt=pt, alphas=fixed_alphas
    )

    fig, ax = plt.subplots(layout="constrained")

    ax.fill_between(
        quark.x,
        np.minimum(quark.y, gluon.y),
        np.maximum(quark.y, gluon.y),
        color=QUARK_COLOR,
        alpha=0.15,
        lw=0,
        label="quark/gluon bracket",
    )
    ax.plot(quark.x, quark.y, color=QUARK_COLOR, lw=2.4, label=r"quark ($C_F = 4/3$)")
    ax.plot(
        gluon.x, gluon.y, color=GLUON_COLOR, lw=2.4, ls="--", label=r"gluon ($C_A = 3$)"
    )

    if groomed:
        xt = softdrop_transition(ZCUT)
        if window[0] < xt < window[1]:
            # Stop the line below the upper-right info block, and keep the tag
            # horizontal above the curve peak (which sits at ~0.65 of the axes
            # height given the 1.55x headroom) so it crosses neither.
            ax.axvline(xt, ymin=0.0, ymax=0.72, color="0.4", lw=1.0, ls=(0, (4, 3)))
            ax.text(
                xt - 0.05,
                0.74,
                r"$\rho^2 = z_{\mathrm{cut}}$",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=16,
                color="0.35",
            )

    grooming_word = "Groomed" if groomed else "Ungroomed"
    ax.set_xlabel(rf"$\log_{{10}}(\rho^2)$, {grooming_word}")
    ax.set_ylabel(r"$\frac{1}{\sigma}\frac{d\sigma}{d\log_{10}(\rho^2)}$")
    ax.set_xlim(*window)

    ymax = max(quark.y.max(), gluon.y.max())
    ax.set_ylim(0.0, ymax * 1.55)

    alphas_line = (
        rf"$\alpha_S = {quark.alphas:.3f}$ at $\mu = p_{{\mathrm{{T}}}}R$"
        if fixed_alphas is None
        else rf"$\alpha_S = {quark.alphas:.2f}$ (fixed)"
    )
    setup = (
        rf"soft drop $\beta = 0$, $z_{{\mathrm{{cut}}}} = {ZCUT}$"
        if groomed
        else "plain jet mass"
    )
    ax.text(
        0.97,
        0.95,
        "LL, " + setup + "\n" + alphas_line + "\n" + rf"AK{int(JET_R * 10)}, unit area over shown range",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        color="0.35",
    )

    ax.legend(loc="upper left", frameon=False, fontsize=17, handlelength=2.4)
    # No CMS label: these are analytic curves, not CMS data or simulation, and
    # "CMS Simulation" would misrepresent them. The pT range still goes where
    # rlabel would sit so the panel self-identifies in a grid.
    ax.text(
        1.0,
        1.005,
        pt_label(pt_bin),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
    )

    tag = "groomed" if groomed else "ungroomed"
    path = outdir / f"theory_ll_{tag}_pt{pt_bin}.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "zjet" / "rho" / "theory",
        help="output directory for the panels",
    )
    parser.add_argument(
        "--fixed-alphas",
        type=float,
        default=None,
        metavar="VALUE",
        help="use a fixed coupling (e.g. 0.12) instead of running at mu = pT*R",
    )
    args = parser.parse_args()

    hep.style.use(hep.style.CMS)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for groomed in (False, True):
        for pt_bin in REPORTED_PT_BINS:
            path = make_panel(pt_bin, groomed, args.outdir, args.fixed_alphas)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
