#!/usr/bin/env python3
"""Overlay the analytic LL quark/gluon bracket on the unfolded rho result.

Reads the saved 2D unfolded output + uncertainty summary and draws, per
reported pT bin and per grooming, the unfolded data (stat + total bands) with
the LL curves from ``unfold.tools.theory_ll`` on top.

Everything is in the analysis rho convention -- rho = m/(pT R), x =
log10(rho^2) -- including the theory, which is translated from the textbook
rho_th = m^2/(pT R)^2 convention inside ``theory_ll`` (soft-drop transition at
x = -1, not -2).

Both sides are normalized to unit area over the *same* per-pT shown window: the
stored spectra are 2D-normalized (a single global integral), so each row is
rescaled here by its own shown-window integral, exactly reproducing the
published per-pT normalization (``normalize_over_shown=True``). A per-row
rescale leaves relative uncertainties untouched.

Caveats worth keeping in the caption: LL + fixed coupling within a pT bin, no
non-perturbative corrections, and pure quark / pure gluon rather than the Z+jet
flavor mixture. The curves are a bracket, not a prediction, and they say nothing
useful below rho ~ Lambda_QCD/(pT R), where the fixed-mass NP bump lives.

Usage
-----
    python notebooks/plot_theory_rho_overlay.py
    python notebooks/plot_theory_rho_overlay.py --tag fixed_jec
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unfold.tools.theory_ll import ZCUT, flavor_bracket, softdrop_transition  # noqa: E402
from unfold.tools.theory_resummed import mixed_prediction  # noqa: E402

from plot_theory_rho_ll import (  # noqa: E402
    GROOMED_FLOORS,
    PT_EDGES,
    REPORTED_PT_BINS,
    REPRESENTATIVE_PT,
    UNGROOMED_FLOOR,
    pt_label,
    window_for,
)


RUN2_LUMI = 138.0

# Kept away from the Pythia/Herwig colors used in the production plots
# (#5790fc / #e42536) so the theory bracket cannot be mistaken for a generator.
QUARK_COLOR = "#7a21dd"
GLUON_COLOR = "#e76300"
THEORY_COLOR = "#7a21dd"

# Gen-level quark fraction of the leading jet per reported pT bin, from our own
# flavor-tagged Pythia validation input (parton_flavor axis, nominal syst):
#   inputs/zjet/validation/validation_pythia_2018_flavortagged.pkl
#   pt_flavor_jet0_gen summed over [200,290], [290,400], [400,1500] GeV.
# f_other is < 1e-3 everywhere and is folded into gluon (conservative).
F_QUARK = {1: 0.743, 2: 0.725, 3: 0.711}


def load_unfolded(data_dir, groomed):
    """Per-pT shown-window-normalized density, plus stat and total bands."""
    tag = "groomed" if groomed else "ungroomed"
    with open(data_dir / f"uncertainty_summary_2d_{tag}.pkl", "rb") as fh:
        payload = pickle.load(fh)

    hist = payload["unfolded"]
    edges = np.asarray(hist.axes[1].edges, dtype=float)
    central = hist.values()
    stat = np.sqrt(hist.variances())
    total_up = payload["unfolded_total_up"].values()
    total_down = payload["unfolded_total_down"].values()
    return {
        "edges": edges,
        "central": central,
        "stat": stat,
        "total_up": total_up,
        "total_down": total_down,
    }


def shown_slice(store, pt_bin, groomed):
    """Restrict a pT row to its shown bins and renormalize to unit area there."""
    edges = store["edges"]
    floor = GROOMED_FLOORS[pt_bin] if groomed else UNGROOMED_FLOOR
    keep = edges[:-1] >= floor - 1e-9
    sub_edges = np.concatenate([edges[:-1][keep], [edges[-1]]])
    widths = np.diff(sub_edges)

    central = store["central"][pt_bin][keep]
    area = float((central * widths).sum())
    if area <= 0:
        raise ValueError(f"non-positive shown area {area} for pT bin {pt_bin}")
    scale = 1.0 / area

    return {
        "edges": sub_edges,
        "centers": 0.5 * (sub_edges[:-1] + sub_edges[1:]),
        "widths": widths,
        "central": central * scale,
        "stat": store["stat"][pt_bin][keep] * scale,
        "total_up": store["total_up"][pt_bin][keep] * scale,
        "total_down": store["total_down"][pt_bin][keep] * scale,
    }


def make_panel(pt_bin, groomed, store, outdir, fixed_alphas=None, model="ll"):
    window = window_for(pt_bin, groomed)
    data = shown_slice(store, pt_bin, groomed)

    fig, ax = plt.subplots(layout="constrained")

    # Data: total band, stat band, then markers on top.
    ax.stairs(
        data["central"] + data["total_up"],
        data["edges"],
        baseline=data["central"] - data["total_down"],
        fill=True,
        color="yellowgreen",
        alpha=0.8,
        label="Total unc.",
    )
    ax.stairs(
        data["central"] + data["stat"],
        data["edges"],
        baseline=data["central"] - data["stat"],
        fill=True,
        color="darkgreen",
        label="Stat. unc.",
    )
    ax.errorbar(
        data["centers"],
        data["central"],
        yerr=data["stat"],
        xerr=data["widths"] / 2,
        fmt="o",
        color="k",
        markersize=6,
        label="Data (unfolded)",
    )

    # Theory curves.
    if model == "ll":
        quark, gluon = flavor_bracket(
            window, groomed, pt=REPRESENTATIVE_PT[pt_bin], alphas=fixed_alphas
        )
        ax.fill_between(
            quark.x,
            np.minimum(quark.y, gluon.y),
            np.maximum(quark.y, gluon.y),
            color=QUARK_COLOR,
            alpha=0.13,
            lw=0,
        )
        ax.plot(quark.x, quark.y, color=QUARK_COLOR, lw=2.4, label=r"LL quark ($C_F$)")
        ax.plot(
            gluon.x, gluon.y, color=GLUON_COLOR, lw=2.4, ls="--",
            label=r"LL gluon ($C_A$)",
        )
        theory_ymax = max(float(quark.y.max()), float(gluon.y.max()))
    else:
        pred = mixed_prediction(
            window, groomed, REPRESENTATIVE_PT[pt_bin], f_quark=F_QUARK[pt_bin]
        )
        ax.fill_between(
            pred["x"], pred["lo"], pred["hi"],
            color=THEORY_COLOR, alpha=0.25, lw=0,
            label=r"$\mu_R$ + IR-freeze env.",
        )
        ax.plot(
            pred["x"], pred["central"], color=THEORY_COLOR, lw=2.4,
            label=rf"NLL (rc), $f_q = {F_QUARK[pt_bin]:.2f}$",
        )
        ax.plot(
            pred["x"], pred["flavors"]["quark"], color=THEORY_COLOR, lw=1.2,
            ls=":", alpha=0.55, label="pure quark / gluon",
        )
        ax.plot(
            pred["x"], pred["flavors"]["gluon"], color=GLUON_COLOR, lw=1.2,
            ls=":", alpha=0.55,
        )
        theory_ymax = float(pred["hi"].max())

    if groomed:
        xt = softdrop_transition(ZCUT)
        if window[0] < xt < window[1]:
            ax.axvline(xt, ymin=0.0, ymax=0.62, color="0.4", lw=1.0, ls=(0, (4, 3)))
            ax.text(
                xt - 0.05,
                0.635,
                r"$\rho^2 = z_{\mathrm{cut}}$",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=15,
                color="0.35",
            )

    grooming_word = "Groomed" if groomed else "Ungroomed"
    ax.set_xlabel(rf"$\log_{{10}}(\rho^2)$, {grooming_word}")
    ax.set_ylabel(r"$\frac{1}{\sigma}\frac{d\sigma}{d\log_{10}(\rho^2)}$")
    ax.set_xlim(*window)

    ymax = max(float((data["central"] + data["total_up"]).max()), theory_ymax)
    ax.set_ylim(0.0, ymax * 1.6)

    ax.legend(loc="upper left", frameon=False, fontsize=15, handlelength=1.8)
    # rlabel keeps the lumi + com (CMS convention), so the pT range goes
    # in-frame at the top right, where nothing else is drawn.
    hep.cms.label("Preliminary", data=True, lumi=RUN2_LUMI, com=13, loc=0, ax=ax)
    ax.text(
        0.97,
        0.95,
        pt_label(pt_bin),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=17,
    )

    tag = "groomed" if groomed else "ungroomed"
    stem = "theory_overlay" if model == "ll" else f"theory_overlay_{model}"
    path = outdir / f"{stem}_{tag}_pt{pt_bin}.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="arc_r2", help="unfolding output tag")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--fixed-alphas",
        type=float,
        default=None,
        metavar="VALUE",
        help="LL model only: fixed coupling instead of running at mu = pT*R",
    )
    parser.add_argument(
        "--model",
        choices=("ll", "resummed"),
        default="ll",
        help="'ll' = fixed-order-in-logs cartoon bracket; 'resummed' = "
        "running-coupling NLL with q/g mixing and a scale-variation band",
    )
    args = parser.parse_args()

    data_dir = ROOT / "outputs" / "zjet" / "rho" / args.tag / "data"
    outdir = args.outdir or ROOT / "outputs" / "zjet" / "rho" / args.tag / "theory"
    outdir.mkdir(parents=True, exist_ok=True)

    hep.style.use(hep.style.CMS)

    for groomed in (False, True):
        store = load_unfolded(data_dir, groomed)
        for pt_bin in REPORTED_PT_BINS:
            print(
                f"wrote {make_panel(pt_bin, groomed, store, outdir, args.fixed_alphas, model=args.model)}"
            )


if __name__ == "__main__":
    main()
