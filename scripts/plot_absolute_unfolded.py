#!/usr/bin/env python3
"""ARC round-3 WS6.1: the unfolded 2D result BEFORE normalization.

One CMS panel per reported pT slice and grooming state, showing the absolute
differential cross section dsigma/dlog10(rho^2) = N_i / (L w_i) over the
reported window: unfolded data points with statistical errors, the total
uncertainty band, and the PYTHIA truth prediction in the same units.

Unlike the published normalized result this carries the luminosity
uncertainty (1.6% Run-2 combined) -- it is a coherent rate term
that the per-pT normalization cancels exactly, so it appears here and only
here.  The band is stat (+) detector systematics (symmetrized per source)
(+) the PS/HAD model envelope (fractional shifts applied to the absolute
spectrum) (+) luminosity.

Output: <tag>/unfold/absolute_unfolded_<mode>_pt<k>.pdf/.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.utils.cms_plot import (
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    stamp_figure,
)

hep.style.use(hep.style.CMS)

DEFAULT_TAG = "jmsjmr_unity_groomed400_floor3"
LUMI_PB = 138_000.0          # 138 fb^-1 in pb^-1
LUMI_FRAC_UNC = 0.016        # combined Run-2 luminosity uncertainty
UPDOWN_KEY = re.compile(r"^(.+)(Up|Down)((?:_corr|_uncorr_\d+)?)$")


def reported_blocks(unfolder):
    counts = [len(e) - 1 for e in unfolder.gen_edges_by_pt]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    floors = unfolder._bl_shown_floors()
    first = max(1, getattr(unfolder, "first_reported_pt_bin", 0))
    out = []
    for i, edges in enumerate(unfolder.gen_edges_by_pt):
        if i < first:
            continue
        edges = np.asarray(edges, float)
        keep = np.flatnonzero(edges[:-1] >= floors[i] - 1e-9)
        if not keep.size:
            continue
        hi = unfolder.pt_edges[i + 1] if i + 1 < len(unfolder.pt_edges) - 1 else None
        label = (rf"${int(unfolder.pt_edges[i])} < p_{{\mathrm{{T}}}} < {int(hi)}$ GeV"
                 if hi else rf"${int(unfolder.pt_edges[i])} < p_{{\mathrm{{T}}}} < \infty$ GeV")
        out.append({
            "flat": starts[i] + keep,
            "edges": edges[keep[0]:keep[-1] + 2],
            "label": label,
            "index": i,
        })
    return out


def detector_syst_band(unfolder):
    """Absolute per-bin systematic uncertainty, symmetrized per source.

    Sources superseded by the model envelope are excluded (the model term is
    added separately from the stored fractional PS/HAD shifts).
    """
    superseded = ("herwig", "fsr", "isr", "model_")
    keys = {k for k in unfolder.y_unf_dict if not k.startswith(superseded)}
    nominal = np.asarray(unfolder.y_unf, float)
    var = np.zeros_like(nominal)
    done = set()
    for key in sorted(keys):
        if key in done:
            continue
        match = UPDOWN_KEY.match(key)
        partner = None
        if match:
            base, direction, suffix = match.groups()
            partner = f"{base}{'Down' if direction == 'Up' else 'Up'}{suffix}"
        if partner in keys:
            done.update({key, partner})
            shift = 0.5 * (
                np.asarray(unfolder.y_unf_dict[key], float)
                - np.asarray(unfolder.y_unf_dict[partner], float)
            )
        else:
            done.add(key)
            shift = np.asarray(unfolder.y_unf_dict[key], float) - nominal
        var += shift ** 2
    return np.sqrt(var)


def run(groomed, tag):
    mode = "groomed" if groomed else "ungroomed"
    unfolder = Unfolder(get_spec("zjet", "rho", tag), groomed, do_syst=True,
                        cms_label="Internal")
    out_dir = Path(unfolder.spec.output_dir) / "unfold"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_abs = np.asarray(unfolder.y_unf, float)
    truth = np.asarray(unfolder.y_true, float)
    stat = np.sqrt(np.clip(np.diag(
        np.asarray(unfolder.cov_data_np) + np.asarray(unfolder.cov_uncorr_np)
    ), 0.0, None))
    syst = detector_syst_band(unfolder)
    ps = np.abs(np.asarray(unfolder.model_ps_shift_flat, float)) * y_abs
    had = np.abs(np.asarray(unfolder.model_had_shift_flat, float)) * y_abs
    lumi = LUMI_FRAC_UNC * y_abs
    total = np.sqrt(stat**2 + syst**2 + ps**2 + had**2 + lumi**2)

    summary = []
    for k, block in enumerate(reported_blocks(unfolder)):
        idx, edges = block["flat"], block["edges"]
        widths = np.diff(edges)
        scale = 1.0 / (LUMI_PB * widths)          # counts -> pb per unit log10(rho^2)
        val = y_abs[idx] * scale
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, ax = plt.subplots(layout="constrained")
        band = total[idx] * scale
        ax.stairs(val + band, edges, baseline=val - band, fill=True,
                  color="yellowgreen",
                  label=r"Total Unc. (incl. lumi)")
        ax.errorbar(centers, val, yerr=stat[idx] * scale,
                    xerr=widths / 2, fmt="o", color="black", ms=6,
                    label="Unfolded data")
        ax.stairs(truth[idx] * scale, edges, color="#5790fc", ls="dotted",
                  lw=3, baseline=None, label="Pythia8")

        ax.set_xlabel(rf"$\log_{{10}}(\rho^2)$, {mode}", fontsize=PUB_LABEL_FONTSIZE)
        ax.set_ylabel(r"$d\sigma / d\log_{10}(\rho^{2})$ (pb)",
                      fontsize=PUB_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        ax.set_xlim(edges[0], edges[-1])
        top = float(np.max(val + band))
        ax.set_ylim(0, top * 1.55)
        # CMS block sits inside at top-left, so the legend moves right and the
        # pT label rides just under the block.
        ax.legend(loc="upper right", fontsize=PUB_LEGEND_FONTSIZE)
        hep.cms.label("Internal", data=True, lumi=138, com=13, loc=2, ax=ax)
        ax.text(0.04, 0.78, block["label"], transform=ax.transAxes,
                ha="left", va="top", fontsize=PUB_ANNOTATION_FONTSIZE)

        path = out_dir / f"absolute_unfolded_{mode}_pt{k}.pdf"
        stamp_figure(fig, inputs=f"zjet rho absolute, {mode} pt{k}")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")

        summary.append({
            "pt_bin": block["label"],
            "sigma_pb_per_bin": (y_abs[idx] / LUMI_PB).tolist(),
            "frac_stat": (stat[idx] / y_abs[idx]).tolist(),
            "frac_total": (total[idx] / y_abs[idx]).tolist(),
        })

    (out_dir / f"absolute_unfolded_summary_{mode}.json").write_text(
        json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()
    for groomed in (True, False):
        run(groomed, args.tag)


if __name__ == "__main__":
    main()
