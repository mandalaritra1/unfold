#!/usr/bin/env python3
"""pT spectrum, Data vs MC (reco), shape-only (density) comparison.

Integrates the unrolled reco mosaic over rho within each pT slice -> 4 pT bins,
then normalizes to a density (counts / total / bin-width, integral = 1) so only
the shape is compared. No unfolding is run.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

from unfold.tools.unfolder_core import Unfolder, get_spec

GROOMED = True
OUT = Path("/tmp/unfold_input_check")
OUT.mkdir(parents=True, exist_ok=True)


def build_inputs_only(spec, groomed):
    u = Unfolder.__new__(Unfolder)
    u.spec = spec
    u.reco_axis = spec.reco_axis
    u.gen_axis = spec.gen_axis
    u.groomed = groomed
    u.cms_label = "Internal"
    u.lumi = 138.0
    u.com = 13.0
    u.stat_propagation = getattr(spec, "stat_propagation", "legacy")
    u.regularization = getattr(spec, "regularization", "none")
    u.tau = getattr(spec, "tau", None)
    u.has_jackknife = True
    u.has_herwig = True
    u.has_validation_inputs = True
    u.response_matrix_stat_available = True
    u.first_reported_pt_bin = 0
    u.closure = False
    u.herwig_closure = False
    u.y_unf_dict = {}
    u._ensure_output_dirs()
    u._setup_binning()
    u._make_inputs_numpy()
    u._configure_systematics(False)
    u._load_data(
        filename_mc=spec.input_dir + spec.mc_file,
        filename_data=spec.input_dir + spec.data_file,
        filename_herwig=spec.input_dir + spec.herwig_file,
    )
    return u


def main():
    spec = get_spec("zjet", "rho")
    u = build_inputs_only(spec, GROOMED)

    data = np.asarray(u.mosaic_2d, dtype=float)
    mc_full = np.asarray(u.mosaic.sum(axis=1), dtype=float) + np.asarray(u.fakes_2d, dtype=float)

    reco_by_pt = u.reco_edges_by_pt
    counts_per_slice = [len(e) - 1 for e in reco_by_pt]
    starts = np.cumsum([0] + counts_per_slice[:-1])

    # integrate over rho within each pT slice -> 4 pT bins
    data_pt = np.array([data[s:s + c].sum() for s, c in zip(starts, counts_per_slice)])
    mc_pt = np.array([mc_full[s:s + c].sum() for s, c in zip(starts, counts_per_slice)])
    data_pt_err = np.sqrt(np.clip(data_pt, 0, None))
    mc_pt_err = np.sqrt(np.clip(mc_pt, 0, None))

    pt_edges = np.asarray(u.pt_edges, dtype=float)
    widths = np.diff(pt_edges)

    # density: integral == 1  -> counts / total / width
    def density(vals, errs):
        tot = vals.sum()
        return vals / tot / widths, errs / tot / widths

    d_dens, d_dens_err = density(data_pt, data_pt_err)
    m_dens, m_dens_err = density(mc_pt, mc_pt_err)

    print("pt_edges:", pt_edges)
    print("pt widths:", widths)
    print("data pt counts:", data_pt, " total:", data_pt.sum())
    print("mc   pt counts:", mc_pt, " total:", mc_pt.sum())
    print("data density:", d_dens, " sum*width =", (d_dens * widths).sum())
    print("mc   density:", m_dens)

    # density normalized to fraction-per-bin too (sum == 1), the cleanest
    # shape comparison when the last pT bin is an open (COM-ceiling) bin.
    d_frac = data_pt / data_pt.sum()
    m_frac = mc_pt / mc_pt.sum()
    d_frac_err = data_pt_err / data_pt.sum()
    m_frac_err = mc_pt_err / mc_pt.sum()

    labels = []
    for i in range(len(pt_edges) - 1):
        lo, hi = pt_edges[i], pt_edges[i + 1]
        labels.append(f"{lo:g}–{'∞' if hi >= 13000 else f'{hi:g}'}")

    # ---- plot: categorical x (4 pT bins), shape = fraction per bin ----
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    xc = np.arange(len(d_frac))

    ax.bar(xc, m_frac, width=0.6, color="red", alpha=0.25, label="PYTHIA8 (reco)")
    ax.errorbar(xc, m_frac, yerr=m_frac_err, fmt="none", color="red", lw=1.5)
    ax.errorbar(xc, d_frac, yerr=d_frac_err, fmt="o", ms=7, color="black", label="Data")
    ax.set_ylabel("Fraction of events per $p_T$ bin  (shape)")
    ax.set_ylim(0, max(d_frac.max(), m_frac.max()) * 1.25)
    ax.legend(loc="upper right", fontsize=14)
    hep.cms.label("Internal", data=True, lumi=138, com=13, ax=ax, fontsize=18)
    for xi, df, mf in zip(xc, d_frac, m_frac):
        ax.text(xi, df, f"  {df:.3f}", va="bottom", ha="left", fontsize=10)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(d_frac, m_frac, out=np.zeros_like(d_frac), where=m_frac > 0)
        ratio_err = np.divide(d_frac_err, m_frac, out=np.zeros_like(d_frac), where=m_frac > 0)
    axr.errorbar(xc, ratio, yerr=ratio_err, fmt="o", ms=7, color="black")
    axr.axhline(1, color="gray", ls="--", lw=1)
    axr.set_ylabel("Data / MC")
    axr.set_ylim(0.8, 1.2)
    axr.set_xticks(xc)
    axr.set_xticklabels(labels)
    axr.set_xlabel(r"Jet $p_T$ bin (GeV)")

    mode = "groomed" if GROOMED else "ungroomed"
    out = OUT / f"pt_density_data_vs_mc_{mode}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)


if __name__ == "__main__":
    main()
