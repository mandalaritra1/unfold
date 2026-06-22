#!/usr/bin/env python3
"""Plot the full unrolled (rho x pt) RECO spectrum, Data vs MC, BEFORE unfolding.

Builds the Unfolder only up through _load_data (no TUnfold run) so we see the
exact arrays that feed the response: raw, un-normalized event counts.
"""
import os, sys
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
    """Replicate Unfolder.__init__ up to (and including) _load_data."""
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
    print("spec tag/input_dir:", spec.input_dir, "| mc:", spec.mc_file, "| data:", spec.data_file)
    u = build_inputs_only(spec, GROOMED)

    data = np.asarray(u.mosaic_2d, dtype=float)                 # raw measured reco (input to TUnfold)
    mc_matched = np.asarray(u.mosaic.sum(axis=1), dtype=float)  # MC matched reco (from response)
    fakes = np.asarray(u.fakes_2d, dtype=float)
    mc_full = mc_matched + fakes                                # MC total reco
    fake_frac = np.asarray(u.fake_fraction_2d, dtype=float)
    data_input = data * (1.0 - fake_frac)                       # fake-subtracted data actually unfolded

    reco_by_pt = u.reco_edges_by_pt
    pt_edges = np.asarray(u.pt_edges, dtype=float)
    counts_per_slice = [len(e) - 1 for e in reco_by_pt]
    N = len(data)

    print("\n--- NORMALIZATION CHECK (is the input normalized?) ---")
    print(f"unrolled length N = {N}, sum(counts_per_slice) = {sum(counts_per_slice)}")
    print(f"TOTAL data counts (sum of mosaic_2d)      = {data.sum():.6g}")
    print(f"TOTAL MC matched-reco counts              = {mc_matched.sum():.6g}")
    print(f"TOTAL MC full-reco counts (matched+fakes) = {mc_full.sum():.6g}")
    print(f"max data bin = {data.max():.6g}  -> NOT unit-area; raw event counts")
    print("per-pt-slice data integrals:", [f"{data[s:s+c].sum():.4g}"
          for s, c in zip(np.cumsum([0]+counts_per_slice[:-1]), counts_per_slice)])

    # ---- plot ----
    hep.style.use("CMS")
    x = np.arange(N)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax.step(x, mc_full, where="mid", color="red", lw=1.4, label="PYTHIA8 reco (matched + fakes)")
    ax.errorbar(x, data, yerr=np.sqrt(np.clip(data, 0, None)), fmt="o", ms=3,
                color="black", lw=0.8, label="Data (raw counts)")
    ax.set_yscale("log")
    ax.set_ylabel("Events (un-normalized)")
    ax.legend(loc="upper right", fontsize=13)
    hep.cms.label("Internal", data=True, lumi=138, com=13, ax=ax, fontsize=18)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(data, mc_full, out=np.zeros_like(data), where=mc_full > 0)
    axr.axhline(1, color="gray", ls="--", lw=1)
    axr.plot(x, ratio, "o", ms=3, color="black")
    axr.set_ylabel("Data / MC")
    axr.set_ylim(0.5, 1.5)
    axr.set_xlabel("Unrolled bin index  (rho within each $p_T$ slice, slices concatenated)")
    axr.set_xlim(-1, N)

    # pt-slice dividers + labels
    boundaries = np.cumsum(counts_per_slice)
    start = 0
    for i, b in enumerate(boundaries):
        for a in (ax, axr):
            a.axvline(b - 0.5, color="steelblue", ls=":", lw=1, alpha=0.7)
        lo = pt_edges[i] if i < len(pt_edges) else None
        hi = pt_edges[i + 1] if i + 1 < len(pt_edges) else np.inf
        lbl = f"{lo:g}-{'∞' if not np.isfinite(hi) else f'{hi:g}'}"
        ax.text((start + b - 1) / 2, ax.get_ylim()[1] * 0.4, lbl,
                ha="center", va="top", fontsize=9, color="steelblue", rotation=90)
        start = b

    mode = "groomed" if GROOMED else "ungroomed"
    out = OUT / f"input_unrolled_data_vs_mc_{mode}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
