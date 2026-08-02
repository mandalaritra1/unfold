#!/usr/bin/env python3
"""Gen-level rho shape comparison: HERWIG gen vs reweighted-PYTHIA gen (per pT).

Shows how well the reweighted PYTHIA reproduces the HERWIG gen rho shape -- the
root of the closure behavior. Nominal PYTHIA gen is overlaid (dotted) as the
"before reweighting" reference. Normalized to unit area per pT slice (analysis
binning). Read-only.

Outputs: outputs/zjet/rho/reweighted_herwig_closure/genshape_<mode>.png
"""
import pickle as pkl
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, RHO_FIXED_JEC_SPEC
from unfold.utils.merge_helpers import unflatten_gen_by_pt

OUTDIR = REPO / "outputs/zjet/rho/reweighted_herwig_closure"
REWEIGHTED = Path.home() / "Downloads" / "reweight_pythia_rho_pythia_all.pkl"


def gen_flat(uf, d, gen_key):
    vals, var = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(d[gen_key]),
        uf.edges_gen, uf.gen_edges_by_pt, ("ptgen", uf.gen_axis))
    return np.asarray(vals, float), np.asarray(var, float)


def run_mode(groomed):
    mode = "groomed" if groomed else "ungroomed"
    spec = RHO_FIXED_JEC_SPEC          # regularization="none" -> fast construction
    uf = Unfolder(spec, groomed, do_syst=False, compute_jackknife_stat=False)
    gen_key = (spec.hist_keys_groomed if groomed else spec.hist_keys_ungroomed)["gen"]

    hw = pkl.load(open(spec.input_dir + spec.herwig_file, "rb"))
    rw = pkl.load(open(REWEIGHTED, "rb"))

    h_val, h_var = gen_flat(uf, hw, gen_key)
    r_val, r_var = gen_flat(uf, rw, gen_key)

    eb = uf.gen_edges_by_pt
    h_pt = unflatten_gen_by_pt(h_val, eb); he_pt = unflatten_gen_by_pt(np.sqrt(np.clip(h_var, 0, None)), eb)
    r_pt = unflatten_gen_by_pt(r_val, eb); re_pt = unflatten_gen_by_pt(np.sqrt(np.clip(r_var, 0, None)), eb)

    hep.style.use("CMS")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for i in uf._reported_pt_indices():
        e = np.asarray(eb[i], float); c = 0.5 * (e[:-1] + e[1:]); bw = np.diff(e)
        def dens(a, ref): s = ref.sum(); return a / bw / s if s > 0 else a
        herwig = dens(h_pt[i], h_pt[i]); herwig_e = dens(he_pt[i], h_pt[i])
        rwp = dens(r_pt[i], r_pt[i]); rwp_e = dens(re_pt[i], r_pt[i])

        fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        # Reweighted Pythia gen: line + shaded stat band.
        hep.histplot(rwp, e, ax=ax, color="#3f90da", lw=2, label="Reweighted Pythia (gen)")
        ax.fill_between(e, np.r_[rwp - rwp_e, (rwp - rwp_e)[-1]],
                        np.r_[rwp + rwp_e, (rwp + rwp_e)[-1]],
                        step="post", color="#3f90da", alpha=0.30, lw=0,
                        label="Reweighted Pythia stat.")
        # Herwig gen: points with error-bar sticks.
        ax.errorbar(c, herwig, yerr=herwig_e, fmt="o", ms=6, color="k",
                    elinewidth=1.5, capsize=2, label="Herwig (gen)")
        ptlab = (f"{int(uf.pt_edges[i])}-"
                 f"{int(uf.pt_edges[i + 1]) if i + 1 < len(uf.pt_edges) - 1 else '∞'} GeV")
        ax.legend(title=ptlab, fontsize=16, loc="lower center")
        ax.set_xlim(*uf._observable_xlim(i))
        ax.margins(y=0.18)
        ax.set_ylabel(uf._normalized_ylabel())
        hep.cms.label("Internal", data=False, rlabel="(13 TeV)", ax=ax, fontsize=20)

        # ratio to Herwig (both uncertainties shown).
        def rel(a):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.divide(a, herwig, out=np.ones_like(a), where=herwig != 0)
        h_rel = rel(herwig_e)
        axr.fill_between(e, np.r_[1 - h_rel, (1 - h_rel)[-1]], np.r_[1 + h_rel, (1 + h_rel)[-1]],
                         step="post", color="k", alpha=0.15, lw=0, label="Herwig stat.")
        rwp_rel = rel(rwp_e)
        axr.fill_between(e, np.r_[rel(rwp) - rwp_rel, (rel(rwp) - rwp_rel)[-1]],
                         np.r_[rel(rwp) + rwp_rel, (rel(rwp) + rwp_rel)[-1]],
                         step="post", color="#3f90da", alpha=0.30, lw=0)
        hep.histplot(rel(rwp), e, ax=axr, color="#3f90da", lw=2)
        axr.axhline(1.0, color="k", ls="dotted", lw=1)
        axr.set_ylim(0.5, 1.5)
        axr.set_xlim(*uf._observable_xlim(i))
        axr.set_xlabel(uf._observable_label())
        axr.set_ylabel("Rew. / Herwig", fontsize=16)

        out = OUTDIR / f"genshape_{mode}_pt{i}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    run_mode(True)
    run_mode(False)
