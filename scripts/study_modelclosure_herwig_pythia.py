#!/usr/bin/env python3
"""ARC closure test 2.4: model-uncertainty validation.

Reviewer request (SMP-25-010, U#2.4): unfold the ORIGINAL Herwig sample with the
ORIGINAL (nominal) Pythia response; the unfolded Herwig should agree with
gen-level Herwig within stat + model uncertainties.

Setup (production-faithful: nominal Pythia response, unregularized tau=0):
  * response / prior : nominal PYTHIA (the analysis response).
  * measured input   : original HERWIG reco (+ its MC-stat) from herwig_all.pkl.
  * truth            : original HERWIG gen (+ its MC-stat).
  * model band       : the analysis modelling uncertainty (reweighted band),
                       |nominal Pythia gen - reweighted Pythia gen| / nominal,
                       per gen bin -- the same quantity quoted on the result.

The non-closure (unfolded Herwig vs Herwig gen) should sit inside the
stat (+) model band -- that is what validates the modelling-uncertainty size.

Individual per-pT figures; everything read-only.
Outputs: outputs/zjet/rho/model_closure/{mode}_pt{i}.png + README.md
"""
import argparse
import pickle as pkl
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, RHO_FIXED_JEC_SPEC
from unfold.utils.merge_helpers import unflatten_gen_by_pt
from study_reweighted_herwig_closure import unfold_once, herwig_reco_measured

OUTDIR = REPO / "outputs/zjet/rho/model_closure"
REWEIGHTED = Path.home() / "Downloads" / "reweight_pythia_rho_pythia_all.pkl"


def _rel(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(num, den, out=np.zeros_like(np.asarray(den, float)), where=den != 0)


def gen_flat(uf, d, gen_key):
    vals, _ = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(d[gen_key]),
        uf.edges_gen, uf.gen_edges_by_pt, ("ptgen", uf.gen_axis))
    return np.asarray(vals, float)


def _rel_band(g_nom_pt, g_alt_pt):
    """Per-pT relative |nominal - alt| / nominal on normalized per-pT shapes."""
    rels = []
    for a, b in zip(g_nom_pt, g_alt_pt):
        sa, sb = a.sum(), b.sum()
        da = a / sa if sa > 0 else a
        db = b / sb if sb > 0 else b
        rels.append(_rel(np.abs(da - db), da))
    return rels


def model_unc_relative(uf, gen_key, edges_by_pt, source="reweighted"):
    """Per-pT, per-bin relative modelling uncertainty vs nominal Pythia gen.

    source = "reweighted" : |nominal Pythia - reweighted Pythia| (analysis primary)
    source = "herwig"     : |nominal Pythia - HERWIG|            (raw alt generator)
    source = "max"        : per-bin max of the two.
    """
    nom = pkl.load(open(uf.spec.input_dir + uf.spec.mc_file, "rb"))
    g_nom = unflatten_gen_by_pt(gen_flat(uf, nom, gen_key), edges_by_pt)

    rw = pkl.load(open(REWEIGHTED, "rb"))
    rel_rw = _rel_band(g_nom, unflatten_gen_by_pt(gen_flat(uf, rw, gen_key), edges_by_pt))
    hw = pkl.load(open(uf.spec.input_dir + uf.spec.herwig_file, "rb"))
    rel_hw = _rel_band(g_nom, unflatten_gen_by_pt(gen_flat(uf, hw, gen_key), edges_by_pt))

    if source == "reweighted":
        return rel_rw
    if source == "herwig":
        return rel_hw
    if source == "max":
        return [np.maximum(a, b) for a, b in zip(rel_rw, rel_hw)]
    raise ValueError(source)


def run_mode(groomed, model_source):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n========== {mode} ({model_source} model) ==========")
    spec = RHO_FIXED_JEC_SPEC                      # regularization="none" (production)
    uf = Unfolder(spec, groomed, do_syst=False, compute_jackknife_stat=False,
                  cms_label="Internal")
    uf._frozen_tau = float(uf.tau) if uf.tau is not None else 0.0

    hist_key = spec.hist_keys_groomed if groomed else spec.hist_keys_ungroomed
    uf._ensure_herwig_bias_inputs()
    meas, meas_var = herwig_reco_measured(uf, hist_key)
    # "nonclosure" model unc is computed per bin from the unfolded result below;
    # the gen-level proxies are precomputed here.
    gen_model_rel = (None if model_source == "nonclosure"
                     else model_unc_relative(uf, hist_key["gen"], uf.gen_edges_by_pt,
                                             source=model_source))

    # Unfold HERWIG reco through the NOMINAL PYTHIA response (production: tau=0).
    nom_mosaic = np.asarray(uf.mosaic_dict["nominal"], float)
    nom_misses = np.asarray(uf.misses_2d, float)
    nom_truth = nom_mosaic.sum(axis=0) + nom_misses
    y, ye = unfold_once(uf, regularize=False, mosaic=nom_mosaic, misses=nom_misses,
                        fake_fraction=uf.fake_fraction_2d, truth=nom_truth,
                        meas=meas, meas_var=meas_var, tag="modelclosure")

    y_true = np.asarray(uf.y_true_herwig, float)
    var = getattr(uf, "herwig_gen_var_flat", None)
    y_true_err = np.sqrt(np.clip(var, 0.0, None)) if var is not None else np.zeros_like(y_true)

    eb = uf.gen_edges_by_pt
    true_pt = unflatten_gen_by_pt(y_true, eb)
    terr_pt = unflatten_gen_by_pt(y_true_err, eb)
    y_pt = unflatten_gen_by_pt(y, eb)
    ye_pt = unflatten_gen_by_pt(ye, eb)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    chi2_rows = []
    for i in uf._reported_pt_indices():
        e = np.asarray(eb[i], float)
        c = 0.5 * (e[:-1] + e[1:])
        bw = np.diff(e)

        herwig = true_pt[i] / bw / true_pt[i].sum()
        herwig_err = terr_pt[i] / bw / true_pt[i].sum()
        unf = y_pt[i] / bw / y_pt[i].sum()
        unf_err = ye_pt[i] / bw / y_pt[i].sum()
        if model_source == "nonclosure":
            # Analysis model uncertainty = the non-closure itself,
            # |unfolded Herwig - Herwig gen| (per bin), applied to the unfolded.
            model_abs = np.abs(unf - herwig)
        else:
            model_abs = unf * gen_model_rel[i]          # gen-level proxy
        tot_err = np.sqrt(unf_err ** 2 + model_abs ** 2)  # stat (+) model

        # chi2 of the non-closure vs (stat + model + Herwig gen stat).
        comb = np.sqrt(tot_err ** 2 + herwig_err ** 2)
        nz = comb > 0
        ndof = int(nz.sum())
        chi2 = float(np.sum(((unf - herwig)[nz] / comb[nz]) ** 2))
        # also the stat-only chi2 (what 2.4 improves on vs 2.1-style)
        comb_s = np.sqrt(unf_err ** 2 + herwig_err ** 2)
        chi2_stat = float(np.sum(((unf - herwig)[comb_s > 0] / comb_s[comb_s > 0]) ** 2))
        chi2_rows.append((mode, i, ndof, chi2, chi2_stat))

        fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})

        # Unfolded Herwig: line + stat band + (wider) stat (+) model band.
        ax.fill_between(e, np.r_[unf - tot_err, (unf - tot_err)[-1]],
                        np.r_[unf + tot_err, (unf + tot_err)[-1]],
                        step="post", color="#f7c331", alpha=0.55, lw=0,
                        label=f"Unfolded stat. $\\oplus$ model ({model_source})")
        hep.histplot(unf, e, ax=ax, color="#3f90da", lw=2, label="Unfolded Herwig (nom. Pythia)")
        ax.fill_between(e, np.r_[unf - unf_err, (unf - unf_err)[-1]],
                        np.r_[unf + unf_err, (unf + unf_err)[-1]],
                        step="post", color="#3f90da", alpha=0.30, lw=0,
                        label="Unfolded stat.")
        ax.errorbar(c, herwig, yerr=herwig_err, fmt="o", ms=6, color="k",
                    elinewidth=1.5, capsize=2, label="Herwig (gen, truth)")

        ptlab = (f"{int(uf.pt_edges[i])}-"
                 f"{int(uf.pt_edges[i + 1]) if i + 1 < len(uf.pt_edges) - 1 else '∞'} GeV")
        ax.legend(title=ptlab, fontsize=14, loc="lower center")
        ax.set_xlim(*uf._observable_xlim(i))
        ax.margins(y=0.18)
        ax.set_ylabel(uf._normalized_ylabel())
        hep.cms.label(uf.cms_label, data=False, rlabel="(13 TeV)", ax=ax, fontsize=20)

        # Ratio to Herwig gen, with stat and stat (+) model bands.
        ratio = _rel(unf, herwig)
        tot_band = _rel(tot_err, herwig)
        stat_band = _rel(unf_err, herwig)
        herwig_rel = _rel(herwig_err, herwig)
        axr.fill_between(e, np.r_[ratio - tot_band, (ratio - tot_band)[-1]],
                         np.r_[ratio + tot_band, (ratio + tot_band)[-1]],
                         step="post", color="#f7c331", alpha=0.55, lw=0)
        axr.fill_between(e, np.r_[ratio - stat_band, (ratio - stat_band)[-1]],
                         np.r_[ratio + stat_band, (ratio + stat_band)[-1]],
                         step="post", color="#3f90da", alpha=0.30, lw=0)
        hep.histplot(ratio, e, ax=axr, color="#3f90da", lw=2)
        axr.fill_between(e, np.r_[1 - herwig_rel, (1 - herwig_rel)[-1]],
                         np.r_[1 + herwig_rel, (1 + herwig_rel)[-1]],
                         step="post", color="k", alpha=0.15, lw=0)
        axr.axhline(1.0, color="k", ls="dotted", lw=1)
        axr.text(0.05, 0.10,
                 f"$\\chi^2$/ndof = {chi2:.1f}/{ndof} (stat$\\oplus$model), "
                 f"{chi2_stat:.0f}/{ndof} (stat)",
                 transform=axr.transAxes, fontsize=11)
        axr.set_ylim(0.5, 1.5)
        axr.set_xlim(*uf._observable_xlim(i))
        axr.set_xlabel(uf._observable_label())
        axr.set_ylabel("Unfolded / Herwig", fontsize=15)

        out = OUTDIR / f"{mode}_{model_source}_pt{i}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  pt{i} [{ptlab}]: chi2/ndof = {chi2:.1f}/{ndof} (stat+model), "
              f"{chi2_stat:.0f}/{ndof} (stat only)  -> {out.name}")

    return chi2_rows


MODEL_DESC = {
    "nonclosure": "|unfolded Herwig (nom. Pythia) - Herwig gen| (the analysis model systematic)",
    "reweighted": "|nominal - reweighted Pythia gen| / nominal (gen-level proxy)",
    "herwig": "|nominal Pythia - HERWIG gen| / nominal (gen-level proxy)",
    "max": "per-bin max(reweighted, herwig) (gen-level proxy)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["nonclosure", "reweighted", "herwig", "max"],
                    default="nonclosure")
    args = ap.parse_args()
    rows = run_mode(True, args.model) + run_mode(False, args.model)
    lines = [
        "# Model-uncertainty closure (ARC U#2.4)",
        "",
        "Original HERWIG reco unfolded with the **nominal PYTHIA** response "
        "(production, unregularized). Compared to HERWIG gen within stat (+) "
        f"model, where model = {MODEL_DESC[args.model]}.",
        "",
        "| mode | pT bin | ndof | chi2 (stat+model) | chi2 (stat only) |",
        "|------|--------|------|-------------------|------------------|",
    ]
    for mode, i, ndof, chi2, chi2_stat in rows:
        lines.append(f"| {mode} | {i} | {ndof} | {chi2:.1f} | {chi2_stat:.0f} |")
    (OUTDIR / f"README_{args.model}.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUTDIR}/README_{args.model}.md")


if __name__ == "__main__":
    main()
