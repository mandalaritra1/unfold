#!/usr/bin/env python3
"""ARC closure test 2.3: unfold the HERWIG sample with the reweighted-PYTHIA model.

Reviewer request (SMP-25-010, U#2.3): unfold the Herwig sample with the
reweighted Pythia sample; ideally the unfolded Herwig agrees with gen-level
Herwig within statistical uncertainties.

Reads the reweighted-PYTHIA response matrix DIRECTLY from the skimmer-produced
reweighted file (no gen-ratio reconstruction):

  * response / prior : reweighted-PYTHIA file -> response_matrix_rho_{g,u}
                       (PYTHIA reweighted at event level to look like HERWIG).
  * measured input   : HERWIG reco (+ its MC-stat) from ``herwig_all.pkl``.
  * truth overlay    : HERWIG gen (+ its MC-stat) from ``herwig_all.pkl``.

Two regularization settings are produced for comparison: the analysis'
ratio-curvature regularization (tau from the nominal L-curve scan) and no
regularization. Plot: unfolded = line + shaded stat band; HERWIG gen = points
with error-bar sticks.

Every pkl is opened read-only; nothing is written back to an input file.

Outputs under outputs/zjet/rho/reweighted_herwig_closure/:
  <mode>_<reg|noreg>_pt<j>.png  and  README.md
"""
import argparse
import pickle as pkl
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, RHO_FIXED_JEC_SPEC
from unfold.utils.merge_helpers import unflatten_gen_by_pt

OUTDIR = REPO / "outputs/zjet/rho/reweighted_herwig_closure"
DEFAULT_REWEIGHTED = Path.home() / "Downloads" / "reweight_pythia_rho_pythia_all.pkl"


def _rel(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(num, den, out=np.zeros_like(np.asarray(den, float)), where=den != 0)


def build_reweighted_inputs(uf, reweighted_path, hist_key):
    """Response mosaic, misses and fake fraction from the reweighted-PYTHIA file."""
    d = pkl.load(open(reweighted_path, "rb"))
    _, mosaic = uf._prepared_response_mosaic(d[hist_key["response"]], "nominal")
    matched_reco = mosaic.sum(axis=1)
    matched_gen = mosaic.sum(axis=0)
    gen_flat, _ = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(d[hist_key["gen"]]),
        uf.edges_gen, uf.gen_edges_by_pt, ("ptgen", uf.gen_axis),
    )
    reco_flat, _ = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(d[hist_key["reco"]]),
        uf.edges, uf.reco_edges_by_pt, ("ptreco", uf.reco_axis),
    )
    misses = gen_flat - matched_gen
    fakes = reco_flat - matched_reco
    fake_fraction = uf._compute_fake_fraction(fakes, matched_reco)
    truth = matched_gen + misses                       # reweighted gen (prior/bias)
    return mosaic, misses, fake_fraction, truth, reco_flat


def pt_scale_to_model(uf, meas, meas_var, model_reco_flat):
    """Scale each HERWIG reco pT slice so the input pT distribution matches the
    model (reweighted PYTHIA). Removes the pT-spectrum mismatch as a confounder
    before unfolding (the rho response has ~28% cross-pT migration). Returns the
    scaled (values, variances) and the per-pT factors."""
    edges = uf.reco_edges_by_pt
    offs = np.cumsum([0] + [len(e) - 1 for e in edges])
    meas = np.array(meas, float); meas_var = np.array(meas_var, float)
    m_tot, mod_tot = meas.sum(), float(np.asarray(model_reco_flat).sum())
    w = np.ones(len(edges))
    for i in range(len(edges)):
        s, e = offs[i], offs[i + 1]
        m_slice = meas[s:e].sum()
        mod_slice = float(np.asarray(model_reco_flat)[s:e].sum())
        if m_slice > 0 and mod_tot > 0:
            w[i] = (mod_slice / mod_tot) / (m_slice / m_tot)
            meas[s:e] *= w[i]
            meas_var[s:e] *= w[i] ** 2
    return meas, meas_var, w


def herwig_reco_measured(uf, hist_key):
    """Full HERWIG reco spectrum + MC-stat variance (the unfolding input)."""
    d = pkl.load(open(uf.spec.input_dir + uf.spec.herwig_file, "rb"))
    vals, var = uf._flatten_prepared_2d(
        uf._select_nominal_histogram(d[hist_key["reco"]]),
        uf.edges, uf.reco_edges_by_pt, ("ptreco", uf.reco_axis),
    )
    return vals, var


def unfold_once(uf, *, regularize, mosaic, misses, fake_fraction, truth,
                meas, meas_var, tag):
    """Unfold HERWIG reco through the reweighted response; return (y, ye)."""
    uf.regularization = "ratio_curvature" if regularize else "none"
    uf.tau = uf._frozen_tau if regularize else None
    uf.misses_2d_dict = getattr(uf, "misses_2d_dict", {})
    uf.misses_2d_dict[tag] = misses
    saved_ff = uf.fake_fraction_2d
    saved_stat = uf.response_matrix_stat_available
    uf.fake_fraction_2d = fake_fraction          # model fakes (reweighted Pythia)
    uf.response_matrix_stat_available = False     # band = HERWIG input stat only
    try:
        uf._perform_unfold(systematic=tag, meas_flat=meas, meas_var=meas_var,
                           resp_np=mosaic, true_flat_override=truth)
    finally:
        uf.fake_fraction_2d = saved_ff
        uf.response_matrix_stat_available = saved_stat
    return (np.asarray(uf.y_unf_dict[tag], float),
            np.asarray(uf.ye_unf_dict[tag], float))


def run_mode(groomed, reweighted_path):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n========== {mode} ==========")
    spec = replace(RHO_FIXED_JEC_SPEC, regularization="ratio_curvature")
    uf = Unfolder(spec, groomed, do_syst=False, compute_jackknife_stat=False,
                  cms_label="Internal")
    uf._frozen_tau = float(uf.tau)
    print(f"  ratio-curvature tau = {uf._frozen_tau:.4g}")

    hist_key = spec.hist_keys_groomed if groomed else spec.hist_keys_ungroomed
    mosaic, misses, fake_fraction, truth, model_reco = build_reweighted_inputs(uf, reweighted_path, hist_key)
    uf._ensure_herwig_bias_inputs()
    meas, meas_var = herwig_reco_measured(uf, hist_key)
    # Match the HERWIG input pT spectrum to the (reweighted-PYTHIA) model.
    meas, meas_var, w_pt = pt_scale_to_model(uf, meas, meas_var, model_reco)
    print(f"  per-pT input scale (Herwig->model): {np.round(w_pt, 3)}")

    results = {}
    for regularize, key in [(True, "reg"), (False, "noreg")]:
        results[key] = unfold_once(
            uf, regularize=regularize, mosaic=mosaic, misses=misses,
            fake_fraction=fake_fraction, truth=truth, meas=meas, meas_var=meas_var,
            tag=f"rw_herwig_{key}",
        )

    y_true = np.asarray(uf.y_true_herwig, float)
    var = getattr(uf, "herwig_gen_var_flat", None)
    y_true_err = np.sqrt(np.clip(var, 0.0, None)) if var is not None else np.zeros_like(y_true)

    edges_by_pt = uf.gen_edges_by_pt
    true_pt = unflatten_gen_by_pt(y_true, edges_by_pt)
    terr_pt = unflatten_gen_by_pt(y_true_err, edges_by_pt)
    prior_pt = unflatten_gen_by_pt(np.asarray(truth, float), edges_by_pt)  # reweighted Pythia gen

    OUTDIR.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    chi2_rows = []
    for key, label in [("reg", "regularized"), ("noreg", "unregularized")]:
        y, ye = results[key]
        y_pt = unflatten_gen_by_pt(y, edges_by_pt)
        ye_pt = unflatten_gen_by_pt(ye, edges_by_pt)

        for i in uf._reported_pt_indices():
            e = np.asarray(edges_by_pt[i], float)
            c = 0.5 * (e[:-1] + e[1:])
            bw = np.diff(e)

            herwig = true_pt[i] / bw / true_pt[i].sum()
            herwig_err = terr_pt[i] / bw / true_pt[i].sum()
            unf = y_pt[i] / bw / y_pt[i].sum()
            unf_err = ye_pt[i] / bw / y_pt[i].sum()
            prior = prior_pt[i] / bw / prior_pt[i].sum()   # reweighted Pythia gen

            comb = np.sqrt(unf_err ** 2 + herwig_err ** 2)
            nz = comb > 0
            ndof = int(nz.sum())
            chi2 = float(np.sum(((unf - herwig)[nz] / comb[nz]) ** 2))
            chi2_rows.append((mode, label, i, ndof, chi2))

            fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

            # Reweighted Pythia gen (the model / unfolding prior).
            hep.histplot(prior, e, ax=ax, color="#964a8b", lw=2, ls="dotted",
                         label="Reweighted Pythia (gen, prior)")
            # Unfolded Herwig: line + shaded stat band -- should land on Herwig gen.
            hep.histplot(unf, e, ax=ax, color="#3f90da", lw=2, label="Unfolded Herwig")
            ax.fill_between(
                e, np.r_[unf - unf_err, (unf - unf_err)[-1]],
                np.r_[unf + unf_err, (unf + unf_err)[-1]],
                step="post", color="#3f90da", alpha=0.30, lw=0,
                label="Unfolded stat. unc.")
            # Herwig gen (truth): points + sticks.
            ax.errorbar(c, herwig, yerr=herwig_err, fmt="o", ms=6, color="k",
                        elinewidth=1.5, capsize=2, label="Herwig (gen, truth)")

            ptlab = (f"{int(uf.pt_edges[i])}-"
                     f"{int(uf.pt_edges[i + 1]) if i + 1 < len(uf.pt_edges) - 1 else '∞'} GeV")
            ax.legend(title=ptlab, fontsize=15, loc="lower center")
            ax.set_xlim(*uf._observable_xlim(i))
            ax.margins(y=0.18)
            ax.set_ylabel(uf._normalized_ylabel())
            hep.cms.label(uf.cms_label, data=False, rlabel="(13 TeV)", ax=ax, fontsize=20)

            # Ratio to Herwig gen: unfolded (blue, band) and prior (purple).
            ratio = _rel(unf, herwig)
            ratio_band = _rel(unf_err, herwig)
            herwig_rel = _rel(herwig_err, herwig)
            axr.fill_between(
                e, np.r_[1 - herwig_rel, (1 - herwig_rel)[-1]],
                np.r_[1 + herwig_rel, (1 + herwig_rel)[-1]],
                step="post", color="k", alpha=0.15, lw=0)
            hep.histplot(_rel(prior, herwig), e, ax=axr, color="#964a8b", lw=2, ls="dotted")
            axr.fill_between(
                e, np.r_[ratio - ratio_band, (ratio - ratio_band)[-1]],
                np.r_[ratio + ratio_band, (ratio + ratio_band)[-1]],
                step="post", color="#3f90da", alpha=0.30, lw=0)
            hep.histplot(ratio, e, ax=axr, color="#3f90da", lw=2)
            axr.axhline(1.0, color="k", ls="dotted", lw=1)
            axr.text(0.05, 0.10, f"$\\chi^2$/ndof={chi2:.1f}/{ndof}",
                     transform=axr.transAxes, fontsize=13)
            axr.set_ylim(0.5, 1.5)
            axr.set_xlim(*uf._observable_xlim(i))
            axr.set_xlabel(uf._observable_label())
            axr.set_ylabel("/ Herwig gen", fontsize=16)

            out = OUTDIR / f"{mode}_{key}_pt{i}.png"
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  {label:13s} pt{i} [{ptlab}]: chi2/ndof = {chi2:.1f}/{ndof}  -> {out.name}")

    return chi2_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reweighted", default=str(DEFAULT_REWEIGHTED))
    args = ap.parse_args()
    rw_path = Path(args.reweighted)
    if not rw_path.exists():
        sys.exit(f"reweighted file not found: {rw_path}")
    print(f"reweighted response from: {rw_path}")

    rows = run_mode(True, rw_path) + run_mode(False, rw_path)
    lines = [
        "# Reweighted-sample HERWIG closure (ARC U#2.3)",
        "",
        f"Response matrix read directly from `{rw_path.name}` (reweighted PYTHIA). "
        "Measured = HERWIG reco (+MC-stat) from `herwig_all.pkl`; truth = HERWIG "
        "gen (+MC-stat). Unfolded = line + stat band; Herwig gen = points + sticks.",
        "",
        "| mode | regularization | pT bin | ndof | chi2 |",
        "|------|----------------|--------|------|------|",
    ]
    for mode, label, i, ndof, c in rows:
        lines.append(f"| {mode} | {label} | {i} | {ndof} | {c:.1f} |")
    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUTDIR}/README.md")


if __name__ == "__main__":
    main()
