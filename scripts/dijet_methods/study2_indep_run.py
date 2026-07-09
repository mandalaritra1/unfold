"""Driver for the dijet-rho independent-methods cross-check study (study2_indep).

Runs, for both groomed and ungroomed:
  (A) two independent central-value cross-checks -- forward-folding spline fit
      and positive regularized (NNLS) fit -- vs D'Agostini n=4;
  (B) a forensic audit of the roounfold_bayes tag's quoted statistical errors;
  (C) direct prior-dependence evidence for D'Agostini n=4 (and n=10).

Writes PNGs, README.md and metrics.json under
  outputs/dijet/2018/rho/method_study2/independent_methods/

HERWIG is not trusted (low stats) and is NOT used as a template or yardstick in
any headline result. It appears only as a thin dotted reference curve on the
comparison plot. No git commits. New files only. Runtime target < ~40 min.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

from scripts.dijet_methods import result as R
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.study2_indep_common import (
    study_binning, REPORTED_PT, gen_rho_centers, slice_gen,
    per_pt_normalized_ratio, MODEL_FLOOR_FRAC,
)
from scripts.dijet_methods import study2_indep_forwardfold as FF
from scripts.dijet_methods import study2_indep_posreg as PR
from scripts.dijet_methods import study2_indep_statforensics as SF
from scripts.dijet_methods import study2_indep_prior as PRI

OUTDIR = Path("outputs/dijet/2018/rho/method_study2/independent_methods")
PT_LABELS = {1: "200-290 GeV", 2: "290-400 GeV", 3: "400-570 GeV",
             4: "570-760 GeV", 5: "760-13000 GeV"}


def log(msg):
    print(f"[study2_indep] {msg}", flush=True)
    sys.stdout.flush()


def cms_label(ax, rlabel):
    hep.cms.label("", data=True, loc=0, ax=ax, rlabel=rlabel)
    ax.grid(alpha=0.25)


def overall(d):
    meds = [v["median_pct"] for v in d.values() if np.isfinite(v["median_pct"])]
    mxs = [v["max_pct"] for v in d.values() if np.isfinite(v["max_pct"])]
    return {"median_of_slice_medians_pct": float(np.median(meds)),
            "max_over_slices_pct": float(np.max(mxs))}


def run_mode(mode, n_toys=200):
    log(f"=== MODE {mode} ===")
    t0 = time.time()
    prep = R.prepare(mode, study_binning(), channel="dijet")
    prob = prep.prob

    # --- central: D'Agostini n=4 ---
    log("D'Agostini n=4 central ...")
    u_dag = M.dagostini(prob, prep.data_matched, n_iter=4)

    # --- Method 1: forward-folding spline fit (per slice) ---
    log("forward-fold spline fit ...")
    x_ff, ff_info = FF.run_forward_fold(prep)

    # --- Method 2: positive regularized (NNLS) fit ---
    log("positive regularized (NNLS) tau scan ...")
    x_pr, tau_pr, pr_scan = PR.run_posreg(prep, osc_thresh=0.05)
    log(f"  chosen tau = {tau_pr:.3g}")

    # --- central-value comparison metrics vs dagostini n=4 ---
    m_ff = per_pt_normalized_ratio(prob, x_ff, u_dag)
    m_pr = per_pt_normalized_ratio(prob, x_pr, u_dag)

    # --- Part 3a: honest toy stat spread (data-only and +response) ---
    log("toy stat spread: data-only ...")
    _, _, rel_data, _ = SF.toy_stat_spread(prep, n_toys=n_toys,
                                           response_toys=False, n_iter=4)
    log("toy stat spread: +response matrix ...")
    _, _, rel_full, _ = SF.toy_stat_spread(prep, n_toys=n_toys,
                                           response_toys=True, n_iter=4)
    mask_rep = SF.reported_mask(prob)
    toy_data = SF.summarize_rel(rel_data, mask_rep)
    toy_full = SF.summarize_rel(rel_full, mask_rep)

    tag_quoted, tag_path = SF.tag_quoted_rel(mode)
    src_defect = SF.sqrtN_vs_sumw2(mode)

    # --- Part 3c: n_iter regularization systematic (honest) ---
    log("n_iter regularization systematic ...")
    niter_sys, _ = SF.n_iter_systematic(prep)

    # ratio: honest toy median / tag quoted median (the gap)
    gap = (toy_data["median_pct"] / tag_quoted["median_pct"]
           if tag_quoted["median_pct"] > 0 else float("nan"))

    # --- Part 4: prior dependence ---
    log("prior-dependence scan (n=4, n=10) ...")
    prior_dep, _ = PRI.prior_dependence(prep, n_iters=(4, 10))

    metrics = {
        "central_crosscheck": {
            "forward_fold_spline": {"per_pt": m_ff, **overall(m_ff)},
            "positive_regularized": {"per_pt": m_pr, "chosen_tau": float(tau_pr),
                                     **overall(m_pr)},
        },
        "forward_fold_fit": {
            str(pt): {"scale": ff_info[pt]["scale"],
                      "scale_rel_err": ff_info[pt]["scale_rel_err"],
                      "n_knots": ff_info[pt]["n_knots"],
                      "chi2_ndf": ff_info[pt]["chi2_ndf"]}
            for pt in REPORTED_PT
        },
        "posreg_tau_scan": [
            {"tau": s["tau"], "osc": s["osc"], "neg": s["neg"]} for s in pr_scan
        ],
        "stat_audit": {
            "tag_quoted_input_err": {**tag_quoted, "source": tag_path},
            "toy_data_only": toy_data,
            "toy_data_plus_response": toy_full,
            "honest_over_quoted_ratio_median": float(gap),
            "source_defect": {
                **src_defect,
                "defect_file_line": "src/unfold/tools/roounfold_backend.py:63",
                "diagnosis": (
                    "run_rho_unfolding.py:134-136 writes "
                    "unfolded_input_errors = sqrt(diag(cov_data_np)). In the bayes "
                    "path (unfolder_core.py:2437-2439,2484) cov_data_np = "
                    "RooUnfoldBayes.Eunfold(kCovariance) (roounfold_backend.py:136), "
                    "which propagates ONLY the data-histogram bin errors. Those bin "
                    "errors are hard-coded to sqrt(|content|)=sqrt(N) at "
                    "roounfold_backend.py:63 (_np_to_th1), discarding the stored "
                    "sumw2. The dijet data is prescale-weighted so sumw2/N ~ 247; the "
                    "honest per-bin stat is sqrt(247) ~ 15.7x larger. The manifest's "
                    "stat_propagation='legacy' (jackknife) is vacuous: dijet has NO "
                    "jackknife inputs (has_jackknife=False, unfolder_core.py:615)."
                ),
            },
            "n_iter_regularization_systematic": niter_sys,
        },
        "prior_dependence": prior_dep,
    }

    # --- plots ---
    log("making comparison plot ...")
    make_comparison_plot(mode, prep, u_dag, x_ff, x_pr, rel_data)
    log("making prior-dependence plot ...")
    make_prior_plot(mode, prep)

    log(f"MODE {mode} done in {time.time()-t0:.0f}s")
    return metrics


def make_comparison_plot(mode, prep, u_dag, x_ff, x_pr, rel_data):
    prob = prep.prob
    xc = gen_rho_centers()
    fig, axes = plt.subplots(2, 3, layout="constrained",
                             figsize=(24, 14))
    axes = axes.ravel()
    for ax_i, pt in enumerate(REPORTED_PT):
        ax = axes[ax_i]
        gidx = slice_gen(prob, pt)

        def norm(v):
            s = v[gidx].sum()
            return v[gidx] / s if s > 0 else v[gidx]
        y_dag = norm(u_dag)
        y_ff = norm(x_ff)
        y_pr = norm(x_pr)
        y_pyth = norm(prob.gen)
        # honest toy stat band on dagostini (relative -> absolute on normalized)
        band = y_dag * (rel_data[gidx] / 100.0)

        ax.step(xc, y_pyth, where="mid", lw=2.0, color="#9c9ca1", ls="--",
                label="Pythia (prior)")
        if prep.her_truth is not None:
            y_her = norm(prep.her_truth)
            ax.step(xc, y_her, where="mid", lw=1.4, color="#7a21dd", ls=":",
                    label="Herwig (ref. only, distrusted)")
        ax.errorbar(xc, y_dag, yerr=band, fmt="o", color="black", ms=6,
                    capsize=3, lw=1.5, label="D'Agostini n=4 (honest toy stat)")
        ax.step(xc, y_ff, where="mid", lw=2.2, color="#5790fc",
                label="forward-fold spline fit")
        ax.step(xc, y_pr, where="mid", lw=2.2, color="#e42536",
                label="positive-reg NNLS fit")
        cms_label(ax, rlabel=f"dijet 2018  {PT_LABELS[pt]}")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        ax.set_ylabel("normalized truth / bin")
        top = max(y_dag.max(), y_ff.max(), y_pr.max()) * 1.55
        ax.set_ylim(0, top)
        if ax_i == 0:
            ax.legend(loc="upper left", fontsize=12, framealpha=0.9)
    for j in range(len(REPORTED_PT), len(axes)):
        axes[j].axis("off")
    out = OUTDIR / f"comparison_{mode}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log(f"  wrote {out}")


def make_prior_plot(mode, prep):
    """Prior-dependence: unfold real data with 3 priors at n=4 and n=10."""
    prob = prep.prob
    xc = gen_rho_centers()
    priors = PRI.build_priors(prob)
    fig, axes = plt.subplots(2, 3, layout="constrained", figsize=(24, 14))
    axes = axes.ravel()
    colors = {"nominal_pythia": "black", "tilt_0p6_1p4": "#f89c20",
              "flat_in_x": "#e42536"}
    labels = {"nominal_pythia": "prior = Pythia (nominal)",
              "tilt_0p6_1p4": "prior = tilt 0.6->1.4",
              "flat_in_x": "prior = flat in x"}
    unf4 = {name: M.dagostini(prob, prep.data_matched, n_iter=4, prior=pr.copy())
            for name, pr in priors.items()}
    unf10 = {name: M.dagostini(prob, prep.data_matched, n_iter=10, prior=pr.copy())
             for name, pr in priors.items()}
    for ax_i, pt in enumerate(REPORTED_PT):
        ax = axes[ax_i]
        gidx = slice_gen(prob, pt)

        def norm(v):
            s = v[gidx].sum()
            return v[gidx] / s if s > 0 else v[gidx]
        for name in priors:
            ax.step(xc, norm(unf4[name]), where="mid", lw=2.0,
                    color=colors[name], label=f"n=4  {labels[name]}")
            ax.step(xc, norm(unf10[name]), where="mid", lw=1.3, ls=":",
                    color=colors[name])
        cms_label(ax, rlabel=f"dijet 2018  {PT_LABELS[pt]}")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        ax.set_ylabel("normalized unfolded / bin")
        ax.set_ylim(bottom=0)
        if ax_i == 0:
            ax.legend(loc="upper left", fontsize=11, framealpha=0.9,
                      title="solid=n4, dotted=n10")
    for j in range(len(REPORTED_PT), len(axes)):
        axes[j].axis("off")
    out = OUTDIR / f"prior_dependence_{mode}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log(f"  wrote {out}")


def make_stat_audit_plot(all_metrics):
    modes = list(all_metrics.keys())
    fig, ax = plt.subplots(layout="constrained", figsize=(11, 8))
    x = np.arange(len(modes))
    w = 0.2
    quoted = [all_metrics[m]["stat_audit"]["tag_quoted_input_err"]["median_pct"] for m in modes]
    toy_d = [all_metrics[m]["stat_audit"]["toy_data_only"]["median_pct"] for m in modes]
    toy_f = [all_metrics[m]["stat_audit"]["toy_data_plus_response"]["median_pct"] for m in modes]
    reg = [all_metrics[m]["stat_audit"]["n_iter_regularization_systematic"]["n6_vs_n4"]["median_pct"] for m in modes]
    ax.bar(x - 1.5 * w, quoted, w, color="#9c9ca1", label="tag quoted (input err)")
    ax.bar(x - 0.5 * w, toy_d, w, color="#5790fc", label="honest toy (data only)")
    ax.bar(x + 0.5 * w, toy_f, w, color="#e42536", label="honest toy (data+response)")
    ax.bar(x + 1.5 * w, reg, w, color="#f89c20", label="n_iter reg syst |n6-n4|")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("median rel. uncertainty [%]")
    hep.cms.label("", data=True, loc=0, ax=ax, rlabel="dijet 2018 rho")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=13)
    out = OUTDIR / "stat_audit_bars.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log(f"wrote {out}")


def write_readme(all_metrics):
    lines = []
    A = lines.append
    A("# Dijet rho: independent-methods cross-check & Bayes stat-error audit\n")
    A("Study `study2_indep`. Two genuinely independent cross-checks of the")
    A("D'Agostini (iterative-Bayes) central value, a forensic audit of the")
    A("`unfolding_bayes` tag's quoted statistical uncertainties, and direct")
    A("prior-dependence evidence for D'Agostini.\n")
    A("Foundation reused read-only: `scripts/dijet_methods/{loader,methods,result}.py`.")
    A("Efficiencies are NEVER floored. Study binning: pt")
    A("`[0,200,290,400,570,760,13000]`, gen rho")
    A("`[-10,-6,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0]` (12 bins), native 25 reco")
    A("rho bins. Reported pt bins 1-5; pt<200 is a migration sink (~90% of the truth)")
    A("and is treated as a nuisance, never reported. **HERWIG is distrusted (low")
    A("stats) and is NOT used as a template/yardstick in any headline number**; it")
    A("appears only as a thin dotted reference curve on the comparison plot.\n")

    A("## A. Independent central-value cross-checks\n")
    A("### Method 1 - forward-folding spline fit (primary)")
    A("Per pt slice, fit the reco data *in that reco-pt block* by folding a")
    A("parametrized gen shape `x_pt = s_pt * pythia_gen_pt * exp(spline(x))` through")
    A("the intra-slice response block (x = log10(rho^2); natural cubic spline, 4")
    A("interior knots across the populated gen range; s_pt a free per-slice norm).")
    A("No matrix inversion, no Bayes. Cross-pt migration into the block is removed")
    A("using the MC (Pythia) out-of-slice prediction, and reco bins are weighted by")
    A("their intra-slice purity so untrusted low-rho / sink migration does not drive")
    A(f"the fit. chi2 uses a {int(100*MODEL_FLOOR_FRAC)}% model-systematic floor added in quadrature to the")
    A("Poisson-floored data variance so the fit does not chase sub-percent reco")
    A("structure. Minimized with `scipy.optimize.least_squares`; per-slice")
    A("normalization errors from the Jacobian (linearized/Gaussian -- caveat).")
    A("(`study2_indep_forwardfold.py`)\n")
    A("### Method 2 - positive regularized NNLS fit")
    A("`min ||(Px-m)/sigma||^2 + tau^2||L_ratio x||^2` s.t. `x>=0`, with L_ratio the")
    A("Pythia-prior-scaled second difference (rows `(1/x0[i-1],-2/x0[i],1/x0[i+1])`),")
    A("solved via a stacked `scipy.optimize.nnls`. tau scanned over `logspace(-2,2,9)`;")
    A("smallest tau with oscillation metric < 0.05 chosen. (`study2_indep_posreg.py`)\n")

    A("| mode | method | median(slice medians) % | max over slices % |")
    A("|---|---|---|---|")
    for mode, mm in all_metrics.items():
        cc = mm["central_crosscheck"]
        for label, key in [("forward-fold spline", "forward_fold_spline"),
                           ("positive-reg NNLS", "positive_regularized")]:
            d = cc[key]
            A(f"| {mode} | {label} | {d['median_of_slice_medians_pct']:.2f} | "
              f"{d['max_over_slices_pct']:.2f} |")
    A("")
    A("Per-slice forward-fold vs D'Agostini n=4 (normalized median |ratio-1| %):\n")
    A("| mode | pt1 | pt2 | pt3 | pt4 | pt5 |")
    A("|---|---|---|---|---|---|")
    for mode, mm in all_metrics.items():
        pp = mm["central_crosscheck"]["forward_fold_spline"]["per_pt"]
        cells = " | ".join(f"{pp[pt]['median_pct']:.1f}" for pt in REPORTED_PT)
        A(f"| {mode} | {cells} |")
    A("")
    A("Per-slice positive-reg NNLS vs D'Agostini n=4 (normalized median |ratio-1| %):\n")
    A("| mode | pt1 | pt2 | pt3 | pt4 | pt5 |")
    A("|---|---|---|---|---|---|")
    for mode, mm in all_metrics.items():
        pp = mm["central_crosscheck"]["positive_regularized"]["per_pt"]
        cells = " | ".join(f"{pp[pt]['median_pct']:.1f}" for pt in REPORTED_PT)
        A(f"| {mode} | {cells} |")
    A("")
    A("**Reading:** the positive-reg NNLS fit agrees with D'Agostini n=4 to")
    A("~4-6% (median) across all five reported slices in both modes -- an")
    A("independent confirmation of the central value. The forward-fold spline")
    A("agrees to ~8-13% in the well-populated, high-purity slices (groomed pt2-5,")
    A("ungroomed pt3-5); the lowest reported slices (pt1, and ungroomed pt2) are")
    A("dominated by the pt<200 migration sink whose low-rho modelling is distrusted,")
    A("so a clean intra-slice forward-fold there is not achievable and those")
    A("numbers should be read as a limitation, not a discrepancy. The two methods")
    A("together bracket D'Agostini n=4 and show no evidence of a biased central value.\n")

    A("## B. Stat-error forensics of the `unfolding_bayes` tag\n")
    A("### The defect (file:line chain)")
    A("- `run_rho_unfolding.py:134-136`:")
    A("  `unfolded_input_errors = sqrt(diag(unfolder.cov_data_np))`.")
    A("- `unfolder_core.py:2437-2439,2484` (bayes path): `want_cov=True` for the")
    A("  nominal unfold, so `cov_data_np = cov` returned by `bayes_unfold(...,")
    A("  with_covariance=True)`.")
    A("- `roounfold_backend.py:136`: that cov is `RooUnfoldBayes.Eunfold(kCovariance)`")
    A("  -- RooUnfold's analytic covariance, which propagates ONLY the data-histogram")
    A("  bin errors.")
    A("- **`roounfold_backend.py:63`** is the bug: `_np_to_th1` sets those bin")
    A("  errors to `sqrt(|content|) = sqrt(N)`, discarding the stored sumw2. The")
    A("  dijet data is prescale-WEIGHTED, so sumw2/N ~ 247 (median); the honest")
    A("  per-bin data stat is therefore `sqrt(247) ~ 15.7x` larger than what")
    A("  RooUnfold sees.")
    A("- The manifest's `stat_propagation='legacy'` (jackknife) is vacuous here:")
    A("  dijet inputs carry NO jackknife replicas (`has_jackknife=False`,")
    A("  `unfolder_core.py:615`), so the code falls through to RooUnfold's analytic")
    A("  `kCovariance` on a sqrt(N) data error.\n")

    A("### The numbers (median rel-stat over reported bins)\n")
    A("| mode | tag quoted % | honest toy data-only % | honest toy data+response % | honest/quoted | reg syst |n6-n4| % |")
    A("|---|---|---|---|---|---|")
    for mode, mm in all_metrics.items():
        sa = mm["stat_audit"]
        A(f"| {mode} | {sa['tag_quoted_input_err']['median_pct']:.4f} | "
          f"{sa['toy_data_only']['median_pct']:.3f} | "
          f"{sa['toy_data_plus_response']['median_pct']:.3f} | "
          f"{sa['honest_over_quoted_ratio_median']:.1f}x | "
          f"{sa['n_iter_regularization_systematic']['n6_vs_n4']['median_pct']:.3f} |")
    A("")
    A("Source-level confirmation (`measured` vs its stored variance in the npz):\n")
    A("| mode | implied sqrt(N)/N % | honest sqrt(sumw2)/N % | sumw2/N | underest. factor |")
    A("|---|---|---|---|---|")
    for mode, mm in all_metrics.items():
        sd = mm["stat_audit"]["source_defect"]
        A(f"| {mode} | {sd['implied_rel_from_sqrtN_median_pct']:.4f} | "
          f"{sd['honest_rel_from_sumw2_median_pct']:.4f} | "
          f"{sd['sumw2_over_N_median']:.1f} | {sd['underestimate_factor_median']:.1f}x |")
    A("")
    A("### Correct scheme")
    A("Quote the honest **toy-propagation** data-stat spread (Gaussian toys on the")
    A("measured spectrum using the STORED sumw2 variance, Poisson-floored, re-unfolded")
    A("through D'Agostini) -- what `result.unfold_with_unc` does and what the honest")
    A("columns above report. Equivalently, fix `_np_to_th1` to accept the per-bin")
    A("sumw2 error and use `kCovToys`. Quote the regularization uncertainty")
    A("separately as the n_iter systematic `|u(n=6)-u(n=4)|` (see table).\n")

    A("## C. Prior-dependence of D'Agostini n=4 (boss-facing)\n")
    A("Same real data unfolded with three priors per pt slice: (i) nominal Pythia")
    A("gen, (ii) a strong linear tilt 0.6->1.4 in x, (iii) flat-in-x (each")
    A("renormalized to the slice's Pythia integral). Spread of the unfolded result")
    A("across priors over reported bins:\n")
    A("| mode | n_iter | prior spread median % | p90 % | max % | flat-vs-nominal median % |")
    A("|---|---|---|---|---|---|")
    for mode, mm in all_metrics.items():
        pd = mm["prior_dependence"]
        for n in (4, 10):
            s = pd[f"n_iter_{n}"]["max_minus_min_spread"]
            flat = pd[f"n_iter_{n}"]["flat_in_x_vs_nominal"]
            A(f"| {mode} | {n} | {s['median_pct']:.2f} | {s['p90_pct']:.2f} | "
              f"{s['max_pct']:.2f} | {flat['median_pct']:.2f} |")
    A("")
    A("**Reading:** at n_iter=4 the unfolded central value moves by several percent")
    A("(median) and much more in the sparse tails when the prior is changed from")
    A("Pythia to flat-in-x -- direct, quantitative confirmation of the analysis")
    A("lead's concern that D'Agostini n=4 is prior-dependent. A gentle tilt moves")
    A("it far less (the tilt stays close to Pythia). Increasing to n_iter=10 reduces")
    A("the prior spread (more iterations wash out the prior, as expected) but does")
    A("not eliminate it; n=10 also amplifies statistical noise. This is exactly the")
    A("regularization/prior-dependence trade-off that motivates the independent")
    A("cross-checks in Part A.\n")

    A("## Files")
    A("- `comparison_<mode>.png` - D'Agostini n=4 (honest toy band) vs forward-fold")
    A("  spline vs positive-reg NNLS, per pt slice (Pythia prior overlaid; Herwig")
    A("  dotted reference only).")
    A("- `prior_dependence_<mode>.png` - D'Agostini unfolded with 3 priors, n=4")
    A("  (solid) and n=10 (dotted), per pt slice.")
    A("- `stat_audit_bars.png` - tag-quoted vs honest-toy vs reg-systematic.")
    A("- `metrics.json` - all numbers (`central_crosscheck`, `stat_audit`,")
    A("  `prior_dependence`).\n")

    (OUTDIR / "README.md").write_text("\n".join(lines))
    log("wrote README.md")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n_toys = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    all_metrics = {}
    for mode in ["groomed", "ungroomed"]:
        all_metrics[mode] = run_mode(mode, n_toys=n_toys)
    make_stat_audit_plot(all_metrics)
    write_readme(all_metrics)
    with open(OUTDIR / "metrics.json", "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    log("wrote metrics.json")

    log("================ SUMMARY ================")
    for mode, mm in all_metrics.items():
        cc = mm["central_crosscheck"]
        sa = mm["stat_audit"]
        pd = mm["prior_dependence"]
        log(f"[{mode}] NNLS vs dag n4 median="
            f"{cc['positive_regularized']['median_of_slice_medians_pct']:.2f}% "
            f"(tau={cc['positive_regularized']['chosen_tau']:.2g}); "
            f"forward-fold median="
            f"{cc['forward_fold_spline']['median_of_slice_medians_pct']:.2f}%")
        log(f"[{mode}] STAT: tag quoted={sa['tag_quoted_input_err']['median_pct']:.4f}% "
            f"vs honest toy data-only={sa['toy_data_only']['median_pct']:.3f}% "
            f"=> {sa['honest_over_quoted_ratio_median']:.1f}x too small "
            f"(source factor {sa['source_defect']['underestimate_factor_median']:.1f}x)")
        log(f"[{mode}] PRIOR spread n4 median="
            f"{pd['n_iter_4']['max_minus_min_spread']['median_pct']:.2f}% "
            f"p90={pd['n_iter_4']['max_minus_min_spread']['p90_pct']:.2f}%; "
            f"n10 median={pd['n_iter_10']['max_minus_min_spread']['median_pct']:.2f}%")


if __name__ == "__main__":
    main()
