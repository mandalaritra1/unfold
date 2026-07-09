"""Plots + README for study2 TUnfold-rescue (v2). Reads metrics.json.

v2: bias axis = data-driven pseudo-truth closure (Herwig yardstick deprecated:
low stats). Adds the prior-dependence table and the prior-free candidate
(unregularized weighted lstsq at the winning coarse binning) to the money plot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/dijet/2018/rho/method_study2/tunfold_rescue")
PROD_TAU = 0.79
PROD_NITER = 4
BIAS_FLOOR = 2e-4   # display floor for the log bias axis (exact closures ~1e-13)


def load():
    with open(OUT / "metrics.json") as fh:
        return json.load(fh)


def _tik_arrays(fam):
    keys = sorted(fam, key=lambda k: fam[k]["tau"])
    tau = np.array([fam[k]["tau"] for k in keys])
    return keys, tau, fam


def _bayes_arrays(fam):
    keys = sorted(fam, key=lambda k: fam[k]["n_iter"])
    ni = np.array([fam[k]["n_iter"] for k in keys])
    return keys, ni, fam


def money_plot(m):
    """Bias-variance plane: x = pseudo-truth closure bias, y = toy spread."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, mode in zip(axes, ("groomed", "ungroomed")):
        fam = m[mode]["families"]
        ks, tau, f = _tik_arrays(fam["tikhonov_ratio"])
        bx = [max(f[k]["closure_bias_med"], BIAS_FLOOR) for k in ks]
        by = [f[k]["toy_spread"] for k in ks]
        ax.plot(bx, by, "-o", ms=3, color="C0", label="Tikhonov ratio-curv (tau scan)")
        it = int(np.argmin(np.abs(tau - PROD_TAU)))
        ax.plot(bx[it], by[it], "*", ms=16, color="C0", mec="k",
                label=f"prod tau~{PROD_TAU}")
        ksc, tauc, fc = _tik_arrays(fam["tikhonov_curvature"])
        bxc = [max(fc[k]["closure_bias_med"], BIAS_FLOOR) for k in ksc]
        byc = [fc[k]["toy_spread"] for k in ksc]
        ax.plot(bxc, byc, "-s", ms=3, color="C3", alpha=0.7,
                label="Tikhonov curv-of-dev (tau scan)")
        ksb, ni, fb = _bayes_arrays(fam["bayes"])
        bbx = [max(fb[k]["closure_bias_med"], BIAS_FLOOR) for k in ksb]
        bby = [fb[k]["toy_spread"] for k in ksb]
        ax.plot(bbx, bby, "-^", ms=4, color="C2", label="D'Agostini (n_iter scan)")
        ib = int(np.argmin(np.abs(ni - PROD_NITER)))
        ax.plot(bbx[ib], bby[ib], "*", ms=16, color="C2", mec="k",
                label=f"prod n_iter={PROD_NITER}")
        # prior-free candidate: unreg lstsq at winning coarse binning
        pf = m[mode]["prior_free_candidate"]
        if pf["toy_spread_med"] is not None:
            ax.plot(max(pf["closure_bias_med"], BIAS_FLOOR), pf["toy_spread_med"],
                    "D", ms=11, color="C1", mec="k",
                    label=f"unreg lstsq @ {pf['binning']} (prior-free, bias=0)")
        ax.set_xlabel("pseudo-truth closure bias  median |u/truth* - 1|")
        ax.set_ylabel("data-stat toy spread  median frac")
        ax.set_title(f"dijet rho {mode}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        ax.annotate(f"bias floor {BIAS_FLOOR:g}\n(exact closures plotted here)",
                    xy=(BIAS_FLOOR, ax.get_ylim()[0]), fontsize=6.5,
                    xytext=(5, 8), textcoords="offset points", color="0.4")
    fig.suptitle("Bias-variance plane, data-driven pseudo-truth closure (dijet rho 2018)")
    fig.tight_layout()
    fig.savefig(OUT / "money_bias_variance.png", dpi=130)
    plt.close(fig)
    print("wrote money_bias_variance.png")


def metric_vs_knob(m):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    metrics = ["neg_bins", "osc", "closure_bias_med", "toy_spread"]
    labels = ["# negative reported bins", "oscillation median|2nd diff u/prior|",
              "pseudo-truth closure bias med", "data-stat toy spread median frac"]
    for r, mode in enumerate(("groomed", "ungroomed")):
        fam = m[mode]["families"]
        ks, tau, f = _tik_arrays(fam["tikhonov_ratio"])
        ksc, tauc, fc = _tik_arrays(fam["tikhonov_curvature"])
        ksb, ni, fb = _bayes_arrays(fam["bayes"])
        for c, (met, lab) in enumerate(zip(metrics, labels)):
            ax = axes[r, c]
            def _v(d):
                x = d[met]
                return max(x, BIAS_FLOOR) if met == "closure_bias_med" else x
            ax.plot(tau, [_v(f[k]) for k in ks], "-o", ms=3, color="C0",
                    label="Tik ratio-curv")
            ax.plot(tauc, [_v(fc[k]) for k in ksc], "-s", ms=3, color="C3",
                    alpha=0.6, label="Tik curv-of-dev")
            ax.set_xscale("log")
            if met in ("osc", "closure_bias_med", "toy_spread"):
                ax.set_yscale("log")
            ax.axvline(PROD_TAU, color="C0", ls=":", alpha=0.6)
            ax.axvline(m[mode]["plateau_tau"], color="k", ls="--", alpha=0.5)
            bvals = [_v(fb[k]) for k in ksb]
            ax.axhspan(min(bvals), max(bvals), color="C2", alpha=0.12,
                       label="D'Agostini n=1..32 range")
            ax.axhline(_v(fb[str(PROD_NITER)]), color="C2", ls="--", alpha=0.7,
                       label=f"D'Agostini n={PROD_NITER}")
            ax.set_xlabel("tau")
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(f"{mode}: {met}", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Metric vs tau; dotted = production tau, dashed = plateau tau; "
                 "green = D'Agostini reference (dijet rho 2018)")
    fig.tight_layout()
    fig.savefig(OUT / "metric_vs_tau.png", dpi=120)
    plt.close(fig)
    print("wrote metric_vs_tau.png")


def bayes_vs_niter(m):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mode in zip(axes, ("groomed", "ungroomed")):
        fam = m[mode]["families"]["bayes"]
        ks, ni, f = _bayes_arrays(fam)
        for met, lab, c in [("osc", "osc median|2nd diff|", "C0"),
                            ("closure_bias_med", "pseudo-truth closure bias med", "C1"),
                            ("toy_spread", "toy spread med", "C2")]:
            ax.plot(ni, [max(f[k][met], BIAS_FLOOR) if met == "closure_bias_med"
                         else f[k][met] for k in ks], "-o", ms=4, color=c, label=lab)
        ax.axvline(PROD_NITER, color="k", ls=":", alpha=0.6, label=f"prod n={PROD_NITER}")
        ax.set_xlabel("n_iter")
        ax.set_yscale("log")
        ax.set_title(f"D'Agostini metrics {mode}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bayes_vs_niter.png", dpi=130)
    plt.close(fig)
    print("wrote bayes_vs_niter.png")


def binning_scan_plot(m):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for ax, mode in zip(axes, ("groomed", "ungroomed")):
        bs = m[mode]["binning_scan"]["binnings"]
        names = list(bs.keys())
        nb = [bs[n]["n_gen_bins_total"] for n in names]
        cond = [bs[n]["cond_P"] for n in names]
        pur = [bs[n]["purity_median"] for n in names]
        neg = [bs[n]["unreg_neg_bins"] for n in names]
        osc = [bs[n]["unreg_osc"] for n in names]
        rd = [bs[n]["unreg_vs_dago4_reldiff_med"] for n in names]
        order = np.argsort(nb)
        srt = lambda arr: [arr[i] for i in order]
        ax.plot(srt(nb), srt(cond), "-o", label="cond(P)")
        ax.plot(srt(nb), [o * 1000 for o in srt(osc)], "-s", label="unreg osc x1e3")
        ax.plot(srt(nb), [n2 * 50 + 1 for n2 in srt(neg)], "-^", label="unreg #neg x50 (+1)")
        ax.plot(srt(nb), [p * 1000 for p in srt(pur)], "-d", label="purity x1e3")
        ax.plot(srt(nb), [r * 1000 for r in srt(rd)], "-v", label="|unreg/dago4-1| x1e3")
        for i in order:
            ax.annotate(names[i], (nb[i], cond[i]), fontsize=7)
        winner = m[mode]["binning_scan"]["winner"]
        if winner is not None:
            wnb = bs[winner["name"]]["n_gen_bins_total"]
            ax.axvline(wnb, color="C1", ls="--", alpha=0.6,
                       label=f"winner {winner['name']}")
        ax.set_xlabel("# gen bins (total, incl sink slice)")
        ax.set_title(f"binning scan {mode} (unreg weighted lstsq)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "binning_scan.png", dpi=130)
    plt.close(fig)
    print("wrote binning_scan.png")


def write_readme(m):
    lines = []
    W = lines.append
    W("# Study 2 (v2): prior-free least-squares vs the Bayes family, dijet rho\n")
    W("x = log10(rho^2); pt blocks [200,290,400,570,760,inf] + pt<200 sink.")
    W("Foundation: scripts/dijet_methods/{loader,methods,result}.py. Data-term")
    W("inverse-variance weighted (Poisson-floored); ratio-curvature L matches the")
    W("production `ratio_curvature`. Reported = pt slices 1..5. Toys: 100,")
    W("data-stat only.\n")
    W("**Bias metric (v2): data-driven pseudo-truth closure.** The HERWIG")
    W("yardstick is DEPRECATED (low MC stats). truth* = smooth iterative")
    W("reweight of the Pythia gen (per pt slice, 4 passes, [0.25,0.5,0.25]")
    W("smoothing, w clipped to [0.2,5], anchored on populated bins) such that")
    W("P @ truth* matches the data matched-reco shape; pseudo-data = P @ truth*")
    W("is unfolded with the NOMINAL Pythia prior/anchor; bias = median")
    W("|u/truth* - 1| over reported bins.\n")

    for mode in ("groomed", "ungroomed"):
        d = m[mode]
        W(f"## {mode}\n")
        pt = d["pseudo_truth"]
        W(f"pseudo-truth reweight: w in [{pt['w_min']:.2f}, {pt['w_max']:.2f}], "
          f"foldback median relres {pt['foldback_median_relres']:.3f}. "
          f"Weighted-lstsq foldback chi2/ndof (study binning) = "
          f"{d['unreg_chi2_ndof_study']:.0f} -- the data is grossly incompatible "
          f"with the nominal response model at these statistics; every metric "
          f"below is shaped by that, not by conditioning alone.\n")

        fam = d["families"]
        b4 = fam["bayes"][str(PROD_NITER)]
        W(f"D'Agostini n={PROD_NITER}: neg={b4['neg_bins']} osc={b4['osc']:.3g} "
          f"closure bias={b4['closure_bias_med']:.3g} spread={b4['toy_spread']:.3g}")
        ks, tau, f = _tik_arrays(fam["tikhonov_ratio"])
        ip = int(np.argmin(np.abs(tau - PROD_TAU)))
        tp = d["plateau_tau"]
        kp = [k for k in ks if f[k]["tau"] == tp][0]
        W(f"Tikhonov ratio-curv, production tau~{PROD_TAU}: "
          f"neg={f[ks[ip]]['neg_bins']} osc={f[ks[ip]]['osc']:.3g} "
          f"bias={f[ks[ip]]['closure_bias_med']:.3g} (still ~unregularized).")
        W(f"Tikhonov ratio-curv, plateau tau={tp:g}: neg={f[kp]['neg_bins']} "
          f"osc={f[kp]['osc']:.3g} bias={f[kp]['closure_bias_med']:.3g} "
          f"spread={f[kp]['toy_spread']:.3g}.\n")

        W("### prior-dependence of the central value (real data, reported bins)\n")
        W("Priors/anchors: nominal Pythia gen, data-reweighted truth*, +-40%")
        W("linear tilt per pt slice. Spread = max-over-priors |u_p - u_nom| /")
        W("max(|u_nom|, 5% MC truth).\n")
        W("| method | median | max |")
        W("|---|---|---|")
        for meth, st in d["prior_dependence"].items():
            extra = f" (tau={st['tau']:g})" if "tau" in st else ""
            W(f"| {meth}{extra} | {st['median']:.4f} | {st['max']:.3f} |")
        W("")
        W("Unregularized lstsq takes NO prior input: zero dependence by")
        W("construction (verified: repeated solves bit-identical).\n")

        W("### binning scan (unregularized weighted lstsq on data)\n")
        W("| binning | #gen | cond(P) | purity med/min | stab med/min | eff | neg | osc | chi2/ndof | vs dago4 med |")
        W("|---|---|---|---|---|---|---|---|---|---|")
        for n, e in d["binning_scan"]["binnings"].items():
            W(f"| {n} | {e['n_gen_bins_total']} | {e['cond_P']:.3g} | "
              f"{e['purity_median']:.2f}/{e['purity_min']:.2f} | "
              f"{e['stability_median']:.2f}/{e['stability_min']:.2f} | "
              f"{e['eff_median']:.2f} | {e['unreg_neg_bins']} | "
              f"{e['unreg_osc']:.3g} | {e['unreg_chi2_ndof']:.0f} | "
              f"{e['unreg_vs_dago4_reldiff_med']:.3g} |")
        W("")
        winner = d["binning_scan"]["winner"]
        if winner is not None:
            W(f"**Winner: {winner['name']}** (gen edges {winner['gen_edges']}).")
            W(f"{winner['note']}")
            W(f"Unreg toy spread (price of prior freedom): median "
              f"{winner['unreg_toy_spread_med']:.3g}, p90 "
              f"{winner['unreg_toy_spread_p90']:.3g}.")
            if winner["negative_bins"]:
                W("Deterministic negative bins (data/model incompatibility, not noise):")
                for nb_ in winner["negative_bins"]:
                    W(f"- pt bin {nb_['pt_bin']}, rho [{nb_['rho_lo']}, {nb_['rho_hi']}]: "
                      f"u/gen = {nb_['u_over_gen']:.2f}, toy pull = {nb_['toy_pull']:.0f}")
            W("")
            W("Per-bin purity/stability/efficiency (reported pt slices):\n")
            W("| pt bin | rho | purity | stability | efficiency |")
            W("|---|---|---|---|---|")
            for b in winner["per_bin"]:
                W(f"| {b['pt_bin']} | [{b['rho_lo']}, {b['rho_hi']}] | "
                  f"{b['purity']:.2f} | {b['stability']:.2f} | {b['efficiency']:.2f} |")
            W("")

        pc = d.get("production_confirmation")
        if pc:
            W("### production confirmation (unfolding_coarse run)\n")
            W("`run_rho_unfolding.py --binning coarse` (TUnfold, tau resolved to 0,")
            W("area constraint on, merged reco sink below the first interior gen")
            W("edge; outputs/dijet/2018/rho/unfolding_coarse/):\n")
            W(f"- production result: **{pc['prod_neg_reported']} negative reported "
              f"bins**, osc = {pc['prod_osc']:.3g}, stat err median "
              f"{pc['prod_stat_err_med']:.3g} (p90 {pc['prod_stat_err_p90']:.3g}).")
            W(f"- numpy unreg lstsq with the SAME merged reco sink: "
              f"{pc['numpy_mergedreco_neg']} negative bins, osc "
              f"{pc['numpy_mergedreco_osc']:.3g}; matches production to "
              f"{pc['numpy_vs_prod_med']:.4f} median ({pc['numpy_vs_prod_p90']:.3f} p90).")
            W(f"- production vs D'Agostini n=4 (independent method, same gen "
              f"edges): {pc['prod_vs_dago4_med']:.3f} median, "
              f"{pc['prod_vs_dago4_p90']:.3f} p90.")
            W(f"- {pc['note']}\n")

    W("## Takeaways\n")
    W("- **Model incompatibility floor**: the weighted foldback chi2/ndof of the")
    W("  best-fit lstsq is 50-950 depending on binning/mode. With 5.1e9 events")
    W("  the 10-35% reco-level data/MC mismatch is thousands of sigma; with the")
    W("  NATIVE fine reco binning no gen binning alone removes the last 1-2")
    W("  deterministic negative bins (toy pulls O(-100)). NNLS parks the same")
    W("  bins at 0 with worse foldback. This is a response-modeling problem")
    W("  (resolution/migration mismatch), not conditioning.")
    W("- **The fix is coarse gen binning + a merged reco sink**: merging the")
    W("  reco bins below the first interior gen edge (the production 'coarse'")
    W("  convention) removes the incompatible low-rho reco constraints, and the")
    W("  fully unregularized inversion becomes sane: 0 negative reported bins,")
    W("  osc ~0.14, in BOTH modes -- confirmed in the production TUnfold run")
    W("  (see production confirmation sections above).")
    W("- push_B reduces the unregularized oscillation to 0.14-0.26 (from")
    W("  0.75-29 on the study binning), agrees with D'Agostini n=4 to 3-8%")
    W("  (median), and costs only ~0.4-0.9% median stat spread (p90 3-6%).")
    W("- **Prior dependence** (the boss-facing number): D'Agostini n=4 moves by")
    W("  3-4% (median) and up to 22-48% (max) across reasonable priors on the")
    W("  study binning; on the coarse binning 1-2% (median), 13-14% (max).")
    W("  Plateau-tau ratio-curvature Tikhonov is worse (median up to 11%, max")
    W("  ~4-5 in near-zero bins). Unregularized lstsq: exactly zero.")
    W("- **Pseudo-truth closure**: the prior-free candidate recovers truth*")
    W("  exactly (bias ~1e-13, by construction on noiseless pseudo-data);")
    W("  D'Agostini n=4 carries 2-3% median closure bias, plateau-tau")
    W("  ratio-curvature Tikhonov 0.6-0.8% (rising to ~5-10% at very large tau).")
    W("  The production tau~0.79 is confirmed to sit at the unregularized")
    W("  plateau (far too weak to matter).")
    W("- Herwig-based numbers from v1 of this study are deprecated and removed.\n")
    W("Files: money_bias_variance.png, metric_vs_tau.png, bayes_vs_niter.png,")
    W("binning_scan.png, metrics.json.")

    with open(OUT / "README.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote README.md")


def main():
    m = load()
    money_plot(m)
    metric_vs_knob(m)
    bayes_vs_niter(m)
    binning_scan_plot(m)
    write_readme(m)


if __name__ == "__main__":
    main()
