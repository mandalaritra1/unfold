"""study2 -- matrix manipulation / toy-MC response strategy for dijet rho.

Does smoothing / parametrizing the response matrix improve the unfolding?
  - stability of matrix-inversion-type methods (lstsq / Tikhonov), and
  - sensitivity of the D'Agostini Bayes result to matrix MC-stat noise.

Three response matrices per mode:
  NOMINAL   : the raw MC response.
  SMOOTH_I  : nonparametric Gaussian-kernel smoothing of the reco-rho shape.
  PARAM_II  : (two-piece) Gaussian kernel fitted per gen bin, mu/sigma smoothed
              with low-order polynomials, re-integrated over reco bins.
Both smoothings preserve per-column pT-migration/efficiency exactly.

Outputs -> outputs/dijet/2018/rho/method_study2/smoothed_response/
  metrics.json, README.md, and PNGs.
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.dijet_methods.loader import (
    Binning, load_native, rebin, build_problem,
)
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.result import prepare, unfold_with_unc, channel_paths
from scripts.dijet_methods import study2_smooth_helpers as H

CHANNEL = "dijet"
YEAR = 2018
OUT = "outputs/dijet/2018/rho/method_study2/smoothed_response"
N_ITER = 4
N_TOYS = 200
SEED = 0

PT_EDGES = np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0])
PT_LABELS = ["0-200", "200-290", "290-400", "400-570", "570-760", "760+"]
RHO_EDGES_RECO = np.array(
    [-10, -8, -7, -6, -5.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25,
     -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
    dtype=float)
RHO_EDGES_GEN = np.array(
    [-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0], dtype=float)

TAUS = [0.0, 1e-3, 1e-2, 1e-1]


# ungroomed physical range starts higher: the two lowest gen bins ([-10,-6],
# [-6,-5]) are ~empty across reported pT, which makes the unrolled P
# rank-deficient (cond=inf). Merge the low tail into a single [-10,-5] gen bin
# (reco keeps native edges >= -5) so inversion methods are actually defined.
RHO_EDGES_GEN_UNGROOMED = np.array(
    [-10, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0], dtype=float)
RHO_EDGES_RECO_UNGROOMED = np.concatenate(
    [[-10.0], RHO_EDGES_RECO[RHO_EDGES_RECO >= -5.0]])


def study_binning(mode="groomed"):
    if mode == "ungroomed":
        return Binning(PT_EDGES.copy(), RHO_EDGES_RECO_UNGROOMED.copy(),
                       RHO_EDGES_GEN_UNGROOMED.copy())
    return Binning(PT_EDGES.copy(), RHO_EDGES_RECO.copy(), RHO_EDGES_GEN.copy())


# --------------------------------------------------------------------------
# Build the three native responses, then rebin each to a Problem.
# --------------------------------------------------------------------------
def build_matrices(mode):
    """Return dict name -> (native_response_array, fitinfo_or_None)."""
    mc_path, _, _ = channel_paths(CHANNEL, YEAR)
    nat = load_native(mc_path, mode)
    R = nat.response
    print(f"  [{mode}] smoothing native response {R.shape} ...", flush=True)
    R_i = H.smooth_response_nonparametric(R, nat.rho_edges_reco, sigma_bins=1.2)
    # Single Gaussian: the reco-rho kernel is skewed+heavy-tailed, so the
    # two-piece form self-closes WORSE (its mode-split biases the peak). Single
    # Gaussian gives the smaller closure bias, so it is the reported PARAM_II.
    R_ii, info = H.fit_response_parametric(
        R, nat.rho_edges_reco, nat.rho_edges_gen, nat.pt_edges_reco,
        deg=2, twopiece=False, min_counts=50.0)
    return nat, {"NOMINAL": (R, None), "SMOOTH_I": (R_i, None),
                 "PARAM_II": (R_ii, info)}


def problem_from_native_R(nat, R_native, binning):
    """Rebin a native response array (keeping nat's reco/gen) into a Problem."""
    import copy
    nat2 = copy.copy(nat)
    nat2.response = R_native
    rb = rebin(nat2, binning)
    return build_problem(rb), rb


def _prep_with_matrix(mode, binning, R_native, nat):
    """A Prepared-like object where prob uses R_native but data/herwig come from
    the same loader path as result.prepare (nominal reco vectors)."""
    from scripts.dijet_methods.result import Prepared
    prob, rb = problem_from_native_R(nat, R_native, binning)
    mc_path, data_path, her_path = channel_paths(CHANNEL, YEAR)
    datn = load_native(data_path, mode, need_gen=False)
    datrb = rebin(datn, binning)
    data_meas = datrb.reco.reshape(-1)
    data_var = (datrb.reco_var.reshape(-1) if datrb.reco_var is not None
                else data_meas.copy())
    data_matched = M.subtract_fakes(data_meas, prob)
    her_truth = her_matched = None
    if her_path:
        her = load_native(her_path, mode)
        herrb = rebin(her, binning)
        her_truth = herrb.gen.reshape(-1)
        her_matched = M.subtract_fakes(herrb.reco.reshape(-1), prob)
    return Prepared(mode, binning, prob, nat, rb, data_meas, data_var,
                    data_matched, her_truth, her_matched, CHANNEL)


# --------------------------------------------------------------------------
# Validation (per rebuilt matrix)
# --------------------------------------------------------------------------
def validate_matrix(prob_nom, prob, prep, name):
    """(a) foldback of Pythia gen through rebuilt P vs nominal matched reco.
       (b) dagostini self-closure of SMOOTHED matrix on ORIGINAL MC reco.
       (c) Herwig-as-data model bias (rebuilt vs herwig truth)."""
    out = {}
    gen = prob_nom.gen                          # same gen truth for all
    # (a) fold Pythia gen through rebuilt P, compare to nominal matched reco
    nominal_matched_reco = prob_nom.P @ gen     # the true matched reco of MC
    fold_rebuilt = prob.P @ gen
    m = nominal_matched_reco > 0
    dev = np.abs(fold_rebuilt[m] - nominal_matched_reco[m]) / nominal_matched_reco[m]
    out["fold_median_dev"] = float(np.median(dev))
    out["fold_p90_dev"] = float(np.percentile(dev, 90))
    # (b) self-closure: smoothed matrix unfolds the ORIGINAL MC matched reco
    #     (nominal fold), with the MC prior. Non-closure = smoothing bias.
    self_est = M.dagostini(prob, nominal_matched_reco, n_iter=N_ITER, prior=gen)
    report = list(range(1, len(prob.binning.pt_edges) - 1))
    rep_mask = np.isin(prob.gen_pt_idx, report) & (gen > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.abs(self_est[rep_mask] / gen[rep_mask] - 1.0)
    out["selfclosure_median_bias"] = float(np.median(cl))
    out["selfclosure_max_bias"] = float(np.max(cl))
    # (c) Herwig-as-data: unfold herwig matched with this matrix vs herwig truth
    if prep.her_matched is not None:
        her_est = M.dagostini(prob, prep.her_matched, n_iter=N_ITER)
        hmask = np.isin(prob.gen_pt_idx, report) & (prep.her_truth > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hb = np.abs(her_est[hmask] / prep.her_truth[hmask] - 1.0)
        out["herwig_model_median_bias"] = float(np.median(hb))
        out["herwig_model_max_bias"] = float(np.max(hb))
    return out


# --------------------------------------------------------------------------
# Money comparison: unfold DATA with each method x matrix
# --------------------------------------------------------------------------
def money_metrics(prob, prep, report, ref_central=None):
    """Return per-method metrics dict for one matrix, and the dagostini central
    (for cross-matrix shift). ref_central = nominal-matrix dagostini n=4."""
    dm = prep.data_matched
    prior = prob.gen
    rep_mask = np.isin(prob.gen_pt_idx, report)

    results = {}
    method_runs = {"dagostini_n4": lambda: M.dagostini(prob, dm, n_iter=N_ITER),
                   "lstsq": lambda: M.tikhonov(prob, dm, tau=0.0)}
    for tau in TAUS:
        method_runs[f"tikhonov_tau{tau:g}"] = (
            lambda t=tau: M.tikhonov(prob, dm, tau=t))

    dago_central = None
    for name, fn in method_runs.items():
        try:
            u = fn()
        except Exception as e:
            results[name] = {"error": str(e)}
            continue
        if name == "dagostini_n4":
            dago_central = u
        neg = int(np.sum(u[rep_mask] < 0))
        osc = H.oscillation_metric(u, prior, prob.gen_pt_idx, report)
        fb = H.foldback_residual(prob.P, np.clip(u, 0, None), dm)
        met = {"neg_bins": neg, "oscillation": osc, "foldback_median": fb}
        if ref_central is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.abs(np.where(ref_central[rep_mask] != 0,
                                     u[rep_mask] / ref_central[rep_mask] - 1.0,
                                     np.nan))
            rr = rr[np.isfinite(rr)]
            met["shift_vs_nomdago_median"] = float(np.median(rr)) if rr.size else None
            met["shift_vs_nomdago_max"] = float(np.max(rr)) if rr.size else None
        results[name] = met
    return results, dago_central


def toy_spread(prep, report):
    """dagostini toy std with response_toys True vs False -> matrix-stat share."""
    prob = prep.prob
    rep_mask = np.isin(prob.gen_pt_idx, report)
    _, std_data, _, _ = unfold_with_unc(
        prep, M.dagostini, n_iter=N_ITER, n_toys=N_TOYS,
        response_toys=False, seed=SEED)
    _, std_both, _, _ = unfold_with_unc(
        prep, M.dagostini, n_iter=N_ITER, n_toys=N_TOYS,
        response_toys=True, seed=SEED)
    central = M.dagostini(prob, prep.data_matched, n_iter=N_ITER)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_data = std_data[rep_mask] / np.abs(central[rep_mask])
        rel_both = std_both[rep_mask] / np.abs(central[rep_mask])
        # matrix-only component in quadrature
        mat = np.sqrt(np.clip(std_both[rep_mask] ** 2 - std_data[rep_mask] ** 2, 0, None))
        rel_mat = mat / np.abs(central[rep_mask])
    fin = lambda a: float(np.nanmedian(a[np.isfinite(a)])) if np.any(np.isfinite(a)) else np.nan
    return {"rel_data_median": fin(rel_data),
            "rel_total_median": fin(rel_both),
            "rel_matrixstat_median": fin(rel_mat)}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _slice(vec, prob, ptg):
    return vec[prob.gen_pt_idx == ptg]


def _norm_shape(y, edges):
    w = np.diff(edges)
    dens = y / w
    area = np.sum(y)
    return (dens / area if area > 0 else dens)


def plot_overlay(mode, matrices_prob, dago_by_matrix, binning):
    edges = binning.rho_edges_gen
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob0 = matrices_prob["NOMINAL"]
    report = list(range(1, len(binning.pt_edges) - 1))
    colors = {"NOMINAL": "k", "SMOOTH_I": "tab:blue", "PARAM_II": "tab:red"}
    fig, axes = plt.subplots(2, len(report), figsize=(4 * len(report), 7),
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for col, ptg in enumerate(report):
        ax = axes[0][col]; rax = axes[1][col]
        ref = _norm_shape(np.clip(_slice(dago_by_matrix["NOMINAL"], prob0, ptg), 0, None), edges)
        for name in ("NOMINAL", "SMOOTH_I", "PARAM_II"):
            u = np.clip(_slice(dago_by_matrix[name], prob0, ptg), 0, None)
            un = _norm_shape(u, edges)
            ax.step(centers, un, where="mid", color=colors[name], lw=1.5,
                    ls="-" if name == "NOMINAL" else "--", label=name)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.where(ref > 0, un / ref, np.nan)
            rax.step(centers, rr, where="mid", color=colors[name], lw=1.2,
                     ls="-" if name == "NOMINAL" else "--")
        ax.set_yscale("log")
        ax.set_title(f"{PT_LABELS[ptg]} GeV")
        ax.set_xlim(edges[0], edges[-1])
        if col == 0:
            ax.set_ylabel("(1/N) dN/dρ (shape)")
            ax.legend(fontsize=8)
        rax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        rax.set_ylim(0.8, 1.2)
        rax.set_xlim(edges[0], edges[-1])
        rax.set_xlabel(r"$\log_{10}(\rho^2)$")
        if col == 0:
            rax.set_ylabel("/ nominal")
    fig.suptitle(f"dijet 2018 {mode} rho — D'Agostini n={N_ITER}: "
                 f"nominal vs smoothed vs parametric response")
    fig.tight_layout()
    path = os.path.join(OUT, f"overlay_dagostini_{mode}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_response_maps(mode, matrices_native, nat, binning):
    """2D response block (nominal/smoothed/parametric) for a low-pt block."""
    # rebin each native response to study binning and show ptreco=ptgen=1 block
    rho_r = binning.rho_edges_reco; rho_g = binning.rho_edges_gen
    ptblk = 1  # 200-290 reco x 200-290 gen
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    names = ["NOMINAL", "SMOOTH_I", "PARAM_II"]
    for k, name in enumerate(names):
        prob, rb = problem_from_native_R(nat, matrices_native[name][0], binning)
        # block: R reshaped -> take ptreco=ptblk, ptgen=ptblk sub-block P
        # P is (Nreco, Ngen); pull rows with reco_pt_idx==ptblk, cols gen_pt_idx==ptblk
        rmask = prob.reco_pt_idx == ptblk
        cmask = prob.gen_pt_idx == ptblk
        Pblk = prob.P[np.ix_(rmask, cmask)]
        ax = axes[0][k]
        im = ax.pcolormesh(rho_g, rho_r, Pblk, cmap="viridis")
        ax.set_title(f"{name}  (pT {PT_LABELS[ptblk]})")
        ax.set_xlabel(r"gen $\log_{10}\rho^2$")
        if k == 0:
            ax.set_ylabel(r"reco $\log_{10}\rho^2$")
        fig.colorbar(im, ax=ax, label="P[reco|gen]")
    fig.suptitle(f"dijet 2018 {mode} rho — response block (folding prob), pT {PT_LABELS[ptblk]}")
    fig.tight_layout()
    path = os.path.join(OUT, f"response_maps_{mode}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


# --------------------------------------------------------------------------
def run_mode(mode, all_metrics):
    print(f"\n=== MODE {mode} ===", flush=True)
    binning = study_binning(mode)
    report = list(range(1, len(binning.pt_edges) - 1))

    nat, mats = build_matrices(mode)          # name -> (R_native, info)

    # build a Problem + Prepared per matrix
    probs = {}; preps = {}
    cond_full = {}; cond_pop = {}
    for name, (Rn, _) in mats.items():
        prep = _prep_with_matrix(mode, binning, Rn, nat)
        preps[name] = prep
        probs[name] = prep.prob
        P = prep.prob.P
        cond_full[name] = float(np.linalg.cond(P))
        # cond restricted to POPULATED gen columns (empty-sink columns give an
        # exact zero singular value -> inf for the full matrix regardless of
        # smoothing; this is the meaningful inversion conditioning).
        pop = P.sum(axis=0) > 0
        cond_pop[name] = float(np.linalg.cond(P[:, pop])) if pop.any() else float("inf")
        print(f"  [{mode}] {name}: cond(P full)={cond_full[name]:.1f} "
              f"cond(P populated)={cond_pop[name]:.1f}", flush=True)

    prob_nom = probs["NOMINAL"]

    all_metrics[mode] = {}
    all_metrics[mode]["_cond"] = {
        name: {"full": cond_full[name], "populated": cond_pop[name]}
        for name in mats}
    # ---- validation per matrix
    val = {}
    for name in mats:
        val[name] = validate_matrix(prob_nom, probs[name], preps[name], name)
        print(f"  [{mode}] validate {name}: "
              f"fold_med={val[name]['fold_median_dev']:.3g} "
              f"selfclose_med={val[name]['selfclosure_median_bias']:.3g} "
              f"herwig_med={val[name].get('herwig_model_median_bias', float('nan')):.3g}",
              flush=True)
    all_metrics[mode]["_validation"] = val

    # ---- money comparison
    # first pass: nominal dagostini central as reference
    _, ref_dago = money_metrics(prob_nom, preps["NOMINAL"], report, ref_central=None)
    dago_by_matrix = {}
    per_matrix_methods = {}
    for name in mats:
        res, dago_c = money_metrics(probs[name], preps[name], report,
                                    ref_central=ref_dago)
        per_matrix_methods[name] = res
        dago_by_matrix[name] = dago_c
        # toy spreads (dagostini only)
        ts = toy_spread(preps[name], report)
        res["dagostini_n4"]["toy_spread"] = ts
        print(f"  [{mode}] {name} dago: neg={res['dagostini_n4']['neg_bins']} "
              f"osc={res['dagostini_n4']['oscillation']:.3g} "
              f"shift_vs_nom={res['dagostini_n4'].get('shift_vs_nomdago_median')} "
              f"matstat={ts['rel_matrixstat_median']:.3g}", flush=True)
        print(f"      lstsq neg={res['lstsq'].get('neg_bins')} "
              f"osc={res['lstsq'].get('oscillation')}", flush=True)

    # reorganize: method -> matrix -> metrics
    method_names = set()
    for name in mats:
        method_names.update(per_matrix_methods[name].keys())
    by_method = {}
    for meth in sorted(method_names):
        by_method[meth] = {name: per_matrix_methods[name].get(meth)
                           for name in mats}
    all_metrics[mode]["methods"] = by_method

    # ---- plots
    p1 = plot_overlay(mode, probs, dago_by_matrix, binning)
    p2 = plot_response_maps(mode, mats, nat, binning)
    print(f"  [{mode}] plots: {p1} | {p2}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    all_metrics = {}
    for mode in ("groomed", "ungroomed"):
        run_mode(mode, all_metrics)

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(OUT, 'metrics.json')}")
    write_readme(all_metrics)


def write_readme(all_metrics):
    lines = ["# study2 — smoothed / parametrized response for dijet rho\n"]
    lines.append(
        "Three response matrices per mode: **NOMINAL** (raw MC), **SMOOTH_I** "
        "(nonparametric Gaussian-kernel smoothing of the reco-rho shape, "
        "sigma=1.2 median-bin-widths), **PARAM_II** (single Gaussian kernel "
        "fitted per gen bin, mu/sigma smoothed with deg-2 polynomials per "
        "ptgen->ptreco pair, re-integrated over reco bins). Both smoothings "
        "preserve per-column pT-migration and efficiency to machine precision.\n")
    lines.append(
        "## TL;DR findings\n"
        "1. **Smoothing does NOT stabilize inversion — it makes it worse.** "
        "cond(P) rises NOMINAL->SMOOTH_I->PARAM_II (groomed 1449->3900->78k; "
        "ungroomed populated 1396->35k->15k). Widening the migration kernel "
        "increases collinearity between neighbouring gen columns, so lstsq / "
        "low-tau Tikhonov get MORE negative bins and MORE oscillation, not less.\n"
        "2. **The D'Agostini central value is barely moved by nonparametric "
        "smoothing** (median shift ~1.6% groomed, ~5% ungroomed over reported "
        "bins) and the matrix-stat toy component is already tiny (~0.1% groomed) "
        "and is NOT meaningfully shrunk by smoothing — the data is systematics-"
        "dominated, matrix MC-stat is negligible either way.\n"
        "3. **A Gaussian parametrization is inadequate.** The reco-rho kernel is "
        "skewed (|skew| up to ~2) and heavy-tailed (excess kurtosis up to ~10), "
        "so PARAM_II self-closes to only ~20% (groomed) and shifts the Bayes "
        "result by ~4%; it is not usable as a drop-in response.\n"
        "**Conclusion: keep the nominal matrix + D'Agostini n=4.** Smoothing "
        "buys no stability for inversion methods and no toy-noise reduction for "
        "Bayes, while parametrization injects a real bias. Stability comes from "
        "*binning* and *iterative regularization*, not from massaging the "
        "matrix.\n")
    lines.append("Study binning: pt " + ",".join(PT_LABELS) +
                 f"; {len(RHO_EDGES_GEN)-1} gen rho bins, "
                 f"{len(RHO_EDGES_RECO)-1} reco rho bins. "
                 f"D'Agostini n_iter={N_ITER}, {N_TOYS} toys.\n")
    for mode, md in all_metrics.items():
        lines.append(f"\n## {mode}\n")
        lines.append("### Response conditioning cond(P)\n")
        lines.append("`populated` drops empty gen columns (the pT<200 sink "
                     "block has unmatched gen bins -> exact-zero singular value "
                     "-> full cond is inf regardless of smoothing).\n")
        lines.append("| matrix | cond full | cond populated |")
        lines.append("|---|---|---|")
        for name, cc in md["_cond"].items():
            lines.append(f"| {name} | {cc['full']:.4g} | {cc['populated']:.4g} |")
        lines.append("\n### Matrix validation\n")
        lines.append("| matrix | fold median dev | fold p90 | self-closure median bias | self-closure max | herwig model median bias |")
        lines.append("|---|---|---|---|---|---|")
        for name, v in md["_validation"].items():
            lines.append(
                f"| {name} | {v['fold_median_dev']:.2e} | {v['fold_p90_dev']:.2e} "
                f"| {v['selfclosure_median_bias']:.2e} | {v['selfclosure_max_bias']:.2e} "
                f"| {v.get('herwig_model_median_bias', float('nan')):.2e} |")
        lines.append("\n### Money comparison (unfolding DATA)\n")
        lines.append("Metrics over reported pT bins (200 GeV+). "
                     "osc = median |2nd diff of u/prior|; "
                     "shift = median|ratio-1| vs nominal-matrix D'Agostini n=4.\n")
        lines.append("| method | matrix | neg bins | oscillation | foldback median | shift vs nom-dago | matrix-stat rel |")
        lines.append("|---|---|---|---|---|---|---|")
        for meth, mm in md["methods"].items():
            for name, met in mm.items():
                if met is None or "error" in met:
                    lines.append(f"| {meth} | {name} | ERR | | | | |")
                    continue
                ts = met.get("toy_spread", {})
                lines.append(
                    f"| {meth} | {name} | {met.get('neg_bins')} "
                    f"| {met.get('oscillation'):.3g} "
                    f"| {met.get('foldback_median'):.3g} "
                    f"| {met.get('shift_vs_nomdago_median')} "
                    f"| {ts.get('rel_matrixstat_median', '')} |")
    lines.append("\n## Files\n- overlay_dagostini_{groomed,ungroomed}.png — "
                 "per-pT D'Agostini shape, nominal vs smoothed vs parametric\n"
                 "- response_maps_{groomed,ungroomed}.png — folding-prob block "
                 "for pT 200-290, three matrices\n- metrics.json — full metric table\n")
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(OUT, 'README.md')}")


if __name__ == "__main__":
    main()
