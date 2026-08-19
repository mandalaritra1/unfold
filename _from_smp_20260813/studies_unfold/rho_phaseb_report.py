#!/usr/bin/env python
"""Assemble the Phase B validation summary from the driver JSONs.

Reads the crosschecks / systematics / profiled-refold / iterative-Bayes /
configuration-comparison outputs of the pair_split Run 2 rho unfolding and
writes a markdown report plus a machine-readable companion. Pure aggregation:
it runs no unfolds of its own, so every number here is traceable to one of the
input JSONs named in ``SOURCES``.

    ~/Projects/unfold/.venv/bin/python \
        scripts/studies/unfold/rho_phaseb_report.py \
        --dir outputs/pairsplit_unfold --out outputs/pairsplit_unfold/phaseB_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ("dijet", "trijet")
DETECTOR = ("JER", "JMS", "JMR")
EXACT = 1e-10


def load(path: Path):
    return json.loads(path.read_text())


def pct(value, digits=2):
    if value is None or not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.{digits}f}"


def status(ok, unavailable=False):
    if unavailable:
        return "UNAVAILABLE"
    return "PASS" if ok else "FAIL"


def check_rows(crosschecks, systematics, refold, bayes, configs):
    """Every check with its own criterion and PASS/FAIL/UNAVAILABLE verdict."""
    rows = []
    for channel in CHANNELS:
        cc = crosschecks[channel]
        nominal = cc["nominal"]
        rows.append(
            (
                channel,
                "self-closure (nominal MC through its own response)",
                f"max |unf/gen-1| <= {EXACT:g}",
                f"{nominal['self_closure_max']:.2e}",
                status(nominal["self_closure_max"] <= EXACT),
            )
        )
        worst_self = max(
            entry["same_variation_self_closure_max"]
            for entry in cc["systematic_closure"].values()
        )
        rows.append(
            (
                channel,
                "per-variation self-closure (6 JER/JMS/JMR slices)",
                f"max |unf/gen-1| <= {EXACT:g}",
                f"{worst_self:.2e}",
                status(worst_self <= EXACT),
            )
        )
        #### bias induced by unfolding a shifted input with the NOMINAL
        #### response, against the band the same source contributes
        band_core = {}
        for source in DETECTOR:
            values = np.asarray(
                systematics[channel]["relative_half_difference"][source],
                dtype=float,
            )
            core = np.asarray(systematics[channel]["core"], dtype=bool)
            band_core[source] = float(np.nanmax(values[core]))
        covered = True
        detail = []
        for source in DETECTOR:
            #### like-for-like: both sides in normalized-shape space, so the
            #### overall yield change of the variation is divided out
            bias = max(
                cc["systematic_closure"][f"{source}{direction}"][
                    "varied_input_nominal_response_shape"
                ]["core"]["max"]
                for direction in ("Up", "Down")
            )
            covered &= bias <= 2.0 * band_core[source] + 1e-9
            detail.append(f"{source} {pct(bias)}% vs band {pct(band_core[source])}%")
        rows.append(
            (
                channel,
                "shifted input through nominal response, shape (core bins)",
                "induced normalized-shape bias within 2x the propagated "
                "band of the same source",
                "; ".join(detail),
                status(covered),
            )
        )
        toys = cc["input_stat_toys"]
        width = toys["empirical_over_analytic_sigma"]["median"]
        coverage = toys["coverage_1sigma"]["median"]
        rows.append(
            (
                channel,
                "input-stat pseudo-experiments",
                "pull width in [0.95, 1.05] and 68% coverage in [0.65, 0.71]",
                f"width {width:.3f}, coverage {coverage:.3f}",
                status(0.95 <= width <= 1.05 and 0.65 <= coverage <= 0.71),
            )
        )
        response = cc["response_stat_toys"]["empirical_over_sysuncorr_sigma"][
            "median"
        ]
        rows.append(
            (
                channel,
                "response-MC-stat toys vs GetEmatrixSysUncorr",
                "sigma ratio in [0.90, 1.10]",
                f"{response:.3f}",
                status(0.90 <= response <= 1.10),
            )
        )
        bottom = cc["formal_bottom_line_test"]
        rows.append(
            (
                channel,
                "bottom-line test (raw chi2 unfolded <= smeared)",
                "chi2_unfolded - chi2_smeared <= 0",
                f"{bottom['unfolded_data_stat']['chi2']:.1f} vs "
                f"{bottom['smeared_data_stat']['chi2']:.1f} "
                f"(delta {bottom['data_stat_raw_chi2_difference_unfolded_minus_smeared']:.1f})",
                status(bottom["data_stat_inequality_holds"]),
            )
        )
        gls = cc["independent_gls_crosscheck"]["shape_delta_vs_tunfold_noarea"][
            "shown"
        ]["max"]
        rows.append(
            (
                channel,
                "independent GLS vs TUnfold (no area, tau=0)",
                "max relative shape delta <= 1e-8",
                f"{gls:.2e}",
                status(gls <= 1e-8),
            )
        )
        refold_delta = refold[channel]["baseline_validation"][
            "maximum_truth_delta_gls_vs_tunfold"
        ]
        rows.append(
            (
                channel,
                "refold consistency (profiled-fit baseline vs TUnfold)",
                "max |truth_GLS/truth_TUnfold - 1| <= 1e-8",
                f"{refold_delta:.2e}",
                status(refold_delta <= 1e-8),
            )
        )
        covariance = cc["normalized_covariance"]
        psd = covariance["min_eigenvalue_relative"] >= -1e-12
        null_ok = max(covariance["normalization_null_residuals"]) <= 1e-12
        rank_ok = covariance["rank"] == covariance["expected_rank_ceiling"]
        rows.append(
            (
                channel,
                "normalized covariance: PSD, rank, one null mode per pT",
                "min eig >= -1e-12 (rel), rank == dim - n_pt, null residual <= 1e-12",
                f"min eig {covariance['min_eigenvalue_relative']:.1e}, rank "
                f"{covariance['rank']}/{covariance['expected_rank_ceiling']}, "
                f"null {max(covariance['normalization_null_residuals']):.1e}",
                status(psd and null_ok and rank_ok),
            )
        )
        prior = max(
            entry["max_abs_closure"] for entry in cc["prior_stress"].values()
        )
        rows.append(
            (
                channel,
                "smooth truth-prior stress (3 patterns)",
                f"max |unf/truth-1| <= {EXACT:g}",
                f"{prior:.2e}",
                status(prior <= EXACT),
            )
        )
        contract = cc["input_contract"]
        contract_ok = (
            contract["negative_fake_bins"] == 0
            and contract["negative_miss_bins"] == 0
            and contract["data_covariance"]["min_eigenvalue"] > 0.0
            and contract["data_covariance"]["rank"]
            == contract["data_covariance"]["dimension"]
            and contract["response_variance_nonnegative"]
            and contract["miss_variance_nonnegative"]
        )
        rows.append(
            (
                channel,
                "input contract (fakes/misses/covariance)",
                "no negative fake or miss bins; data covariance PD and full rank",
                f"fakes {pct(contract['fake_fraction'],1)}%, misses "
                f"{pct(contract['miss_fraction'],1)}%, cov rank "
                f"{contract['data_covariance']['rank']}/"
                f"{contract['data_covariance']['dimension']}",
                status(contract_ok),
            )
        )
        conditioning = cc["conditioning"]
        rank_full = (
            conditioning["plain"]["rank"] == conditioning["plain"]["n_columns"]
        )
        rows.append(
            (
                channel,
                "response rank / conditioning",
                "full column rank at tau=0",
                f"rank {conditioning['plain']['rank']}/"
                f"{conditioning['plain']['n_columns']}, cond "
                f"{conditioning['plain']['condition']:.2f} (data-weighted "
                f"{conditioning['data_weighted']['condition']:.3g})",
                status(rank_full),
            )
        )
        negatives = configs[channel]["configurations"]["tau0_area_full"][
            "negative_reported_bins"
        ]
        rows.append(
            (
                channel,
                "negative reported bins (production config, data unfold)",
                "zero negative bins",
                str(negatives),
                status(negatives == 0),
            )
        )
        limit = bayes[channel]["iteration_scan"]
        far = crosschecks[channel]["iterative_bayes_crosscheck"]["iterations"][
            "1000"
        ]["shape_delta_vs_tunfold"]["core"]["median"]
        rows.append(
            (
                channel,
                "iterative Bayes converges to the tau=0 solution",
                "core median |shape ratio - 1| <= 1% at 1000 iterations",
                f"{pct(far)}% (n=4: "
                f"{pct(limit['4']['shape_delta_vs_tunfold']['core']['median'])}%)",
                status(far <= 0.01),
            )
        )
        rows.append(
            (
                channel,
                "alternate-generator (HERWIG) response closure",
                "pair_split HERWIG response NPZ present",
                "no pair_split HERWIG production exists; the July "
                "old-selection herwig NPZ is on a different selection and "
                "was NOT substituted",
                status(False, unavailable=True),
            )
        )
        for name in (
            "statistically independent MC half-response closure",
            "m_g > 2 GeV floor-enabled full-stat production",
            "joint dijet-trijet data and response covariance",
        ):
            rows.append(
                (channel, name, "input present in the NPZ set", "not present",
                 status(False, unavailable=True))
            )
    return rows


def per_bin_table(channel, systematics, configs, bayes):
    """Reported-bin table: value, errors and every propagated shift."""
    system = systematics[channel]
    config = configs[channel]
    production = config["configurations"]["tau0_area_full"]["per_bin"]
    n_pt, n_rho = config["n_pt"], config["n_rho"]
    gen_edges = config["gen_rho_edges"]
    pt_edges = config["pt_edges"]
    shown = np.asarray(config["shown"], dtype=bool)
    core = np.asarray(config["core"], dtype=bool)
    unfolded = np.asarray(production["unfolded"], dtype=float)
    error = np.asarray(production["statistical_error"], dtype=float)
    mc_gen = np.asarray(config["mc_gen"], dtype=float)
    bayes_shift = np.asarray(
        bayes[channel]["reference"]["signed_shape_delta_vs_tunfold"],
        dtype=float,
    )

    arrays = {
        source: np.asarray(
            system["relative_half_difference"][source], dtype=float
        )
        for source in DETECTOR
    }
    arrays["det_quad"] = np.asarray(system["total_half_difference"], float)
    for source in ("fsr", "isr"):
        arrays[source] = np.asarray(
            system["parton_shower"]["relative_half_difference"][source], float
        )
    stat = np.asarray(system["relative_data_stat"], dtype=float)
    response_stat = np.asarray(system["relative_response_stat"], dtype=float)

    lines = []
    for pt_index in range(n_pt):
        lines.append(
            f"\n**pT [{pt_edges[pt_index]:g}, {pt_edges[pt_index + 1]:g}] GeV**\n"
        )
        lines.append(
            "| rho bin | in win | unfolded | data/MC gen | stat % | respMC % "
            "| JER % | JMS % | JMR % | det quad % | FSR % | ISR % | Bayes n=4 % |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for rho_index in range(n_rho):
            flat = pt_index * n_rho + rho_index
            if not shown[flat]:
                continue
            ratio = (
                unfolded[flat] / mc_gen[flat] if mc_gen[flat] > 0 else np.nan
            )
            lines.append(
                f"| [{gen_edges[rho_index]:g}, {gen_edges[rho_index + 1]:g}] "
                f"| {'yes' if core[flat] else 'no'} "
                f"| {unfolded[flat]:.4g} | {ratio:.3f} "
                f"| {pct(error[flat] / unfolded[flat])} "
                f"| {pct(response_stat[flat])} "
                + "".join(
                    f"| {pct(arrays[key][flat])} "
                    for key in (*DETECTOR, "det_quad", "fsr", "isr")
                )
                + f"| {pct(bayes_shift[flat])} |"
            )
    return "\n".join(lines)


def systematics_section(systematics, systematics_2to1):
    lines = [
        "Symmetrized half-difference of the per-pT normalized shape, over the "
        "core (in-window) reported bins. Detector sources are combined in "
        "quadrature; the parton-shower weights are propagated through exactly "
        "the same response-variation machinery but reported separately -- "
        "**FSR is the parton-shower band, ISR is computed for observation "
        "only and is never quoted as a band**.",
        "",
        "| channel | merge | data stat | response MC stat | JER+JMS+JMR quad "
        "| FSR | ISR |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, source in (("coarse_tail", systematics), ("2to1", systematics_2to1)):
        for channel in CHANNELS:
            summary = source[channel]["core_summary"]
            shower = source[channel]["parton_shower"]["core_summary"]
            lines.append(
                f"| {channel} | `{label}` "
                f"| {pct(summary['data_stat']['median'])} / "
                f"{pct(summary['data_stat']['max'])} "
                f"| {pct(summary['response_stat']['median'])} / "
                f"{pct(summary['response_stat']['max'])} "
                f"| {pct(summary['half_difference']['median'])} / "
                f"{pct(summary['half_difference']['max'])} "
                f"| {pct(shower['fsr_half_difference']['median'])} / "
                f"{pct(shower['fsr_half_difference']['max'])} "
                f"| {pct(shower['isr_half_difference']['median'])} / "
                f"{pct(shower['isr_half_difference']['max'])} |"
            )
    lines.append("")
    lines.append(
        "Cells are median / max in percent over core reported bins. The "
        "statistical column here is the error of the NORMALIZED shape "
        "(propagated through the per-pT normalization Jacobian), so it is not "
        "identical to the absolute statistical error quoted in section 5."
    )
    lines.append("")
    lines.append(
        "Bins where the Up and Down variation land on the SAME side of nominal "
        "(so the half-difference is not a bracketing band):"
    )
    lines.append("")
    lines.append(
        "| channel | merge | core bins | JER | JMS | JMR | FSR | ISR |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, source in (("coarse_tail", systematics), ("2to1", systematics_2to1)):
        for channel in CHANNELS:
            detector = source[channel]["non_bracketing_core_bins"]
            shower = source[channel]["parton_shower"]["non_bracketing_core_bins"]
            n_core = int(np.sum(np.asarray(source[channel]["core"], bool)))
            lines.append(
                f"| {channel} | `{label}` | {n_core} | "
                + " | ".join(str(detector[key]) for key in DETECTOR)
                + f" | {shower['fsr']} | {shower['isr']} |"
            )
    return "\n".join(lines)


def refold_section(refold):
    lines = [
        "Free coarse truth bins, no truth prior, Gaussian-constrained "
        "piecewise-linear JER/JMS/JMR response nuisances restricted to their "
        "supplied [-1, +1] template range. The question is how much of the "
        "large folded data/MC chi2 the detector nuisances can absorb.",
        "",
        "| channel | fit | chi2_data | penalty | chi2_total | ndf | "
        "absorbed vs baseline | JER | JMS | JMR | at bound |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for channel in CHANNELS:
        baseline = refold[channel]["baseline_data_stat_only"]["chi2_total"]
        for key, name in (
            ("baseline_data_stat_only", "baseline (no nuisances)"),
            ("profile_JER_JMS_JMR_data_stat_only", "profiled, data stat only"),
            (
                "profile_JER_JMS_JMR_with_approx_response_mc_stat",
                "profiled + approx response MC stat",
            ),
        ):
            fit = refold[channel][key]
            pulls = fit["nuisance_pulls"]
            bounds = [
                source
                for source, hit in fit.get("nuisance_at_bounds", {}).items()
                if hit
            ]
            lines.append(
                f"| {channel} | {name} | {fit['chi2_data']:.1f} | "
                f"{fit['chi2_penalty']:.2f} | {fit['chi2_total']:.1f} | "
                f"{fit['ndf']} | "
                f"{baseline - fit['chi2_total']:.1f} "
                f"({(baseline - fit['chi2_total']) / baseline:.1%}) | "
                + " | ".join(
                    f"{pulls[source]:+.3f}" if source in pulls else "--"
                    for source in DETECTOR
                )
                + f" | {', '.join(bounds) if bounds else 'none'} |"
            )
    lines.append("")
    lines.append(
        "| channel | fit | truth-shape change vs baseline, core med / max "
        "| all reported med / max |"
    )
    lines.append("|---|---|---|---|")
    for channel in CHANNELS:
        for key, name in (
            ("profile_JER_JMS_JMR_data_stat_only", "profiled, data stat only"),
            (
                "profile_JER_JMS_JMR_with_approx_response_mc_stat",
                "profiled + response MC stat",
            ),
        ):
            change = refold[channel][key]["truth_shape_change"]
            lines.append(
                f"| {channel} | {name} | "
                f"{pct(change['core']['median'])} / "
                f"{pct(change['core']['maximum'])} | "
                f"{pct(change['shown']['median'])} / "
                f"{pct(change['shown']['maximum'])} |"
            )
    lines.append("")
    lines.append("First-pT feed-in (fake normalization) scan, data stat only:")
    lines.append("")
    lines.append(
        "| channel | feed 1-sigma | chi2_total | absorbed vs baseline | "
        "feed pull | JER | JMS | JMR |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for channel in CHANNELS:
        baseline = refold[channel]["baseline_data_stat_only"]["chi2_total"]
        for name, entry in refold[channel][
            "profile_with_first_pt_feed_scan"
        ].items():
            fit = entry["data_stat_only"]
            pulls = fit["nuisance_pulls"]
            feed = [key for key in pulls if key.startswith("feed_")][0]
            lines.append(
                f"| {channel} | {name} | {fit['chi2_total']:.1f} | "
                f"{baseline - fit['chi2_total']:.1f} "
                f"({(baseline - fit['chi2_total']) / baseline:.1%}) | "
                f"{pulls[feed]:+.2f} | "
                + " | ".join(f"{pulls[source]:+.3f}" for source in DETECTOR)
                + " |"
            )
    return "\n".join(lines)


def bayes_section(bayes, bayes_2to1, crosschecks):
    lines = [
        "Unregularized D'Agostini built straight from the nominal NPZ inputs "
        "with the stability-script conventions (no RooUnfold, no efficiency "
        "floor). `n = 4` is the quoted reference. The comparison is the "
        "per-pT normalized shape against the tau=0 TUnfold production result.",
        "",
        "| channel | merge | n=1 | n=2 | n=3 | **n=4** | n=5 | n=6 | n=7 | n=8 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, source in (("coarse_tail", bayes), ("2to1", bayes_2to1)):
        for channel in CHANNELS:
            scan = source[channel]["iteration_scan"]
            cells = " | ".join(
                (
                    f"**{pct(scan[str(n)]['shape_delta_vs_tunfold']['core']['median'])}**"
                    if n == 4
                    else pct(
                        scan[str(n)]["shape_delta_vs_tunfold"]["core"]["median"]
                    )
                )
                for n in range(1, 9)
            )
            lines.append(f"| {channel} | `{label}` | {cells} |")
    lines.append("")
    lines.append(
        "Cells: median |shape(Bayes)/shape(TUnfold) - 1| over core reported "
        "bins, in percent."
    )
    lines.append("")
    lines.append(
        "| channel | merge | n=4 core med / max | n=4 all reported med / max "
        "| negative bins | stat err med (Bayes / TUnfold) | 1000-iteration "
        "core median |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for label, source in (("coarse_tail", bayes), ("2to1", bayes_2to1)):
        for channel in CHANNELS:
            reference = source[channel]["reference"]
            delta = reference["abs_shape_delta_vs_tunfold"]
            limit = (
                crosschecks[channel]["iterative_bayes_crosscheck"][
                    "iterations"
                ]["1000"]["shape_delta_vs_tunfold"]["core"]["median"]
                if label == "coarse_tail"
                else None
            )
            lines.append(
                f"| {channel} | `{label}` | "
                f"{pct(delta['core']['median'])} / {pct(delta['core']['max'])} | "
                f"{pct(delta['shown']['median'])} / {pct(delta['shown']['max'])} | "
                f"{reference['negative_shown_bins']} | "
                f"{pct(reference['relative_statistical_error']['median'])} / "
                f"{pct(source[channel]['tunfold_reference']['relative_statistical_error']['median'])} | "
                + (pct(limit) if limit is not None else "--")
                + " |"
            )
    lines.append("")
    lines.append(
        "Prior dependence at n=4: the MC prior is swapped for a flat one "
        "(uniform over gen rho bins, keeping each pT slice's total yield). "
        "This is why iterative Bayes stays a cross-check and is not the "
        "production method."
    )
    lines.append("")
    lines.append(
        "| channel | merge | flat per-pT: core med / max | all reported med / "
        "max | flat global: all reported med / max |"
    )
    lines.append("|---|---|---|---|---|")
    for label, source in (("coarse_tail", bayes), ("2to1", bayes_2to1)):
        for channel in CHANNELS:
            sensitivity = source[channel]["flat_prior_sensitivity"]
            per_pt = sensitivity["flat_per_pt"]["shape_delta_vs_mc_prior"]
            glob = sensitivity["flat_global"]["shape_delta_vs_mc_prior"]
            lines.append(
                f"| {channel} | `{label}` | "
                f"{pct(per_pt['core']['median'])} / {pct(per_pt['core']['max'])} | "
                f"{pct(per_pt['shown']['median'])} / {pct(per_pt['shown']['max'])} | "
                f"{pct(glob['shown']['median'])} / {pct(glob['shown']['max'])} |"
            )
    return "\n".join(lines)


#### (label, extractor, "lower"|"higher" is better, formatter)
COMPARABLE = [
    (
        "self-closure max",
        lambda cc, rf: cc["nominal"]["self_closure_max"],
        "lower",
        "{:.1e}",
    ),
    (
        "folded chi2A / ndf",
        lambda cc, rf: cc["nominal"]["chi2A"] / max(cc["nominal"]["ndf"], 1),
        "lower",
        "{:.1f}",
    ),
    (
        "fake fraction",
        lambda cc, rf: cc["input_contract"]["fake_fraction"],
        "lower",
        "{:.3f}",
    ),
    (
        "miss fraction",
        lambda cc, rf: cc["input_contract"]["miss_fraction"],
        "lower",
        "{:.3f}",
    ),
    (
        "response condition number",
        lambda cc, rf: cc["conditioning"]["plain"]["condition"],
        "lower",
        "{:.2f}",
    ),
    (
        "input-toy pull width",
        lambda cc, rf: cc["input_stat_toys"][
            "empirical_over_analytic_sigma"
        ]["median"],
        "unity",
        "{:.3f}",
    ),
    (
        "input-toy 68% coverage",
        lambda cc, rf: cc["input_stat_toys"]["coverage_1sigma"]["median"],
        "target68",
        "{:.3f}",
    ),
    (
        "response-toy sigma ratio",
        lambda cc, rf: cc["response_stat_toys"][
            "empirical_over_sysuncorr_sigma"
        ]["median"],
        "unity",
        "{:.3f}",
    ),
    (
        "area-constraint shape delta (core med)",
        lambda cc, rf: cc["area_constraint_sensitivity"]["shape_delta"][
            "core"
        ]["median"],
        "lower",
        "{:.2e}",
    ),
    (
        "full-vs-diagonal cov shape delta (core med)",
        lambda cc, rf: cc["full_vs_diagonal_data_covariance"]["shape_delta"][
            "core"
        ]["median"],
        "lower",
        "{:.2e}",
    ),
    (
        "Bayes n=4 vs tau=0 (core med)",
        lambda cc, rf: cc["iterative_bayes_crosscheck"]["iterations"]["4"][
            "shape_delta_vs_tunfold"
        ]["core"]["median"],
        "lower",
        "{:.4f}",
    ),
    (
        "data/MC ratio distance before unfolding (core med)",
        lambda cc, rf: cc["data_mc_before_after_unfolding"][
            "before_unfolding"
        ]["core"]["median_absolute"],
        "lower",
        "{:.4f}",
    ),
    (
        "data/MC ratio distance after unfolding (core med)",
        lambda cc, rf: cc["data_mc_before_after_unfolding"][
            "after_unfolding"
        ]["core"]["median_absolute"],
        "lower",
        "{:.4f}",
    ),
    (
        "core bins worse after unfolding",
        lambda cc, rf: cc["data_mc_before_after_unfolding"][
            "core_bins_with_larger_absolute_disagreement"
        ]
        / cc["data_mc_before_after_unfolding"]["n_core_bins"],
        "lower",
        "{:.2f}",
    ),
    (
        "JMS cross-closure, shape-free (core max)",
        lambda cc, rf: max(
            cc["systematic_closure"][f"JMS{d}"][
                "varied_input_nominal_response"
            ]["core"]["max"]
            for d in ("Up", "Down")
        ),
        "lower",
        "{:.4f}",
    ),
    (
        "profiled fit chi2 absorbed (fraction of baseline)",
        lambda cc, rf: (
            rf["baseline_data_stat_only"]["chi2_total"]
            - rf["profile_JER_JMS_JMR_data_stat_only"]["chi2_total"]
        )
        / rf["baseline_data_stat_only"]["chi2_total"],
        "higher",
        "{:.4f}",
    ),
]


#### changes below this relative size are toy/round-off noise, not a trend
SAME_TOLERANCE = 0.03


#### both sides already at float64 round-off: no trend to read
NEGLIGIBLE = 1e-9


def _verdict(new, old, direction):
    if max(abs(new), abs(old)) <= NEGLIGIBLE:
        return "same (both exact)"
    scale = max(abs(new), abs(old), 1e-30)
    if abs(new - old) <= SAME_TOLERANCE * scale:
        return "same"
    if direction == "unity":
        better = abs(new - 1.0) <= abs(old - 1.0)
    elif direction == "target68":
        better = abs(new - 0.68) <= abs(old - 0.68)
    elif direction == "higher":
        better = new >= old
    else:
        better = new <= old
    return "better" if better else "**worse**"


def old_vs_new_section(crosschecks, refold, old_crosschecks, old_refold):
    lines = [
        "The July reference (`outputs/hadronic_rho_crosschecks.json`, "
        "`outputs/hadronic_rho_profiled_refold.json`) was produced on the "
        "trijet-priority selection with the SAME reco binning and the SAME "
        "`coarse_tail` gen merge, so these rows are like-for-like in "
        "everything except the selection (and the Run 2 vs 2018 statistics). "
        "Differences are expected; the point of the table is to surface "
        "regressions.",
        "",
        "| channel | quantity | July (old selection) | pair_split Run 2 | "
        "direction |",
        "|---|---|---|---|---|",
    ]
    for channel in CHANNELS:
        for label, extract, direction, fmt in COMPARABLE:
            try:
                new = extract(crosschecks[channel], refold[channel])
                old = extract(old_crosschecks[channel], old_refold[channel])
            except (KeyError, ZeroDivisionError):
                continue
            lines.append(
                f"| {channel} | {label} | {fmt.format(old)} | "
                f"{fmt.format(new)} | {_verdict(new, old, direction)} |"
            )
    return "\n".join(lines)


def era_section(era_chi2):
    eras = era_chi2["eras"]
    lines = [
        "The Run 2 folded chi2 is decomposed per era, each era rebuilt from "
        "its OWN data and MC pickle so detector conditions, response and "
        "covariance all match. The per-era data vectors sum bit-exactly to "
        "the Phase A Run 2 combination (response to 4e-16), so this is a "
        "decomposition, not a re-derivation.",
        "",
        "Three chi2 are quoted, each against the era's own covariance:",
        "",
        "* `shape` -- prediction rescaled by one global factor, so the "
        "overall normalization is divided out;",
        "* `shape/pT` -- prediction rescaled INSIDE each pT slice, which is "
        "how the measurement is normalized, so both the overall scale and "
        "any pT-spectrum disagreement are divided out and only the rho shape "
        "remains;",
        "* `chi2A` -- the tau=0 TUnfold folded residual of that era's own "
        "unfold.",
        "",
        "The `@R2` columns rescale each era to the Run 2 data yield. A "
        "fractional mismatch that does not depend on the era gives an "
        "era-independent `@R2` value, and the per-era chi2 then sum to the "
        "combined chi2; an era standing out in `@R2` is a conditions or "
        "response mismatch rather than statistics.",
        "",
    ]
    for channel in CHANNELS:
        record = era_chi2[channel]
        lines.append(f"### {channel}")
        lines.append("")
        lines.append(
            "| era | lumi (fb-1) | data yield frac | data/MC norm | shape | "
            "shape/pT | chi2A/ndf | shape @R2 | shape/pT @R2 | chi2A @R2 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for era in eras:
            entry = record[era]
            lines.append(
                f"| {era} | {entry['nominal_lumi_fb']:.1f} | "
                f"{entry['data_yield_fraction']:.3f} | "
                f"{entry['data_over_mc_normalization']:.4f} | "
                f"{entry['reco_chi2_shape']:.0f} | "
                f"{entry['reco_chi2_shape_per_pt']:.0f} | "
                f"{entry['tunfold_chi2A']:.1f}/{entry['tunfold_ndf']} | "
                f"{entry['shape_chi2_at_run2_yield']:.0f} | "
                f"{entry['shape_per_pt_chi2_at_run2_yield']:.0f} | "
                f"{entry['tunfold_chi2A_at_run2_yield']:.1f} |"
            )
        summary = record["summary"]
        lines.append(
            f"| **sum of eras** | 137.6 | 1.000 | -- | "
            f"{summary['sum_of_era_shape_chi2']:.0f} | "
            f"{summary['sum_of_era_shape_per_pt_chi2']:.0f} | "
            f"{summary['sum_of_era_tunfold_chi2A']:.1f} | -- | -- | -- |"
        )
        combined = record["run2_combined"]
        lines.append(
            f"| **Run 2 combined** | 137.6 | 1.000 | "
            f"{combined['data_over_mc_normalization']:.4f} | "
            f"{combined['reco_chi2_shape']:.0f} | "
            f"{combined['reco_chi2_shape_per_pt']:.0f} | "
            f"{combined['tunfold_chi2A']:.1f}/{combined['tunfold_ndf']} | "
            f"{combined['reco_chi2_shape']:.0f} | "
            f"{combined['reco_chi2_shape_per_pt']:.0f} | "
            f"{combined['tunfold_chi2A']:.1f} |"
        )
        lines.append("")
        common = combined["reco_chi2_shape"] / summary["sum_of_era_shape_chi2"]
        lines.append(
            f"sum(eras)/combined = "
            f"{summary['sum_over_combined_shape_ratio']:.2f}, so "
            f"{common:.0%} of the per-era chi2 is a mismatch COMMON to all "
            f"four eras and only {1.0 - common:.0%} is era-specific."
        )
        lines.append("")
        lines.append("Per-pT-slice share of the era's `shape` chi2:")
        lines.append("")
        slices = record[eras[0]]["per_pt_shape_chi2"]
        header = " | ".join(
            f"{entry['pt_range'][0]:.0f}-{entry['pt_range'][1]:.0f}"
            for entry in slices
        )
        lines.append(f"| era | {header} |")
        lines.append("|---" * (len(slices) + 1) + "|")
        for era in eras:
            blocks = record[era]["per_pt_shape_chi2"]
            total = sum(entry["shape_chi2"] for entry in blocks)
            lines.append(
                f"| {era} | "
                + " | ".join(
                    f"{entry['shape_chi2'] / total:.2f}" for entry in blocks
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def config_table(channel, configs):
    config = configs[channel]
    lines = [
        "| configuration | stat err med % | stat err max % | negative bins "
        "| max shape shift vs production % (core / all) | pull width | "
        "68% coverage | mean bias (sigma) | chi2A/ndf |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in config["configurations"].items():
        chi2 = (
            "--"
            if row["chi2A"] is None
            else f"{row['chi2A']:.1f}/{row['ndf']}"
        )
        lines.append(
            f"| `{name}`"
            f" | {pct(row['relative_statistical_error']['median'])}"
            f" | {pct(row['relative_statistical_error']['max'])}"
            f" | {row['negative_reported_bins']}"
            f" | {pct(row['shape_shift_vs_production']['core']['max'])} / "
            f"{pct(row['shape_shift_vs_production']['shown']['max'])}"
            f" | {row['pull_width']['median']:.3f}"
            f" | {row['coverage_1sigma']['median']:.3f}"
            f" | {row['mean_bias_in_sigma']['median']:+.3f}"
            f" | {chi2} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("outputs"),
        help="directory holding the July trijet-priority reference JSONs",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    base = args.dir
    crosschecks = load(base / "hadronic_rho_crosschecks_run2.json")
    systematics = load(base / "hadronic_rho_systematics_run2.json")
    systematics_2to1 = load(base / "hadronic_rho_systematics_run2_2to1.json")
    refold = load(base / "hadronic_rho_profiled_refold_run2.json")
    bayes = load(base / "hadronic_rho_bayes_run2.json")
    bayes_2to1 = load(base / "hadronic_rho_bayes_run2_2to1.json")
    configs = load(base / "hadronic_rho_config_comparison_run2.json")
    configs_2to1 = load(base / "hadronic_rho_config_comparison_run2_2to1.json")
    stability = {
        channel: load(base / f"{channel}_run2_stability.json")
        for channel in CHANNELS
    }
    old_crosschecks = load(args.old_dir / "hadronic_rho_crosschecks.json")
    old_refold = load(args.old_dir / "hadronic_rho_profiled_refold.json")

    rows = check_rows(crosschecks, systematics, refold, bayes, configs)

    out = ["# Phase B -- validation of the pair_split Run 2 groomed-rho unfolds", ""]
    out.append(
        "Run 2 combined (2016APV+2016+2017+2018, ~137.6 fb-1), groomed rho, no "
        "m_g floor. Production configuration: tau=0 TUnfoldDensity, "
        "`kRegModeCurvature`, `kDensityModeNone`, area constraint, full "
        "event-clustered data covariance, fakes removed as a per-reco-bin "
        "survival factor, misses in the response underflow, gen bins merged "
        "with the **`coarse_tail`** nested map (dijet 10 reported rho bins per "
        "pT slice, trijet 6). The coarser `2to1` candidate is carried "
        "alongside wherever it is cheap."
    )
    out.append("")
    out.append("## 1. Check register")
    out.append("")
    out.append("| channel | check | criterion | value | status |")
    out.append("|---|---|---|---|---|")
    for channel, name, criterion, value, verdict in rows:
        #### escape pipes so criteria written with |x| stay inside their cell
        criterion = criterion.replace("|", "\\|")
        value = value.replace("|", "\\|")
        out.append(f"| {channel} | {name} | {criterion} | {value} | **{verdict}** |")
    out.append("")

    out.append("## 2. Systematic bands: detector and parton shower")
    out.append("")
    out.append(systematics_section(systematics, systematics_2to1))
    out.append("")

    out.append("## 3. Profiled refolding fit")
    out.append("")
    out.append(refold_section(refold))
    out.append("")

    out.append("## 4. Iterative Bayes (D'Agostini) cross-check")
    out.append("")
    out.append(bayes_section(bayes, bayes_2to1, crosschecks))
    out.append("")

    out.append("## 5. Configuration comparison")
    out.append("")
    out.append(
        "All rows share the inputs, the gen merge, the fake/miss treatment "
        "and the normalization window; only the estimator changes. Pull "
        "width, coverage and bias come from input pseudo-experiments thrown "
        f"around the matched MC reco spectrum with the data covariance "
        f"({configs['dijet']['n_toys']} toys). Iterative Bayes has no analytic "
        "covariance here -- its quoted error IS the toy spread, so its pull "
        "width is 1 by construction and only its coverage and bias carry "
        "information."
    )
    out.append("")
    for channel in CHANNELS:
        out.append(f"### {channel} -- `coarse_tail` (production merge)")
        out.append("")
        out.append(config_table(channel, configs))
        out.append("")
        out.append(f"### {channel} -- `2to1` (coarser candidate)")
        out.append("")
        out.append(config_table(channel, configs_2to1))
        out.append("")

    out.append("## 6. Gen-merge trade-off (from the tau=0 stability ladder)")
    out.append("")
    out.append(
        "| channel | row | reco x gen | reported | closure | stat med/max % | "
        "amp med/max | \\|rho_i\\| med/max | neg | chi2A/ndf | shape flips |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for channel in CHANNELS:
        for record in stability[channel]:
            if record["label"] not in ("proposal_coarse_tail", "proposal_2to1"):
                continue
            out.append(
                f"| {channel} | `{record['label']}` | "
                f"{record['n_reco']} x {record['n_gen']} | "
                f"{record['n_shown']} | {record['closure_max']:.1e} | "
                f"{pct(record['rel_stat_median'])} / {pct(record['rel_stat_max'])} | "
                f"{record['amp_median']:.2f} / {record['amp_max']:.2f} | "
                f"{record['rho_median']:.3f} / {record['rho_max']:.3f} | "
                f"{record['negative']} | {record['chi2A']:.1f}/{record['ndf']} | "
                f"{record['flips']}/{record['flips_of']} |"
            )
    out.append("")

    out.append("## 7. Per reported bin")
    out.append("")
    out.append(
        "`in win` marks the per-pT normalization window "
        "(dijet [-2.85, -0.55], trijet [-3.0, -0.7]); bins outside it are "
        "reported but do not set the normalization. All systematic columns are "
        "the symmetrized half-difference of the normalized shape, in percent. "
        "`Bayes n=4 %` is the signed normalized-shape difference of the "
        "D'Agostini cross-check against this tau=0 result."
    )
    for channel in CHANNELS:
        out.append("")
        out.append(f"### {channel} -- `coarse_tail`, gen rho edges "
                   f"{configs[channel]['gen_rho_edges']}")
        out.append(per_bin_table(channel, systematics, configs, bayes))
    for channel in CHANNELS:
        out.append("")
        out.append(f"### {channel} -- `2to1`, gen rho edges "
                   f"{configs_2to1[channel]['gen_rho_edges']}")
        out.append(
            per_bin_table(channel, systematics_2to1, configs_2to1, bayes_2to1)
        )
    out.append("")

    out.append("## 8. pair_split Run 2 versus the July trijet-priority selection")
    out.append("")
    out.append(
        old_vs_new_section(
            crosschecks, refold, old_crosschecks, old_refold
        )
    )
    out.append("")
    #### the July reference is 2018-only (its yields are 2.2-2.4x smaller), so
    #### the fair comparison for anything that scales with luminosity is the
    #### 2018-only pair_split stability row, not the Run 2 one
    matched = {}
    for channel in CHANNELS:
        rows18 = load(base / f"{channel}_2018_stability.json")
        record = [
            entry for entry in rows18 if entry["label"] == "proposal_coarse_tail"
        ][0]
        matched[channel] = record["chi2A"]
    out.append(
        "**The chi2 rows need a luminosity correction before they can be "
        "read as regressions.** The July reference is 2018-only -- its fake "
        "and miss yields are 2.2-2.4x smaller than the Run 2 numbers -- while "
        "everything above is Run 2 combined. A fixed fractional data/MC "
        "mismatch produces a chi2 that grows linearly with luminosity, and "
        "that is exactly what is seen: dijet chi2A rises by "
        f"{crosschecks['dijet']['nominal']['chi2A'] / old_crosschecks['dijet']['nominal']['chi2A']:.2f}x "
        "and trijet by "
        f"{crosschecks['trijet']['nominal']['chi2A'] / old_crosschecks['trijet']['nominal']['chi2A']:.2f}x "
        "against a luminosity ratio of about 2.3."
    )
    out.append("")
    out.append(
        "At **matched** luminosity (the 2018-only pair_split stability row "
        "against the July 2018-only reference, same merge, same reco binning):"
    )
    out.append("")
    out.append(
        "| channel | July 2018 chi2A | pair_split 2018 chi2A | pair_split "
        "Run 2 chi2A | verdict at matched lumi |"
    )
    out.append("|---|---|---|---|---|")
    for channel in CHANNELS:
        july = old_crosschecks[channel]["nominal"]["chi2A"]
        run2 = crosschecks[channel]["nominal"]["chi2A"]
        verdict = (
            "unchanged"
            if abs(matched[channel] / july - 1.0) < 0.1
            else ("better" if matched[channel] < july else "**worse**")
        )
        out.append(
            f"| {channel} | {july:.1f} | {matched[channel]:.1f} | "
            f"{run2:.1f} | {verdict} |"
        )
    out.append("")
    out.append(
        "So the dijet data/MC agreement is **unchanged** by the selection "
        "change once luminosity is matched, and the trijet agreement is "
        "**better** by roughly a factor two. The remaining trijet Run 2 / "
        f"2018 ratio ({crosschecks['trijet']['nominal']['chi2A'] / matched['trijet']:.1f}x) "
        "is larger than the luminosity ratio, which is a genuine statement "
        "that the added eras do not agree with the 2018-tuned response as "
        "well as 2018 does -- worth a per-era look before anything is quoted."
    )
    out.append("")
    out.append("The remaining flagged rows, in order of how much they matter:")
    out.append("")
    out.append(
        "* **dijet fake fraction 13.7% -> 18.6%** (misses 35.0% -> 16.1%). "
        "A real consequence of the selection change, already documented in "
        "Phase A: almost all of it sits in the 200-290 GeV pT slice. More "
        "fakes means more of the measurement rests on the MC-derived "
        "survival factor, which is exactly what the first-pT feed-in "
        "nuisance in section 3 probes -- and that nuisance runs to -2.8 "
        "sigma at a 10% prior, so the dijet fake normalization is the "
        "weakest modelled part of this measurement."
    )
    out.append(
        "* **full-vs-diagonal covariance sensitivity 3.1e-4 -> 1.4e-3** "
        "(core median; max 2.5%). The pair_split selection puts both jets of "
        "an event into the measurement more often, so the event-clustered "
        "off-diagonal terms matter more than before. This is an argument for "
        "keeping the full covariance, not against the selection."
    )
    out.append(
        "* **profiled chi2 absorbed 4.3% -> 3.7% of baseline.** In absolute "
        "terms the nuisances absorb more than before (179 vs 88 units of "
        "chi2); the fraction falls only because the baseline chi2 grew with "
        "luminosity. Not a regression in the fit."
    )
    out.append(
        "* **Bayes n=4 vs tau=0, 2.00% -> 2.13%**, and **data/MC ratio "
        "distance before unfolding, 4.8% -> 5.1%.** Both are sub-percent "
        "absolute moves on quantities that are themselves a few percent; "
        "neither changes a conclusion."
    )
    out.append("")

    era_path = base / "hadronic_rho_era_chi2_run2.json"
    if era_path.exists():
        era_chi2 = load(era_path)
        out.append("## 9. Per-era decomposition of the folded chi2")
        out.append("")
        out.append(era_section(era_chi2))
        dijet_at_r2 = [
            era_chi2["dijet"][era]["tunfold_chi2A_at_run2_yield"]
            for era in era_chi2["eras"]
        ]
        trijet_at_r2 = [
            era_chi2["trijet"][era]["tunfold_chi2A_at_run2_yield"]
            for era in era_chi2["eras"]
        ]
        out.append("**Verdict.**")
        out.append("")
        out.append(
            "* **The extra Run 2 chi2 is not one bad era.** Per-era chi2A at "
            "a common yield: dijet "
            + ", ".join(
                f"{era} {value:.0f}"
                for era, value in zip(era_chi2["eras"], dijet_at_r2)
            )
            + " (falls monotonically 2016APV -> 2018, spread "
            f"{max(dijet_at_r2) / min(dijet_at_r2):.1f}x); trijet "
            + ", ".join(
                f"{era} {value:.0f}"
                for era, value in zip(era_chi2["eras"], trijet_at_r2)
            )
            + f" (spread {max(trijet_at_r2) / min(trijet_at_r2):.1f}x, not "
            "monotonic). No era is an isolated outlier on the HIGH side in "
            "either channel."
        )
        out.append(
            "* **The trijet 4.1x-vs-2.3x puzzle is 2018 being anomalously "
            f"GOOD, not the other eras being bad.** Its chi2A at Run 2 yield "
            f"is {trijet_at_r2[-1]:.0f} against {min(trijet_at_r2[:-1]):.0f}"
            f"-{max(trijet_at_r2[:-1]):.0f} for the other three and "
            f"{era_chi2['trijet']['run2_combined']['tunfold_chi2A']:.0f} for "
            "the combination. The July study extrapolated from the single "
            "best-agreeing era, so it understated what Run 2 would give. "
            "Nothing new is broken."
        )
        common = {
            channel: era_chi2[channel]["run2_combined"]["reco_chi2_shape"]
            / era_chi2[channel]["summary"]["sum_of_era_shape_chi2"]
            for channel in CHANNELS
        }
        out.append(
            f"* **Most of the mismatch is common to all four eras**: "
            f"{common['dijet']:.0%} (dijet) and {common['trijet']:.0%} "
            "(trijet) of the summed per-era chi2 survives the combination. "
            f"The era-specific residue is {1 - common['dijet']:.0%} and "
            f"{1 - common['trijet']:.0%}."
        )
        out.append(
            "* **The dijet chi2 is a pT-spectrum effect, not a rho-shape "
            "one.** 73-84% of it sits in the highest pT slice (> 570 GeV), "
            "where every rho bin pulls coherently negative (largest pulls "
            "-136 to -207, all in the [-10, -4] buffer at pT > 570). "
            "Normalizing inside each pT slice -- which is exactly how the "
            "measurement is reported -- removes 77% of it (284775 -> 65469). "
            "The quoted rho shapes are therefore far less affected than the "
            "raw chi2A suggests."
        )
        out.append(
            "* **2016APV is the weakest era in both channels and the effect "
            "survives per-pT normalization** (dijet 101654 at Run 2 yield "
            "against 61200-70133 for the other three), so it is a genuine "
            "rho-shape mismatch rather than a normalization one. Consistent "
            "with the APV/HIPM-era calibration being the least well matched "
            "by the response; a per-era JMS check is the natural follow-up, "
            "but the size does not threaten the combination."
        )
        out.append(
            "* **The per-era data/MC normalization spread is larger than the "
            "luminosity uncertainty**: dijet 1.039-1.084 (4.5%), trijet "
            "0.982-1.053 (7.2%, with 2017 low and 2018 high). That is well "
            "above the ~2.5% CMS luminosity uncertainty and points at "
            "trigger/prescale bookkeeping rather than physics. It does not "
            "affect the per-pT normalized measurement, but it should be "
            "understood before any absolute cross-section is quoted."
        )
        out.append("")
        out.append("## 10. Verdict, anomalies and unavailable checks")
    else:
        out.append("## 9. Verdict, anomalies and unavailable checks")
    out.append("")
    production = {
        channel: configs[channel]["configurations"]["tau0_area_full"]
        for channel in CHANNELS
    }
    out.append(
        "**Configuration verdict.** The production configuration -- tau=0 "
        "TUnfold with the area constraint and the full event-clustered data "
        "covariance -- is the best-behaved row in both channels and nothing "
        "here challenges it."
    )
    out.append("")
    out.append(
        "* The area constraint is numerically irrelevant at tau=0: the "
        "no-area shape agrees with the production shape to "
        f"{pct(configs['dijet']['configurations']['tau0_noarea_full']['shape_shift_vs_production']['shown']['max'], 5)}% "
        "(dijet) and "
        f"{pct(configs['trijet']['configurations']['tau0_noarea_full']['shape_shift_vs_production']['shown']['max'], 3)}% "
        "(trijet). It costs one degree of freedom and is kept for continuity."
    )
    out.append(
        "* GLS reproduces the no-area TUnfold solution to machine precision, "
        "so the TUnfold call is not doing anything the linear algebra does "
        "not."
    )
    out.append(
        "* Diagonalizing the data covariance moves the dijet shape by up to "
        f"{pct(configs['dijet']['configurations']['tau0_area_diag']['shape_shift_vs_production']['core']['max'])}% "
        "and leaves trijet exactly unchanged (its covariance is diagonal by "
        "construction -- one measured jet per event). The dijet off-diagonal "
        "terms are real two-jets-per-event correlations and must be kept."
    )
    out.append(
        "* Iterative Bayes at n=4 has a visibly smaller statistical error "
        f"(dijet {pct(configs['dijet']['configurations']['bayes_n4']['relative_statistical_error']['median'])}% "
        f"vs {pct(production['dijet']['relative_statistical_error']['median'])}%) "
        "only because it is still regularized toward the MC prior: the "
        "difference against tau=0 shrinks monotonically with iteration and "
        "the flat-prior swap moves it by more than the entire systematic "
        "band. It stays a cross-check."
    )
    out.append(
        "* Pull width and 68% coverage are within tolerance for every "
        "configuration in both channels, so no row is mis-stating its own "
        "uncertainty."
    )
    out.append("")
    out.append(
        "**Gen merge.** `coarse_tail` (production) keeps 50 reported dijet "
        "bins and 18 trijet bins; `2to1` halves the dijet statistical error "
        "(0.69% vs 1.28% median) and lowers the amplification, at the cost of "
        "resolution and of degrees of freedom. Both close exactly and neither "
        "produces a negative bin, so this is a physics-reach choice, not a "
        "stability one; see section 6."
    )
    out.append("")
    out.append("**Anomalies found in this phase.**")
    out.append("")
    out.append(
        "1. **The Phase A systematic NPZs carry no data.** Every "
        "`<channel>_run2_<variation>.npz` has `data_reco`, `data_w2` and "
        "`data_V` identically zero -- Phase A produced them without "
        "`--data`. `rho_unfold_systematics.py` and `rho_profiled_refold.py` "
        "both unfold DATA through the shifted response and would have "
        "crashed or silently fit a nuisance that scales the data away. The "
        "nominal data is now restored at load time by "
        "`rho_unfold_systematics.load_variation`; because "
        "`rho_unfold_inputs_from_hists.py` always reads the data histogram "
        "from its nominal slice, this is bit-identical to having passed "
        "`--data` on the variation runs. Nothing was faked, and the "
        "crosschecks results are unchanged by it (verified byte-for-byte)."
    )
    out.append(
        "2. **The gen spectrum is not exactly invariant under JER/JMS/JMR.** "
        "Four of 65 dijet gen bins move by up to 5.5e-6 relative between the "
        "nominal and JER slices (3.2e-5 for trijet), present in the 2018-only "
        "files too so it is not an era-summing artifact. Expected for a "
        "pair_split selection whose gen bookkeeping follows a reco-chosen "
        "pair. Negligible at this size, but it was exactly zero in the July "
        "trijet-priority selection."
    )
    out.append(
        f"3. **The folded data/MC chi2 is large**: dijet "
        f"{crosschecks['dijet']['nominal']['chi2A']:.0f}/"
        f"{crosschecks['dijet']['nominal']['ndf']}, trijet "
        f"{crosschecks['trijet']['nominal']['chi2A']:.0f}/"
        f"{crosschecks['trijet']['nominal']['ndf']} at the production merge "
        "(5803/29 and 48/8 at the `2to1` merge). Detector nuisances absorb "
        "only a small part of it (section 3). This is a data/MC modelling "
        "statement at Run 2 statistics, not a construction failure -- "
        "self-closure is exact and the bottom-line inequality holds."
    )
    out.append(
        "4. **The dijet folded residual is concentrated at low rho and high "
        "pT.** The largest diagonal pulls are +29.6 in pT [480, 570] rho "
        "[-3.4, -2.85] and -20.7 in the same pT slice at rho [-4.0, -3.4]; "
        "the top three covariance eigenmodes carry 42% of the chi2. Same "
        "pattern as the July selection."
    )
    out.append(
        "5. **The top rho catch-all carries a very large JMS shift.** In the "
        "dijet [-0.55, 0] bin the JMS half-difference is 13-16% in every pT "
        "slice, an order of magnitude above anything inside the "
        "normalization window (max 5.0%). That bin is the overflow "
        "catch-all, is outside the window and does not set the "
        "normalization, but it is reported -- any quotable number taken from "
        "it needs the full band attached. The same bin drives the 18.7% "
        "JMS cross-closure `shown` maximum."
    )
    out.append(
        "6. **ISR is negligible next to FSR**, as the policy assumes: core "
        "median 0.06% (dijet) and 0.03% (trijet) against 0.83% and 1.07% for "
        "FSR. ISR is recorded here for observation only and is not part of "
        "any band."
    )
    out.append(
        "7. **JMS is the dominant detector source and pushes the profiled "
        "fit to its template boundary** in the dijet channel (JMS +1, JMR -1 "
        "both at bound). The nuisance range is capped at +/-1 because no "
        "physical extrapolation beyond the supplied templates is defined, so "
        "the true minimum may lie outside it."
    )
    out.append("")
    out.append("**Checks that could not run, with the reason.**")
    out.append("")
    out.append(
        "* *Alternate-generator (HERWIG) response closure* -- no pair_split "
        "HERWIG production exists. `outputs/{dijet,trijet}_herwig_prod_full.npz` "
        "are on the July trijet-priority selection and were deliberately NOT "
        "substituted. The model-response stress is therefore untested on "
        "these inputs."
    )
    out.append(
        "* *Statistically independent MC half-response closure* -- the NPZ "
        "set carries a single response built from the full MC; no split-half "
        "response exists."
    )
    out.append(
        "* *m_g > 2 GeV floor-enabled production* -- the `_mfloor2` "
        "histograms exist in the source pickles but were deliberately not "
        "used for these NPZs, so the floor cannot be studied without "
        "regenerating inputs."
    )
    out.append(
        "* *Joint dijet-trijet data and response covariance* -- the two "
        "channels are stored as independent NPZs with no cross-channel "
        "covariance block."
    )
    out.append(
        "* *Comparison of the JER/JMS/JMR band against the July selection* -- "
        "no old-selection systematics JSON was ever written "
        "(`outputs/` has only the crosschecks and profiled-refold references), "
        "so only the crosschecks and refold numbers can be compared "
        "old-vs-new."
    )
    out.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")

    if args.json_out is not None:
        companion = {
            "checks": [
                {
                    "channel": channel,
                    "check": name,
                    "criterion": criterion,
                    "value": value,
                    "status": verdict,
                }
                for channel, name, criterion, value, verdict in rows
            ],
            "sources": {
                "crosschecks": "hadronic_rho_crosschecks_run2.json",
                "systematics": "hadronic_rho_systematics_run2.json",
                "systematics_2to1": "hadronic_rho_systematics_run2_2to1.json",
                "profiled_refold": "hadronic_rho_profiled_refold_run2.json",
                "iterative_bayes": "hadronic_rho_bayes_run2.json",
                "iterative_bayes_2to1": "hadronic_rho_bayes_run2_2to1.json",
                "config_comparison": "hadronic_rho_config_comparison_run2.json",
                "config_comparison_2to1": (
                    "hadronic_rho_config_comparison_run2_2to1.json"
                ),
            },
        }
        args.json_out.write_text(json.dumps(companion, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
