#!/usr/bin/env python3
"""Quantify the Z+jet rho bottom-line test and preconditioning invariance.

The test compares the same fixed PYTHIA model in two spaces:

  detector: fake-corrected data vs. matched reconstructed PYTHIA;
  unfolded: unfolded data vs. PYTHIA truth (matched + misses).

Only data statistical covariance is included.  The script deliberately does
not turn a response reweighting, a bin merge, or a matrix preconditioner into a
new central result; it reports whether each is a valid robustness check.

Run from the repository root:

    source scripts/setup_root.sh
    .venv/bin/python scripts/study_bottom_line.py

The default tags compare the nominal response to the two low-rho binning/area
constraint experiments.  No jackknife replicas are run because this study uses
the propagated input-data covariance, not the publication statistical band.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unfold.tools.unfolder_core import Unfolder, get_spec


DEFAULT_TAGS = ("original", "original_merge25", "original_merge25_noarea")
OUTPUT_DIR = REPO_ROOT / "outputs/zjet/rho/bottom_line_test"


def condition_number(matrix: np.ndarray) -> float:
    """Return a finite condition number when the matrix has resolved modes."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    nonzero = singular_values[singular_values > np.finfo(float).eps * singular_values[0]]
    if nonzero.size == 0:
        return float("inf")
    return float(nonzero[0] / nonzero[-1])


def weighted_least_squares_solution(
    response: np.ndarray,
    measured: np.ndarray,
    uncertainties: np.ndarray,
) -> np.ndarray:
    """Solve a data-stat-weighted, unregularized response equation.

    This is used only to demonstrate preconditioning invariance below.  The
    analysis result itself continues to come from TUnfold.
    """
    valid = uncertainties > 0
    whitened_response = response[valid] / uncertainties[valid, None]
    whitened_measured = measured[valid] / uncertainties[valid]
    return np.linalg.lstsq(whitened_response, whitened_measured, rcond=None)[0]


def preconditioning_check(unfolder: Unfolder) -> dict[str, float]:
    """Show why a diagonal reco-row factor cannot alter the physical result.

    Let ``D`` be a diagonal reco-space factor.  Transforming ``y -> D y``,
    ``A -> D A``, and ``V -> D V D^T`` changes numerical units only.  The
    whitened system and hence the unregularized estimator are unchanged.
    """
    model_truth = np.asarray(unfolder.y_true, dtype=float)
    response = np.divide(
        np.asarray(unfolder.mosaic, dtype=float),
        model_truth[None, :],
        out=np.zeros_like(unfolder.mosaic, dtype=float),
        where=model_truth[None, :] > 0,
    )
    measured = np.asarray(unfolder.y_meas, dtype=float)
    variance = np.asarray(unfolder.corrected_measured_variances, dtype=float)
    uncertainty = np.sqrt(np.clip(variance, 0.0, None))
    valid = uncertainty > 0

    # A typical "make it look more diagonal" row normalization.  It can alter
    # the ordinary matrix condition number, but it cannot alter the covariance-
    # weighted system after D is applied consistently.
    row_norm = np.linalg.norm(response, axis=1)
    row_scale = np.divide(
        1.0,
        row_norm,
        out=np.ones_like(row_norm),
        where=row_norm > 0,
    )
    scaled_response = row_scale[:, None] * response
    scaled_measured = row_scale * measured
    scaled_uncertainty = row_scale * uncertainty

    solution = weighted_least_squares_solution(response, measured, uncertainty)
    scaled_solution = weighted_least_squares_solution(
        scaled_response,
        scaled_measured,
        scaled_uncertainty,
    )
    residual = measured - response @ model_truth
    scaled_residual = row_scale * residual
    covariance = np.diag(variance)
    scaled_covariance = (
        row_scale[:, None] * covariance * row_scale[None, :]
    )
    original_chi2 = Unfolder._chi2_from_covariance(residual, covariance)["chi2"]
    scaled_chi2 = Unfolder._chi2_from_covariance(
        scaled_residual,
        scaled_covariance,
    )["chi2"]

    weighted_response = response[valid] / uncertainty[valid, None]
    weighted_scaled_response = (
        scaled_response[valid] / scaled_uncertainty[valid, None]
    )
    return {
        "plain_response_condition": condition_number(response),
        "row_scaled_response_condition": condition_number(scaled_response),
        "data_weighted_response_condition": condition_number(weighted_response),
        "data_weighted_row_scaled_response_condition": condition_number(
            weighted_scaled_response
        ),
        "max_abs_unregularized_solution_change": float(
            np.max(np.abs(solution - scaled_solution))
        ),
        "smeared_chi2_before": original_chi2,
        "smeared_chi2_after": scaled_chi2,
        "abs_smeared_chi2_change": abs(scaled_chi2 - original_chi2),
    }


def evaluate_tag(tag: str, groomed: bool, output_dir: Path) -> dict:
    spec = get_spec("zjet", "rho", tag)
    mode = "groomed" if groomed else "ungroomed"
    # Avoid writing production figures while still keeping a traceable folder
    # for ROOT/TUnfold's temporary artifacts.
    study_spec = replace(
        spec,
        output_dir=str(output_dir / "work" / tag / mode) + "/",
    )
    unfolder = Unfolder(
        study_spec,
        groomed,
        do_syst=False,
        cms_label="Internal",
        compute_jackknife_stat=False,
    )
    return {
        "tag": tag,
        "mode": mode,
        "binning": {
            "n_reco": int(unfolder.mosaic.shape[0]),
            "n_truth": int(unfolder.mosaic.shape[1]),
            "gen_edges_by_pt": unfolder.gen_edges_by_pt,
        },
        "bottom_line": unfolder.bottom_line_test(),
        "preconditioning": preconditioning_check(unfolder),
    }


def write_summary(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bottom_line_summary.json"
    json_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    lines = [
        "# Z+jet rho bottom-line test (data-stat-only)",
        "",
        "The detector-space statistic compares fake-corrected data to matched reconstructed PYTHIA. The unfolded-space statistic compares TUnfold data to PYTHIA truth including misses. No detector/model systematic, response-MC statistical, or unfolding-bias covariance is included.",
        "",
        "A passing unregularized test has $\\chi^2_\\mathrm{unfolded} \\leq \\chi^2_\\mathrm{smeared}$. The ndf is the numerical covariance rank; no model normalization is fitted.",
        "",
        "| tag | mode | $\\chi^2_\\mathrm{smeared}$/ndf (p) | $\\chi^2_\\mathrm{unfolded}$/ndf (p) | inequality |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        test = result["bottom_line"]
        smeared = test["smeared"]
        unfolded = test["unfolded"]
        lines.append(
            f"| {result['tag']} | {result['mode']} | "
            f"{smeared['chi2']:.2f}/{smeared['ndof']} ({smeared['pvalue']:.3g}) | "
            f"{unfolded['chi2']:.2f}/{unfolded['ndof']} ({unfolded['pvalue']:.3g}) | "
            f"{'PASS' if test['inequality_holds'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Preconditioning check",
            "",
            "For each case the script multiplies every reco row by a diagonal factor and transforms the covariance with the same factor. This can change the ordinary condition number printed for the raw response, but must leave both the data-weighted condition number, the unregularized solution, and the smeared chi-square unchanged. It is therefore a numerical unit change, not a physics correction or a way to diagonalize migrations.",
            "",
            "| tag | mode | raw cond. | row-scaled cond. | weighted cond. | max $|\\Delta x|$ | $|\\Delta\\chi^2|$ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        check = result["preconditioning"]
        lines.append(
            f"| {result['tag']} | {result['mode']} | "
            f"{check['plain_response_condition']:.3g} | "
            f"{check['row_scaled_response_condition']:.3g} | "
            f"{check['data_weighted_response_condition']:.3g} | "
            f"{check['max_abs_unregularized_solution_change']:.3g} | "
            f"{check['abs_smeared_chi2_change']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This validates information preservation for the fixed PYTHIA model. It does **not** claim PYTHIA describes the data, and it does not validate unmodelled detector effects. Response-to-data reweighting and the alternative-generator closure remain separate model-dependence studies.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def plot_chi2(results: list[dict], output_dir: Path) -> None:
    short_tag = {
        "original": "original",
        "original_merge25": r"merge $\rho<-2.5$",
        "original_merge25_noarea": r"merge $\rho<-2.5$ (no area)",
    }
    labels = [f"{short_tag.get(r['tag'], r['tag'])}\n{r['mode']}" for r in results]
    smeared = [
        r["bottom_line"]["smeared"]["chi2"]
        for r in results
    ]
    unfolded = [
        r["bottom_line"]["unfolded"]["chi2"]
        for r in results
    ]
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(12, 2.8 * len(results)), 6.2))
    width = 0.38
    ax.bar(x - width / 2, smeared, width, label=r"smeared $\chi^2$", color="#e42536")
    ax.bar(x + width / 2, unfolded, width, label=r"unfolded $\chi^2$", color="#5790fc")
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel(r"$\chi^2$")
    ax.set_title(r"Z+jet $\rho$ bottom-line test (data statistical covariance only)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "bottom_line_chi2.pdf")
    fig.savefig(output_dir / "bottom_line_chi2.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags",
        nargs="+",
        default=list(DEFAULT_TAGS),
        help="Z+jet rho response tags to test.",
    )
    parser.add_argument(
        "--mode",
        choices=("groomed", "ungroomed", "both"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = (False, True) if args.mode == "both" else (args.mode == "groomed",)
    output_dir = args.output_dir.resolve()
    results = []
    for tag in args.tags:
        for groomed in modes:
            mode = "groomed" if groomed else "ungroomed"
            print(f"\n=== {tag} | {mode} ===")
            result = evaluate_tag(tag, groomed, output_dir)
            bottom_line = result["bottom_line"]
            print(
                "smeared: "
                f"{bottom_line['smeared']['chi2']:.3f}/"
                f"{bottom_line['smeared']['ndof']}; unfolded: "
                f"{bottom_line['unfolded']['chi2']:.3f}/"
                f"{bottom_line['unfolded']['ndof']}; "
                f"inequality={'PASS' if bottom_line['inequality_holds'] else 'FAIL'}"
            )
            results.append(result)
    write_summary(results, output_dir)
    plot_chi2(results, output_dir)
    print(f"\nWrote {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
