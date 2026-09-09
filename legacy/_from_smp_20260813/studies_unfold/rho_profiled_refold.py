#!/usr/bin/env python
"""Profile response and feed-in nuisances in the hadronic rho refolding fit.

This diagnostic keeps every adopted coarse truth-bin yield free.  It does not
regularize toward the PYTHIA prior.  Instead it minimizes

    (y(theta) - P(theta) x)^T V^-1 (y(theta) - P(theta) x) + theta^T theta

where ``x`` is the particle-level spectrum, ``P`` is the response probability
matrix, and ``theta`` contains Gaussian-constrained response/feed-in nuisance
parameters.  JER/JMS/JMR use the supplied Up/Down response inputs.  The
first-pT feed-in nuisance coherently varies the nominal MC-derived fake yield
only in the 200--290 GeV reco slice.

The fit is a goodness-of-fit and nuisance-model diagnostic.  It is not yet a
complete experimental likelihood: JES, trigger/prescale, pileup, and
alternative-generator response variations are absent from the current inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SOURCES = ("JER", "JMS", "JMR")


@dataclass(frozen=True)
class Nuisance:
    name: str
    response_derivative: np.ndarray
    data_derivative: np.ndarray
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    response_up_shift: np.ndarray | None = None
    response_down_shift: np.ndarray | None = None
    data_up_shift: np.ndarray | None = None
    data_down_shift: np.ndarray | None = None


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return {key: value for key, value in np.load(path).items()}


def whitening_matrix(
    covariance: np.ndarray, rcond: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Return W with ||W r||^2 = r^T pinv(V) r and the retained rank."""
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not eigenvalues.size or eigenvalues[-1] <= 0.0:
        raise ValueError("Input covariance has no positive eigenvalues")
    keep = eigenvalues > rcond * eigenvalues[-1]
    whitening = (
        eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    ).T
    return whitening, int(np.sum(keep))


def solve_unconstrained_gls(
    data: np.ndarray,
    covariance: np.ndarray,
    response: np.ndarray,
) -> dict:
    """Solve the tau=0 generalized least-squares problem."""
    whitening, covariance_rank = whitening_matrix(covariance)
    design = whitening @ response
    target = whitening @ data
    truth, _, design_rank, singular_values = np.linalg.lstsq(
        design, target, rcond=1e-12
    )
    residual = data - response @ truth
    chi2_data = float(np.linalg.norm(whitening @ residual) ** 2)
    covariance_truth = np.linalg.pinv(design.T @ design, rcond=1e-12)
    return {
        "truth": truth,
        "truth_covariance": covariance_truth,
        "residual": residual,
        "chi2_data": chi2_data,
        "chi2_total": chi2_data,
        "covariance_rank": covariance_rank,
        "design_rank": int(design_rank),
        "ndf": int(covariance_rank - design_rank),
        "condition": float(singular_values[0] / singular_values[-1]),
        "nuisance_pulls": {},
        "success": True,
        "message": "unconstrained GLS",
    }


def solve_profiled(
    data: np.ndarray,
    covariance: np.ndarray,
    response: np.ndarray,
    nuisances: list[Nuisance],
    initial_truth: np.ndarray | None = None,
) -> dict:
    """Fit free truth yields and Gaussian-constrained nuisance parameters."""
    if not nuisances:
        return solve_unconstrained_gls(data, covariance, response)

    whitening, covariance_rank = whitening_matrix(covariance)
    n_truth = response.shape[1]
    n_nuisance = len(nuisances)
    if initial_truth is None:
        initial_truth = solve_unconstrained_gls(
            data, covariance, response
        )["truth"]

    def unpack(parameters):
        return parameters[:n_truth], parameters[n_truth:]

    def varied_inputs(nuisance_values):
        varied_response = response.copy()
        varied_data = data.copy()
        for nuisance, value in zip(nuisances, nuisance_values):
            if (
                nuisance.response_up_shift is not None
                and nuisance.response_down_shift is not None
                and nuisance.data_up_shift is not None
                and nuisance.data_down_shift is not None
            ):
                # Piecewise-linear interpolation passes exactly through the
                # nominal and supplied +/-1 templates and preserves
                # non-negative response cells between those templates.
                if value >= 0.0:
                    varied_response += value * nuisance.response_up_shift
                    varied_data += value * nuisance.data_up_shift
                else:
                    varied_response += (-value) * nuisance.response_down_shift
                    varied_data += (-value) * nuisance.data_down_shift
            else:
                varied_response += value * nuisance.response_derivative
                varied_data += value * nuisance.data_derivative
        # Simultaneous additive template morphs can make cells that are zero
        # in the nominal response infinitesimally negative.  A response
        # probability cannot be negative, so enforce the physical boundary.
        varied_response = np.maximum(varied_response, 0.0)
        return varied_data, varied_response

    def residual_vector(parameters):
        truth, nuisance_values = unpack(parameters)
        varied_data, varied_response = varied_inputs(nuisance_values)
        data_residual = varied_data - varied_response @ truth
        return np.concatenate(
            [whitening @ data_residual, nuisance_values]
        )

    initial = np.concatenate(
        [np.asarray(initial_truth, dtype=float), np.zeros(n_nuisance)]
    )
    lower_bounds = np.concatenate(
        [
            np.full(n_truth, -np.inf),
            np.asarray(
                [nuisance.lower_bound for nuisance in nuisances],
                dtype=float,
            ),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(n_truth, np.inf),
            np.asarray(
                [nuisance.upper_bound for nuisance in nuisances],
                dtype=float,
            ),
        ]
    )
    fit = least_squares(
        residual_vector,
        initial,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=4000,
    )
    truth, nuisance_values = unpack(fit.x)
    varied_data, varied_response = varied_inputs(nuisance_values)
    residual = varied_data - varied_response @ truth
    chi2_data = float(np.linalg.norm(whitening @ residual) ** 2)
    penalty = float(np.sum(nuisance_values**2))
    hessian = fit.jac.T @ fit.jac
    covariance_parameters = np.linalg.pinv(hessian, rcond=1e-12)
    design_rank = int(np.linalg.matrix_rank(whitening @ varied_response))

    def at_bound(value, bound):
        if not np.isfinite(bound):
            return False
        tolerance = max(1.0e-5, 1.0e-3 * max(1.0, abs(bound)))
        return bool(abs(value - bound) <= tolerance)

    nuisance_at_bounds = {
        nuisance.name: (
            at_bound(value, nuisance.lower_bound)
            or at_bound(value, nuisance.upper_bound)
        )
        for nuisance, value in zip(nuisances, nuisance_values)
    }
    return {
        "truth": truth,
        "truth_covariance": covariance_parameters[:n_truth, :n_truth],
        "residual": residual,
        "chi2_data": chi2_data,
        "chi2_penalty": penalty,
        "chi2_total": chi2_data + penalty,
        "covariance_rank": covariance_rank,
        "design_rank": design_rank,
        "ndf": int(covariance_rank - design_rank),
        "condition": float(np.linalg.cond(whitening @ varied_response)),
        "nuisance_pulls": {
            nuisance.name: float(value)
            for nuisance, value in zip(nuisances, nuisance_values)
        },
        "nuisance_at_bounds": nuisance_at_bounds,
        "negative_response_cells": int(
            np.count_nonzero(varied_response < 0.0)
        ),
        "minimum_response_probability": float(np.min(varied_response)),
        "efficiency_range": [
            float(np.min(np.sum(varied_response, axis=0))),
            float(np.max(np.sum(varied_response, axis=0))),
        ],
        "success": bool(fit.success),
        "message": str(fit.message),
        "n_function_evaluations": int(fit.nfev),
    }


def response_mc_covariance(prepared: dict, truth: np.ndarray) -> np.ndarray:
    """Approximate independent response-cell MC statistics in reco space."""
    gen = np.asarray(prepared["gen"], dtype=float)
    coefficient = np.divide(
        truth,
        gen,
        out=np.zeros_like(truth),
        where=gen > 0.0,
    )
    variance = np.sum(
        prepared["response_w2"] * coefficient[None, :] ** 2,
        axis=1,
    )
    return np.diag(np.maximum(variance, 0.0))


def detector_nuisances(
    channel: str,
    input_dir: Path,
    nominal_prepared: dict,
    nominal_inputs: dict[str, np.ndarray],
) -> list[Nuisance]:
    """Build piecewise-linear Up/Down response nuisance templates."""
    from scripts.studies.unfold.rho_unfold_crosschecks import _prepare
    from scripts.studies.unfold.rho_unfold_systematics import load_variation

    nuisances = []
    for source in SOURCES:
        up = _prepare(
            load_variation(
                input_dir / f"{channel}_{source}Up.npz", nominal_inputs
            )[0],
            channel,
        )
        down = _prepare(
            load_variation(
                input_dir / f"{channel}_{source}Down.npz", nominal_inputs
            )[0],
            channel,
        )
        nuisances.append(
            Nuisance(
                name=source,
                response_derivative=0.5
                * (up["probability"] - down["probability"]),
                data_derivative=0.5 * (up["data"] - down["data"]),
                # The available templates only define this linear morph
                # between the quoted Down and Up variations. Extrapolating
                # farther can produce negative response probabilities.
                lower_bound=-1.0,
                upper_bound=1.0,
                response_up_shift=up["probability"]
                - nominal_prepared["probability"],
                response_down_shift=down["probability"]
                - nominal_prepared["probability"],
                data_up_shift=up["data"] - nominal_prepared["data"],
                data_down_shift=down["data"] - nominal_prepared["data"],
            )
        )
    return nuisances


def first_pt_feed_nuisance(
    nominal_inputs: dict[str, np.ndarray],
    prepared: dict,
    relative_uncertainty: float,
) -> Nuisance:
    """Coherently vary the fake/feed-in yield in the first reco pT slice."""
    n_rho_reco = len(nominal_inputs["rho_edges"]) - 1
    fake_fraction = np.divide(
        prepared["fakes"],
        prepared["reco"],
        out=np.zeros_like(prepared["fakes"]),
        where=prepared["reco"] > 0.0,
    )
    derivative = np.zeros_like(prepared["data"])
    first_pt = slice(0, n_rho_reco)
    derivative[first_pt] = (
        -relative_uncertainty
        * nominal_inputs["data_reco"][first_pt]
        * fake_fraction[first_pt]
    )
    maximum_fake_fraction = float(np.max(fake_fraction[first_pt]))
    lower_bound = -1.0 / relative_uncertainty
    upper_bound = (
        np.inf
        if maximum_fake_fraction <= 0.0
        else (1.0 / maximum_fake_fraction - 1.0)
        / relative_uncertainty
    )
    return Nuisance(
        name=f"feed_firstpt_{100.0 * relative_uncertainty:g}pct",
        response_derivative=np.zeros_like(prepared["probability"]),
        data_derivative=derivative,
        # Keep the varied fake yield and fake-subtracted data non-negative.
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def normalized_shape(values, n_pt, n_rho, in_window):
    values_2d = np.asarray(values).reshape(n_pt, n_rho)
    totals = values_2d[:, in_window].sum(axis=1, keepdims=True)
    return (values_2d / totals).reshape(-1)


def relative_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values)[mask]
    selected = selected[np.isfinite(selected)]
    return {
        "median": float(np.median(selected)),
        "maximum": float(np.max(selected)),
    }


def shape_change_records(
    relative_delta: np.ndarray,
    prepared: dict,
    limit: int = 8,
) -> list[dict]:
    n_rho = prepared["n_rho"]
    eligible = np.flatnonzero(
        prepared["shown"] & np.isfinite(relative_delta)
    )
    order = eligible[
        np.argsort(relative_delta[eligible])[::-1]
    ][:limit]
    output = []
    for flat_index in order:
        pt_index, rho_index = divmod(int(flat_index), n_rho)
        output.append(
            {
                "pt_range": [
                    float(prepared["pt_edges"][pt_index]),
                    float(prepared["pt_edges"][pt_index + 1]),
                ],
                "rho_range": [
                    float(prepared["gen_edges"][rho_index]),
                    float(prepared["gen_edges"][rho_index + 1]),
                ],
                "relative_change": float(relative_delta[flat_index]),
                "inside_normalization_window": bool(
                    prepared["core"][flat_index]
                ),
            }
        )
    return output


def residual_records(
    residual: np.ndarray,
    covariance: np.ndarray,
    prepared: dict,
    limit: int = 8,
) -> list[dict]:
    diagonal = np.diag(covariance)
    pulls = np.divide(
        residual,
        np.sqrt(np.maximum(diagonal, 0.0)),
        out=np.zeros_like(residual),
        where=diagonal > 0.0,
    )
    fractional = np.divide(
        residual,
        prepared["data"],
        out=np.zeros_like(residual),
        where=prepared["data"] != 0.0,
    )
    n_rho_reco = len(prepared["reco_edges"]) - 1
    order = np.argsort(np.abs(pulls))[::-1]
    output = []
    for flat_index in order[:limit]:
        pt_index, rho_index = divmod(int(flat_index), n_rho_reco)
        output.append(
            {
                "flat_reco_bin": int(flat_index),
                "pt_range": [
                    float(prepared["pt_edges"][pt_index]),
                    float(prepared["pt_edges"][pt_index + 1]),
                ],
                "rho_range": [
                    float(prepared["reco_edges"][rho_index]),
                    float(prepared["reco_edges"][rho_index + 1]),
                ],
                "diagonal_pull": float(pulls[flat_index]),
                "fractional_residual": float(fractional[flat_index]),
            }
        )
    return output


def serializable_fit(
    fit: dict,
    baseline_truth: np.ndarray,
    prepared: dict,
    covariance: np.ndarray,
) -> dict:
    n_pt = prepared["n_pt"]
    n_rho = prepared["n_rho"]
    baseline_shape = normalized_shape(
        baseline_truth, n_pt, n_rho, prepared["in_window"]
    )
    fitted_shape = normalized_shape(
        fit["truth"], n_pt, n_rho, prepared["in_window"]
    )
    relative_delta = np.abs(
        np.divide(
            fitted_shape,
            baseline_shape,
            out=np.full_like(fitted_shape, np.nan),
            where=baseline_shape != 0.0,
        )
        - 1.0
    )
    ndf = fit["ndf"]
    return {
        "success": fit["success"],
        "message": fit["message"],
        "chi2_data": fit["chi2_data"],
        "chi2_penalty": fit.get("chi2_penalty", 0.0),
        "chi2_total": fit["chi2_total"],
        "ndf": ndf,
        "pvalue_approx": float(
            chi2_distribution.sf(fit["chi2_total"], ndf)
        )
        if ndf > 0
        else float("nan"),
        "nuisance_pulls": fit["nuisance_pulls"],
        "nuisance_at_bounds": fit.get("nuisance_at_bounds", {}),
        "response_physicality": {
            "negative_cells": fit.get("negative_response_cells", 0),
            "minimum_probability": fit.get(
                "minimum_response_probability",
                float(np.min(prepared["probability"])),
            ),
            "efficiency_range": fit.get(
                "efficiency_range",
                [
                    float(
                        np.min(np.sum(prepared["probability"], axis=0))
                    ),
                    float(
                        np.max(np.sum(prepared["probability"], axis=0))
                    ),
                ],
            ),
        },
        "truth_shape_change": {
            "core": relative_summary(relative_delta, prepared["core"]),
            "shown": relative_summary(relative_delta, prepared["shown"]),
            "largest_shown_bins": shape_change_records(
                relative_delta, prepared
            ),
        },
        "largest_diagonal_residuals": residual_records(
            fit["residual"], covariance, prepared
        ),
    }


def analyse_channel(
    channel: str,
    nominal_path: Path,
    input_dir: Path,
    feed_relative_values: list[float],
) -> dict:
    from scripts.studies.unfold.rho_unfold_crosschecks import _prepare, _run

    nominal_inputs = load_npz(nominal_path)
    prepared = _prepare(nominal_inputs, channel)
    baseline = solve_unconstrained_gls(
        prepared["data"],
        prepared["data_covariance"],
        prepared["probability"],
    )
    tunfold_no_area = _run(
        prepared,
        prepared["data"],
        prepared["data_covariance"],
        f"_{channel}_profile_baseline",
        area=False,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_vs_tunfold = np.abs(
            baseline["truth"] / tunfold_no_area["x"] - 1.0
        )

    detector = detector_nuisances(
        channel, input_dir, prepared, nominal_inputs
    )
    detector_fit = solve_profiled(
        prepared["data"],
        prepared["data_covariance"],
        prepared["probability"],
        detector,
        baseline["truth"],
    )
    response_covariance = response_mc_covariance(
        prepared, baseline["truth"]
    )
    covariance_with_response_mc = (
        prepared["data_covariance"] + response_covariance
    )
    detector_with_response_mc = solve_profiled(
        prepared["data"],
        covariance_with_response_mc,
        prepared["probability"],
        detector,
        baseline["truth"],
    )

    feed_scans = {}
    for relative_uncertainty in feed_relative_values:
        feed = first_pt_feed_nuisance(
            nominal_inputs, prepared, relative_uncertainty
        )
        nuisances = [*detector, feed]
        data_stat_fit = solve_profiled(
            prepared["data"],
            prepared["data_covariance"],
            prepared["probability"],
            nuisances,
            baseline["truth"],
        )
        response_stat_fit = solve_profiled(
            prepared["data"],
            covariance_with_response_mc,
            prepared["probability"],
            nuisances,
            baseline["truth"],
        )
        feed_scans[f"{100.0 * relative_uncertainty:g}pct"] = {
            "data_stat_only": serializable_fit(
                data_stat_fit,
                baseline["truth"],
                prepared,
                prepared["data_covariance"],
            ),
            "with_approx_response_mc_stat": serializable_fit(
                response_stat_fit,
                baseline["truth"],
                prepared,
                covariance_with_response_mc,
            ),
        }

    return {
        "model": (
            "free coarse truth bins; Gaussian-constrained linear Up/Down "
            "response nuisances restricted to their supplied template range; "
            "combined response probabilities clamped at zero; first-pT feed-in "
            "constrained to non-negative yields; no truth prior"
        ),
        "baseline_validation": {
            "gls_chi2": baseline["chi2_data"],
            "gls_ndf": baseline["ndf"],
            "tunfold_noarea_chi2A": float(tunfold_no_area["chi2A"]),
            "tunfold_noarea_ndf": int(tunfold_no_area["ndf"]),
            "maximum_truth_delta_gls_vs_tunfold": float(
                np.nanmax(baseline_vs_tunfold)
            ),
        },
        "baseline_data_stat_only": serializable_fit(
            baseline,
            baseline["truth"],
            prepared,
            prepared["data_covariance"],
        ),
        "profile_JER_JMS_JMR_data_stat_only": serializable_fit(
            detector_fit,
            baseline["truth"],
            prepared,
            prepared["data_covariance"],
        ),
        "profile_JER_JMS_JMR_with_approx_response_mc_stat": serializable_fit(
            detector_with_response_mc,
            baseline["truth"],
            prepared,
            covariance_with_response_mc,
        ),
        "profile_with_first_pt_feed_scan": feed_scans,
        "limitations": [
            "response dependence is interpolated linearly between Up and Down",
            "detector nuisance interpolation is restricted to [-1, +1] because no "
            "physical extrapolation beyond the supplied templates is defined",
            "combined additive response morphs are clamped at zero probability",
            "response-MC covariance neglects gen-normalization correlations",
            "JES, trigger/prescale, pileup, and alternative-response nuisances absent",
            "feed-in scan varies one correlated normalization template, not its rho shape",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--nominal-dijet", required=True, type=Path)
    parser.add_argument("--nominal-trijet", required=True, type=Path)
    parser.add_argument(
        "--feed-relative",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.5, 1.0],
        help="relative 1-sigma uncertainty scans for first-pT fake/feed yield",
    )
    parser.add_argument(
        "--binning",
        choices=("coarse_tail", "2to1"),
        default="coarse_tail",
        help="gen-bin merge; coarse_tail is the production choice",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/hadronic_rho_profiled_refold.json"),
    )
    args = parser.parse_args(argv)
    if any(value <= 0.0 for value in args.feed_relative):
        parser.error("--feed-relative values must be positive")
    from scripts.studies.unfold.rho_unfold_systematics import use_binning

    use_binning(args.binning)

    result = {}
    for channel, nominal_path in (
        ("dijet", args.nominal_dijet),
        ("trijet", args.nominal_trijet),
    ):
        print(f"\n{'=' * 78}\n{channel}")
        result[channel] = analyse_channel(
            channel,
            nominal_path,
            args.input_dir,
            args.feed_relative,
        )
        baseline = result[channel]["baseline_validation"]
        profiled = result[channel][
            "profile_JER_JMS_JMR_with_approx_response_mc_stat"
        ]
        print(
            f"  baseline GLS/TUnfold chi2 = {baseline['gls_chi2']:.2f} / "
            f"{baseline['tunfold_noarea_chi2A']:.2f}"
        )
        print(
            f"  profiled JER/JMS/JMR + response MC stat: "
            f"{profiled['chi2_total']:.2f}/{profiled['ndf']}  "
            f"pulls={profiled['nuisance_pulls']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
