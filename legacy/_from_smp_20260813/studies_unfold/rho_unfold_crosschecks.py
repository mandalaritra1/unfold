#!/usr/bin/env python
"""Comprehensive numerical cross-checks for the locked hadronic rho unfolds.

This driver validates the adopted tau=0 TUnfold construction without changing
the locked reco/gen binning, nested gen merge, area-constraint production
choice, or rho-only purity criterion.  It consumes the same NPZ inputs as
``rho_unfold_stability.py`` and the seven nominal/JER/JMS/JMR files used by
``rho_unfold_systematics.py``.

The checks are deliberately separated by what they can establish:

* input/marginal/covariance invariants;
* exact nominal and same-variation self-closure;
* shifted-input through nominal-response closure;
* independent generalized-least-squares agreement at tau=0;
* full versus diagonal input covariance and area/no-area sensitivity;
* response rank, singular values, and covariance-weighted conditioning;
* normalized-covariance symmetry, PSD, and one normalization null mode per pT;
* input-stat pseudo-experiment pull width and 68% coverage;
* uncorrelated response-bin toys versus ``GetEmatrixSysUncorr``;
* smooth truth-prior stresses (a prior-independence check, not an alternate
  generator closure);
* folded-residual chi2 eigenmodes and largest diagonal pulls.

Optional alternate-generator NPZs enable a conditional-response model stress.
A true independent-MC split, the m_g > 2 GeV production, and the
dijet/trijet joint event covariance need inputs that are not present in the
nominal NPZ files; they are reported explicitly as unavailable, not silently
treated as passing.

Example
-------
source ~/Projects/unfold/scripts/setup_root.sh
python scripts/rho_unfold_crosschecks.py \
  --input-dir /path/to/syst \
  --nominal-dijet /path/to/dijet_prod_full.npz \
  --nominal-trijet /path/to/trijet_prod_full.npz \
  --toys 300 --response-toys 200 \
  --out outputs/hadronic_rho_crosschecks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.studies.unfold.rho_unfold_stability import gen_map, tunfold
from scripts.studies.unfold.rho_unfold_systematics import (
    BINNINGS,
    SOURCES,
    SPECS,
    load_variation,
    normalized_covariance,
    normalized_shape,
    use_binning,
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return {key: value for key, value in np.load(path).items()}


def _summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values, dtype=float)[mask]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(selected)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
    }


def _relative_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 0.0, a / b - 1.0, np.nan)


def _ratio_disagreement(
    ratio: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Summarize the visible departure of a normalized data/MC ratio from one."""
    selected = np.asarray(ratio, dtype=float)[mask]
    selected = selected[np.isfinite(selected)]
    delta = selected - 1.0
    return {
        "median_absolute": float(np.median(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "maximum_absolute": float(np.max(np.abs(delta))),
        "minimum_ratio": float(np.min(selected)),
        "maximum_ratio": float(np.max(selected)),
    }


def _shape_chi2(
    delta: np.ndarray, covariance: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    """Evaluate a shape-only chi2 on the nonsingular covariance subspace."""
    selected_delta = np.asarray(delta, dtype=float)[mask]
    selected_covariance = np.asarray(covariance, dtype=float)[
        np.ix_(mask, mask)
    ]
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (selected_covariance + selected_covariance.T)
    )
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    rank = int(np.sum(eigenvalues > 1e-12 * scale))
    chi2 = selected_delta @ np.linalg.pinv(
        selected_covariance, rcond=1e-12
    ) @ selected_delta
    return {"chi2": float(chi2), "rank": rank}


def _eigh_psd(matrix: np.ndarray, rtol: float = 1e-12):
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    keep = eigenvalues > rtol * scale
    return eigenvalues, eigenvectors, keep


def _draw_gaussian(
    rng: np.random.Generator, mean: np.ndarray, covariance: np.ndarray, n: int
) -> np.ndarray:
    eigenvalues, eigenvectors, keep = _eigh_psd(covariance)
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    return mean[None, :] + rng.normal(size=(n, int(keep.sum()))) @ factor.T


def _prepare(inputs: dict[str, np.ndarray], channel: str):
    groups = SPECS[channel]["rho_groups"]
    n_pt = len(inputs["pt_edges"]) - 1
    n_rho_reco = len(inputs["rho_edges"]) - 1
    mapping = gen_map(
        n_pt,
        n_rho_reco,
        [[index] for index in range(n_pt)],
        groups,
    )
    response = inputs["A"] @ mapping
    response_w2 = inputs["A_w2"] @ mapping
    gen = inputs["gen"] @ mapping
    gen_w2 = inputs["gen_w2"] @ mapping
    reco = inputs["reco"]
    fakes = reco - response.sum(axis=1)
    misses = gen - response.sum(axis=0)
    misses_w2 = np.maximum(gen_w2 - response_w2.sum(axis=0), 0.0)
    fake_w2 = np.maximum(inputs["reco_w2"] - response_w2.sum(axis=1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fake_fraction = np.where(reco > 0.0, fakes / reco, 0.0)
    survival = 1.0 - np.clip(fake_fraction, 0.0, 1.0)
    data = inputs["data_reco"] * survival
    data_covariance = inputs["data_V"] * np.outer(survival, survival)

    gen_edges = np.r_[
        [inputs["rho_edges"][group[0]] for group in groups],
        inputs["rho_edges"][-1],
    ]
    n_rho_gen = len(groups)
    norm = SPECS[channel]["norm"]
    in_window = (
        (gen_edges[:-1] >= norm[0] - 1e-9)
        & (gen_edges[1:] <= norm[1] + 1e-9)
    )
    shown = (
        np.tile(gen_edges[:-1] >= -4.0, n_pt)
        & (gen > 0.0)
        & (response.sum(axis=0) > 0.0)
    )
    core = shown & np.tile(in_window, n_pt)
    denominator = response.sum(axis=0) + misses
    probability = np.divide(
        response,
        denominator[None, :],
        out=np.zeros_like(response),
        where=denominator[None, :] > 0.0,
    )
    return {
        "groups": groups,
        "n_pt": n_pt,
        "n_rho": n_rho_gen,
        "response": response,
        "response_w2": response_w2,
        "gen": gen,
        "gen_w2": gen_w2,
        "reco": reco,
        "fakes": fakes,
        "fake_w2": fake_w2,
        "misses": misses,
        "misses_w2": misses_w2,
        "survival": survival,
        "data": data,
        "data_covariance": data_covariance,
        "gen_edges": gen_edges,
        "pt_edges": np.asarray(inputs["pt_edges"], dtype=float),
        "reco_edges": np.asarray(inputs["rho_edges"], dtype=float),
        "in_window": in_window,
        "shown": shown,
        "core": core,
        "probability": probability,
    }


def _run(prepared, y, covariance, tag, *, area=True, response=None, response_w2=None,
         misses=None, misses_w2=None):
    return tunfold(
        prepared["response"] if response is None else response,
        prepared["response_w2"] if response_w2 is None else response_w2,
        prepared["misses"] if misses is None else misses,
        prepared["misses_w2"] if misses_w2 is None else misses_w2,
        y,
        covariance,
        tag,
        area_constraint=area,
        quiet=True,
        include_rho=False,
    )


def _covariance_diagnostics(covariance: np.ndarray) -> dict[str, float | int]:
    eigenvalues, _, keep = _eigh_psd(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    return {
        "max_asymmetry": float(np.max(np.abs(covariance - covariance.T))),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "min_eigenvalue_relative": float(np.min(eigenvalues) / scale),
        "rank": int(keep.sum()),
        "dimension": int(covariance.shape[0]),
    }


def _conditioning(prepared) -> dict:
    probability = prepared["probability"]
    singular = np.linalg.svd(probability, compute_uv=False)
    eigenvalues, eigenvectors, keep = _eigh_psd(prepared["data_covariance"])
    whitener = (
        eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    ).T
    weighted = whitener @ probability
    weighted_singular = np.linalg.svd(weighted, compute_uv=False)

    def details(values):
        threshold = max(values[0], 1.0) * 1e-12
        kept = values > threshold
        return {
            "rank": int(kept.sum()),
            "n_columns": int(probability.shape[1]),
            "largest": float(values[0]),
            "smallest_resolved": float(values[kept][-1]),
            "condition": float(values[0] / values[kept][-1]),
            "singular_values": values.tolist(),
        }

    return {"plain": details(singular), "data_weighted": details(weighted_singular)}


def _gls(probability: np.ndarray, y: np.ndarray, covariance: np.ndarray):
    eigenvalues, eigenvectors, keep = _eigh_psd(covariance)
    inverse = (
        eigenvectors[:, keep] / eigenvalues[keep]
    ) @ eigenvectors[:, keep].T
    fisher = probability.T @ inverse @ probability
    fisher_inverse = np.linalg.pinv(fisher, rcond=1e-12)
    return fisher_inverse @ probability.T @ inverse @ y, fisher_inverse


def _normalized_covariance_checks(prepared, nominal) -> dict:
    covariance = normalized_covariance(
        nominal["x"],
        nominal["Ein"],
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )
    diagnostics = _covariance_diagnostics(covariance)
    null_residuals = []
    for pt_index in range(prepared["n_pt"]):
        selector = np.zeros(nominal["x"].size)
        offset = pt_index * prepared["n_rho"]
        selector[offset : offset + prepared["n_rho"]] = prepared["in_window"]
        residual = selector @ covariance
        scale = max(float(np.max(np.abs(covariance))), 1.0)
        null_residuals.append(float(np.max(np.abs(residual)) / scale))
    diagnostics["normalization_null_residuals"] = null_residuals
    diagnostics["expected_rank_ceiling"] = int(
        covariance.shape[0] - prepared["n_pt"]
    )
    return diagnostics


def _input_toys(prepared, nominal_mc, n_toys, rng, channel):
    covariance = prepared["data_covariance"]
    mean = prepared["response"].sum(axis=1)
    toys = _draw_gaussian(rng, mean, covariance, n_toys)
    unfolded = np.empty((n_toys, prepared["gen"].size))
    for index, toy in enumerate(toys):
        unfolded[index] = _run(
            prepared,
            toy,
            covariance,
            f"_{channel}_inputtoy_{index}",
        )["x"]
    sigma = np.sqrt(np.maximum(np.diag(nominal_mc["Ein"]), 0.0))
    empirical_sigma = np.std(unfolded, axis=0, ddof=1)
    pull_mean = np.divide(
        np.mean(unfolded, axis=0) - prepared["gen"],
        sigma,
        out=np.full_like(sigma, np.nan),
        where=sigma > 0.0,
    )
    width_ratio = np.divide(
        empirical_sigma,
        sigma,
        out=np.full_like(sigma, np.nan),
        where=sigma > 0.0,
    )
    coverage = np.mean(
        np.abs(unfolded - prepared["gen"][None, :]) <= sigma[None, :],
        axis=0,
    )
    return {
        "n_toys": n_toys,
        "pull_mean": _summary(pull_mean, prepared["shown"]),
        "empirical_over_analytic_sigma": _summary(
            width_ratio, prepared["shown"]
        ),
        "coverage_1sigma": _summary(coverage, prepared["shown"]),
    }


def _response_toys(prepared, nominal_data, n_toys, rng, channel):
    unfolded = np.empty((n_toys, prepared["gen"].size))
    response_sigma = np.sqrt(np.maximum(prepared["response_w2"], 0.0))
    misses_sigma = np.sqrt(np.maximum(prepared["misses_w2"], 0.0))
    for index in range(n_toys):
        response = np.maximum(
            prepared["response"] + rng.normal(size=response_sigma.shape) * response_sigma,
            0.0,
        )
        misses = np.maximum(
            prepared["misses"] + rng.normal(size=misses_sigma.shape) * misses_sigma,
            0.0,
        )
        unfolded[index] = _run(
            prepared,
            prepared["data"],
            prepared["data_covariance"],
            f"_{channel}_resptoy_{index}",
            response=response,
            misses=misses,
        )["x"]
    empirical_sigma = np.std(unfolded, axis=0, ddof=1)
    analytic_sigma = np.sqrt(np.maximum(np.diag(nominal_data["Esys"]), 0.0))
    ratio = np.divide(
        empirical_sigma,
        analytic_sigma,
        out=np.full_like(analytic_sigma, np.nan),
        where=analytic_sigma > 0.0,
    )
    bias = np.divide(
        np.mean(unfolded, axis=0) - nominal_data["x"],
        analytic_sigma,
        out=np.full_like(analytic_sigma, np.nan),
        where=analytic_sigma > 0.0,
    )
    return {
        "n_toys": n_toys,
        "toy_model": "independent Gaussian response cells and miss bins",
        "empirical_over_sysuncorr_sigma": _summary(ratio, prepared["shown"]),
        "mean_bias_in_sysuncorr_sigma": _summary(bias, prepared["shown"]),
    }


def _prior_stresses(prepared, channel):
    centers = 0.5 * (prepared["gen_edges"][:-1] + prepared["gen_edges"][1:])
    center = float(np.mean(centers[prepared["in_window"]]))
    span = max(float(np.ptp(centers[prepared["in_window"]])), 1.0)
    local = (centers - center) / span
    patterns = {
        "rho_tilt_up": np.tile(
            np.clip(np.exp(0.6 * local), 0.5, 1.5), prepared["n_pt"]
        ),
        "rho_tilt_down": np.tile(
            np.clip(np.exp(-0.6 * local), 0.5, 1.5), prepared["n_pt"]
        ),
        "alternating_20pct": np.tile(
            1.0 + 0.2 * np.where(np.arange(prepared["n_rho"]) % 2, 1.0, -1.0),
            prepared["n_pt"],
        ),
    }
    output = {}
    tiny_covariance = np.diag(
        np.maximum(prepared["response"].sum(axis=1), 1.0)
    )
    for name, weights in patterns.items():
        truth = prepared["gen"] * weights
        pseudo_data = prepared["response"] @ weights
        result = _run(
            prepared,
            pseudo_data,
            tiny_covariance,
            f"_{channel}_prior_{name}",
        )
        deviation = np.abs(_relative_delta(result["x"], truth))
        output[name] = {
            "weight_range": [float(np.min(weights)), float(np.max(weights))],
            "max_abs_closure": float(np.nanmax(deviation[prepared["shown"]])),
        }
    return output


def _iterative_bayes_crosscheck(prepared, tunfold_shape):
    """Unregularized iterative-Bayes limit, with no efficiency flooring.

    Finite iteration counts are included only to show their regularization
    dependence.  The high-iteration result is the independent method check.
    """
    probability = prepared["probability"]
    efficiency = probability.sum(axis=0)
    estimate = prepared["gen"].copy()
    output = {}
    for iteration in range(1, 1001):
        folded = probability @ estimate
        estimate = np.divide(
            estimate,
            efficiency,
            out=np.zeros_like(estimate),
            where=efficiency > 0.0,
        ) * (
            probability.T
            @ np.divide(
                prepared["data"],
                folded,
                out=np.zeros_like(folded),
                where=folded > 0.0,
            )
        )
        if iteration in (4, 16, 64, 1000):
            shape = normalized_shape(
                estimate,
                prepared["n_pt"],
                prepared["n_rho"],
                prepared["in_window"],
            )
            delta = np.abs(_relative_delta(shape, tunfold_shape))
            output[str(iteration)] = {
                "shape_delta_vs_tunfold": {
                    "core": _summary(delta, prepared["core"]),
                    "shown": _summary(delta, prepared["shown"]),
                },
                "negative_shown_bins": int(
                    np.sum(estimate[prepared["shown"]] < 0.0)
                ),
            }
    return {
        "efficiency_range": [
            float(np.min(efficiency)),
            float(np.max(efficiency)),
        ],
        "no_efficiency_floor": True,
        "iterations": output,
    }


def _systematic_closure(
    prepared,
    nominal_inputs,
    input_dir: Path,
    channel: str,
):
    output = {}
    for source in SOURCES:
        for direction in ("Up", "Down"):
            name = f"{source}{direction}"
            varied_inputs, _ = load_variation(
                input_dir / f"{channel}_{name}.npz", nominal_inputs
            )
            varied = _prepare(varied_inputs, channel)
            gen_delta = np.max(
                np.abs(_relative_delta(varied["gen"], prepared["gen"]))
            )
            same_response = _run(
                varied,
                varied["response"].sum(axis=1),
                np.diag(np.maximum(varied["response"].sum(axis=1), 1.0)),
                f"_{channel}_{name}_self",
            )
            self_deviation = np.abs(
                _relative_delta(same_response["x"], varied["gen"])
            )
            shifted_input = _run(
                prepared,
                varied["response"].sum(axis=1),
                np.diag(np.maximum(varied["response"].sum(axis=1), 1.0)),
                f"_{channel}_{name}_through_nominal",
            )
            nonclosure = np.abs(
                _relative_delta(shifted_input["x"], prepared["gen"])
            )
            #### The propagated systematic band lives in normalized-shape
            #### space, so the induced bias has to be measured there too;
            #### the absolute non-closure above still carries the overall
            #### normalization change of the variation.
            shape_nonclosure = np.abs(
                _relative_delta(
                    normalized_shape(
                        shifted_input["x"],
                        prepared["n_pt"],
                        prepared["n_rho"],
                        prepared["in_window"],
                    ),
                    normalized_shape(
                        prepared["gen"],
                        prepared["n_pt"],
                        prepared["n_rho"],
                        prepared["in_window"],
                    ),
                )
            )
            output[name] = {
                "gen_max_relative_change": float(gen_delta),
                "same_variation_self_closure_max": float(
                    np.nanmax(self_deviation[varied["shown"]])
                ),
                "varied_input_nominal_response": {
                    "core": _summary(nonclosure, prepared["core"]),
                    "shown": _summary(nonclosure, prepared["shown"]),
                },
                "varied_input_nominal_response_shape": {
                    "core": _summary(shape_nonclosure, prepared["core"]),
                    "shown": _summary(shape_nonclosure, prepared["shown"]),
                },
            }
    return output


def _alternate_generator_closure(
    prepared,
    alternate_inputs: dict[str, np.ndarray] | None,
    channel: str,
):
    """Unfold an alternate sample with the nominal conditional response.

    The nominal fake survival correction is intentional: this is the complete
    response-model stress that would be applied to data, rather than a
    same-generator self-closure with the alternate fake fraction.
    """
    if alternate_inputs is None:
        return {"available": False}

    alternate = _prepare(alternate_inputs, channel)
    raw_reco = (
        alternate_inputs["data_reco"]
        if np.any(alternate_inputs["data_reco"])
        else alternate_inputs["reco"]
    )
    raw_covariance = (
        alternate_inputs["data_V"]
        if np.any(alternate_inputs["data_V"])
        else np.diag(alternate_inputs["reco_w2"])
    )
    survival = prepared["survival"]
    result = _run(
        prepared,
        raw_reco * survival,
        raw_covariance * np.outer(survival, survival),
        f"_{channel}_alternate_through_nominal",
    )
    unfolded_shape = normalized_shape(
        result["x"],
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )
    truth_shape = normalized_shape(
        alternate["gen"],
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )
    delta = np.abs(_relative_delta(unfolded_shape, truth_shape))
    return {
        "available": True,
        "nominal_fake_survival_applied": True,
        "alternate_fake_fraction": float(
            np.sum(alternate["fakes"]) / np.sum(alternate["reco"])
        ),
        "shape_nonclosure": {
            "core": _summary(delta, prepared["core"]),
            "shown": _summary(delta, prepared["shown"]),
        },
        "negative_shown_bins": int(
            np.sum(result["x"][prepared["shown"]] < 0.0)
        ),
    }


def _chi2_modes(prepared, nominal_data) -> dict:
    residual = prepared["data"] - prepared["probability"] @ nominal_data["x"]
    covariance = prepared["data_covariance"]
    eigenvalues, eigenvectors, keep = _eigh_psd(covariance)
    projections = eigenvectors[:, keep].T @ residual
    contributions = projections**2 / eigenvalues[keep]
    order = np.argsort(contributions)[::-1]
    diagonal_pulls = np.divide(
        residual,
        np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        out=np.zeros_like(residual),
        where=np.diag(covariance) > 0.0,
    )
    pull_order = np.argsort(np.abs(diagonal_pulls))[::-1]
    n_rho_reco = len(prepared["reco_edges"]) - 1

    def reco_bin_record(bin_index):
        pt_index, rho_index = divmod(int(bin_index), n_rho_reco)
        return {
            "flat_reco_bin": int(bin_index),
            "pt_range": [
                float(prepared["pt_edges"][pt_index]),
                float(prepared["pt_edges"][pt_index + 1]),
            ],
            "rho_range": [
                float(prepared["reco_edges"][rho_index]),
                float(prepared["reco_edges"][rho_index + 1]),
            ],
            "pull": float(diagonal_pulls[bin_index]),
        }

    return {
        "manual_chi2": float(np.sum(contributions)),
        "tunfold_chi2A": float(nominal_data["chi2A"]),
        "ndf": int(nominal_data["ndf"]),
        "top_eigenmode_contributions": [
            {
                "rank": int(index + 1),
                "chi2": float(contributions[mode]),
                "fraction": float(contributions[mode] / np.sum(contributions)),
            }
            for index, mode in enumerate(order[:5])
        ],
        "largest_diagonal_pulls": [
            reco_bin_record(bin_index) for bin_index in pull_order[:8]
        ],
    }


def _data_mc_before_after(
    channel: str,
    prepared: dict,
    nominal_inputs: dict[str, np.ndarray],
    nominal_data: dict,
) -> dict:
    """Compare normalized data/MC disagreement before and after unfolding.

    The reco spectra are first merged onto the adopted gen-bin map, so both
    sides use identical reported bins and per-pT normalization windows.
    """
    n_pt = prepared["n_pt"]
    n_rho_reco = len(nominal_inputs["rho_edges"]) - 1
    mapping = gen_map(
        n_pt,
        n_rho_reco,
        [[index] for index in range(n_pt)],
        SPECS[channel]["rho_groups"],
    )

    reco_data = mapping.T @ prepared["data"]
    reco_mc = mapping.T @ prepared["response"].sum(axis=1)
    reco_covariance = (
        mapping.T @ prepared["data_covariance"] @ mapping
    )

    before_data_shape = normalized_shape(
        reco_data, n_pt, prepared["n_rho"], prepared["in_window"]
    )
    before_mc_shape = normalized_shape(
        reco_mc, n_pt, prepared["n_rho"], prepared["in_window"]
    )
    after_data_shape = normalized_shape(
        nominal_data["x"], n_pt, prepared["n_rho"], prepared["in_window"]
    )
    after_mc_shape = normalized_shape(
        prepared["gen"], n_pt, prepared["n_rho"], prepared["in_window"]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        before_ratio = np.where(
            before_mc_shape > 0.0,
            before_data_shape / before_mc_shape,
            np.nan,
        )
        after_ratio = np.where(
            after_mc_shape > 0.0,
            after_data_shape / after_mc_shape,
            np.nan,
        )

    before_shape_covariance = normalized_covariance(
        reco_data,
        reco_covariance,
        n_pt,
        prepared["n_rho"],
        prepared["in_window"],
    )
    after_shape_covariance = normalized_covariance(
        nominal_data["x"],
        nominal_data["Ein"],
        n_pt,
        prepared["n_rho"],
        prepared["in_window"],
    )

    core = prepared["core"]
    shown = prepared["shown"]
    before_absolute = np.abs(before_ratio[core] - 1.0)
    after_absolute = np.abs(after_ratio[core] - 1.0)
    return {
        "test_type": (
            "visible ratio-distance diagnostic; not the formal raw-chi2 "
            "bottom-line inequality"
        ),
        "definition": (
            "same adopted reported-bin map and per-pT normalization window; "
            "before = fake-corrected data / matched reco MC; "
            "after = unfolded data / particle-level MC"
        ),
        "before_unfolding": {
            "core": _ratio_disagreement(before_ratio, core),
            "shown": _ratio_disagreement(before_ratio, shown),
            "data_stat_shape_test_core": _shape_chi2(
                before_data_shape - before_mc_shape,
                before_shape_covariance,
                core,
            ),
        },
        "after_unfolding": {
            "core": _ratio_disagreement(after_ratio, core),
            "shown": _ratio_disagreement(after_ratio, shown),
            "data_stat_shape_test_core": _shape_chi2(
                after_data_shape - after_mc_shape,
                after_shape_covariance,
                core,
            ),
        },
        "core_bins_with_larger_absolute_disagreement": int(
            np.sum(after_absolute > before_absolute)
        ),
        "n_core_bins": int(core.sum()),
        "caveat": (
            "shape chi2 values use data statistical covariance only; "
            "MC statistics and detector/model systematic covariances are omitted"
        ),
    }


def _formal_bottom_line_test(prepared: dict, nominal_data: dict) -> dict:
    """Evaluate the data-processing inequality for the tau=0 unfolding.

    The formal TUnfold bottom-line test compares raw chi2 values, not chi2 per
    degree of freedom or visible ratio excursions. The unfolded statistic must
    not exceed the detector-level statistic when the covariance is propagated
    consistently.
    """
    reco_residual = (
        prepared["data"] - prepared["response"].sum(axis=1)
    )
    truth_residual = nominal_data["x"] - prepared["gen"]
    smeared = _shape_chi2(
        reco_residual,
        prepared["data_covariance"],
        np.ones(reco_residual.size, dtype=bool),
    )
    unfolded_input_stat = _shape_chi2(
        truth_residual,
        nominal_data["Ein"],
        np.ones(truth_residual.size, dtype=bool),
    )
    unfolded_with_response_stat = _shape_chi2(
        truth_residual,
        nominal_data["Ein"] + nominal_data["Esys"],
        np.ones(truth_residual.size, dtype=bool),
    )
    return {
        "definition": (
            "raw chi2_unfolded <= raw chi2_smeared with consistently "
            "propagated covariance; compare raw chi2, not chi2/rank"
        ),
        "smeared_data_stat": smeared,
        "unfolded_data_stat": unfolded_input_stat,
        "unfolded_input_plus_response_mc_stat": (
            unfolded_with_response_stat
        ),
        "data_stat_raw_chi2_difference_unfolded_minus_smeared": float(
            unfolded_input_stat["chi2"] - smeared["chi2"]
        ),
        "data_stat_inequality_holds": bool(
            unfolded_input_stat["chi2"] <= smeared["chi2"] + 1e-9
        ),
        "with_response_mc_stat_inequality_holds": bool(
            unfolded_with_response_stat["chi2"]
            <= smeared["chi2"] + 1e-9
        ),
    }


def analyse_channel(
    channel: str,
    nominal_path: Path,
    input_dir: Path,
    n_toys: int,
    n_response_toys: int,
    seed: int,
    alternate_path: Path | None = None,
):
    nominal_inputs = load_npz(nominal_path)
    alternate_inputs = (
        load_npz(alternate_path) if alternate_path is not None else None
    )
    prepared = _prepare(nominal_inputs, channel)
    rng = np.random.default_rng(seed)

    matched = prepared["response"].sum(axis=1)
    nominal_mc = _run(
        prepared,
        matched,
        prepared["data_covariance"],
        f"_{channel}_mcclosure",
    )
    nominal_data = _run(
        prepared,
        prepared["data"],
        prepared["data_covariance"],
        f"_{channel}_data",
    )
    no_area = _run(
        prepared,
        prepared["data"],
        prepared["data_covariance"],
        f"_{channel}_noarea",
        area=False,
    )
    diagonal_covariance = np.diag(np.diag(prepared["data_covariance"]))
    diagonal_result = _run(
        prepared,
        prepared["data"],
        diagonal_covariance,
        f"_{channel}_diagcov",
    )

    shape = normalized_shape(
        nominal_data["x"],
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )
    no_area_shape = normalized_shape(
        no_area["x"],
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )
    diagonal_shape = normalized_shape(
        diagonal_result["x"],
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )
    gls_x, gls_covariance = _gls(
        prepared["probability"],
        prepared["data"],
        prepared["data_covariance"],
    )
    gls_shape = normalized_shape(
        gls_x,
        prepared["n_pt"],
        prepared["n_rho"],
        prepared["in_window"],
    )

    self_deviation = np.abs(
        _relative_delta(nominal_mc["x"], prepared["gen"])
    )
    fakes = prepared["fakes"]
    misses = prepared["misses"]
    input_contract = {
        "min_fake_yield": float(np.min(fakes)),
        "min_miss_yield": float(np.min(misses)),
        "negative_fake_bins": int(np.sum(fakes < -1e-8)),
        "negative_miss_bins": int(np.sum(misses < -1e-8)),
        "fake_fraction": float(np.sum(fakes) / np.sum(prepared["reco"])),
        "miss_fraction": float(np.sum(misses) / np.sum(prepared["gen"])),
        "data_covariance": _covariance_diagnostics(
            prepared["data_covariance"]
        ),
        "response_variance_nonnegative": bool(
            np.all(prepared["response_w2"] >= 0.0)
        ),
        "miss_variance_nonnegative": bool(
            np.all(prepared["misses_w2"] >= 0.0)
        ),
    }

    area_delta = np.abs(_relative_delta(no_area_shape, shape))
    diagonal_delta = np.abs(_relative_delta(diagonal_shape, shape))
    gls_delta = np.abs(_relative_delta(gls_shape, no_area_shape))
    no_area_cov_delta = np.abs(
        _relative_delta(np.diag(gls_covariance), np.diag(no_area["Ein"]))
    )

    return {
        "input_contract": input_contract,
        "conditioning": _conditioning(prepared),
        "nominal": {
            "n_reco": int(prepared["response"].shape[0]),
            "n_gen": int(prepared["response"].shape[1]),
            "n_shown": int(prepared["shown"].sum()),
            "n_core": int(prepared["core"].sum()),
            "self_closure_max": float(
                np.nanmax(self_deviation[prepared["shown"]])
            ),
            "chi2A": float(nominal_data["chi2A"]),
            "ndf": int(nominal_data["ndf"]),
            "negative_unfolded_bins": int(
                np.sum(nominal_data["x"][prepared["shown"]] < 0.0)
            ),
        },
        "independent_gls_crosscheck": {
            "shape_delta_vs_tunfold_noarea": {
                "core": _summary(gls_delta, prepared["core"]),
                "shown": _summary(gls_delta, prepared["shown"]),
            },
            "absolute_covariance_diagonal_delta": {
                "core": _summary(no_area_cov_delta, prepared["core"]),
                "shown": _summary(no_area_cov_delta, prepared["shown"]),
            },
        },
        "area_constraint_sensitivity": {
            "shape_delta": {
                "core": _summary(area_delta, prepared["core"]),
                "shown": _summary(area_delta, prepared["shown"]),
            },
            "chi2A_noarea": float(no_area["chi2A"]),
            "ndf_noarea": int(no_area["ndf"]),
        },
        "full_vs_diagonal_data_covariance": {
            "shape_delta": {
                "core": _summary(diagonal_delta, prepared["core"]),
                "shown": _summary(diagonal_delta, prepared["shown"]),
            },
            "chi2A_diagonal": float(diagonal_result["chi2A"]),
            "ndf_diagonal": int(diagonal_result["ndf"]),
        },
        "normalized_covariance": _normalized_covariance_checks(
            prepared, nominal_data
        ),
        "input_stat_toys": _input_toys(
            prepared, nominal_mc, n_toys, rng, channel
        ),
        "response_stat_toys": _response_toys(
            prepared, nominal_data, n_response_toys, rng, channel
        ),
        "prior_stress": _prior_stresses(prepared, channel),
        "iterative_bayes_crosscheck": _iterative_bayes_crosscheck(
            prepared, shape
        ),
        "systematic_closure": _systematic_closure(
            prepared, nominal_inputs, input_dir, channel
        ),
        "alternate_generator_closure": _alternate_generator_closure(
            prepared, alternate_inputs, channel
        ),
        "folded_residual": _chi2_modes(prepared, nominal_data),
        "data_mc_before_after_unfolding": _data_mc_before_after(
            channel, prepared, nominal_inputs, nominal_data
        ),
        "formal_bottom_line_test": _formal_bottom_line_test(
            prepared, nominal_data
        ),
        "not_executable_from_current_npz": [
            "statistically independent MC half-response closure",
            "m_g > 2 GeV floor-enabled full-stat production",
            "joint dijet-trijet data and response covariance",
        ] + ([] if alternate_inputs is not None else [
            "alternative-generator closure / conditional-response model bias",
        ]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--nominal-dijet", required=True, type=Path)
    parser.add_argument("--nominal-trijet", required=True, type=Path)
    parser.add_argument("--alternate-dijet", type=Path)
    parser.add_argument("--alternate-trijet", type=Path)
    parser.add_argument("--toys", type=int, default=300)
    parser.add_argument("--response-toys", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--binning",
        choices=tuple(BINNINGS),
        default="coarse_tail",
        help="gen-bin merge; coarse_tail is the production choice",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.toys < 2 or args.response_toys < 2:
        parser.error("toy counts must both be at least two")
    use_binning(args.binning)

    result = {}
    for channel, path, alternate_path in (
        ("dijet", args.nominal_dijet, args.alternate_dijet),
        ("trijet", args.nominal_trijet, args.alternate_trijet),
    ):
        print(f"\n{'=' * 78}\n{channel}")
        result[channel] = analyse_channel(
            channel,
            path,
            args.input_dir,
            args.toys,
            args.response_toys,
            args.seed + (0 if channel == "dijet" else 1),
            alternate_path,
        )
        nominal = result[channel]["nominal"]
        toys = result[channel]["input_stat_toys"]
        print(
            f"  closure {nominal['self_closure_max']:.3e}; "
            f"chi2A/ndf {nominal['chi2A']:.2f}/{nominal['ndf']}\n"
            f"  input-toy sigma ratio "
            f"{toys['empirical_over_analytic_sigma']['median']:.3f} median; "
            f"coverage {toys['coverage_1sigma']['median']:.3f} median"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
