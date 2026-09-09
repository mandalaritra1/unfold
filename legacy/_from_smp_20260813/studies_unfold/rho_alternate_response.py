#!/usr/bin/env python
"""Compare nominal and alternate-generator responses on the same rho data.

This separates three quantities that should not be conflated:

* the particle-level generator difference;
* conditional detector-response differences at fixed truth bin;
* the change in unfolded data when the complete alternate response, fake
  correction, and miss efficiency are used.

Both unfolds use the locked tau=0 binning and area constraint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.studies.unfold.rho_unfold_crosschecks import _prepare, _run, load_npz
from scripts.studies.unfold.rho_unfold_systematics import (
    normalized_covariance,
    normalized_shape,
)


def _summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values, dtype=float)[mask]
    selected = selected[np.isfinite(selected)]
    return {
        "median": float(np.median(selected)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
    }


def _relative_change(
    varied: np.ndarray, nominal: np.ndarray
) -> np.ndarray:
    return np.abs(
        np.divide(
            varied,
            nominal,
            out=np.full_like(varied, np.nan),
            where=nominal != 0.0,
        )
        - 1.0
    )


def _bin_records(
    values: np.ndarray,
    prepared: dict,
    mask: np.ndarray,
    limit: int = 10,
) -> list[dict]:
    eligible = np.flatnonzero(mask & np.isfinite(values))
    order = eligible[np.argsort(values[eligible])[::-1]][:limit]
    records = []
    for flat_index in order:
        pt_index, rho_index = divmod(
            int(flat_index), prepared["n_rho"]
        )
        records.append(
            {
                "pt_range": [
                    float(prepared["pt_edges"][pt_index]),
                    float(prepared["pt_edges"][pt_index + 1]),
                ],
                "rho_range": [
                    float(prepared["gen_edges"][rho_index]),
                    float(prepared["gen_edges"][rho_index + 1]),
                ],
                "absolute_relative_change": float(values[flat_index]),
                "inside_normalization_window": bool(
                    prepared["core"][flat_index]
                ),
            }
        )
    return records


def _ratio_records(
    varied: np.ndarray,
    nominal: np.ndarray,
    prepared: dict,
) -> list[dict]:
    ratio = np.divide(
        varied,
        nominal,
        out=np.full_like(varied, np.nan),
        where=nominal != 0.0,
    )
    records = []
    for flat_index in np.flatnonzero(
        prepared["shown"] & np.isfinite(ratio)
    ):
        pt_index, rho_index = divmod(
            int(flat_index), prepared["n_rho"]
        )
        records.append(
            {
                "pt_range": [
                    float(prepared["pt_edges"][pt_index]),
                    float(prepared["pt_edges"][pt_index + 1]),
                ],
                "rho_range": [
                    float(prepared["gen_edges"][rho_index]),
                    float(prepared["gen_edges"][rho_index + 1]),
                ],
                "alternate_over_nominal": float(ratio[flat_index]),
                "inside_normalization_window": bool(
                    prepared["core"][flat_index]
                ),
            }
        )
    return records


def _model_data(
    raw_data: np.ndarray,
    raw_covariance: np.ndarray,
    prepared: dict,
) -> tuple[np.ndarray, np.ndarray]:
    survival = prepared["survival"]
    return (
        raw_data * survival,
        raw_covariance * np.outer(survival, survival),
    )


def _fake_fraction_by_pt(prepared: dict) -> list[float]:
    n_pt = prepared["n_pt"]
    n_rho_reco = len(prepared["reco_edges"]) - 1
    reco = prepared["reco"].reshape(n_pt, n_rho_reco).sum(axis=1)
    fakes = prepared["fakes"].reshape(n_pt, n_rho_reco).sum(axis=1)
    return np.divide(
        fakes,
        reco,
        out=np.zeros_like(fakes),
        where=reco != 0.0,
    ).tolist()


def _miss_fraction_by_pt(prepared: dict) -> list[float]:
    gen = prepared["gen"].reshape(
        prepared["n_pt"], prepared["n_rho"]
    ).sum(axis=1)
    misses = prepared["misses"].reshape(
        prepared["n_pt"], prepared["n_rho"]
    ).sum(axis=1)
    return np.divide(
        misses,
        gen,
        out=np.zeros_like(misses),
        where=gen != 0.0,
    ).tolist()


def analyse(channel: str, nominal_path: Path, alternate_path: Path) -> dict:
    nominal_inputs = load_npz(nominal_path)
    alternate_inputs = load_npz(alternate_path)
    nominal = _prepare(nominal_inputs, channel)
    alternate = _prepare(alternate_inputs, channel)

    if nominal["probability"].shape != alternate["probability"].shape:
        raise ValueError("Nominal and alternate response shapes differ")
    raw_data = np.asarray(nominal_inputs["data_reco"], dtype=float)
    raw_covariance = np.asarray(nominal_inputs["data_V"], dtype=float)
    if not np.any(raw_data):
        raise ValueError("Nominal input does not contain data")

    nominal_data, nominal_covariance = _model_data(
        raw_data, raw_covariance, nominal
    )
    alternate_data, alternate_covariance = _model_data(
        raw_data, raw_covariance, alternate
    )
    nominal_result = _run(
        nominal,
        nominal_data,
        nominal_covariance,
        f"_{channel}_nominal_response_data",
    )
    alternate_result = _run(
        alternate,
        alternate_data,
        alternate_covariance,
        f"_{channel}_alternate_response_data",
    )

    def run_probability(
        probability: np.ndarray,
        data: np.ndarray,
        covariance: np.ndarray,
        tag: str,
    ):
        # TUnfold depends on the response through the column-normalized
        # probability and efficiency. Use the nominal truth counts as an
        # arbitrary common column normalization for hybrid component tests.
        column_normalization = nominal["gen"]
        response = probability * column_normalization[None, :]
        misses = (
            1.0 - probability.sum(axis=0)
        ) * column_normalization
        return _run(
            nominal,
            data,
            covariance,
            tag,
            response=response,
            response_w2=np.zeros_like(response),
            misses=misses,
            misses_w2=np.zeros_like(misses),
        )

    nominal_probability = nominal["probability"]
    alternate_probability = alternate["probability"]
    nominal_efficiency = nominal_probability.sum(axis=0)
    alternate_efficiency = alternate_probability.sum(axis=0)
    nominal_migration = np.divide(
        nominal_probability,
        nominal_efficiency[None, :],
        out=np.zeros_like(nominal_probability),
        where=nominal_efficiency[None, :] > 0.0,
    )
    alternate_migration = np.divide(
        alternate_probability,
        alternate_efficiency[None, :],
        out=np.zeros_like(alternate_probability),
        where=alternate_efficiency[None, :] > 0.0,
    )
    component_results = {
        "alternate_fakes_only": run_probability(
            nominal_probability,
            alternate_data,
            alternate_covariance,
            f"_{channel}_alternate_fakes_only",
        ),
        "alternate_efficiency_only": run_probability(
            nominal_migration * alternate_efficiency[None, :],
            nominal_data,
            nominal_covariance,
            f"_{channel}_alternate_efficiency_only",
        ),
        "alternate_migration_only": run_probability(
            alternate_migration * nominal_efficiency[None, :],
            nominal_data,
            nominal_covariance,
            f"_{channel}_alternate_migration_only",
        ),
        "alternate_response_only": run_probability(
            alternate_probability,
            nominal_data,
            nominal_covariance,
            f"_{channel}_alternate_response_only",
        ),
    }

    nominal_shape = normalized_shape(
        nominal_result["x"],
        nominal["n_pt"],
        nominal["n_rho"],
        nominal["in_window"],
    )
    alternate_shape = normalized_shape(
        alternate_result["x"],
        alternate["n_pt"],
        alternate["n_rho"],
        alternate["in_window"],
    )
    data_result_change = _relative_change(
        alternate_shape, nominal_shape
    )
    component_changes = {}
    for name, result in component_results.items():
        component_shape = normalized_shape(
            result["x"],
            nominal["n_pt"],
            nominal["n_rho"],
            nominal["in_window"],
        )
        change = _relative_change(component_shape, nominal_shape)
        component_changes[name] = {
            "core": _summary(change, nominal["core"]),
            "shown": _summary(change, nominal["shown"]),
            "largest_shown_bins": _bin_records(
                change, nominal, nominal["shown"]
            ),
            "refolding": {
                "chi2A": float(result["chi2A"]),
                "ndf": int(result["ndf"]),
            },
        }

    nominal_truth_shape = normalized_shape(
        nominal["gen"],
        nominal["n_pt"],
        nominal["n_rho"],
        nominal["in_window"],
    )
    alternate_truth_shape = normalized_shape(
        alternate["gen"],
        alternate["n_pt"],
        alternate["n_rho"],
        alternate["in_window"],
    )
    particle_level_change = _relative_change(
        alternate_truth_shape, nominal_truth_shape
    )

    nominal_miss_probability = 1.0 - nominal_probability.sum(axis=0)
    alternate_miss_probability = 1.0 - alternate_probability.sum(axis=0)
    conditional_total_variation = 0.5 * (
        np.sum(
            np.abs(alternate_probability - nominal_probability),
            axis=0,
        )
        + np.abs(
            alternate_miss_probability - nominal_miss_probability
        )
    )

    def fit_record(result):
        ndf = int(result["ndf"])
        return {
            "chi2A": float(result["chi2A"]),
            "ndf": ndf,
            "pvalue": float(
                chi2_distribution.sf(float(result["chi2A"]), ndf)
            ),
        }

    def response_mc_stat_record(result, shape, prepared):
        shape_covariance = normalized_covariance(
            result["x"],
            result["Esys"],
            prepared["n_pt"],
            prepared["n_rho"],
            prepared["in_window"],
        )
        relative_uncertainty = np.divide(
            np.sqrt(np.maximum(np.diag(shape_covariance), 0.0)),
            np.abs(shape),
            out=np.full_like(shape, np.nan),
            where=shape != 0.0,
        )
        return {
            "core": _summary(relative_uncertainty, prepared["core"]),
            "shown": _summary(relative_uncertainty, prepared["shown"]),
        }

    return {
        "channel": channel,
        "definition": (
            "same real data unfolded with each model's complete fake "
            "correction, miss efficiency, and response; tau=0; area constraint"
        ),
        "refolding_goodness_of_fit": {
            "nominal": fit_record(nominal_result),
            "alternate": fit_record(alternate_result),
        },
        "response_mc_stat_relative_uncertainty": {
            "nominal": response_mc_stat_record(
                nominal_result, nominal_shape, nominal
            ),
            "alternate": response_mc_stat_record(
                alternate_result, alternate_shape, alternate
            ),
        },
        "unfolded_data_shape_change": {
            "core": _summary(data_result_change, nominal["core"]),
            "shown": _summary(data_result_change, nominal["shown"]),
            "largest_shown_bins": _bin_records(
                data_result_change, nominal, nominal["shown"]
            ),
            "binwise_alternate_over_nominal": _ratio_records(
                alternate_shape, nominal_shape, nominal
            ),
        },
        "unfolded_data_component_swaps": component_changes,
        "particle_level_generator_shape_change": {
            "core": _summary(particle_level_change, nominal["core"]),
            "shown": _summary(particle_level_change, nominal["shown"]),
            "largest_shown_bins": _bin_records(
                particle_level_change, nominal, nominal["shown"]
            ),
        },
        "conditional_response_total_variation": {
            "core": _summary(
                conditional_total_variation, nominal["core"]
            ),
            "shown": _summary(
                conditional_total_variation, nominal["shown"]
            ),
            "largest_shown_bins": _bin_records(
                conditional_total_variation,
                nominal,
                nominal["shown"],
            ),
        },
        "fake_fraction_by_pt": {
            "nominal": _fake_fraction_by_pt(nominal),
            "alternate": _fake_fraction_by_pt(alternate),
        },
        "miss_fraction_by_pt": {
            "nominal": _miss_fraction_by_pt(nominal),
            "alternate": _miss_fraction_by_pt(alternate),
        },
        "caveat": (
            "MG+Pythia8 CP5 versus MG+Herwig7 CH3 changes shower, "
            "hadronization, and tune together; it is a model stress, not a "
            "pure parton-shower-scale variation"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=["dijet", "trijet"], required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = analyse(args.channel, args.nominal, args.alternate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "nominal chi2A/ndf = "
        f"{result['refolding_goodness_of_fit']['nominal']['chi2A']:.2f}/"
        f"{result['refolding_goodness_of_fit']['nominal']['ndf']}"
    )
    print(
        "alternate chi2A/ndf = "
        f"{result['refolding_goodness_of_fit']['alternate']['chi2A']:.2f}/"
        f"{result['refolding_goodness_of_fit']['alternate']['ndf']}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
