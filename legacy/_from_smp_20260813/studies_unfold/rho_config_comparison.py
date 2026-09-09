#!/usr/bin/env python
"""Compare unfolding configurations on one common set of reported rho bins.

Every configuration below is applied to the SAME nominal NPZ inputs, the same
adopted nested gen merge, the same fake-survival treatment and the same
per-pT normalization window, so the rows differ only by the estimator:

  ``tau0_area_full``   tau=0 TUnfold, area constraint, full data covariance
                       -- the production configuration, the reference row;
  ``tau0_noarea_full`` the same without the area constraint;
  ``tau0_area_diag``   area constraint with the data covariance diagonalized;
  ``gls``              generalized least squares solved directly in numpy
                       (independent of TUnfold, equivalent to no-area at tau=0);
  ``bayes_n4``         unregularized D'Agostini at the reference 4 iterations.

For each row: median/max relative statistical error, negative reported bins,
the largest normalized-shape shift against the production row, and -- from
input pseudo-experiments thrown around the matched MC reco spectrum with the
data covariance -- the pull width, the 68% coverage and the mean bias.

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python \
        scripts/studies/unfold/rho_config_comparison.py \
        --nominal-dijet outputs/pairsplit_unfold/npz/dijet_run2_nominal.npz \
        --nominal-trijet outputs/pairsplit_unfold/npz/trijet_run2_nominal.npz \
        --out outputs/pairsplit_unfold/hadronic_rho_config_comparison_run2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.studies.unfold.rho_iterative_bayes import (  # noqa: E402
    REFERENCE_ITERATION,
    iterative_bayes,
)
from scripts.studies.unfold.rho_unfold_systematics import (  # noqa: E402
    BINNINGS,
    load_npz,
    normalized_shape,
    use_binning,
)

PRODUCTION = "tau0_area_full"


def _summary(values, mask):
    selected = np.asarray(values, dtype=float)[mask]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(selected)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
    }


def _relative(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 0.0, a / b - 1.0, np.nan)


def _estimators(prepared, channel):
    """Return {name: callable(y, V) -> (x, sigma)} for every configuration."""
    from scripts.studies.unfold.rho_unfold_crosschecks import _gls, _run

    probability = prepared["probability"]
    efficiency = probability.sum(axis=0)
    prior = prepared["gen"].copy()

    def tunfold_factory(area, diagonal, tag):
        counter = {"n": 0}

        def run(y, covariance):
            used = (
                np.diag(np.diag(covariance)) if diagonal else covariance
            )
            counter["n"] += 1
            result = _run(
                prepared,
                y,
                used,
                f"_{channel}_{tag}_{counter['n']}",
                area=area,
            )
            return (
                result["x"],
                np.sqrt(np.maximum(np.diag(result["Ein"]), 0.0)),
                result,
            )

        return run

    def gls(y, covariance):
        x, covariance_x = _gls(probability, y, covariance)
        return x, np.sqrt(np.maximum(np.diag(covariance_x), 0.0)), None

    def bayes(y, covariance):
        x = iterative_bayes(
            probability, efficiency, y, prior, REFERENCE_ITERATION
        )
        return x, None, None

    return {
        "tau0_area_full": tunfold_factory(True, False, "areafull"),
        "tau0_noarea_full": tunfold_factory(False, False, "noareafull"),
        "tau0_area_diag": tunfold_factory(True, True, "areadiag"),
        "gls": gls,
        "bayes_n4": bayes,
    }


def analyse_channel(channel, nominal_path, n_toys, seed):
    from scripts.studies.unfold.rho_unfold_crosschecks import (
        _draw_gaussian,
        _prepare,
    )

    prepared = _prepare(load_npz(nominal_path), channel)
    n_pt, n_rho = prepared["n_pt"], prepared["n_rho"]
    shown, core, in_window = (
        prepared["shown"],
        prepared["core"],
        prepared["in_window"],
    )
    estimators = _estimators(prepared, channel)

    #### data unfolds
    data_results = {}
    for name, estimator in estimators.items():
        x, sigma, extra = estimator(
            prepared["data"], prepared["data_covariance"]
        )
        data_results[name] = {
            "x": x,
            "sigma": sigma,
            "chi2A": None if extra is None else float(extra["chi2A"]),
            "ndf": None if extra is None else int(extra["ndf"]),
        }

    #### input pseudo-experiments: throw around the matched MC reco spectrum
    #### with the data covariance, unfold each toy with every estimator, and
    #### compare against the MC truth the toys were generated from.
    rng = np.random.default_rng(seed)
    mean = prepared["response"].sum(axis=1)
    toys = _draw_gaussian(rng, mean, prepared["data_covariance"], n_toys)
    toy_unfolded = {name: [] for name in estimators}
    for toy in toys:
        for name, estimator in estimators.items():
            toy_unfolded[name].append(
                estimator(toy, prepared["data_covariance"])[0]
            )
    toy_unfolded = {
        name: np.stack(values) for name, values in toy_unfolded.items()
    }

    production_shape = normalized_shape(
        data_results[PRODUCTION]["x"], n_pt, n_rho, in_window
    )

    rows = {}
    for name in estimators:
        x = data_results[name]["x"]
        sigma = data_results[name]["sigma"]
        toy_sigma = np.std(toy_unfolded[name], axis=0, ddof=1)
        sigma_source = "analytic"
        if sigma is None:
            #### iterative Bayes has no analytic covariance here; its quoted
            #### error IS the toy spread, so its pull width is 1 by
            #### construction and only coverage/bias carry information.
            sigma = toy_sigma
            sigma_source = "input toys (pull width is 1 by construction)"
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_error = np.where(x > 0.0, sigma / x, np.nan)
            width = np.where(sigma > 0.0, toy_sigma / sigma, np.nan)
            bias = np.where(
                sigma > 0.0,
                (toy_unfolded[name].mean(axis=0) - prepared["gen"]) / sigma,
                np.nan,
            )
        coverage = np.mean(
            np.abs(toy_unfolded[name] - prepared["gen"][None, :])
            <= sigma[None, :],
            axis=0,
        )
        shape = normalized_shape(x, n_pt, n_rho, in_window)
        shift = _relative(shape, production_shape)
        rows[name] = {
            "sigma_source": sigma_source,
            "chi2A": data_results[name]["chi2A"],
            "ndf": data_results[name]["ndf"],
            "relative_statistical_error": _summary(relative_error, shown),
            "negative_reported_bins": int(np.sum(x[shown] < 0.0)),
            "shape_shift_vs_production": {
                "core": _summary(np.abs(shift), core),
                "shown": _summary(np.abs(shift), shown),
            },
            "pull_width": _summary(width, shown),
            "coverage_1sigma": _summary(coverage, shown),
            "mean_bias_in_sigma": _summary(bias, shown),
            "per_bin": {
                "unfolded": np.asarray(x).tolist(),
                "statistical_error": np.asarray(sigma).tolist(),
                "normalized_shape": shape.tolist(),
                "signed_shape_shift_vs_production": shift.tolist(),
            },
        }

    return {
        "n_toys": n_toys,
        "production_configuration": PRODUCTION,
        "gen_rho_edges": prepared["gen_edges"].tolist(),
        "pt_edges": prepared["pt_edges"].tolist(),
        "n_pt": n_pt,
        "n_rho": n_rho,
        "shown": shown.tolist(),
        "core": core.tolist(),
        "in_window": in_window.tolist(),
        "mc_gen": prepared["gen"].tolist(),
        "configurations": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-dijet", required=True, type=Path)
    parser.add_argument("--nominal-trijet", required=True, type=Path)
    parser.add_argument("--toys", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--binning",
        choices=tuple(BINNINGS),
        default="coarse_tail",
        help="gen-bin merge; coarse_tail is the production choice",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    use_binning(args.binning)

    result = {"binning": args.binning}
    for offset, (channel, path) in enumerate(
        (("dijet", args.nominal_dijet), ("trijet", args.nominal_trijet))
    ):
        print(f"\n{'=' * 78}\n{channel}")
        record = analyse_channel(channel, path, args.toys, args.seed + offset)
        result[channel] = record
        header = (
            f"  {'configuration':<18}{'stat med':>10}{'stat max':>10}"
            f"{'neg':>5}{'|dshape| max':>14}{'pull w':>9}{'cover':>8}"
        )
        print(header)
        for name, row in record["configurations"].items():
            print(
                f"  {name:<18}"
                f"{row['relative_statistical_error']['median']:>9.2%} "
                f"{row['relative_statistical_error']['max']:>9.2%} "
                f"{row['negative_reported_bins']:>4d}"
                f"{row['shape_shift_vs_production']['shown']['max']:>13.2%} "
                f"{row['pull_width']['median']:>8.3f}"
                f"{row['coverage_1sigma']['median']:>8.3f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
