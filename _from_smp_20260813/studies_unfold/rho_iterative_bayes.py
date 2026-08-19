#!/usr/bin/env python
"""Iterative-Bayes (D'Agostini) cross-check of the tau=0 TUnfold rho result.

This is a CROSS-CHECK, not a production method. It is built straight from the
nominal NPZ inputs with the same conventions as ``rho_unfold_stability.py``:

  * gen bins merged with the adopted nested map (``rho_unfold_systematics``
    ``SPECS``; production is the ``coarse_tail`` merge);
  * ``fakes = reco - A.sum(gen)`` removed from data as a per-reco-bin survival
    factor, ``misses = gen - A.sum(reco)`` carried in the efficiency;
  * conditional response ``P(r|g) = A[r, g] / gen[g]``, efficiency
    ``eps[g] = sum_r P(r|g)``, with NO efficiency floor.

The iteration is the standard unregularized D'Agostini step

    x^(n+1)[g] = x^(n)[g] / eps[g] * sum_r P(r|g) * y[r] / (P x^(n))[r]

seeded with the MC prior. ``n = 4`` is the quoted reference; ``n = 1..8`` is
scanned to expose the residual regularization, and the same n = 4 unfold is
repeated from a flat prior to size the prior dependence that keeps this method
a cross-check rather than the production choice.

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python \
        scripts/studies/unfold/rho_iterative_bayes.py \
        --nominal-dijet outputs/pairsplit_unfold/npz/dijet_run2_nominal.npz \
        --nominal-trijet outputs/pairsplit_unfold/npz/trijet_run2_nominal.npz \
        --out outputs/pairsplit_unfold/hadronic_rho_bayes_run2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.studies.unfold.rho_unfold_systematics import (  # noqa: E402
    BINNINGS,
    load_npz,
    normalized_shape,
    use_binning,
)

REFERENCE_ITERATION = 4
MAX_ITERATION = 8


def iterative_bayes(probability, efficiency, data, prior, n_iterations):
    """Unregularized D'Agostini iteration; returns the estimate after n steps."""
    estimate = np.array(prior, dtype=float)
    for _ in range(n_iterations):
        folded = probability @ estimate
        ratio = np.divide(
            data, folded, out=np.zeros_like(folded), where=folded > 0.0
        )
        estimate = (
            np.divide(
                estimate,
                efficiency,
                out=np.zeros_like(estimate),
                where=efficiency > 0.0,
            )
            * (probability.T @ ratio)
        )
    return estimate


def flat_prior(gen, n_pt, n_rho, mode="per_pt"):
    """MC-free seed: uniform over gen rho bins, keeping the yield scale."""
    gen_2d = np.asarray(gen, dtype=float).reshape(n_pt, n_rho)
    if mode == "per_pt":
        totals = gen_2d.sum(axis=1, keepdims=True)
        return np.repeat(totals / n_rho, n_rho, axis=1).reshape(-1)
    if mode == "global":
        return np.full(gen_2d.size, gen_2d.sum() / gen_2d.size)
    raise ValueError(f"unknown flat-prior mode {mode!r}")


def bayes_statistical_errors(
    prepared, prior, n_iterations, n_toys, rng
):
    """Data-statistical error of the Bayes estimate from input toys."""
    from scripts.studies.unfold.rho_unfold_crosschecks import _draw_gaussian

    toys = _draw_gaussian(
        rng, prepared["data"], prepared["data_covariance"], n_toys
    )
    efficiency = prepared["probability"].sum(axis=0)
    unfolded = np.stack(
        [
            iterative_bayes(
                prepared["probability"], efficiency, toy, prior, n_iterations
            )
            for toy in toys
        ]
    )
    return np.std(unfolded, axis=0, ddof=1)


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


def analyse_channel(channel, nominal_path, n_toys, seed):
    from scripts.studies.unfold.rho_unfold_crosschecks import _prepare, _run

    prepared = _prepare(load_npz(nominal_path), channel)
    n_pt, n_rho = prepared["n_pt"], prepared["n_rho"]
    in_window, shown, core = (
        prepared["in_window"],
        prepared["shown"],
        prepared["core"],
    )
    probability = prepared["probability"]
    efficiency = probability.sum(axis=0)

    #### production reference: tau=0 TUnfold, area constraint, full covariance
    tunfold_result = _run(
        prepared,
        prepared["data"],
        prepared["data_covariance"],
        f"_{channel}_bayes_reference",
    )
    tunfold_shape = normalized_shape(
        tunfold_result["x"], n_pt, n_rho, in_window
    )
    tunfold_error = np.sqrt(
        np.maximum(np.diag(tunfold_result["Ein"]), 0.0)
    )

    mc_prior = prepared["gen"].copy()
    scan = {}
    estimates = {}
    for iteration in range(1, MAX_ITERATION + 1):
        estimate = iterative_bayes(
            probability, efficiency, prepared["data"], mc_prior, iteration
        )
        estimates[iteration] = estimate
        shape = normalized_shape(estimate, n_pt, n_rho, in_window)
        delta = np.abs(_relative(shape, tunfold_shape))
        scan[str(iteration)] = {
            "shape_delta_vs_tunfold": {
                "core": _summary(delta, core),
                "shown": _summary(delta, shown),
            },
            "negative_shown_bins": int(np.sum(estimate[shown] < 0.0)),
            "absolute_delta_vs_tunfold": _summary(
                np.abs(_relative(estimate, tunfold_result["x"])), shown
            ),
        }

    reference = estimates[REFERENCE_ITERATION]
    reference_shape = normalized_shape(reference, n_pt, n_rho, in_window)
    reference_delta = _relative(reference_shape, tunfold_shape)

    rng = np.random.default_rng(seed)
    reference_error = bayes_statistical_errors(
        prepared, mc_prior, REFERENCE_ITERATION, n_toys, rng
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_error = np.where(
            reference > 0.0, reference_error / reference, np.nan
        )
        tunfold_relative_error = np.where(
            tunfold_result["x"] > 0.0,
            tunfold_error / tunfold_result["x"],
            np.nan,
        )

    prior_sensitivity = {}
    for mode in ("per_pt", "global"):
        seed_prior = flat_prior(prepared["gen"], n_pt, n_rho, mode)
        flat_estimate = iterative_bayes(
            probability,
            efficiency,
            prepared["data"],
            seed_prior,
            REFERENCE_ITERATION,
        )
        flat_shape = normalized_shape(flat_estimate, n_pt, n_rho, in_window)
        delta = np.abs(_relative(flat_shape, reference_shape))
        prior_sensitivity[f"flat_{mode}"] = {
            "shape_delta_vs_mc_prior": {
                "core": _summary(delta, core),
                "shown": _summary(delta, shown),
            },
            "shape_delta_vs_tunfold": {
                "core": _summary(
                    np.abs(_relative(flat_shape, tunfold_shape)), core
                ),
                "shown": _summary(
                    np.abs(_relative(flat_shape, tunfold_shape)), shown
                ),
            },
            "per_bin_shape_delta_vs_mc_prior": delta.tolist(),
        }

    return {
        "method": (
            "unregularized D'Agostini; MC prior seed; no efficiency floor; "
            "fakes removed as a per-reco-bin survival factor; misses in the "
            "efficiency"
        ),
        "reference_iteration": REFERENCE_ITERATION,
        "n_toys_for_statistical_error": n_toys,
        "gen_rho_edges": prepared["gen_edges"].tolist(),
        "pt_edges": prepared["pt_edges"].tolist(),
        "n_pt": n_pt,
        "n_rho": n_rho,
        "shown": shown.tolist(),
        "core": core.tolist(),
        "in_window": in_window.tolist(),
        "efficiency_range": [
            float(np.min(efficiency)),
            float(np.max(efficiency)),
        ],
        "iteration_scan": scan,
        "reference": {
            "unfolded": reference.tolist(),
            "statistical_error": reference_error.tolist(),
            "relative_statistical_error": _summary(relative_error, shown),
            "normalized_shape": reference_shape.tolist(),
            "signed_shape_delta_vs_tunfold": reference_delta.tolist(),
            "abs_shape_delta_vs_tunfold": {
                "core": _summary(np.abs(reference_delta), core),
                "shown": _summary(np.abs(reference_delta), shown),
            },
            "negative_shown_bins": int(np.sum(reference[shown] < 0.0)),
        },
        "tunfold_reference": {
            "unfolded": tunfold_result["x"].tolist(),
            "statistical_error": tunfold_error.tolist(),
            "relative_statistical_error": _summary(
                tunfold_relative_error, shown
            ),
            "normalized_shape": tunfold_shape.tolist(),
            "chi2A": float(tunfold_result["chi2A"]),
            "ndf": int(tunfold_result["ndf"]),
        },
        "flat_prior_sensitivity": prior_sensitivity,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-dijet", required=True, type=Path)
    parser.add_argument("--nominal-trijet", required=True, type=Path)
    parser.add_argument("--toys", type=int, default=400)
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
        record = analyse_channel(
            channel, path, args.toys, args.seed + offset
        )
        result[channel] = record
        scan = record["iteration_scan"]
        print(
            "  |shape(Bayes n)/shape(TUnfold) - 1| over reported bins "
            "(median / max):"
        )
        for iteration in range(1, MAX_ITERATION + 1):
            entry = scan[str(iteration)]["shape_delta_vs_tunfold"]["shown"]
            print(
                f"    n={iteration}: {entry['median']:.2%} / "
                f"{entry['max']:.2%}"
                + ("   <- reference" if iteration == REFERENCE_ITERATION else "")
            )
        flat = record["flat_prior_sensitivity"]["flat_per_pt"][
            "shape_delta_vs_mc_prior"
        ]["shown"]
        print(
            f"  flat-prior (per-pT) sensitivity at n=4: "
            f"{flat['median']:.2%} median / {flat['max']:.2%} max"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
