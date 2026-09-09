#!/usr/bin/env python
"""Per-bin reruns behind the pair_split Run 2 appendix figure set.

The Phase B JSONs under ``outputs/pairsplit_unfold`` store SUMMARIES
(median/min/max) for several checks that the appendix wants drawn bin by bin.
Everything here is a deterministic re-execution of the very same driver logic
on the very same NPZ inputs, dumping the per-bin vectors the summaries were
made from.  Nothing existing is overwritten; the single product is

    outputs/pairsplit_unfold/appendix_rerun_run2.json

What is rerun, and against which driver:

* ``closure``      -- ``rho_unfold_crosschecks.analyse_channel`` nominal MC
                      through its own response (matched reco, data covariance).
* ``varclosure``   -- ``rho_unfold_crosschecks._systematic_closure``, extended
                      to the parton-shower slices, per bin instead of summarized.
* ``input_toys``   -- ``rho_unfold_crosschecks._input_toys`` with the SAME rng
                      stream (``default_rng(seed + 0/1)``, first consumer), so
                      the per-bin pull widths and coverages are the ones the
                      Phase B medians were taken over.
* ``bayes_iter``   -- ``rho_iterative_bayes.iterative_bayes`` at n = 1, 2, 4, 8.
* ``refold``       -- ``rho_profiled_refold`` baseline GLS and the profiled
                      JER/JMS/JMR fit, keeping the folded reco vectors.

Run under ROOT:

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python \\
        scripts/plots/appendix_rerun_pairsplit_run2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.studies.unfold.rho_unfold_systematics import (  # noqa: E402
    load_npz,
    load_variation,
    normalized_shape,
)
from scripts.studies.unfold.rho_unfold_crosschecks import (  # noqa: E402
    _draw_gaussian,
    _prepare,
    _run,
)
from scripts.studies.unfold import rho_profiled_refold as refold  # noqa: E402
from scripts.studies.unfold.rho_iterative_bayes import (  # noqa: E402
    iterative_bayes,
)

IN = REPO / "outputs" / "pairsplit_unfold"
NPZ = IN / "npz"
SYST_DIR = IN / "syst_run2"
OUT = IN / "appendix_rerun_run2.json"

CHANNELS = ("dijet", "trijet")
DET_SOURCES = ("JER", "JMS", "JMR")
SHOWER_SOURCES = ("fsr", "isr")
VARIATIONS = [
    f"{source}{direction}"
    for source in (*DET_SOURCES, *SHOWER_SOURCES)
    for direction in ("Up", "Down")
]
BAYES_ITERATIONS = (1, 2, 4, 8)
SEED = 20260726


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 0.0, np.asarray(a, float) / b, np.nan)


def closure(prepared, channel):
    """tau=0 self-closure: nominal MC folded through its own response."""
    matched = prepared["response"].sum(axis=1)
    result = _run(
        prepared, matched, prepared["data_covariance"], f"_{channel}_apx_mc"
    )
    return {
        "unfolded": result["x"].tolist(),
        "gen": prepared["gen"].tolist(),
        "ratio": _ratio(result["x"], prepared["gen"]).tolist(),
    }


def variation_closure(prepared, nominal_inputs, channel):
    """Shifted matched-reco input pushed through the NOMINAL response.

    Two per-bin curves per variation: the absolute ratio against the nominal
    truth (which still carries the variation's overall normalization change)
    and the per-pT normalized-shape ratio, which is the space the systematic
    band actually lives in.
    """
    nominal_shape = normalized_shape(
        prepared["gen"], prepared["n_pt"], prepared["n_rho"],
        prepared["in_window"],
    )
    output = {}
    for name in VARIATIONS:
        varied_inputs, restored = load_variation(
            SYST_DIR / f"{channel}_{name}.npz", nominal_inputs
        )
        varied = _prepare(varied_inputs, channel)
        shifted = varied["response"].sum(axis=1)
        result = _run(
            prepared,
            shifted,
            np.diag(np.maximum(shifted, 1.0)),
            f"_{channel}_apx_{name}",
        )
        shape = normalized_shape(
            result["x"], prepared["n_pt"], prepared["n_rho"],
            prepared["in_window"],
        )
        output[name] = {
            "ratio_absolute": _ratio(result["x"], prepared["gen"]).tolist(),
            "ratio_shape": _ratio(shape, nominal_shape).tolist(),
            "gen_max_relative_change": float(
                np.nanmax(np.abs(_ratio(varied["gen"], prepared["gen"]) - 1.0))
            ),
            "nominal_data_restored": bool(restored),
        }
    return output


def input_toys(prepared, channel, n_toys, seed):
    """``_input_toys`` with its per-bin vectors and the raw pull matrix kept."""
    rng = np.random.default_rng(seed)
    covariance = prepared["data_covariance"]
    mean = prepared["response"].sum(axis=1)
    nominal_mc = _run(
        prepared, mean, covariance, f"_{channel}_apx_toyref"
    )
    toys = _draw_gaussian(rng, mean, covariance, n_toys)
    unfolded = np.empty((n_toys, prepared["gen"].size))
    for index, toy in enumerate(toys):
        unfolded[index] = _run(
            prepared, toy, covariance, f"_{channel}_apx_inputtoy_{index}"
        )["x"]
    sigma = np.sqrt(np.maximum(np.diag(nominal_mc["Ein"]), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = np.where(
            sigma[None, :] > 0.0,
            (unfolded - prepared["gen"][None, :]) / sigma[None, :],
            np.nan,
        )
    empirical = np.std(unfolded, axis=0, ddof=1)
    coverage = np.mean(
        np.abs(unfolded - prepared["gen"][None, :]) <= sigma[None, :], axis=0
    )
    return {
        "n_toys": int(n_toys),
        "seed": int(seed),
        "sigma_analytic": sigma.tolist(),
        "sigma_empirical": empirical.tolist(),
        "pull_width": _ratio(empirical, sigma).tolist(),
        "pull_mean": np.nanmean(pulls, axis=0).tolist(),
        "coverage_1sigma": coverage.tolist(),
        "pulls": pulls.tolist(),
    }


def bayes_iterations(prepared, channel):
    probability = prepared["probability"]
    efficiency = probability.sum(axis=0)
    tunfold_result = _run(
        prepared,
        prepared["data"],
        prepared["data_covariance"],
        f"_{channel}_apx_bayesref",
    )
    tunfold_shape = normalized_shape(
        tunfold_result["x"], prepared["n_pt"], prepared["n_rho"],
        prepared["in_window"],
    )
    output = {"tunfold_shape": tunfold_shape.tolist(), "iterations": {}}
    for iteration in BAYES_ITERATIONS:
        estimate = iterative_bayes(
            probability, efficiency, prepared["data"], prepared["gen"].copy(),
            iteration,
        )
        shape = normalized_shape(
            estimate, prepared["n_pt"], prepared["n_rho"],
            prepared["in_window"],
        )
        output["iterations"][str(iteration)] = {
            "normalized_shape": shape.tolist(),
            "ratio_vs_tunfold": _ratio(shape, tunfold_shape).tolist(),
        }
    return output


def profiled_refold(prepared, nominal_inputs, channel):
    """Baseline GLS and the profiled JER/JMS/JMR fit, keeping folded vectors."""
    baseline = refold.solve_unconstrained_gls(
        prepared["data"], prepared["data_covariance"], prepared["probability"]
    )
    nuisances = refold.detector_nuisances(
        channel, SYST_DIR, prepared, nominal_inputs
    )
    fit = refold.solve_profiled(
        prepared["data"],
        prepared["data_covariance"],
        prepared["probability"],
        nuisances,
        baseline["truth"],
    )
    #### rebuild the fitted templates so the POSTFIT folded prediction is
    #### compared against the data the fit actually saw (the fake-subtracted
    #### data moves with the detector nuisances too)
    varied_response = prepared["probability"].copy()
    varied_data = prepared["data"].copy()
    for nuisance in nuisances:
        value = fit["nuisance_pulls"][nuisance.name]
        if value >= 0.0:
            varied_response = varied_response + value * nuisance.response_up_shift
            varied_data = varied_data + value * nuisance.data_up_shift
        else:
            varied_response = (
                varied_response + (-value) * nuisance.response_down_shift
            )
            varied_data = varied_data + (-value) * nuisance.data_down_shift
    varied_response = np.maximum(varied_response, 0.0)

    folded_prefit = prepared["probability"] @ baseline["truth"]
    folded_postfit = varied_response @ fit["truth"]
    sigma = np.sqrt(np.maximum(np.diag(prepared["data_covariance"]), 0.0))
    return {
        "data_prefit": prepared["data"].tolist(),
        "data_postfit": varied_data.tolist(),
        "folded_prefit": folded_prefit.tolist(),
        "folded_postfit": folded_postfit.tolist(),
        "sigma_data": sigma.tolist(),
        "ratio_prefit": _ratio(prepared["data"], folded_prefit).tolist(),
        "ratio_postfit": _ratio(varied_data, folded_postfit).tolist(),
        "truth_prefit": baseline["truth"].tolist(),
        "truth_postfit": fit["truth"].tolist(),
        "nuisance_pulls": fit["nuisance_pulls"],
        "nuisance_at_bounds": fit["nuisance_at_bounds"],
        "chi2_prefit": float(baseline["chi2_data"]),
        "chi2_postfit_data": float(fit["chi2_data"]),
        "chi2_postfit_total": float(fit["chi2_total"]),
        "ndf": int(fit["ndf"]),
    }


def analyse(channel, n_toys, seed):
    nominal_inputs = load_npz(NPZ / f"{channel}_run2_nominal.npz")
    prepared = _prepare(nominal_inputs, channel)
    print(f"\n{channel}: {int(prepared['shown'].sum())} reported bins")
    out = {
        "gen_rho_edges": prepared["gen_edges"].tolist(),
        "pt_edges": prepared["pt_edges"].tolist(),
        "reco_rho_edges": prepared["reco_edges"].tolist(),
        "n_pt": int(prepared["n_pt"]),
        "n_rho": int(prepared["n_rho"]),
        "shown": prepared["shown"].tolist(),
        "core": prepared["core"].tolist(),
        "in_window": prepared["in_window"].tolist(),
        "gen": prepared["gen"].tolist(),
    }
    out["closure"] = closure(prepared, channel)
    dev = np.abs(np.asarray(out["closure"]["ratio"])[prepared["shown"]] - 1.0)
    print(f"  self-closure max |unf/gen - 1| = {np.nanmax(dev):.2e}")

    out["varclosure"] = variation_closure(prepared, nominal_inputs, channel)
    worst = max(
        (
            float(np.nanmax(np.abs(
                np.asarray(v["ratio_shape"])[prepared["core"]] - 1.0))),
            name,
        )
        for name, v in out["varclosure"].items()
    )
    print(f"  worst core shape non-closure: {worst[1]} {worst[0]:.3%}")

    out["input_toys"] = input_toys(prepared, channel, n_toys, seed)
    width = np.asarray(out["input_toys"]["pull_width"])[prepared["shown"]]
    cover = np.asarray(out["input_toys"]["coverage_1sigma"])[prepared["shown"]]
    print(
        f"  toys: pull width median {np.nanmedian(width):.3f} "
        f"[{np.nanmin(width):.3f}, {np.nanmax(width):.3f}]; "
        f"coverage median {np.median(cover):.3f} "
        f"[{cover.min():.3f}, {cover.max():.3f}]"
    )

    out["bayes_iter"] = bayes_iterations(prepared, channel)
    for iteration in BAYES_ITERATIONS:
        r = np.asarray(
            out["bayes_iter"]["iterations"][str(iteration)]["ratio_vs_tunfold"]
        )
        print(
            f"  bayes n={iteration}: core median |ratio-1| "
            f"{np.nanmedian(np.abs(r[prepared['core']] - 1.0)):.2%}"
        )

    out["refold"] = profiled_refold(prepared, nominal_inputs, channel)
    print(
        f"  refold: chi2 {out['refold']['chi2_prefit']:.1f} -> "
        f"{out['refold']['chi2_postfit_data']:.1f}; pulls "
        + ", ".join(
            f"{k} {v:+.3f}" for k, v in out["refold"]["nuisance_pulls"].items()
        )
    )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toys", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    result = {
        channel: analyse(
            channel,
            args.toys,
            args.seed + (0 if channel == "dijet" else 1),
        )
        for channel in CHANNELS
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
