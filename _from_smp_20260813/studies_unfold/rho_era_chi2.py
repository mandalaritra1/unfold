#!/usr/bin/env python
"""Decompose the folded data/MC chi2 of the hadronic rho unfold per Run 2 era.

The Run 2 combined folded chi2 is large, and for trijet it grew by more than
the luminosity ratio between 2018-only and Run 2. Either the fractional data/MC
mismatch is the same everywhere and the chi2 is pure statistics, or one era's
detector conditions disagree with its own response and the chi2 is
concentrated there. This driver answers that directly.

For every (channel, era) it rebuilds the adopted inputs from the era's OWN data
and MC pickles -- so detector conditions, response and covariance all match --
using ``rho_unfold_inputs_from_hists.py``'s rebinning verbatim, and reports:

  * the reco-level folded chi2 against the era's own covariance, both raw and
    after rescaling the prediction to the era's data normalization (shape only);
  * the same chi2 rescaled to the Run 2 data yield. For a fractional mismatch
    that is identical in every era this quantity is era-independent and the
    per-era chi2 values SUM to the Run 2 combined chi2; an era that stands out
    here is a conditions/response mismatch, not statistics;
  * the tau=0 TUnfold chi2A/ndf of the era's own unfold;
  * the data/MC normalization factor and the per-pT-slice breakdown.

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python \
        scripts/studies/unfold/rho_era_chi2.py \
        --out outputs/pairsplit_unfold/hadronic_rho_era_chi2_run2.json
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.studies.unfold.rho_unfold_inputs_from_hists import (  # noqa: E402
    ADOPTED,
    HIST_KEYS,
    _sel,
    edge_groups,
    regroup,
)
from scripts.studies.unfold.rho_unfold_systematics import (  # noqa: E402
    BINNINGS,
    use_binning,
)

BASE = Path("/Users/aritra/cernbox (2)/hadronic_minimal_rho_pairsplit")
ERAS = ("2016APV", "2016", "2017", "2018")
CHANNELS = ("dijet", "trijet")
#### CMS UL integrated luminosities, quoted for reference only -- every number
#### below is scaled by the measured data yield, not by these.
NOMINAL_LUMI = {"2016APV": 19.5, "2016": 16.8, "2017": 41.5, "2018": 59.8}


def era_path(era, channel, kind):
    stem = (
        f"minimal_rho_{channel}_mg_pythia8_{era}.pkl"
        if kind == "mc"
        else f"minimal_rho_{channel}_data_{era}.pkl"
    )
    return BASE / era / f"{channel}_{kind}" / stem


def build_era_inputs(channel: str, era: str, groom: str = "g") -> dict:
    """Rebin one era onto the adopted axes; same schema as the NPZ files."""
    spec = ADOPTED[channel]
    rho_edges = np.asarray(spec["rho"], dtype=float)
    pt_groups = spec["pt_groups"]
    keys = HIST_KEYS[groom]

    mc = pickle.load(open(era_path(era, channel, "mc"), "rb"))
    reco_values, reco_w2, h_reco = _sel(mc, keys["reco"])
    gen_values, gen_w2, h_gen = _sel(mc, keys["gen"])
    matrix_values, matrix_w2, _ = _sel(mc, keys["matrix"])

    pt_fine = np.asarray(h_reco.axes["ptreco"].edges)
    reco_rho_fine = np.asarray(h_reco.axes["mpt_reco"].edges)
    gen_rho_fine = np.asarray(h_gen.axes["mpt_gen"].edges)
    reco_groups = edge_groups(reco_rho_fine, rho_edges)
    gen_groups = edge_groups(gen_rho_fine, rho_edges)
    pt_edges = np.array(
        [pt_fine[group[0]] for group in pt_groups]
        + [pt_fine[pt_groups[-1][-1] + 1]]
    )

    interior = slice(1, -1)

    def rebin_1d(values, groups):
        merged = regroup(
            regroup(values[interior, interior], 0, pt_groups), 1, groups
        )
        return merged.reshape(-1)

    def rebin_matrix(values):
        #### (ptreco, ptgen, rho_reco, rho_gen) -> (ptreco, rho_reco, ptgen, rho_gen)
        block = np.transpose(
            values[interior, interior, interior, interior], (0, 2, 1, 3)
        )
        block = regroup(
            regroup(
                regroup(regroup(block, 0, pt_groups), 1, reco_groups),
                2,
                pt_groups,
            ),
            3,
            gen_groups,
        )
        return block.reshape(
            block.shape[0] * block.shape[1], block.shape[2] * block.shape[3]
        )

    inputs = {
        "A": rebin_matrix(matrix_values),
        "A_w2": rebin_matrix(matrix_w2),
        "reco": rebin_1d(reco_values, reco_groups),
        "reco_w2": rebin_1d(reco_w2, reco_groups),
        "gen": rebin_1d(gen_values, gen_groups),
        "gen_w2": rebin_1d(gen_w2, gen_groups),
        "pt_edges": pt_edges,
        "rho_edges": rho_edges,
    }
    del mc, reco_values, reco_w2, gen_values, gen_w2, matrix_values, matrix_w2
    gc.collect()

    data = pickle.load(open(era_path(era, channel, "data"), "rb"))
    data_values, data_w2, _ = _sel(data, keys["reco"])
    inputs["data_reco"] = rebin_1d(data_values, reco_groups)
    inputs["data_w2"] = rebin_1d(data_w2, reco_groups)
    if keys["cov"] in data:
        covariance_values, _, _ = _sel(data, keys["cov"])
        block = covariance_values[interior, interior, interior, interior]
        block = regroup(
            regroup(
                regroup(regroup(block, 0, pt_groups), 1, reco_groups),
                2,
                pt_groups,
            ),
            3,
            reco_groups,
        )
        size = block.shape[0] * block.shape[1]
        inputs["data_V"] = block.reshape(size, size)
        inputs["covariance_source"] = "reco_cov (event-clustered)"
    else:
        inputs["data_V"] = np.diag(inputs["data_w2"])
        inputs["covariance_source"] = "diagonal sumw2 (one measured jet/event)"
    del data, data_values, data_w2
    gc.collect()
    return inputs


def _chi2(residual, covariance):
    values = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    scale = max(float(np.max(np.abs(values))), 1.0)
    rank = int(np.sum(values > 1e-12 * scale))
    chi2 = float(
        residual @ np.linalg.pinv(covariance, rcond=1e-12) @ residual
    )
    return chi2, rank


def analyse(channel, era, inputs):
    from scripts.studies.unfold.rho_unfold_crosschecks import _prepare, _run

    prepared = _prepare(inputs, channel)
    data = prepared["data"]
    prediction = prepared["response"].sum(axis=1)
    covariance = prepared["data_covariance"]

    normalization = float(data.sum() / prediction.sum())
    raw_chi2, rank = _chi2(data - prediction, covariance)
    shape_chi2, _ = _chi2(data - normalization * prediction, covariance)

    #### The measurement reports a shape normalized INSIDE each pT slice, so
    #### rescale the prediction the same way: this divides out both the global
    #### normalization and any disagreement in the pT spectrum itself, and
    #### leaves only the rho shape the analysis actually quotes.
    n_rho_reco_all = len(np.asarray(inputs["rho_edges"])) - 1
    n_pt_all = len(np.asarray(inputs["pt_edges"])) - 1
    per_pt_prediction = prediction.copy().reshape(n_pt_all, n_rho_reco_all)
    data_2d = data.reshape(n_pt_all, n_rho_reco_all)
    slice_factors = np.divide(
        data_2d.sum(axis=1),
        per_pt_prediction.sum(axis=1),
        out=np.ones(n_pt_all),
        where=per_pt_prediction.sum(axis=1) > 0.0,
    )
    per_pt_prediction = (per_pt_prediction * slice_factors[:, None]).reshape(-1)
    shape_per_pt_chi2, _ = _chi2(data - per_pt_prediction, covariance)

    unfolded = _run(prepared, data, covariance, f"_{channel}_{era}_era")

    n_rho_reco = len(prepared["reco_edges"]) - 1
    diagonal = np.diag(covariance)
    pulls = np.divide(
        data - normalization * prediction,
        np.sqrt(np.maximum(diagonal, 0.0)),
        out=np.zeros_like(data),
        where=diagonal > 0.0,
    )
    per_pt = []
    for pt_index in range(prepared["n_pt"]):
        window = slice(pt_index * n_rho_reco, (pt_index + 1) * n_rho_reco)
        block_chi2, _ = _chi2(
            (data - normalization * prediction)[window],
            covariance[window, window],
        )
        per_pt.append(
            {
                "pt_range": [
                    float(prepared["pt_edges"][pt_index]),
                    float(prepared["pt_edges"][pt_index + 1]),
                ],
                "shape_chi2": block_chi2,
                "n_bins": n_rho_reco,
            }
        )
    order = np.argsort(np.abs(pulls))[::-1][:4]
    largest = []
    for flat in order:
        pt_index, rho_index = divmod(int(flat), n_rho_reco)
        largest.append(
            {
                "pt_range": [
                    float(prepared["pt_edges"][pt_index]),
                    float(prepared["pt_edges"][pt_index + 1]),
                ],
                "rho_range": [
                    float(prepared["reco_edges"][rho_index]),
                    float(prepared["reco_edges"][rho_index + 1]),
                ],
                "shape_pull": float(pulls[flat]),
            }
        )

    return {
        "covariance_source": inputs["covariance_source"],
        "nominal_lumi_fb": NOMINAL_LUMI.get(era, sum(NOMINAL_LUMI.values())),
        "data_yield": float(data.sum()),
        "raw_data_yield": float(np.asarray(inputs["data_reco"]).sum()),
        "matched_mc_yield": float(prediction.sum()),
        "data_over_mc_normalization": normalization,
        "fake_fraction": float(
            prepared["fakes"].sum() / prepared["reco"].sum()
        ),
        "miss_fraction": float(prepared["misses"].sum() / prepared["gen"].sum()),
        "reco_chi2_raw": raw_chi2,
        "reco_chi2_shape": shape_chi2,
        "reco_chi2_shape_per_pt": shape_per_pt_chi2,
        "reco_ndf": rank,
        "reco_ndf_shape_per_pt": rank - n_pt_all,
        "per_pt_normalization_factors": slice_factors.tolist(),
        "tunfold_chi2A": float(unfolded["chi2A"]),
        "tunfold_ndf": int(unfolded["ndf"]),
        "negative_reported_bins": int(
            np.sum(unfolded["x"][prepared["shown"]] < 0.0)
        ),
        "per_pt_shape_chi2": per_pt,
        "largest_shape_pulls": largest,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--npz-dir", type=Path, default=None,
                        help="also cache each era's rebinned inputs here")
    parser.add_argument("--era", action="append", choices=ERAS)
    parser.add_argument("--channel", action="append", choices=CHANNELS)
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=Path("outputs/pairsplit_unfold/npz"),
        help="directory holding <channel>_run2_nominal.npz for the "
             "combined reference row",
    )
    parser.add_argument(
        "--binning",
        choices=tuple(BINNINGS),
        default="coarse_tail",
        help="gen-bin merge; coarse_tail is the production choice",
    )
    args = parser.parse_args(argv)
    use_binning(args.binning)

    eras = args.era or list(ERAS)
    channels = args.channel or list(CHANNELS)
    result = {"binning": args.binning, "eras": eras}
    for channel in channels:
        result[channel] = {}
        for era in eras:
            print(f"  {channel} {era} ...", flush=True)
            inputs = build_era_inputs(channel, era)
            if args.npz_dir is not None:
                args.npz_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.npz_dir / f"{channel}_{era}_nominal.npz",
                    **{
                        key: value
                        for key, value in inputs.items()
                        if isinstance(value, np.ndarray)
                    },
                )
            result[channel][era] = analyse(channel, era, inputs)
            del inputs
            gc.collect()

        #### rescale every era to the Run 2 data yield: for a fractional
        #### mismatch that does not depend on the era this is a constant, and
        #### the per-era chi2 then sum to the combined chi2
        total = sum(
            result[channel][era]["data_yield"] for era in eras
        )
        for era in eras:
            record = result[channel][era]
            factor = total / record["data_yield"]
            record["data_yield_fraction"] = record["data_yield"] / total
            record["shape_chi2_at_run2_yield"] = (
                record["reco_chi2_shape"] * factor
            )
            record["shape_per_pt_chi2_at_run2_yield"] = (
                record["reco_chi2_shape_per_pt"] * factor
            )
            record["tunfold_chi2A_at_run2_yield"] = (
                record["tunfold_chi2A"] * factor
            )
            record["raw_chi2_at_run2_yield"] = record["reco_chi2_raw"] * factor
        #### the Run 2 combined row, computed identically from the Phase A NPZ
        combined_path = args.combined_dir / f"{channel}_run2_nominal.npz"
        combined = None
        if combined_path.exists():
            combined = analyse(
                channel,
                "run2",
                {key: value for key, value in np.load(combined_path).items()}
                | {"covariance_source": "from Phase A NPZ"},
            )
            result[channel]["run2_combined"] = combined

        result[channel]["summary"] = {
            "total_data_yield": total,
            "sum_of_era_shape_chi2": sum(
                result[channel][era]["reco_chi2_shape"] for era in eras
            ),
            "sum_of_era_raw_chi2": sum(
                result[channel][era]["reco_chi2_raw"] for era in eras
            ),
            "sum_of_era_shape_per_pt_chi2": sum(
                result[channel][era]["reco_chi2_shape_per_pt"] for era in eras
            ),
            "sum_of_era_tunfold_chi2A": sum(
                result[channel][era]["tunfold_chi2A"] for era in eras
            ),
            "spread_of_shape_per_pt_chi2_at_run2_yield": {
                "min": min(
                    result[channel][era]["shape_per_pt_chi2_at_run2_yield"]
                    for era in eras
                ),
                "max": max(
                    result[channel][era]["shape_per_pt_chi2_at_run2_yield"]
                    for era in eras
                ),
            },
            "spread_of_tunfold_chi2A_at_run2_yield": {
                "min": min(
                    result[channel][era]["tunfold_chi2A_at_run2_yield"]
                    for era in eras
                ),
                "max": max(
                    result[channel][era]["tunfold_chi2A_at_run2_yield"]
                    for era in eras
                ),
            },
            "run2_combined_shape_chi2": (
                None if combined is None else combined["reco_chi2_shape"]
            ),
            "run2_combined_raw_chi2": (
                None if combined is None else combined["reco_chi2_raw"]
            ),
            "sum_over_combined_shape_ratio": (
                None
                if combined is None
                else sum(
                    result[channel][era]["reco_chi2_shape"] for era in eras
                )
                / combined["reco_chi2_shape"]
            ),
            "spread_of_shape_chi2_at_run2_yield": {
                "min": min(
                    result[channel][era]["shape_chi2_at_run2_yield"]
                    for era in eras
                ),
                "max": max(
                    result[channel][era]["shape_chi2_at_run2_yield"]
                    for era in eras
                ),
            },
        }

        print(f"\n{channel}")
        print(
            f"  {'era':<9}{'lumi':>7}{'yfrac':>8}{'data/MC':>9}"
            f"{'shape':>10}{'shape/pT':>10}{'chi2A/ndf':>12}"
            f"{'shape@R2':>10}{'shp/pT@R2':>11}{'chi2A@R2':>10}"
        )
        for era in eras:
            record = result[channel][era]
            print(
                f"  {era:<9}{record['nominal_lumi_fb']:>7.1f}"
                f"{record['data_yield_fraction']:>8.3f}"
                f"{record['data_over_mc_normalization']:>9.4f}"
                f"{record['reco_chi2_shape']:>10.0f}"
                f"{record['reco_chi2_shape_per_pt']:>10.0f}"
                f"{record['tunfold_chi2A']:>8.1f}/{record['tunfold_ndf']:<3d}"
                f"{record['shape_chi2_at_run2_yield']:>10.0f}"
                f"{record['shape_per_pt_chi2_at_run2_yield']:>11.0f}"
                f"{record['tunfold_chi2A_at_run2_yield']:>10.1f}"
            )
        summary = result[channel]["summary"]
        if summary["run2_combined_shape_chi2"] is not None:
            print(
                f"  {'sum eras':<9}{'':>7}{'':>8}{'':>9}"
                f"{summary['sum_of_era_shape_chi2']:>10.0f}"
                f"{summary['sum_of_era_shape_per_pt_chi2']:>10.0f}"
                f"{summary['sum_of_era_tunfold_chi2A']:>8.1f}{'':>4}"
            )
            combined_record = result[channel]["run2_combined"]
            print(
                f"  {'RUN 2':<9}{137.6:>7.1f}{1.0:>8.3f}"
                f"{combined_record['data_over_mc_normalization']:>9.4f}"
                f"{combined_record['reco_chi2_shape']:>10.0f}"
                f"{combined_record['reco_chi2_shape_per_pt']:>10.0f}"
                f"{combined_record['tunfold_chi2A']:>8.1f}/"
                f"{combined_record['tunfold_ndf']:<3d}"
                f"{combined_record['reco_chi2_shape']:>10.0f}"
                f"{combined_record['reco_chi2_shape_per_pt']:>11.0f}"
                f"{combined_record['tunfold_chi2A']:>10.1f}"
            )
            print(
                f"  sum(eras)/combined shape chi2 = "
                f"{summary['sum_over_combined_shape_ratio']:.2f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
