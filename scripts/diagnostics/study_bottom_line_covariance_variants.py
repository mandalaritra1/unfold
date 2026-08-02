#!/usr/bin/env python3
"""Diagnose the ARC-r2 global bottom-line covariance assumptions.

This does not alter the production result. It compares the full propagated
statistical covariance with a pT-block-diagonal diagnostic, and the global
two-vector PS/HAD model covariance with the per-pT two-vector alternative.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unfold.tools.unfolder_core import Unfolder, get_spec


def _offsets(edges_by_pt):
    counts = [len(edges) - 1 for edges in edges_by_pt]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def _shown_indices(unfolder):
    starts, counts = _offsets(unfolder.gen_edges_by_pt)
    floors = unfolder._bl_shown_floors()
    blocks = []
    for i, edges in enumerate(unfolder.gen_edges_by_pt):
        if unfolder.pt_edges[i] < 200:
            continue
        lower_edges = np.asarray(edges[:-1], float)
        block = [
            starts[i] + j for j in range(counts[i])
            if lower_edges[j] >= floors[i] - 1e-9
        ]
        if block:
            blocks.append(block)
    return blocks


def _block_diagonal(covariance, blocks):
    output = np.zeros_like(covariance)
    offset = 0
    for block in blocks:
        size = len(block)
        output[offset:offset + size, offset:offset + size] = (
            covariance[offset:offset + size, offset:offset + size]
        )
        offset += size
    return output


def evaluate(groomed, output_dir):
    base = get_spec("zjet", "rho", "arc_r2")
    spec = replace(
        base,
        model_covariance_scope="global_shown",
        # ARC-r2 research-note definition: 5 GeV groomed-mass display floor,
        # aligned with [185,200,290,400,inf] pT bins.
        bl_shown_floors_groomed=[-2.5, -3.0, -3.0, -3.5],
    )
    unfolder = Unfolder(spec, groomed, do_syst=True, cms_label="Internal")

    blocks = _shown_indices(unfolder)
    gidx = [index for block in blocks for index in block]
    covariance_stat = np.asarray(unfolder.cov_data_np)[np.ix_(gidx, gidx)]
    covariance_stat_block = _block_diagonal(covariance_stat, blocks)
    residual = (
        np.asarray(unfolder.y_unf) - np.asarray(unfolder.y_true)
    )[gidx]

    reco_starts, reco_counts = _offsets(unfolder.reco_edges_by_pt)
    reco_indices = []
    floors = unfolder._bl_shown_floors()
    for i, edges in enumerate(unfolder.reco_edges_by_pt):
        if unfolder.pt_edges[i] < 200:
            continue
        lower_edges = np.asarray(edges[:-1], float)
        reco_indices.extend(
            reco_starts[i] + j for j in range(reco_counts[i])
            if lower_edges[j] >= floors[i] - 1e-9
        )
    ndof = unfolder._effective_ndof(reco_indices)

    unfolder.spec = replace(spec, model_covariance_scope="global_shown")
    covariance_model_global = unfolder._model_cov_unfolded(gidx)
    unfolder.spec = replace(spec, model_covariance_scope="per_pt")
    covariance_model_per_pt = unfolder._model_cov_unfolded(gidx)

    variants = {
        "full stat + global PS/HAD": covariance_stat + covariance_model_global,
        "full stat + per-pT PS/HAD": covariance_stat + covariance_model_per_pt,
        "pT-block stat + global PS/HAD": covariance_stat_block + covariance_model_global,
        "pT-block stat + per-pT PS/HAD": covariance_stat_block + covariance_model_per_pt,
        "full stat only": covariance_stat,
        "pT-block stat only": covariance_stat_block,
    }
    results = {}
    for name, covariance in variants.items():
        metric = unfolder._chi2_from_covariance(
            residual, covariance, ndof=ndof)
        results[name] = {
            "chi2": metric["chi2"],
            "ndof": metric["ndof"],
            "chi2_per_ndof": metric["chi2"] / metric["ndof"],
        }

    # Publication-shape diagnostic: normalize data and prediction independently
    # over the SHOWN bins in every pT slice and propagate the covariance with
    # the exact normalization Jacobian. This is the quantity displayed by the
    # normalized differential measurement; it differs from the historical BLT,
    # which tests absolute unfolded yields.
    def _normalize_blocks(values, covariance, selected_blocks, widths_blocks):
        values = np.asarray(values, float)
        covariance = np.asarray(covariance, float)
        normalized_parts = []
        jacobian = np.zeros_like(covariance)
        offset = 0
        for block, widths in zip(selected_blocks, widths_blocks):
            size = len(block)
            local_values = values[offset:offset + size]
            total = local_values.sum()
            local_jacobian = (
                np.eye(size) * total - np.outer(local_values, np.ones(size))
            ) / total**2
            density_scale = np.diag(1.0 / np.asarray(widths, float))
            local_jacobian = density_scale @ local_jacobian
            jacobian[offset:offset + size, offset:offset + size] = local_jacobian
            normalized_parts.append(local_values / total / widths)
            offset += size
        return np.concatenate(normalized_parts), jacobian @ covariance @ jacobian.T

    gen_widths = []
    gen_starts, _ = _offsets(unfolder.gen_edges_by_pt)
    for block in blocks:
        pt_index = next(
            i for i, start in enumerate(gen_starts)
            if start <= block[0] < start + len(unfolder.gen_edges_by_pt[i]) - 1
        )
        local = [index - gen_starts[pt_index] for index in block]
        all_widths = np.diff(np.asarray(unfolder.gen_edges_by_pt[pt_index], float))
        gen_widths.append(all_widths[local])
    gen_data, gen_stat_normalized = _normalize_blocks(
        np.asarray(unfolder.y_unf)[gidx], covariance_stat, blocks, gen_widths)
    gen_truth, _ = _normalize_blocks(
        np.asarray(unfolder.y_true)[gidx], np.zeros_like(covariance_stat),
        blocks, gen_widths)
    gen_residual_normalized = gen_data - gen_truth
    ps_normalized = np.asarray(unfolder.model_ps_shift_flat)[gidx] * gen_data
    had_normalized = np.asarray(unfolder.model_had_shift_flat)[gidx] * gen_data
    gen_model_normalized = (
        np.outer(ps_normalized, ps_normalized)
        + np.outer(had_normalized, had_normalized)
    )
    normalized_unfolded = unfolder._chi2_from_covariance(
        gen_residual_normalized,
        gen_stat_normalized + gen_model_normalized,
    )

    reco_starts, reco_counts = _offsets(unfolder.reco_edges_by_pt)
    reco_blocks = []
    reco_widths = []
    for i, edges in enumerate(unfolder.reco_edges_by_pt):
        if unfolder.pt_edges[i] < 200:
            continue
        lower = np.asarray(edges[:-1], float)
        local = [j for j in range(reco_counts[i]) if lower[j] >= floors[i] - 1e-9]
        reco_blocks.append([reco_starts[i] + j for j in local])
        reco_widths.append(np.diff(np.asarray(edges, float))[local])
    reco_flat = [index for block in reco_blocks for index in block]
    reco_covariance = np.diag(np.asarray(unfolder.corrected_measured_variances)[reco_flat])
    reco_data, reco_stat_normalized = _normalize_blocks(
        np.asarray(unfolder.y_meas)[reco_flat], reco_covariance,
        reco_blocks, reco_widths)
    reco_mc, _ = _normalize_blocks(
        np.asarray(unfolder.mosaic.sum(axis=1))[reco_flat],
        np.zeros_like(reco_covariance), reco_blocks, reco_widths)
    normalized_smeared = unfolder._chi2_from_covariance(
        reco_data - reco_mc, reco_stat_normalized)

    results["normalized shown shapes + global PS/HAD"] = {
        "unfolded_chi2": normalized_unfolded["chi2"],
        "unfolded_ndof": normalized_unfolded["ndof"],
        "unfolded_chi2_per_ndof": (
            normalized_unfolded["chi2"] / normalized_unfolded["ndof"]
        ),
        "smeared_chi2": normalized_smeared["chi2"],
        "smeared_ndof": normalized_smeared["ndof"],
        "smeared_chi2_per_ndof": (
            normalized_smeared["chi2"] / normalized_smeared["ndof"]
        ),
    }

    tag = "groomed" if groomed else "ungroomed"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"covariance_variants_{tag}.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    names = list(results)
    plot_names = [name for name in names if "chi2_per_ndof" in results[name]]
    values = [results[name]["chi2_per_ndof"] for name in plot_names]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(plot_names, values)
    ax.axvline(330.99 if groomed else 364.22, color="black", linestyle="--",
               label="smeared reference")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xlabel(r"Unfolded $\chi^2/n_\mathrm{dof}$")
    ax.set_title(f"{tag.capitalize()}: covariance diagnostics in shown space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"covariance_variants_{tag}.pdf")
    fig.savefig(output_dir / f"covariance_variants_{tag}.png", dpi=160)
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/zjet/rho/model_covariance_diagnostics"),
    )
    args = parser.parse_args()
    for groomed in (False, True):
        result = evaluate(groomed, args.output_dir)
        tag = "groomed" if groomed else "ungroomed"
        print(tag, json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
