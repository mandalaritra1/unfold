#!/usr/bin/env python3
"""Run the 2018 dijet or trijet rho unfolding with the shared Unfolder."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import ROOT

from unfold.tools.rho_channel_inputs import (
    CHANNELS,
    build_prepared_rho_inputs,
    discover_rho_channel_files,
)
from unfold.tools.unfolder_core import RHO_FIXED_JEC_SPEC, Unfolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=CHANNELS, required=True)
    parser.add_argument("--year", default="2018")
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "inputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cms-label", default="Internal")
    parser.add_argument("--lumi", type=float, default=59.7)
    parser.add_argument("--com", type=float, default=13.0)
    parser.add_argument(
        "--jacobian",
        action="store_true",
        help=(
            "Propagate normalized-result statistics through the normalization "
            "Jacobian (errors + correlation of the normalized spectrum). The "
            "default output dir gets a '_jacobian' suffix."
        ),
    )
    parser.add_argument(
        "--regularization",
        choices=("none", "ratio_curvature"),
        default="none",
        help=(
            "Unfolding regularization; 'ratio_curvature' penalizes the "
            "curvature of x/x_MC per pT slice (zero penalty for spectra "
            "proportional to the MC prior). tau from an L-curve scan unless "
            "--tau is given. The default output dir gets a '_reg' suffix."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Fixed regularization strength (skips the L-curve scan).",
    )
    parser.add_argument(
        "--method",
        choices=("tunfold", "roounfold_bayes"),
        default="tunfold",
        help=(
            "Unfolding backend. 'roounfold_bayes' uses iterative Bayes "
            "(D'Agostini) via RooUnfold with --n-iter iterations; these channels "
            "have no jackknife inputs, so the stat uncertainty falls back to "
            "RooUnfold's propagated covariance. Needs a built libRooUnfold "
            "(scripts/setup_roounfold.sh). The output dir gets a '_bayes' suffix."
        ),
    )
    parser.add_argument(
        "--n-iter", type=int, default=4,
        help="D'Agostini iterations for --method roounfold_bayes (default 4).",
    )
    args = parser.parse_args()
    if args.tau is not None and args.regularization == "none":
        parser.error("--tau requires --regularization ratio_curvature")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_artifact(unfolder: Unfolder, mode: str, output_dir: Path) -> Path:
    path = output_dir / "artifacts" / f"{mode}_results.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    systematic_names = [
        name for name in unfolder.systematics if name != "nominal"
    ]
    systematic_values = (
        np.stack(
            [unfolder.y_unf_dict[name] for name in systematic_names],
            axis=0,
        )
        if systematic_names
        else np.empty((0, len(unfolder.y_unf)))
    )
    np.savez_compressed(
        path,
        pt_edges=np.asarray(unfolder.pt_edges, dtype=float),
        rho_edges_reco=np.asarray(unfolder.edges, dtype=float),
        rho_edges_gen=np.asarray(unfolder.edges_gen, dtype=float),
        response_mosaic=unfolder.mosaic,
        measured=unfolder.y_meas,
        measured_variances_raw=unfolder.measured_variances,
        measured_variances_fake_corrected=unfolder.corrected_measured_variances,
        unfolded=unfolder.y_unf,
        unfolded_input_errors=np.sqrt(
            np.clip(np.diag(unfolder.cov_data_np), 0.0, None)
        ),
        truth_prior=unfolder.y_true,
        folded=unfolder.x_folded,
        fake_fraction=unfolder.fake_fraction_2d,
        misses=unfolder.misses_2d,
        covariance=unfolder.cov_np,
        input_covariance=unfolder.cov_data_np,
        systematic_names=np.asarray(systematic_names),
        systematic_unfolded=systematic_values,
    )
    return path


def mode_summary(unfolder: Unfolder, mode: str) -> dict:
    return {
        "mode": mode,
        "response_shape": list(unfolder.mosaic.shape),
        "unfolded_bins": int(len(unfolder.y_unf)),
        "unfolded_finite": bool(np.isfinite(unfolder.y_unf).all()),
        "folded_finite": bool(np.isfinite(unfolder.x_folded).all()),
        "fake_fraction_min": float(np.min(unfolder.fake_fraction_2d)),
        "fake_fraction_max": float(np.max(unfolder.fake_fraction_2d)),
        "negative_fake_bins_before_fraction": int(
            np.count_nonzero(unfolder.fakes_2d < 0)
        ),
        "negative_miss_bins": int(np.count_nonzero(unfolder.misses_2d < 0)),
        "zero_data_variance_bins": int(
            np.count_nonzero(unfolder.measured_variances == 0)
        ),
    }


def build_gallery(output_dir: Path) -> Path:
    gallery_path = output_dir / "index.html"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "outputs" / "build_rho_gallery.py"),
            "--root",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return gallery_path


def main() -> None:
    args = parse_args()
    # Option runs get suffixed default dirs so they never overwrite the
    # baseline outputs (mirrors the zjet '<tag>_jacobian[_reg]' convention).
    option_suffix = ("_jacobian" if args.jacobian else "") + (
        "_reg" if args.regularization != "none" else ""
    ) + ("_bayes" if args.method == "roounfold_bayes" else "")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT
        / "outputs"
        / args.channel
        / str(args.year)
        / "rho"
        / f"unfolding{option_suffix}"
    )
    output_dir = output_dir.resolve()
    unfolder_output_dir = os.path.relpath(output_dir, REPO_ROOT) + "/"
    for subdirectory in (
        "unfold",
        "uncertainties",
        "diagnostics",
        "artifacts",
        "_previews",
    ):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    files = discover_rho_channel_files(
        args.input_root,
        args.channel,
        args.year,
    )
    prepared = build_prepared_rho_inputs(files)
    summaries = []
    artifacts = []
    resolved_taus = {}

    ROOT.gErrorIgnoreLevel = ROOT.kError
    for mode, groomed in (("ungroomed", False), ("groomed", True)):
        spec = replace(
            RHO_FIXED_JEC_SPEC,
            output_dir=unfolder_output_dir,
            stat_propagation="jacobian" if args.jacobian else RHO_FIXED_JEC_SPEC.stat_propagation,
            regularization=args.regularization,
            tau=args.tau,
            method=args.method,
            n_iter=args.n_iter,
            xlim_lower_groomed=(
                prepared.binning[mode].gen_rho_edges_by_pt[0][0]
                if args.channel == "dijet" and groomed
                else RHO_FIXED_JEC_SPEC.xlim_lower_groomed
            ),
        )
        unfolder = Unfolder.from_prepared_inputs(
            spec,
            groomed,
            mc_inputs=prepared.mc,
            data_inputs=prepared.data,
            analysis_binning=prepared.binning[mode],
            systematics=prepared.systematics,
            cms_label=args.cms_label,
            lumi=args.lumi,
            com=args.com,
        )

        # Keep plot definitions and formatting in the shared Z+jet Unfolder.
        unfolder.run_all_plots(show=False)
        artifacts.append(write_artifact(unfolder, mode, output_dir))
        summaries.append(mode_summary(unfolder, mode))
        resolved_taus[mode] = float(unfolder.tau or 0.0)

    gallery_path = build_gallery(output_dir)
    input_paths = [files.data, files.mc]
    generated_outputs = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"run_manifest.json", "run_summary.txt"}
    )
    manifest = {
        "workflow": "dijet/trijet rho unfolding",
        "channel": args.channel,
        "year": str(args.year),
        "command": shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "cms_label": args.cms_label,
        "integrated_luminosity_fb-1": args.lumi,
        "center_of_mass_energy_TeV": args.com,
        "stat_propagation": "jacobian" if args.jacobian else "legacy",
        "regularization": args.regularization,
        "tau": {"requested": args.tau, "resolved_by_mode": resolved_taus},
        "root_version": ROOT.gROOT.GetVersion(),
        "plotting": {
            "implementation": "Unfolder.run_all_plots",
            "format": "shared Z+jet unfolder_core plotting methods",
            "first_reported_pt_bin": 1,
        },
        "inputs": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for path in input_paths
        },
        "binning": {
            mode: {
                "pt_edges_GeV": binning.pt_edges,
                "rho_edges_reco": binning.rho_edges,
                "rho_edges_gen": binning.rho_edges_gen,
                "rho_edges_reco_by_pt": binning.reco_rho_edges_by_pt,
                "rho_edges_gen_by_pt": binning.gen_rho_edges_by_pt,
                "first_reported_pt_bin": 1,
            }
            for mode, binning in prepared.binning.items()
        },
        "systematics": prepared.systematics,
        "uncertainty_scope": {
            "included": [
                "TUnfold-propagated input data statistical covariance",
                "available MG+PYTHIA8 response variations",
            ],
            "unavailable": [
                "response-matrix statistical uncertainty (no jackknife inputs)",
                "alternate-generator/model uncertainty (HERWIG intentionally skipped)",
            ],
            "label": "partial",
        },
        "validation": {
            "status": "deferred",
            "reason": "dedicated dijet/trijet validation inputs are not available",
            "reserved_output": str(
                REPO_ROOT
                / "outputs"
                / args.channel
                / str(args.year)
                / "rho"
                / "validation"
            ),
        },
        "modes": summaries,
        "outputs": [display_path(path) for path in generated_outputs],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        f"{args.channel} {args.year} rho unfolding",
        f"ROOT: {manifest['root_version']}",
        f"data: {files.data}",
        f"MC: {files.mc}",
        f"systematics: {len(prepared.systematics)} including nominal",
        "plots: shared Unfolder.run_all_plots Z+jet format",
        "uncertainties: partial (input statistics + available response variations)",
        "excluded: response MC statistics, HERWIG/model uncertainty",
        "validation: deferred",
    ]
    for summary in summaries:
        summary_lines.append(
            f"{summary['mode']}: response={summary['response_shape']}, "
            f"unfolded bins={summary['unfolded_bins']}, "
            f"fake fraction=[{summary['fake_fraction_min']:.4g}, "
            f"{summary['fake_fraction_max']:.4g}]"
        )
    summary_path = output_dir / "run_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(summary_lines))
    print(f"manifest: {manifest_path}")
    print(f"gallery: {gallery_path}")
    print(f"artifacts: {', '.join(str(path) for path in artifacts)}")


if __name__ == "__main__":
    main()
