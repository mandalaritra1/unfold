#!/usr/bin/env python3
"""Run Combine ML unfolding for the 2018 dijet/trijet rho workflows.

This reuses the prepared-input path used by ``run_rho_unfolding.py`` and writes a
Combine model analogous to ``run_combine_rho_full.py``:

* signal processes are gen-bin response columns;
* raw reco data are fitted, while matched-response templates are scaled by the
  nominal fake-survival correction, matching the current TUnfold convention;
* fitted POIs are converted back to full particle-level spectra by multiplying
  the nominal full truth (matched gen + misses);
* paired response variations become Combine shape nuisances when available.

The dijet/trijet inputs do not currently provide response-matrix jackknife
replicas, so the uncertainty scope is profile likelihood stat + available shape
systematics only.  Dijet 2018 includes HERWIG and therefore gets the model
nuisance; trijet 2018 does not.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

from run_combine_rho_full import (  # noqa: E402
    DEFAULT_CMSENV,
    DEFAULT_CONTAINER,
    docker_command,
    docker_is_running,
    gen_offsets,
    load_fit,
    normalize_by_pt,
    run_in_docker,
    selected_pairs,
    write_extractor,
    write_shapes_and_card,
)
from unfold.tools.rho_channel_inputs import (  # noqa: E402
    CHANNELS,
    build_prepared_rho_inputs,
    discover_rho_channel_files,
)
from unfold.tools.unfolder_core import RHO_FIXED_JEC_SPEC, Unfolder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=CHANNELS, required=True)
    parser.add_argument("--year", default="2018")
    parser.add_argument("--mode", choices=("groomed", "ungroomed"), default="groomed")
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "inputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--cmsenv", default=DEFAULT_CMSENV)
    parser.add_argument("--run-docker", action="store_true")
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--systematics",
        default="all",
        help="Comma-separated nuisance bases, 'all', or 'none'. Default: all paired variations.",
    )
    parser.add_argument("--max-nuisances", type=int, default=None)
    return parser.parse_args()


def build_channel_unfolder(args: argparse.Namespace):
    import matplotlib

    matplotlib.use("Agg")
    files = discover_rho_channel_files(args.input_root, args.channel, args.year)
    prepared = build_prepared_rho_inputs(files)
    mode = args.mode
    groomed = mode == "groomed"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / "outputs"
            / args.channel
            / str(args.year)
            / "rho"
            / "combine_full"
            / mode
        )
    output_dir = output_dir.resolve()

    spec = replace(
        RHO_FIXED_JEC_SPEC,
        output_dir=str(output_dir / "unfolder_inputs") + "/",
        stat_propagation="jacobian",
        regularization="none",
        method="tunfold",
    )
    unfolder = Unfolder.from_prepared_inputs(
        spec,
        groomed,
        mc_inputs=prepared.mc,
        data_inputs=prepared.data,
        analysis_binning=prepared.binning[mode],
        systematics=prepared.systematics,
        herwig_inputs=prepared.herwig,
        lumi=59.7,
        com=13.0,
    )
    return unfolder, prepared, files, output_dir


def normalize_density_by_pt(values: np.ndarray, edges_by_pt: list[list[float]]) -> np.ndarray:
    zeros = np.zeros((len(values), len(values)), dtype=float)
    normalized, _ = normalize_by_pt(values, zeros, edges_by_pt)
    return normalized


def write_channel_plot(output_dir: Path, stem: str, unfolder, channel: str, year: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = np.load(output_dir / f"{stem}_combine_unfolded.npz", allow_pickle=True)
    meta = np.load(output_dir / f"{stem}_meta.npz", allow_pickle=True)
    gen_edges_by_pt = json.loads(str(result["gen_edges_json"].item()))
    starts, counts = gen_offsets(gen_edges_by_pt)
    pt_edges = np.asarray(result["pt_edges"], dtype=float)

    combine_norm = np.asarray(result["normalized"], dtype=float)
    combine_stat = np.sqrt(np.clip(np.diag(result["cov_norm_stat"]), 0.0, None))
    combine_total = np.sqrt(np.clip(np.diag(result["cov_norm_total"]), 0.0, None))
    truth_norm = normalize_density_by_pt(np.asarray(meta["full_truth"], dtype=float), gen_edges_by_pt)
    herwig_norm = None
    if getattr(unfolder, "y_true_herwig", None) is not None:
        herwig_norm = normalize_density_by_pt(np.asarray(unfolder.y_true_herwig, dtype=float), gen_edges_by_pt)

    reported = list(range(1, len(gen_edges_by_pt)))
    fig, axes = plt.subplots(2, len(reported), figsize=(4.25 * len(reported), 6.8), sharex="col")
    for panel, ipt in enumerate(reported):
        ax = axes[0, panel]
        rax = axes[1, panel]
        idx = np.arange(starts[ipt], starts[ipt] + counts[ipt])
        edges = np.asarray(gen_edges_by_pt[ipt], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * np.diff(edges)

        ax.fill_between(
            centers,
            combine_norm[idx] - combine_total[idx],
            combine_norm[idx] + combine_total[idx],
            step="mid",
            color="#5790fc",
            alpha=0.24,
            label="Combine stat+syst" if panel == 0 else None,
        )
        ax.errorbar(
            centers,
            combine_norm[idx],
            yerr=combine_stat[idx],
            xerr=half_widths,
            fmt="o",
            color="#111827",
            markersize=3.5,
            label="Combine" if panel == 0 else None,
        )
        ax.step(
            edges,
            np.r_[truth_norm[idx], truth_norm[idx][-1]],
            where="post",
            color="#008060",
            lw=1.6,
            label="MG+PYTHIA8 truth" if panel == 0 else None,
        )
        if herwig_norm is not None:
            ax.step(
                edges,
                np.r_[herwig_norm[idx], herwig_norm[idx][-1]],
                where="post",
                color="#7c3aed",
                lw=1.5,
                ls="--",
                label="HERWIG truth" if panel == 0 else None,
            )

        denom = np.where(truth_norm[idx] != 0.0, truth_norm[idx], np.nan)
        rax.axhline(1.0, color="#6b7280", lw=1)
        rax.errorbar(
            centers,
            combine_norm[idx] / denom,
            yerr=combine_total[idx] / denom,
            xerr=half_widths,
            fmt="o",
            color="#111827",
            markersize=3.5,
        )
        if herwig_norm is not None:
            rax.step(edges, np.r_[herwig_norm[idx] / denom, (herwig_norm[idx] / denom)[-1]], where="post", color="#7c3aed", lw=1.5, ls="--")
        rax.set_ylim(0.25, 1.85)
        rax.grid(True, alpha=0.25)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$")

        hi = int(pt_edges[ipt + 1]) if ipt + 1 < len(pt_edges) - 1 else None
        ax.set_title(f"{int(pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+") + " GeV")
        ax.grid(True, alpha=0.25)
        if panel == 0:
            ax.set_ylabel(unfolder.spec.normalized_ylabel)
            rax.set_ylabel("ratio to PYTHIA")

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{channel} {year} rho Combine ML unfolding ({'groomed' if unfolder.groomed else 'ungroomed'})", y=0.99)
    fig.tight_layout()
    plot_path = output_dir / f"{stem}_combine_summary.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def run_nominal_in_docker(output_dir: Path, stem: str, n_gen: int, args: argparse.Namespace) -> None:
    """Run one no-nuisance fit and mirror it into statonly/total JSON outputs."""
    if not docker_is_running(args.container):
        raise RuntimeError(f"Docker container {args.container!r} is not running")
    write_extractor(output_dir)
    container_dir = f"/tmp/unfold_{stem}_nominal"
    subprocess.run(["docker", "exec", args.container, "rm", "-rf", container_dir], check=True)
    subprocess.run(["docker", "exec", args.container, "mkdir", "-p", container_dir], check=True)
    subprocess.run(["docker", "cp", str(output_dir) + "/.", f"{args.container}:{container_dir}/"], check=True)

    command = docker_command(stem, n_gen, args.r_min, args.r_max, args.cmsenv, False, args.timeout)
    wrapped = f"rm -rf /work && ln -s {container_dir} /work && {command}"
    subprocess.run(["docker", "exec", "-i", args.container, "bash", "-lc", wrapped], check=True)

    tmp_copy = output_dir.parent / f".{output_dir.name}.docker_copy_nominal"
    if tmp_copy.exists():
        shutil.rmtree(tmp_copy)
    subprocess.run(["docker", "cp", f"{args.container}:{container_dir}/.", str(tmp_copy)], check=True)
    for item in tmp_copy.iterdir():
        target = output_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), target)
    tmp_copy.rmdir()
    # The plotting helper expects both files. With no nuisances, total == stat.
    shutil.copyfile(output_dir / f"{stem}_total.json", output_dir / f"{stem}_statonly.json")


def fit_quality(output_dir: Path, stem: str) -> dict[str, dict[str, float | int]]:
    quality = {}
    for label in ("statonly", "total"):
        path = output_dir / f"{stem}_{label}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        quality[label] = {key: payload[key] for key in ("status", "covQual", "edm", "minNll")}
    return quality


def update_summary(
    output_dir: Path,
    stem: str,
    args: argparse.Namespace,
    prepared,
    files,
    card_summary: dict,
    plot_path: Path | None,
) -> None:
    summary_path = output_dir / f"{stem}_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update(
        {
            "workflow": "dijet/trijet rho Combine ML unfolding",
            "channel": args.channel,
            "year": str(args.year),
            "mode": args.mode,
            "inputs": {
                "data": str(files.data.resolve()),
                "mc": str(files.mc.resolve()),
                "herwig": None if files.herwig is None else str(files.herwig.resolve()),
            },
            "available_systematics": list(prepared.systematics),
            "card": card_summary,
            "outputs": {
                **summary.get("outputs", {}),
                "summary_plot": None if plot_path is None else plot_path.name,
            },
            "physics_notes": [
                "Raw reco data are fitted; matched-response templates are scaled by 1/(1-fake_fraction).",
                "Fitted POIs scale nominal full truth = matched gen + misses for the reported unfolded spectrum.",
                "Paired prepared-input response variations enter as shape nuisances; dijet has HERWIG model if the pickle is present.",
                "No TUnfold regularization term and no response-matrix jackknife uncertainty are included.",
            ],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    unfolder, prepared, files, output_dir = build_channel_unfolder(args)
    stem = f"{args.channel}_rho_{args.year}_{args.mode}_combine"

    pairs = selected_pairs(list(unfolder.systematics), args.systematics, args.max_nuisances)
    card_summary = write_shapes_and_card(
        unfolder,
        output_dir,
        stem,
        pairs,
        data_mode="data",
    )
    (output_dir / f"{stem}_card_summary.json").write_text(json.dumps(card_summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Combine card: {output_dir / card_summary['card']}")
    print(f"Nuisance pairs: {len(pairs)}")
    print(f"Response shape: {card_summary['n_reco']} reco x {card_summary['n_gen']} gen")
    print(f"Fake survival min: {float(np.min(1.0 - unfolder.fake_fraction_2d)):.6g}")

    plot_path = None
    if args.run_docker:
        if pairs:
            run_in_docker(output_dir, stem, card_summary["n_gen"], args)
        else:
            run_nominal_in_docker(output_dir, stem, card_summary["n_gen"], args)
        from run_combine_rho_full import write_results_and_plot

        write_results_and_plot(output_dir, stem, unfolder, card_summary)
        plot_path = write_channel_plot(output_dir, stem, unfolder, args.channel, str(args.year))
        print(f"Wrote plot: {plot_path}")
    else:
        print("Use --run-docker to build the workspace, run Combine, and make the plot.")
    update_summary(output_dir, stem, args, prepared, files, card_summary, plot_path)
    if args.run_docker:
        print(json.dumps({"fit_quality": fit_quality(output_dir, stem)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
