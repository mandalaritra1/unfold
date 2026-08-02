#!/usr/bin/env python3
"""Run a matched-only 50:50 PYTHIA split closure with Combine.

This is the Combine analogue of ``study_5050_bias.py``.  The delete-one-tenth
jackknife response mosaics are inverted into two statistically independent
half-samples:

    R_A = sum(tenths 0..4),   R_B = sum(tenths 5..9).

The Combine response templates are built from half A.  The observed spectrum is
half B's matched reco projection, and the result is compared to half B's matched
gen truth.  This intentionally stays matched-only so it tests response/statistical
unfolding closure, not fake correction, miss efficiency, or detector systematics.
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
import uproot

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

from run_combine_rho_full import (  # noqa: E402
    DEFAULT_CMSENV,
    DEFAULT_CONTAINER,
    docker_is_running,
    gen_offsets,
    load_fit,
    normalize_by_pt,
    write_extractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="original_jacobian_reg")
    parser.add_argument("--mode", choices=("groomed", "ungroomed"), default="groomed")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--cmsenv", default=DEFAULT_CMSENV)
    parser.add_argument("--run-docker", action="store_true")
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args()


def build_unfolder(tag: str, groomed: bool):
    import matplotlib

    matplotlib.use("Agg")
    from unfold.tools.unfolder_core import Unfolder, get_spec

    spec = get_spec("zjet", "rho", tag)
    spec = replace(
        spec,
        output_dir=str(REPO_ROOT / "outputs" / "zjet" / "rho" / "combine_full" / "5050_inputs") + "/",
    )
    return Unfolder(spec, groomed=groomed, do_syst=False, compute_jackknife_stat=False)


def reconstruct_halves(unfolder) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct independent half-response matrices from delete-one-tenth JKs."""
    jk_matrices = [np.asarray(matrix, dtype=float) for matrix in unfolder.mosaic_jk_list]
    if len(jk_matrices) != 10:
        raise ValueError(f"expected 10 jackknife response matrices, got {len(jk_matrices)}")

    full_from_jk = np.sum(jk_matrices, axis=0) / 9.0
    tenths = [full_from_jk - matrix for matrix in jk_matrices]
    response_a = np.sum(tenths[:5], axis=0)
    response_b = np.sum(tenths[5:], axis=0)
    if not np.allclose(response_a + response_b, full_from_jk, rtol=1.0e-6, atol=1.0e-3):
        raise RuntimeError("jackknife half-response reconstruction failed")
    return response_a, response_b


def write_shapes_and_card(
    output_dir: Path,
    stem: str,
    response_a: np.ndarray,
    response_b: np.ndarray,
    unfolder,
    *,
    floor: float = 1.0e-9,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_path = output_dir / f"{stem}_shapes.root"
    card_path = output_dir / f"{stem}_card.txt"
    meta_path = output_dir / f"{stem}_meta.npz"

    templates = np.clip(np.asarray(response_a, dtype=float), floor, None)
    observed = np.asarray(response_b, dtype=float).sum(axis=1)
    truth_a = np.asarray(response_a, dtype=float).sum(axis=0)
    truth_b = np.asarray(response_b, dtype=float).sum(axis=0)
    n_reco, n_gen = templates.shape
    reco_edges = np.arange(n_reco + 1, dtype=float)

    with uproot.recreate(root_path) as root_file:
        root_file["data_obs"] = (observed, reco_edges)
        root_file["dummy_bkg"] = (np.full(n_reco, floor), reco_edges)
        for gen_index in range(n_gen):
            root_file[f"gen_{gen_index:03d}"] = (templates[:, gen_index], reco_edges)

    process_names = ["dummy_bkg"] + [f"gen_{index:03d}" for index in range(n_gen)]
    process_ids = ["1"] + [str(-index) for index in range(n_gen)]
    with card_path.open("w") as handle:
        handle.write("imax 1 number of channels\n")
        handle.write("jmax * number of processes minus one\n")
        handle.write("kmax 0 number of nuisance parameters\n")
        handle.write("---------------\n")
        handle.write(f"shapes * rho_5050 {root_path.name} $PROCESS\n")
        handle.write(f"shapes data_obs rho_5050 {root_path.name} data_obs\n")
        handle.write("---------------\n")
        handle.write("bin rho_5050\n")
        handle.write("observation -1\n")
        handle.write("---------------\n")
        handle.write("bin " + " ".join(["rho_5050"] * len(process_names)) + "\n")
        handle.write("process " + " ".join(process_names) + "\n")
        handle.write("process " + " ".join(process_ids) + "\n")
        handle.write("rate " + " ".join(["-1"] * len(process_names)) + "\n")
        handle.write("---------------\n")

    with np.errstate(divide="ignore", invalid="ignore"):
        start_values = np.divide(
            truth_b,
            truth_a,
            out=np.ones_like(truth_b),
            where=truth_a > 0.0,
        )
    start_values = np.clip(start_values, 0.0, 8.0)
    np.savez(
        meta_path,
        observed=observed,
        response_a=response_a,
        response_b=response_b,
        truth_a=truth_a,
        truth_b=truth_b,
        start_values=start_values,
        pt_edges=np.asarray(unfolder.pt_edges, dtype=float),
        gen_counts=np.asarray([len(edges) - 1 for edges in unfolder.gen_edges_by_pt], dtype=int),
        gen_edges_json=np.asarray(json.dumps(unfolder.gen_edges_by_pt), dtype=object),
    )

    return {
        "card": card_path.name,
        "root": root_path.name,
        "meta": meta_path.name,
        "n_reco": n_reco,
        "n_gen": n_gen,
        "observed_sum": float(np.sum(observed)),
        "template_sum": float(np.sum(templates)),
        "truth_a_sum": float(np.sum(truth_a)),
        "truth_b_sum": float(np.sum(truth_b)),
        "negative_template_bins_before_floor": int(np.count_nonzero(response_a < 0.0)),
        "zero_or_negative_template_bins_before_floor": int(np.count_nonzero(response_a <= 0.0)),
    }


def docker_command(stem: str, n_gen: int, r_min: float, r_max: float, cmsenv: str, timeout: int) -> str:
    workspace = f"{stem}_workspace.root"
    fit_name = f"{stem}_5050"
    poi_loop = (
        f"po=(); for i in $(seq 0 {n_gen - 1}); do "
        "idx=$(printf '%03d' \"$i\"); "
        f"po+=(--PO \"map=.*/gen_${{idx}}:r_gen_${{idx}}[1,{r_min:g},{r_max:g}]\"); "
        "done; "
    )
    poi_list = (
        f"pois=$(for i in $(seq 0 {n_gen - 1}); do "
        "printf 'r_gen_%03d,' \"$i\"; done | sed 's/,$//'); "
    )
    start_params = (
        "starts=$(python3 - <<'PY'\n"
        "import numpy as np\n"
        f"m=np.load('{stem}_meta.npz', allow_pickle=True)\n"
        "print(','.join(f'r_gen_{i:03d}={v:.6g}' for i,v in enumerate(m['start_values'])))\n"
        "PY\n"
        "); "
    )
    return (
        f"source {cmsenv} >/dev/null; cd /work; "
        f"{poi_loop}"
        f"text2workspace.py {stem}_card.txt "
        "-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel "
        "\"${po[@]}\" --for-fits --no-wrappers --X-pack-asympows "
        f"--optimize-simpdf-constraints=cms --use-histsum -o {workspace} "
        "> text2workspace_5050.log 2>&1; "
        f"{poi_list}"
        f"{start_params}"
        f"timeout {timeout:d} combine -M MultiDimFit {workspace} --algo none "
        "--redefineSignalPOIs \"$pois\" --floatOtherPOIs 1 --saveFitResult "
        "--cminDefaultMinimizerStrategy 0 "
        f"--setParameters \"$starts\" -n .{fit_name} "
        "> combine_5050.log 2>&1; "
        f"python3 extract_fit.py multidimfit.{fit_name}.root {fit_name}.json"
    )


def run_in_docker(output_dir: Path, stem: str, n_gen: int, args: argparse.Namespace) -> None:
    if not docker_is_running(args.container):
        raise RuntimeError(f"Docker container {args.container!r} is not running")

    write_extractor(output_dir)
    container_dir = f"/tmp/unfold_{stem}"
    subprocess.run(["docker", "exec", args.container, "rm", "-rf", container_dir], check=True)
    subprocess.run(["docker", "exec", args.container, "mkdir", "-p", container_dir], check=True)
    subprocess.run(["docker", "cp", str(output_dir) + "/.", f"{args.container}:{container_dir}/"], check=True)

    command = docker_command(stem, n_gen, args.r_min, args.r_max, args.cmsenv, args.timeout)
    wrapped = f"rm -rf /work && ln -s {container_dir} /work && {command}"
    subprocess.run(["docker", "exec", "-i", args.container, "bash", "-lc", wrapped], check=True)

    tmp_copy = output_dir.parent / f".{output_dir.name}.docker_copy"
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


def density(values: np.ndarray, edges_by_pt: list[list[float]]) -> np.ndarray:
    zeros = np.zeros((len(values), len(values)), dtype=float)
    normalized, _ = normalize_by_pt(values, zeros, edges_by_pt)
    return normalized


def write_results_and_plot(output_dir: Path, stem: str, unfolder, card_summary: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_gen = int(card_summary["n_gen"])
    fit_values, fit_cov, fit_payload = load_fit(output_dir / f"{stem}_5050.json", n_gen)
    meta = np.load(output_dir / f"{stem}_meta.npz", allow_pickle=True)
    truth_a = np.asarray(meta["truth_a"], dtype=float)
    truth_b = np.asarray(meta["truth_b"], dtype=float)
    gen_edges_by_pt = json.loads(str(meta["gen_edges_json"].item()))

    unfolded = fit_values * truth_a
    cov_abs = np.diag(truth_a) @ fit_cov @ np.diag(truth_a)
    unfolded_norm, cov_norm = normalize_by_pt(unfolded, cov_abs, gen_edges_by_pt)
    truth_b_norm = density(truth_b, gen_edges_by_pt)
    err_norm = np.sqrt(np.clip(np.diag(cov_norm), 0.0, None))

    result_npz = output_dir / f"{stem}_combine_5050.npz"
    np.savez(
        result_npz,
        central=unfolded,
        normalized=unfolded_norm,
        cov_abs=cov_abs,
        cov_norm=cov_norm,
        truth_a=truth_a,
        truth_b=truth_b,
        truth_b_normalized=truth_b_norm,
        poi_values=fit_values,
        pt_edges=np.asarray(unfolder.pt_edges, dtype=float),
        gen_edges_json=np.asarray(json.dumps(gen_edges_by_pt), dtype=object),
    )

    starts, counts = gen_offsets(gen_edges_by_pt)
    mode = "groomed" if unfolder.groomed else "ungroomed"
    reported = list(range(1, len(gen_edges_by_pt)))
    fig, axes = plt.subplots(2, len(reported), figsize=(5.2 * len(reported), 6.8), sharex="col")
    summary_slices = []
    for panel, ipt in enumerate(reported):
        ax = axes[0, panel]
        rax = axes[1, panel]
        idx = np.arange(starts[ipt], starts[ipt] + counts[ipt])
        edges = np.asarray(gen_edges_by_pt[ipt], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * np.diff(edges)
        denom = np.where(truth_b_norm[idx] != 0.0, truth_b_norm[idx], np.nan)
        ratio = unfolded_norm[idx] / denom
        ratio_err = err_norm[idx] / denom

        ax.step(
            edges,
            np.r_[truth_b_norm[idx], truth_b_norm[idx][-1]],
            where="post",
            color="#008060",
            lw=1.8,
            label="half-B truth" if panel == 0 else None,
        )
        ax.errorbar(
            centers,
            unfolded_norm[idx],
            yerr=err_norm[idx],
            xerr=half_widths,
            fmt="o",
            color="#111827",
            markersize=4,
            label="Combine half-B unfolded" if panel == 0 else None,
        )
        rax.axhline(1.0, color="#6b7280", lw=1)
        rax.errorbar(centers, ratio, yerr=ratio_err, xerr=half_widths, fmt="o", color="#111827", markersize=4)
        rax.set_ylim(0.75, 1.25)
        rax.grid(True, alpha=0.25)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$")

        finite = np.isfinite(ratio[1:])
        median_abs = float(np.nanmedian(np.abs(ratio[1:][finite] - 1.0))) if np.any(finite) else float("nan")
        max_abs = float(np.nanmax(np.abs(ratio[1:][finite] - 1.0))) if np.any(finite) else float("nan")
        hi = int(unfolder.pt_edges[ipt + 1]) if ipt + 1 < len(unfolder.pt_edges) - 1 else None
        summary_slices.append(
            {
                "pt_low": float(unfolder.pt_edges[ipt]),
                "pt_high": None if hi is None else float(unfolder.pt_edges[ipt + 1]),
                "median_abs_closure_excluding_underflow": median_abs,
                "max_abs_closure_excluding_underflow": max_abs,
            }
        )
        ax.set_title(f"{int(unfolder.pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+") + " GeV")
        ax.grid(True, alpha=0.25)
        if panel == 0:
            ax.set_ylabel(unfolder.spec.normalized_ylabel)
            rax.set_ylabel("unfolded / truth")

    axes[0, 0].legend(frameon=False, fontsize=9)
    max_slice_closure = max(item["max_abs_closure_excluding_underflow"] for item in summary_slices)
    fig.suptitle(
        f"Z+jet rho Combine 50:50 PYTHIA split ({mode}), max closure = {100 * max_slice_closure:.1f}%",
        y=0.99,
    )
    fig.tight_layout()
    plot_path = output_dir / f"{stem}_combine_5050_split.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "workflow": "Z+jet rho Combine matched-only 50:50 PYTHIA split closure",
        "mode": mode,
        "card": card_summary,
        "fit": {key: fit_payload[key] for key in ("status", "covQual", "edm", "minNll")},
        "closure_by_pt": summary_slices,
        "outputs": {"npz": result_npz.name, "plot": plot_path.name},
        "physics_notes": [
            "Templates use half-A matched response from jackknife-inverted PYTHIA tenths.",
            "Observed spectrum is half-B matched reco; reference truth is half-B matched gen.",
            "No fake correction, misses, detector systematics, or regularization are included in this specific closure test.",
        ],
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return plot_path


def main() -> None:
    args = parse_args()
    groomed = args.mode == "groomed"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "zjet" / "rho" / "combine_full" / f"{args.tag}_{args.mode}_5050"
    output_dir = output_dir.resolve()
    stem = f"zjet_rho_{args.tag}_{args.mode}_combine"

    unfolder = build_unfolder(args.tag, groomed)
    response_a, response_b = reconstruct_halves(unfolder)
    card_summary = write_shapes_and_card(output_dir, stem, response_a, response_b, unfolder)
    (output_dir / f"{stem}_card_summary.json").write_text(json.dumps(card_summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Combine 50:50 card: {output_dir / card_summary['card']}")
    print(f"Half-A template sum: {card_summary['template_sum']:.6g}")
    print(f"Half-B observed sum: {card_summary['observed_sum']:.6g}")

    if args.run_docker:
        run_in_docker(output_dir, stem, card_summary["n_gen"], args)
        plot_path = write_results_and_plot(output_dir, stem, unfolder, card_summary)
        print(f"Wrote plot: {plot_path}")
    else:
        print("Use --run-docker to build the workspace, run Combine, and make the plot.")


if __name__ == "__main__":
    main()
