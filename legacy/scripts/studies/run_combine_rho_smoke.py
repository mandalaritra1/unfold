#!/usr/bin/env python3
"""Build and optionally run a Combine ML-fit smoke test for Z+jet rho unfolding.

This is intentionally a *technical* Combine prototype, not a replacement for the
production TUnfold/RooUnfold workflow.  The default card unfolds one rho spectrum
inside a single pT slice using the same-pT response block.  Cross-pT migrations,
fake corrections, miss corrections, and all systematic nuisances are omitted on
purpose so the Combine mechanics can be tested quickly and visibly.

The likelihood model follows the Combine unfolding tutorial pattern: each gen
rho bin is represented as a signal process, and `multiSignalModel` maps each
process to its own POI (`r_gen_000`, ...).  The fitted POI is the data/nominal-MC
scale factor for that gen-bin template under the simplified model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import uproot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTAINER = "bgttbar-2dalphabet"
DEFAULT_CMSENV = "/opt/bgttbar-docker/cmsenv.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="original", help="Z+jet rho input tag")
    parser.add_argument("--mode", choices=("groomed", "ungroomed"), default="groomed")
    parser.add_argument(
        "--pt-index",
        type=int,
        default=1,
        help="pT-bin index to unfold inside; default 1 = 200-290 GeV for current zjet rho inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/zjet/rho/combine_smoke/<tag>_<mode>_pt<idx>",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--cmsenv", default=DEFAULT_CMSENV)
    parser.add_argument("--run-docker", action="store_true", help="Run text2workspace.py and combine in Docker")
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=10.0)
    return parser.parse_args()


def nominal_values(hist_obj) -> np.ndarray:
    values = hist_obj.values(flow=False)
    if "systematic" in hist_obj.axes.name:
        systematic_axis = hist_obj.axes.name.index("systematic")
        # The rho producer stores nominal as the first systematic category.
        values = np.take(values, 0, axis=systematic_axis)
    return np.asarray(values, dtype=float)


def load_pt_slice(tag: str, mode: str, pt_index: int) -> dict[str, np.ndarray]:
    key_suffix = "g" if mode == "groomed" else "u"
    input_dir = REPO_ROOT / "inputs" / "zjet" / "rho" / tag
    data_path = input_dir / "data_all.pkl"
    mc_path = input_dir / "pythia_all.pkl"
    if not data_path.exists() or not mc_path.exists():
        raise FileNotFoundError(f"Missing inputs under {input_dir}")

    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    with mc_path.open("rb") as handle:
        mc = pickle.load(handle)

    data_hist = data[f"ptjet_rhojet_{key_suffix}_reco"]
    response_hist = mc[f"response_matrix_rho_{key_suffix}"]
    gen_hist = mc[f"ptjet_rhojet_{key_suffix}_gen"]

    data_values = nominal_values(data_hist).sum(axis=0)  # sum data-taking datasets
    response = nominal_values(response_hist)
    gen = nominal_values(gen_hist)

    if pt_index < 0 or pt_index >= response.shape[0] or pt_index >= response.shape[2]:
        raise ValueError(f"pt-index {pt_index} is outside response shape {response.shape}")

    observed_all = data_values[pt_index, :]
    # Smoke-test simplification: use only the same-pT response block.
    response_rates_all = response[pt_index, :, pt_index, :]
    gen_nominal = gen[pt_index, :]

    keep_reco = response_rates_all.sum(axis=0) > 0.0
    if not np.any(keep_reco):
        raise RuntimeError("Selected pT slice has no nonzero response columns")

    return {
        "observed": observed_all[keep_reco],
        "observed_all": observed_all,
        "response_rates": response_rates_all[:, keep_reco],
        "response_rates_all": response_rates_all,
        "gen_nominal": gen_nominal,
        "keep_reco": keep_reco,
        "pt_edges": np.asarray(response_hist.axes["ptgen"].edges, dtype=float),
        "rho_gen_edges": np.asarray(response_hist.axes["mpt_gen"].edges, dtype=float),
        "rho_reco_edges": np.asarray(response_hist.axes["mpt_reco"].edges, dtype=float),
    }


def write_shape_card(payload: dict[str, np.ndarray], output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = payload["observed"].astype(float)
    response_rates = payload["response_rates"].astype(float)
    edges = np.arange(len(observed) + 1, dtype=float)

    root_path = output_dir / f"{stem}_shapes.root"
    card_path = output_dir / f"{stem}_card.txt"
    meta_path = output_dir / f"{stem}_meta.npz"

    with uproot.recreate(root_path) as root_file:
        root_file["data_obs"] = (observed, edges)
        # Combine's datacard parser expects at least one background process.
        # This tiny template is numerically irrelevant for the smoke test.
        root_file["dummy_bkg"] = (np.full_like(observed, 1.0e-6), edges)
        for gen_index, values in enumerate(response_rates):
            root_file[f"gen_{gen_index:03d}"] = (values, edges)

    processes = [f"gen_{index:03d}" for index in range(response_rates.shape[0])]
    names = ["dummy_bkg"] + processes
    process_ids = ["1"] + [str(-index) for index in range(len(processes))]

    with card_path.open("w") as handle:
        handle.write("imax 1 number of channels\n")
        handle.write("jmax * number of processes minus one\n")
        handle.write("kmax * number of nuisance parameters\n")
        handle.write("---------------\n")
        handle.write(f"shapes * rho_pt {root_path.name} $PROCESS\n")
        handle.write(f"shapes data_obs rho_pt {root_path.name} data_obs\n")
        handle.write("---------------\n")
        handle.write("bin rho_pt\n")
        handle.write("observation -1\n")
        handle.write("---------------\n")
        handle.write("bin " + " ".join(["rho_pt"] * len(names)) + "\n")
        handle.write("process " + " ".join(names) + "\n")
        handle.write("process " + " ".join(process_ids) + "\n")
        handle.write("rate " + " ".join(["-1"] * len(names)) + "\n")
        handle.write("---------------\n")
        handle.write("* autoMCStats 0\n")

    np.savez(meta_path, **payload)
    return card_path, root_path, meta_path


def docker_is_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "running"


def run_combine_in_docker(
    output_dir: Path,
    card_path: Path,
    stem: str,
    n_gen: int,
    container: str,
    cmsenv: str,
    r_min: float,
    r_max: float,
) -> None:
    if not docker_is_running(container):
        raise RuntimeError(f"Docker container {container!r} is not running")

    container_dir = f"/tmp/unfold_{stem}"
    subprocess.run(["docker", "exec", container, "rm", "-rf", container_dir], check=True)
    subprocess.run(["docker", "cp", str(output_dir), f"{container}:{container_dir}"], check=True)

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
    workspace = f"{stem}_workspace.root"
    fit_name = stem
    command = (
        f"source {cmsenv} >/dev/null; cd {container_dir}; "
        f"{poi_loop}"
        f"text2workspace.py {card_path.name} "
        "-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel "
        f"\"${{po[@]}}\" -o {workspace} > text2workspace.log 2>&1; "
        f"{poi_list}"
        f"combine -M MultiDimFit {workspace} --algo none "
        "--redefineSignalPOIs \"$pois\" --floatOtherPOIs 1 --saveFitResult "
        f"--cminDefaultMinimizerStrategy 0 -n .{fit_name} > combine_fit.log 2>&1"
    )
    subprocess.run(["docker", "exec", "-i", container, "bash", "-lc", command], check=True)

    tmp_copy = output_dir.parent / f".{output_dir.name}.docker_copy"
    if tmp_copy.exists():
        shutil.rmtree(tmp_copy)
    subprocess.run(["docker", "cp", f"{container}:{container_dir}/.", str(tmp_copy)], check=True)
    for item in tmp_copy.iterdir():
        target = output_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), target)
    tmp_copy.rmdir()


def extract_fit_summary(output_dir: Path, stem: str, payload: dict[str, np.ndarray]) -> Path | None:
    matches = sorted(output_dir.glob(f"higgsCombine.{stem}.MultiDimFit.mH*.root"))
    if not matches:
        return None
    fit_root = matches[-1]
    with uproot.open(fit_root) as root_file:
        tree = root_file["limit"]
        poi_names = [name for name in tree.keys() if name.startswith("r_gen_")]
        arrays = tree.arrays(poi_names + ["deltaNLL"], library="np")

    fit_values = {name: float(arrays[name][0]) for name in poi_names}
    gen_nominal = payload["gen_nominal"]
    unfolded = np.array([fit_values[f"r_gen_{i:03d}"] for i in range(len(gen_nominal))]) * gen_nominal
    summary = {
        "fit_root": fit_root.name,
        "deltaNLL": float(arrays["deltaNLL"][0]),
        "observed_sum": float(payload["observed"].sum()),
        "response_sum": float(payload["response_rates"].sum()),
        "gen_nominal_sum": float(gen_nominal.sum()),
        "fitted_scale_factors": fit_values,
        "fitted_unfolded_sum_same_pt_model": float(unfolded.sum()),
        "physics_caveat": (
            "Smoke test only: same-pT response block, no fake/miss corrections, "
            "no cross-pT migrations, no systematic nuisances."
        ),
    }
    summary_path = output_dir / f"{stem}_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "zjet" / "rho" / "combine_smoke" / f"{args.tag}_{args.mode}_pt{args.pt_index}"
    output_dir = output_dir.resolve()
    stem = f"zjet_rho_{args.tag}_{args.mode}_pt{args.pt_index}"

    payload = load_pt_slice(args.tag, args.mode, args.pt_index)
    card_path, root_path, meta_path = write_shape_card(payload, output_dir, stem)
    print(f"wrote {card_path}")
    print(f"wrote {root_path}")
    print(f"wrote {meta_path}")
    print(
        "inputs: "
        f"observed={payload['observed'].sum():.3f}, "
        f"response={payload['response_rates'].sum():.3f}, "
        f"gen={payload['gen_nominal'].sum():.3f}"
    )

    if args.run_docker:
        run_combine_in_docker(
            output_dir=output_dir,
            card_path=card_path,
            stem=stem,
            n_gen=payload["response_rates"].shape[0],
            container=args.container,
            cmsenv=args.cmsenv,
            r_min=args.r_min,
            r_max=args.r_max,
        )
        summary_path = extract_fit_summary(output_dir, stem, payload)
        if summary_path is None:
            print("Combine finished, but no higgsCombine output ROOT file was found", file=sys.stderr)
            sys.exit(2)
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
