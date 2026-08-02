#!/usr/bin/env python3
"""Build and run a Combine ML unfolding model for Z+jet rho.

This is the full-physics counterpart to ``run_combine_rho_smoke.py``:

* uses the full reco x gen response mosaic from the existing ``Unfolder`` input
  preparation, including cross-pT migrations;
* fits the raw reco data by scaling matched-response templates with the nominal
  fake-survival correction, matching the current TUnfold fake-corrected-data
  convention without changing the data histogram;
* converts fitted gen-bin scale factors back to the full particle-level
  spectrum with the nominal miss correction;
* writes detector/systematic response variations as Combine shape nuisances;
* runs two fits, one with nuisances frozen (data-stat component) and one with
  nuisances profiled (stat+syst component), then propagates the fitted POI
  covariance to the normalized rho spectrum.

Physics caveats:
  - Combine does not apply the TUnfold ratio-curvature regularization here.
    The model uncertainty is represented by the HERWIG response shape nuisance.
  - The covariance split is ``total - stat`` from two profile fits. This is the
    standard practical decomposition but is not guaranteed positive definite
    bin-by-bin in ill-conditioned spectra.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import uproot

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

DEFAULT_CONTAINER = "bgttbar-2dalphabet"
DEFAULT_CMSENV = "/opt/bgttbar-docker/cmsenv.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", default="original_jacobian_reg")
    parser.add_argument("--mode", choices=("groomed", "ungroomed"), default="groomed")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--cmsenv", default=DEFAULT_CMSENV)
    parser.add_argument("--run-docker", action="store_true")
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument(
        "--data-mode",
        choices=("data", "self-closure"),
        default="data",
        help=(
            "Observed spectrum in the datacard. 'data' uses the real reco data; "
            "'self-closure' uses the nominal PYTHIA reco prediction from the "
            "same Combine templates, so the expected fitted POIs are one."
        ),
    )
    parser.add_argument(
        "--systematics",
        default="all",
        help=(
            "Comma-separated nuisance bases to include, or 'all', or 'none'. "
            "Examples: herwig,JMS,JMR,pu. Default: all paired variations."
        ),
    )
    parser.add_argument(
        "--max-nuisances",
        type=int,
        default=None,
        help="Debug limiter after filtering. Omit for the full nuisance model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each Combine fit inside Docker.",
    )
    parser.add_argument(
        "--skip-unfolder-output",
        action="store_true",
        help="Do not call plotting/output methods on the existing TUnfold object.",
    )
    return parser.parse_args()


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def variation_base(name: str) -> tuple[str, str] | None:
    """Return (base, direction) for names like JES_FooUp_corr or puDown."""
    for direction in ("Up", "Down"):
        if direction in name:
            return name.replace(direction, "", 1), direction
    return None


def selected_pairs(systematics: list[str], selector: str, max_nuisances: int | None) -> list[tuple[str, str, str]]:
    if selector.strip().lower() == "none":
        return []
    requested = None
    if selector.strip().lower() != "all":
        requested = {item.strip().lower() for item in selector.split(",") if item.strip()}

    by_base: dict[str, dict[str, str]] = {}
    for name in systematics:
        if name == "nominal":
            continue
        parsed = variation_base(name)
        if parsed is None:
            continue
        base, direction = parsed
        if requested is not None and not any(
            base.lower().startswith(prefix) or name.lower().startswith(prefix)
            for prefix in requested
        ):
            continue
        by_base.setdefault(base, {})[direction] = name

    pairs = [
        (sanitize(base), directions["Up"], directions["Down"])
        for base, directions in sorted(by_base.items())
        if "Up" in directions and "Down" in directions
    ]
    if max_nuisances is not None:
        pairs = pairs[:max_nuisances]
    return pairs


def docker_is_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "running"


def build_unfolder(tag: str, groomed: bool):
    import matplotlib

    matplotlib.use("Agg")

    from unfold.tools.unfolder_core import Unfolder, get_spec

    spec = get_spec("zjet", "rho", tag)
    spec = replace(
        spec,
        output_dir=str(REPO_ROOT / "outputs" / "zjet" / "rho" / "combine_full" / tag) + "/",
    )
    return Unfolder(spec, groomed=groomed, do_syst=True, compute_jackknife_stat=False)


def safe_template(values: np.ndarray, *, floor: float = 1.0e-9) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(values, dtype=float)
    diagnostics = {
        "negative_bins": int(np.count_nonzero(raw < 0.0)),
        "zero_or_negative_bins": int(np.count_nonzero(raw <= 0.0)),
    }
    return np.clip(raw, floor, None), diagnostics


def template_matrix(unfolder, systematic: str) -> tuple[np.ndarray, dict[str, int]]:
    matrix = np.asarray(unfolder.mosaic_dict[systematic], dtype=float)
    fake_fraction = np.asarray(unfolder.fake_fraction_2d, dtype=float)
    if systematic in {"herwigUp", "herwigDown"} and hasattr(unfolder, "fake_fraction_2d_herwig"):
        fake_fraction = np.asarray(unfolder.fake_fraction_2d_herwig, dtype=float)
    survival = np.clip(1.0 - fake_fraction, 1.0e-6, None)
    # TUnfold corrects data by multiplying by survival. For an equivalent raw
    # data likelihood, scale the matched-response prediction by 1/survival.
    templates = matrix / survival[:, None]
    clipped, diagnostics = safe_template(templates)
    diagnostics["fake_survival_le_0"] = int(np.count_nonzero((1.0 - fake_fraction) <= 0.0))
    return clipped, diagnostics


def gen_offsets(edges_by_pt: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(edges) - 1 for edges in edges_by_pt], dtype=int)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def normalize_by_pt(values: np.ndarray, covariance: np.ndarray, edges_by_pt: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    normalized = np.zeros_like(values)
    jacobian = np.zeros((values.size, values.size), dtype=float)
    starts, counts = gen_offsets(edges_by_pt)
    for start, count, edges in zip(starts, counts, edges_by_pt):
        indices = np.arange(start, start + count)
        widths = np.diff(np.asarray(edges, dtype=float))
        block = values[indices]
        total = float(np.sum(block))
        if total <= 0.0:
            continue
        normalized[indices] = block / total / widths
        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                delta = 1.0 if local_i == local_j else 0.0
                jacobian[global_i, global_j] = (
                    (delta * total - block[local_i]) / (total * total) / widths[local_i]
                )
    return normalized, jacobian @ covariance @ jacobian.T


def write_shapes_and_card(
    unfolder,
    output_dir: Path,
    stem: str,
    nuisance_pairs: list[tuple[str, str, str]],
    *,
    data_mode: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_path = output_dir / f"{stem}_shapes.root"
    card_path = output_dir / f"{stem}_card.txt"
    meta_path = output_dir / f"{stem}_meta.npz"

    nominal_templates, nominal_diag = template_matrix(unfolder, "nominal")
    n_reco, n_gen = nominal_templates.shape
    if data_mode == "self-closure":
        observed = np.sum(nominal_templates, axis=1)
    else:
        observed = np.asarray(unfolder.mosaic_2d, dtype=float)
    reco_edges = np.arange(n_reco + 1, dtype=float)

    template_diagnostics: dict[str, dict[str, int]] = {"nominal": nominal_diag}
    with uproot.recreate(root_path) as root_file:
        root_file["data_obs"] = (observed, reco_edges)
        root_file["dummy_bkg"] = (np.full(n_reco, 1.0e-6), reco_edges)
        for gen_index in range(n_gen):
            process = f"gen_{gen_index:03d}"
            root_file[process] = (nominal_templates[:, gen_index], reco_edges)

        for nuisance, up_key, down_key in nuisance_pairs:
            for direction, key in (("Up", up_key), ("Down", down_key)):
                varied_templates, diag = template_matrix(unfolder, key)
                template_diagnostics[f"{nuisance}{direction}"] = diag
                for gen_index in range(n_gen):
                    root_file[f"gen_{gen_index:03d}__{nuisance}{direction}"] = (
                        varied_templates[:, gen_index],
                        reco_edges,
                    )

    processes = [f"gen_{index:03d}" for index in range(n_gen)]
    names = ["dummy_bkg"] + processes
    process_ids = ["1"] + [str(-index) for index in range(n_gen)]

    with card_path.open("w") as handle:
        handle.write("imax 1 number of channels\n")
        handle.write("jmax * number of processes minus one\n")
        handle.write("kmax * number of nuisance parameters\n")
        handle.write("---------------\n")
        handle.write(f"shapes * rho_full {root_path.name} $PROCESS $PROCESS__$SYSTEMATIC\n")
        handle.write(f"shapes data_obs rho_full {root_path.name} data_obs\n")
        handle.write("---------------\n")
        handle.write("bin rho_full\n")
        handle.write("observation -1\n")
        handle.write("---------------\n")
        handle.write("bin " + " ".join(["rho_full"] * len(names)) + "\n")
        handle.write("process " + " ".join(names) + "\n")
        handle.write("process " + " ".join(process_ids) + "\n")
        handle.write("rate " + " ".join(["-1"] * len(names)) + "\n")
        handle.write("---------------\n")
        for nuisance, _, _ in nuisance_pairs:
            handle.write(nuisance + " shape " + " ".join(["-"] + ["1"] * n_gen) + "\n")
        if nuisance_pairs:
            handle.write("syst group = " + " ".join(n for n, _, _ in nuisance_pairs) + "\n")

    matched_gen = np.asarray(unfolder.mosaic.sum(axis=0), dtype=float)
    full_truth = matched_gen + np.asarray(unfolder.misses_2d, dtype=float)
    start_values = np.ones(n_gen, dtype=float)
    if hasattr(unfolder, "y_unf"):
        with np.errstate(divide="ignore", invalid="ignore"):
            start_values = np.divide(
                np.asarray(unfolder.y_unf, dtype=float),
                full_truth,
                out=np.ones_like(full_truth),
                where=full_truth > 0,
            )
        start_values = np.clip(start_values, 0.0, 8.0)

    np.savez(
        meta_path,
        observed=observed,
        nominal_templates=nominal_templates,
        matched_gen=matched_gen,
        misses=np.asarray(unfolder.misses_2d, dtype=float),
        full_truth=full_truth,
        start_values=start_values,
        data_mode=np.asarray(data_mode),
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
        "data_mode": data_mode,
        "nominal_template_sum": float(np.sum(nominal_templates)),
        "full_truth_sum": float(np.sum(full_truth)),
        "nuisances": [{"name": n, "up": u, "down": d} for n, u, d in nuisance_pairs],
        "template_diagnostics": template_diagnostics,
    }


def docker_command(stem: str, n_gen: int, r_min: float, r_max: float, cmsenv: str, freeze_syst: bool, timeout: int) -> str:
    workspace = f"{stem}_workspace.root"
    fit_name = f"{stem}_{'statonly' if freeze_syst else 'total'}"
    freeze = "--freezeNuisanceGroups syst" if freeze_syst else ""
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
        f"> text2workspace_{'statonly' if freeze_syst else 'total'}.log 2>&1; "
        f"{poi_list}"
        f"{start_params}"
        f"timeout {timeout:d} combine -M MultiDimFit {workspace} --algo none "
        "--redefineSignalPOIs \"$pois\" --floatOtherPOIs 1 --saveFitResult "
        "--X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerStrategy 0 "
        f"--setParameters \"$starts\" {freeze} -n .{fit_name} "
        f"> combine_{'statonly' if freeze_syst else 'total'}.log 2>&1; "
        "python3 extract_fit.py "
        f"multidimfit.{fit_name}.root {fit_name}.json"
    )


def write_extractor(output_dir: Path) -> None:
    extractor = output_dir / "extract_fit.py"
    extractor.write_text(
        """#!/usr/bin/env python3
import json
import sys
import ROOT

root_path, out_path = sys.argv[1], sys.argv[2]
handle = ROOT.TFile.Open(root_path)
fit = handle.Get("fit_mdf")
if not fit:
    raise SystemExit(f"missing fit_mdf in {root_path}")
pars = fit.floatParsFinal()
names, values, errors = [], [], []
for i in range(pars.getSize()):
    obj = pars.at(i)
    name = obj.GetName()
    if name.startswith("r_gen_"):
        names.append(name)
        values.append(float(obj.getVal()))
        errors.append(float(obj.getError()))
order = sorted(range(len(names)), key=lambda i: names[i])
names = [names[i] for i in order]
values = [values[i] for i in order]
errors = [errors[i] for i in order]
cov = fit.covarianceMatrix()
covariance = []
for i_name in names:
    row = []
    for j_name in names:
        row.append(float(cov[pars.index(i_name)][pars.index(j_name)]))
    covariance.append(row)
payload = {
    "status": int(fit.status()),
    "covQual": int(fit.covQual()),
    "edm": float(fit.edm()),
    "minNll": float(fit.minNll()),
    "names": names,
    "values": values,
    "errors": errors,
    "covariance": covariance,
}
with open(out_path, "w") as handle_out:
    json.dump(payload, handle_out, indent=2, sort_keys=True)
    handle_out.write("\\n")
""",
    )


def run_in_docker(output_dir: Path, stem: str, n_gen: int, args: argparse.Namespace) -> None:
    if not docker_is_running(args.container):
        raise RuntimeError(f"Docker container {args.container!r} is not running")
    write_extractor(output_dir)
    container_dir = f"/tmp/unfold_{stem}"
    subprocess.run(["docker", "exec", args.container, "rm", "-rf", container_dir], check=True)
    subprocess.run(["docker", "exec", args.container, "mkdir", "-p", container_dir], check=True)
    subprocess.run(["docker", "cp", str(output_dir) + "/.", f"{args.container}:{container_dir}/"], check=True)

    for freeze in (True, False):
        command = docker_command(stem, n_gen, args.r_min, args.r_max, args.cmsenv, freeze, args.timeout)
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


def load_fit(path: Path, n_gen: int) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = json.loads(path.read_text())
    values = np.ones(n_gen, dtype=float)
    for name, value in zip(payload["names"], payload["values"]):
        values[int(name.rsplit("_", 1)[1])] = float(value)
    covariance = np.asarray(payload["covariance"], dtype=float)
    return values, covariance, payload


def write_results_and_plot(output_dir: Path, stem: str, unfolder, card_summary: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_gen = card_summary["n_gen"]
    stat_values, cov_r_stat, stat_payload = load_fit(output_dir / f"{stem}_statonly.json", n_gen)
    total_values, cov_r_total, total_payload = load_fit(output_dir / f"{stem}_total.json", n_gen)
    full_truth = np.load(output_dir / f"{stem}_meta.npz", allow_pickle=True)["full_truth"]

    central = total_values * full_truth
    cov_abs_stat = np.diag(full_truth) @ cov_r_stat @ np.diag(full_truth)
    cov_abs_total = np.diag(full_truth) @ cov_r_total @ np.diag(full_truth)
    cov_abs_syst = cov_abs_total - cov_abs_stat
    norm, cov_norm_total = normalize_by_pt(central, cov_abs_total, unfolder.gen_edges_by_pt)
    _, cov_norm_stat = normalize_by_pt(stat_values * full_truth, cov_abs_stat, unfolder.gen_edges_by_pt)
    cov_norm_syst = cov_norm_total - cov_norm_stat

    result_npz = output_dir / f"{stem}_combine_unfolded.npz"
    np.savez(
        result_npz,
        central=central,
        normalized=norm,
        cov_abs_stat=cov_abs_stat,
        cov_abs_syst=cov_abs_syst,
        cov_abs_total=cov_abs_total,
        cov_norm_stat=cov_norm_stat,
        cov_norm_syst=cov_norm_syst,
        cov_norm_total=cov_norm_total,
        poi_values=total_values,
        poi_values_statonly=stat_values,
        pt_edges=np.asarray(unfolder.pt_edges, dtype=float),
        gen_edges_json=np.asarray(json.dumps(unfolder.gen_edges_by_pt), dtype=object),
    )

    mode = "groomed" if unfolder.groomed else "ungroomed"
    starts, counts = gen_offsets(unfolder.gen_edges_by_pt)
    n_panels = len(starts) - 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.2), sharey=False)
    if n_panels == 1:
        axes = [axes]
    for ax, ipt in zip(axes, range(1, len(starts))):
        start, count = starts[ipt], counts[ipt]
        idx = np.arange(start, start + count)
        edges = np.asarray(unfolder.gen_edges_by_pt[ipt], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        y = norm[idx]
        stat = np.sqrt(np.clip(np.diag(cov_norm_stat)[idx], 0.0, None))
        total = np.sqrt(np.clip(np.diag(cov_norm_total)[idx], 0.0, None))
        ax.fill_between(centers, y - total, y + total, step="mid", color="#5790fc", alpha=0.28, label="stat+syst")
        ax.errorbar(centers, y, yerr=stat, xerr=0.5 * widths, fmt="o", color="#1f2937", markersize=4, label="data stat")
        hi = int(unfolder.pt_edges[ipt + 1]) if ipt + 1 < len(unfolder.pt_edges) - 1 else None
        title = f"{int(unfolder.pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+") + " GeV"
        ax.set_title(title)
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        ax.grid(True, alpha=0.25)
        if ipt == 1:
            ax.set_ylabel(unfolder.spec.normalized_ylabel)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Z+jet rho Combine ML unfolding ({mode})", y=1.02)
    fig.tight_layout()
    plot_path = output_dir / f"{stem}_combine_unfolded.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "workflow": "Z+jet rho Combine ML unfolding",
        "mode": mode,
        "card": card_summary,
        "fit_statonly": {k: stat_payload[k] for k in ("status", "covQual", "edm", "minNll")},
        "fit_total": {k: total_payload[k] for k in ("status", "covQual", "edm", "minNll")},
        "outputs": {
            "npz": result_npz.name,
            "plot": plot_path.name,
        },
        "physics_notes": [
            "Raw reco data are fitted; matched-response templates are scaled by 1/(1-fake_fraction).",
            "Fitted POIs scale nominal full truth = matched gen + misses for the reported unfolded spectrum.",
            "Detector/model response variations enter as shape nuisances; stat-only is a refit with the nuisance group frozen.",
            "No TUnfold regularization term is included in this Combine likelihood.",
        ],
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return plot_path


def main() -> None:
    args = parse_args()
    mode = args.mode
    groomed = mode == "groomed"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "zjet" / "rho" / "combine_full" / f"{args.tag}_{mode}"
    output_dir = output_dir.resolve()
    stem = f"zjet_rho_{args.tag}_{mode}_combine"

    unfolder = build_unfolder(args.tag, groomed)
    if args.skip_unfolder_output:
        pass

    pairs = selected_pairs(list(unfolder.systematics), args.systematics, args.max_nuisances)
    card_summary = write_shapes_and_card(
        unfolder,
        output_dir,
        stem,
        pairs,
        data_mode=args.data_mode,
    )
    (output_dir / f"{stem}_card_summary.json").write_text(json.dumps(card_summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Combine card: {output_dir / card_summary['card']}")
    print(f"Nuisance pairs: {len(pairs)}")

    if args.run_docker:
        run_in_docker(output_dir, stem, card_summary["n_gen"], args)
        plot_path = write_results_and_plot(output_dir, stem, unfolder, card_summary)
        print(f"Wrote plot: {plot_path}")
    else:
        print("Use --run-docker to build the workspace, run Combine, and make the plot.")


if __name__ == "__main__":
    main()
