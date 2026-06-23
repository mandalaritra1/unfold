#!/usr/bin/env python3
"""Unified unfolding runner across the channel x observable matrix.

    run_unfolding.py --channel {zjet,dijet,trijet} --observable {rho,mass} [...]

- zjet rho/mass run from the merged-era spec inputs (inputs/zjet/...) through
  the shared Unfolder (Unfolder(spec, groomed).run_all_plots()).
- dijet/trijet rho run from the prepared channel inputs (inputs/<channel>/rho/)
  and are delegated to run_rho_unfolding.py, the producer-compatible path.

Availability is driven by unfold.tools.unfolder_core.CHANNEL_OBSERVABLES.
ROOT must be importable (source scripts/setup_root.sh) for an actual run; only
``--help`` works without it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

CHANNELS = ("zjet", "dijet", "trijet")
OBSERVABLES = ("rho", "mass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--channel", choices=CHANNELS, required=True)
    parser.add_argument("--observable", choices=OBSERVABLES, default="rho")
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "zjet spec tag, e.g. rho: original|fixed_jec|fixed_miss, mass: "
            "nominal (defaults per DEFAULT_TAGS). Every tag also has a "
            "'<tag>_jacobian' twin with Jacobian-propagated normalized "
            "statistics, and a '<tag>_jacobian_reg' twin that additionally "
            "turns on ratio-curvature regularization (L-curve tau). Each "
            "writes to a sibling output dir. Ignored elsewhere."
        ),
    )
    parser.add_argument(
        "--jacobian",
        action="store_true",
        help=(
            "Propagate the normalized-result statistics through the "
            "normalization Jacobian (errors + correlation matrix of the "
            "normalized spectrum). Equivalent to picking the '<tag>_jacobian' "
            "twin; the output dir gets a '_jacobian' suffix when this flag "
            "changes the spec."
        ),
    )
    parser.add_argument(
        "--regularization",
        choices=("none", "ratio_curvature"),
        default=None,
        help=(
            "Unfolding regularization. 'ratio_curvature' penalizes the "
            "curvature of x/x_MC per pT slice (zero penalty for spectra "
            "proportional to the MC prior); tau from an L-curve scan unless "
            "--tau is given. The output dir gets a '_reg' suffix when this "
            "flag changes the spec. Default: whatever the tag specifies."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help=(
            "Fixed regularization strength, skipping the L-curve scan "
            "(requires a regularized spec or --regularization). Does not "
            "change the output dir; combine with --output-dir to keep "
            "scanned-tau results."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("tunfold", "roounfold_bayes"),
        default=None,
        help=(
            "Unfolding backend. 'roounfold_bayes' uses iterative Bayes "
            "(D'Agostini) via RooUnfold with --n-iter iterations, reusing the "
            "jackknife for the statistical uncertainty; needs a built "
            "libRooUnfold (see scripts/setup_roounfold.sh). The output dir gets "
            "a '_bayes' suffix. Default: the tag's method (tunfold)."
        ),
    )
    parser.add_argument(
        "--n-iter", type=int, default=None,
        help="D'Agostini iterations for --method roounfold_bayes (default 4).",
    )
    parser.add_argument("--year", default="2018", help="dijet/trijet only")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cms-label", default="Internal")
    parser.add_argument("--lumi", type=float, default=None)
    parser.add_argument("--com", type=float, default=13.0)
    parser.add_argument(
        "--no-syst", action="store_true", help="zjet: skip systematic variations"
    )
    parser.add_argument("--no-gallery", action="store_true")
    return parser.parse_args()


def build_gallery(output_dir: Path, observable: str) -> None:
    builder = "build_mass_gallery.py" if observable == "mass" else "build_rho_gallery.py"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "outputs" / builder), "--root", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )


def resolve_zjet_spec(args: argparse.Namespace):
    """Resolve the tag to a spec and apply the option flags.

    Flags that change the spec also suffix the output dir (mirroring the
    '<tag>_jacobian' / '<tag>_jacobian_reg' registry twins) so option runs
    never overwrite the base tag's outputs. An explicit --output-dir wins.
    """
    from dataclasses import replace

    from unfold.tools.unfolder_core import get_spec

    spec = get_spec("zjet", args.observable, args.tag)

    suffix = ""
    if args.jacobian and spec.stat_propagation != "jacobian":
        spec = replace(spec, stat_propagation="jacobian")
        suffix += "_jacobian"
    if args.regularization is not None and spec.regularization != args.regularization:
        spec = replace(spec, regularization=args.regularization)
        # suffix in both directions so option runs never overwrite the tag's
        # own outputs (e.g. '..._jacobian_reg' run with --regularization none)
        suffix += "_reg" if args.regularization != "none" else "_noreg"
    if args.tau is not None:
        if spec.regularization == "none":
            sys.exit("--tau requires a regularized spec (use --regularization "
                     "ratio_curvature or a '<tag>_jacobian_reg' tag).")
        spec = replace(spec, tau=args.tau)
    if args.method is not None and args.method != getattr(spec, "method", "tunfold"):
        spec = replace(spec, method=args.method)
        if args.method == "roounfold_bayes":
            suffix += "_bayes"
    if args.n_iter is not None:
        spec = replace(spec, n_iter=args.n_iter)
    if suffix:
        spec = replace(spec, output_dir=spec.output_dir.rstrip("/") + suffix + "/")

    if args.output_dir is not None:
        rel = os.path.relpath(args.output_dir.resolve(), REPO_ROOT) + "/"
        spec = replace(spec, output_dir=rel)
    return spec


def run_zjet(args: argparse.Namespace) -> Path:
    """Run the zjet spec-input path for both grooming modes."""
    import matplotlib

    matplotlib.use("Agg")

    from unfold.tools.unfolder_core import Unfolder

    spec = resolve_zjet_spec(args)
    output_dir = (REPO_ROOT / spec.output_dir).resolve()

    for mode, groomed in (("ungroomed", False), ("groomed", True)):
        unfolder = Unfolder(
            spec, groomed, do_syst=not args.no_syst, cms_label=args.cms_label
        )
        if args.lumi is not None:
            unfolder.lumi = args.lumi
        unfolder.com = args.com
        unfolder.run_all_plots(show=False)
        print(f"[zjet {args.observable} {mode}] wrote plots to {spec.output_dir}")

    if not args.no_gallery:
        build_gallery(output_dir, args.observable)
    return output_dir


def run_channel(args: argparse.Namespace) -> None:
    """Delegate dijet/trijet rho to the producer-compatible runner."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rho_unfolding.py"),
        "--channel", args.channel,
        "--year", str(args.year),
        "--cms-label", args.cms_label,
        "--com", str(args.com),
    ]
    if args.lumi is not None:
        cmd += ["--lumi", str(args.lumi)]
    if args.output_dir is not None:
        cmd += ["--output-dir", str(args.output_dir)]
    if args.jacobian:
        cmd += ["--jacobian"]
    if args.regularization is not None:
        cmd += ["--regularization", args.regularization]
    if args.tau is not None:
        cmd += ["--tau", str(args.tau)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    from unfold.tools.unfolder_core import CHANNEL_OBSERVABLES

    how = CHANNEL_OBSERVABLES.get((args.channel, args.observable))
    if how is None:
        runnable = [f"{c}/{o}" for (c, o), v in CHANNEL_OBSERVABLES.items() if v]
        sys.exit(
            f"({args.channel}, {args.observable}) is not available. "
            f"Runnable: {', '.join(runnable)}"
        )
    if how == "spec":
        run_zjet(args)
    else:  # "channel_inputs"
        if args.observable != "rho":
            sys.exit(f"{args.channel} only supports rho (channel-inputs path).")
        run_channel(args)


if __name__ == "__main__":
    main()
