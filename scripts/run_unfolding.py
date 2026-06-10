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
        help="zjet rho tag: original|fixed_jec (default: original). Ignored elsewhere.",
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


def run_zjet(args: argparse.Namespace) -> Path:
    """Run the zjet spec-input path for both grooming modes."""
    import matplotlib

    matplotlib.use("Agg")
    from dataclasses import replace

    from unfold.tools.unfolder_core import Unfolder, get_spec

    spec = get_spec("zjet", args.observable, args.tag)
    if args.output_dir is not None:
        rel = os.path.relpath(args.output_dir.resolve(), REPO_ROOT) + "/"
        spec = replace(spec, output_dir=rel)
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
