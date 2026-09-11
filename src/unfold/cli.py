"""Command line: ``unfold run``, ``unfold tags``, ``unfold gallery``.

    unfold run --channel zjet|dijet|trijet [--observable rho|mass] [--tag original] [options]

Run from the repository root with ROOT on the path (``source setup_root.sh``).
Every run writes ``run_manifest.json`` next to its outputs with the resolved
configuration, the command, the git revision and the input files.
"""

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

from unfold.config import (
    CHANNELS, ChannelTag, DEFAULT_TAG, OBSERVABLES, ObservableSpec, PairSplitTag, describe, get_tag,
    list_tags, option_suffix, with_options,
)
from unfold.paths import REPO_ROOT


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_describe():
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True,
                              text=True, cwd=REPO_ROOT, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def command_line():
    return shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])


def resolve_output_dir(tag, args):
    """<tag output dir><option suffix>, or the explicit --output-dir."""
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    base_reg = getattr(tag, "regularization", "none")
    suffix = option_suffix(jacobian=args.jacobian, regularization=args.regularization,
                           method=args.method, base_regularization=base_reg)
    if isinstance(tag, PairSplitTag) and args.stat_method is not None and args.stat_method != tag.stat_method:
        suffix += "_stat_" + args.stat_method
    return (REPO_ROOT / (tag.output_dir.rstrip("/") + suffix)).resolve()


def base_manifest(args, tag, output_dir):
    return {
        "channel": args.channel, "observable": args.observable, "tag": args.tag or DEFAULT_TAG,
        "command": command_line(), "git": git_describe(), "output_dir": str(output_dir),
        "configuration": describe(tag),
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
    }


# ---------------------------------------------------------------------------
# Z+jet (merged-era pickles)
# ---------------------------------------------------------------------------
def run_zjet(args, tag, output_dir):
    from unfold import plots
    from unfold.binning import ZJET_BINNINGS
    from unfold.engine import Unfolder
    from unfold.zjet_inputs import load_zjet_inputs

    spec = with_options(tag, jacobian=args.jacobian, regularization=args.regularization, tau=args.tau,
                        method=args.method, n_iter=args.n_iter, model_covariance_scope=args.model_covariance_scope)
    spec = replace(spec, output_dir=os.path.relpath(output_dir, REPO_ROOT) + "/")
    lumi = 138.0 if args.lumi is None else args.lumi
    manifest = base_manifest(args, spec, output_dir)
    manifest["modes"] = {}
    for groomed in grooming_modes(args):
        mode = "groomed" if groomed else "ungroomed"
        binning = ZJET_BINNINGS[spec.binning_name(groomed)]
        if spec.gen_merge_below is not None:
            binning = binning.merge_gen_below(spec.gen_merge_below)
        inputs = load_zjet_inputs(spec, groomed, binning, do_syst=not args.no_syst, era_split=args.era_split)
        u = Unfolder(inputs, spec, groomed, cms_label=args.cms_label, lumi=lumi, com=args.com).run()
        if args.no_validation_plots:
            u.has_validation_inputs = False
        plots.run_all_plots(u, show=False)
        manifest["modes"][mode] = {
            "systematics": list(u.systematics), "tau": float(u.tau or 0.0),
            "stat_uncertainty": "jackknife" if u.has_jackknife else "TUnfold input + matrix covariance",
            "binning": {"pt_edges": list(binning.pt_edges), "gen_edges_by_pt": [list(e) for e in binning.gen_edges_by_pt],
                        "reco_edges_by_pt": [list(e) for e in binning.reco_edges_by_pt]},
        }
        print(f"[zjet {args.observable} {mode}] wrote plots to {spec.output_dir}")
    manifest["inputs"] = sorted(str(p) for p in Path(spec.input_dir).glob("*.pkl"))
    return manifest


# ---------------------------------------------------------------------------
# dijet / trijet, single year (minimal_rho pickles under inputs/<channel>/rho/)
# ---------------------------------------------------------------------------
def run_channel_year(args, tag, output_dir):
    import numpy as np
    import ROOT
    from unfold import plots
    from unfold.channel_inputs import build_prepared_rho_inputs, discover_rho_channel_files, to_binning
    from unfold.config import RHO_BASE
    from unfold.engine import Unfolder
    from unfold.inputs import prepared_inputs

    files = discover_rho_channel_files(REPO_ROOT / "inputs", tag.channel, tag.year)
    prepared = build_prepared_rho_inputs(files)
    lumi = tag.lumi if args.lumi is None else args.lumi
    ROOT.gErrorIgnoreLevel = ROOT.kError
    manifest = base_manifest(args, tag, output_dir)
    manifest["inputs"] = {p.name: {"path": str(p.resolve()), "sha256": file_sha256(p)} for p in (files.data, files.mc)}
    manifest["systematics"] = prepared.systematics
    manifest["modes"] = {}
    for groomed in grooming_modes(args):
        mode = "groomed" if groomed else "ungroomed"
        binning = to_binning(prepared.binning[mode])
        spec = with_options(RHO_BASE, jacobian=args.jacobian, regularization=args.regularization, tau=args.tau,
                            method=args.method, n_iter=args.n_iter)
        spec = replace(
            spec,
            output_dir=os.path.relpath(output_dir, REPO_ROOT) + "/",
            xlim_lower_groomed=(binning.gen_edges_by_pt[0][0] if tag.channel == "dijet" and groomed
                                else RHO_BASE.xlim_lower_groomed),
        )
        systematics = ["nominal"] if args.no_syst else prepared.systematics
        inputs = prepared_inputs(spec, groomed, binning, mc_inputs=prepared.mc, data_inputs=prepared.data,
                                 systematics=systematics, herwig_inputs=prepared.herwig, first_reported_pt_bin=1)
        u = Unfolder(inputs, spec, groomed, cms_label=args.cms_label, lumi=lumi, com=args.com).run()
        plots.run_all_plots(u, show=False)
        artifact = output_dir / "artifacts" / f"{mode}_results.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        names = [n for n in u.systematics if n != "nominal"]
        np.savez_compressed(
            artifact, pt_edges=np.asarray(u.pt_edges, float), rho_edges_reco=np.asarray(u.edges, float),
            rho_edges_gen=np.asarray(u.edges_gen, float), response_mosaic=u.mosaic, measured=u.y_meas,
            measured_variances_raw=u.measured_variances, measured_variances_fake_corrected=u.corrected_measured_variances,
            unfolded=u.y_unf, unfolded_input_errors=np.sqrt(np.clip(np.diag(u.cov_data_np), 0.0, None)),
            truth_prior=u.y_true, folded=u.x_folded, fake_fraction=u.fake_fraction_2d, misses=u.misses_2d,
            covariance=u.cov_np, input_covariance=u.cov_data_np, systematic_names=np.asarray(names),
            systematic_unfolded=(np.stack([u.y_unf_dict[n] for n in names]) if names else np.empty((0, len(u.y_unf)))),
        )
        manifest["modes"][mode] = {"spec": describe(spec), "tau": float(u.tau or 0.0), "artifact": str(artifact),
                                   "binning": {"pt_edges": list(binning.pt_edges),
                                               "gen_edges_by_pt": [list(e) for e in binning.gen_edges_by_pt],
                                               "reco_edges_by_pt": [list(e) for e in binning.reco_edges_by_pt]}}
    return manifest


# ---------------------------------------------------------------------------
# dijet / trijet, Run 2 pair-split
# ---------------------------------------------------------------------------
def run_pairsplit(args, tag, output_dir):
    from unfold.pairsplit.run import PairSplitOptions, run_all

    if args.method not in (None, "tunfold"):
        sys.exit("the pair-split path has no RooUnfold backend")
    regularization = args.regularization if args.regularization is not None else tag.regularization
    if regularization == "ratio_curvature":
        sys.exit("the pair-split path supports --regularization none|curvature")
    tau = args.tau if args.tau is not None else tag.tau
    if tau is not None and regularization == "none":
        sys.exit("--tau requires --regularization curvature")
    options = PairSplitOptions(
        channel=tag.channel, grooming_mode=args.grooming_mode, binning=tag.binning,
        regularization=regularization, normalization_window=tag.normalization_window, tau=tau,
        systematics="nominal" if args.no_syst else tag.systematics, output_dir=output_dir,
        no_plots=args.no_plots, model_envelope=tag.model_envelope, model_covariance=tag.model_covariance,
        cms_label=args.cms_label, lumi=tag.lumi if args.lumi is None else args.lumi, com=args.com,
        command=command_line(),
        stat_method=args.stat_method or tag.stat_method,
        **({"jackknife_input_root": args.jackknife_input_root}
           if args.jackknife_input_root is not None else {}),
    )
    manifests = run_all(options)
    manifest = base_manifest(args, tag, output_dir)
    manifest["modes"] = {p.parent.name: str(p) for p in manifests}
    return manifest


def grooming_modes(args):
    return {"both": (False, True), "groomed": (True,), "ungroomed": (False,)}[args.grooming_mode]


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    from unfold.cms_plot import set_stamp

    set_stamp(not args.no_stamp)
    tag = get_tag(args.channel, args.observable, args.tag)
    if not isinstance(tag, PairSplitTag) and (
        args.stat_method is not None or args.jackknife_input_root is not None
    ):
        sys.exit("--stat-method and --jackknife-input-root apply to Run-2 dijet/trijet tags only")
    output_dir = resolve_output_dir(tag, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(tag, ObservableSpec):
        manifest = run_zjet(args, tag, output_dir)
    elif isinstance(tag, PairSplitTag):
        manifest = run_pairsplit(args, tag, output_dir)
    elif isinstance(tag, ChannelTag):
        manifest = run_channel_year(args, tag, output_dir)
    else:
        raise TypeError(type(tag))
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if not args.no_gallery and not args.no_plots:
        from unfold.gallery import build_gallery
        build_gallery(output_dir)
    print(f"done: {output_dir}")
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(prog="unfold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="unfold one channel / observable / tag")
    r.add_argument("--channel", choices=CHANNELS, required=True)
    r.add_argument("--observable", choices=OBSERVABLES, default="rho")
    r.add_argument("--tag", default=None, help=f"name in config.TAGS (default {DEFAULT_TAG!r}); see `unfold tags`")
    r.add_argument("--grooming-mode", choices=("both", "groomed", "ungroomed"), default="both")
    r.add_argument("--no-syst", action="store_true", help="nominal only (fast diagnostic run)")
    r.add_argument("--jacobian", action="store_true",
                   help="propagate the normalized-result statistics through the normalization Jacobian (Z+jet, 2018 tags)")
    r.add_argument("--regularization", choices=("none", "ratio_curvature", "curvature"), default=None)
    r.add_argument("--tau", type=float, default=None, help="fixed regularization strength (skips the L-curve scan)")
    r.add_argument("--method", choices=("tunfold", "roounfold_bayes"), default=None)
    r.add_argument("--stat-method", choices=("analytic", "jackknife"), default=None,
                   help="Run-2 dijet/trijet: jackknife by default, analytic if replica files are absent")
    r.add_argument("--jackknife-input-root", type=Path, default=None,
                   help="Run-2 replica campaign containing data/ and mc/; overrides UNFOLD_PAIRSPLIT_JACKKNIFE_INPUTS")
    r.add_argument("--n-iter", type=int, default=None, help="D'Agostini iterations for roounfold_bayes")
    r.add_argument("--era-split", choices=("sqrt", "linear"), default="sqrt",
                   help="Z+jet JES year-correlation split: 'sqrt' is the JetMET prescription (default); "
                        "'linear' reproduces the pre-2026-09-09 production")
    r.add_argument("--model-covariance-scope", choices=("global_shown", "global_all", "per_pt"), default=None)
    r.add_argument("--no-validation-plots", action="store_true", help="Z+jet: skip the reco-level data/MC input plots")
    r.add_argument("--output-dir", type=Path, default=None, help="override outputs/<channel>/<observable>/<tag>/")
    r.add_argument("--cms-label", default="Internal")
    r.add_argument("--lumi", type=float, default=None)
    r.add_argument("--com", type=float, default=13.0)
    r.add_argument("--no-stamp", action="store_true",
                   help="no provenance stamp (date | git revision | inputs) on the figures; for publication")
    r.add_argument("--no-plots", action="store_true", help="arrays and manifest only")
    r.add_argument("--no-gallery", action="store_true")
    r.set_defaults(func=run)

    t = sub.add_parser("tags", help="list the registered channel / observable / tag combinations")
    t.set_defaults(func=lambda a: print(list_tags()))

    g = sub.add_parser("gallery", help="rebuild the HTML gallery of an output directory")
    g.add_argument("--root", type=Path, required=True)
    g.set_defaults(func=lambda a: __import__("unfold.gallery", fromlist=["build_gallery"]).build_gallery(a.root))
    return parser


def main(argv=None):
    if os.environ.get("ROOTSYS"):
        sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
