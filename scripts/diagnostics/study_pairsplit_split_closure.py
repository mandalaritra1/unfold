#!/usr/bin/env python3
"""Split-sample closure for the pair-split channels.

The simulation is divided into two statistically independent halves at the
producer (``rho_split`` mode of smp_jetmass_run2: chunk index parity on the
``jk`` axis).  Half A builds the response matrix, fake fractions, and misses;
the reconstruction-level spectrum of half B is unfolded through it exactly as
the data are (same prepared-input path, same fake correction, same TUnfold
configuration, same per-slice normalization), and the result is compared
with the generator-level truth of half B.  The nominal self-closure cannot
see input/response correlations because both come from the same events; this
test can.

Per channel and grooming mode this writes, under
``outputs/pairsplit_run2/validation_split_closure/<channel>_<mode>/``:

  - ``split_closure_<mode>_pt<i>.pdf``  per reported pT slice: normalized
                                        truth (half B) and unfolded (half B
                                        through half A); lower panel the
                                        deviation with the NOMINAL
                                        measurement's statistical band
  - ``split_closure_metrics.txt``       per-slice max |unfolded/truth - 1|,
                                        pull against the nominal stat
                                        uncertainty, pull against the test's
                                        own stat uncertainty
  - ``split_closure_metrics.json``      the same, machine readable

The nominal statistical band is read from the canonical run artifact
(``artifacts/<mode>_results.npz``: ``normalized_result`` and
``norm_cov_stat``), never recomputed.

Usage:
  source scripts/setup_root.sh
  .venv/bin/python scripts/diagnostics/study_pairsplit_split_closure.py \
      [--channel dijet trijet] [--grooming-mode groomed ungroomed] \
      [--split-root DIR] [--synthetic]

``--synthetic`` thins the nominal production into two halves bin by bin
(binomial on the effective event count) so the script can be exercised
without the split production; it is a code check, not a physics result, and
its outputs land in a ``*_synthetic`` directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")

import hist
import numpy as np

import run_pairsplit_unfolding as runner
from unfold.tools.pairsplit_run2_inputs import (
    JET_RADIUS,
    LEGACY_HISTOGRAM_KEYS,
    PAIR_SPLIT_ERAS,
    PAIR_SPLIT_FINE_AXES,
    PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS,
    PHYSICAL_RHO_DEFINITION,
    TRANSFORMED_COORDINATE_DEFINITION,
    TRANSFORMED_COORDINATE_NAME,
    PairSplitModeArrays,
    PairSplitRun2Inputs,
    PairSplitSourceFiles,
    _ARRAY_AXES,
    load_pairsplit_run2_inputs,
    prepare_pairsplit_inputs,
)
from unfold.utils.merge_helpers import unflatten_gen_by_pt

OUT_ROOT = REPO_ROOT / "outputs" / "pairsplit_run2" / "validation_split_closure"
#### rho_split production (nominal only, jk axis = the two halves), one pickle
#### per era and channel, mirroring the aligned all_syst tree layout
SPLIT_INPUT_ROOT = Path.home() / "cernbox (2)" / "hadronic_minimal_rho_pairsplit_split"
#### canonical run directories whose artifacts define the nominal stat band
CANONICAL_RUN_FINGERPRINTS = {
    ("dijet", "groomed"): "f64617bc992988f4",
    ("dijet", "ungroomed"): "ac5dca45279f978c",
    ("trijet", "groomed"): "9e20a3857f23f611",
    ("trijet", "ungroomed"): "57d6a4e8cadbc415",
}


def canonical_run_dir(channel: str, grooming_mode: str) -> Path:
    parent = REPO_ROOT / "outputs" / "pairsplit_run2" / channel / "aligned"
    if grooming_mode == "ungroomed":
        parent = parent / "ungroomed"
    fingerprint = CANONICAL_RUN_FINGERPRINTS[(channel, grooming_mode)]
    matches = sorted(parent.glob(f"*__systematics-{fingerprint}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one canonical run directory for {channel} {grooming_mode} "
            f"(fingerprint {fingerprint}) under {parent}; found {matches}"
        )
    return matches[0]


ROLES = ("response", "reco", "gen")
HALF_LABELS = {0: "A", 1: "B"}


# ---------------------------------------------------------------------------
# input halves
# ---------------------------------------------------------------------------
def _empty_half():
    return {role: None for role in ROLES} | {f"{role}_var": None for role in ROLES}


def _accumulate(target, role, values, variances):
    if target[role] is None:
        target[role] = np.array(values, dtype=float)
        target[f"{role}_var"] = np.array(variances, dtype=float)
    else:
        target[role] += values
        target[f"{role}_var"] += variances


def _check_edges(histogram, axis_name, expected, context):
    edges = tuple(float(x) for x in histogram.axes[axis_name].edges)
    expected = tuple(float(x) for x in expected)
    if len(edges) != len(expected) or not np.allclose(edges, expected):
        raise ValueError(f"{context}: axis {axis_name!r} edges {edges} != expected {expected}")


def discover_split_file(channel: str, era: str, split_root: Path) -> Path:
    era_dir = split_root / era
    if not era_dir.is_dir():
        raise FileNotFoundError(f"Missing split-production era directory: {era_dir}")
    pattern = f"**/{channel}_mc/**/rho_split_{channel}_mg_pythia8_{era}.pkl"
    matches = sorted(era_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one rho_split pickle for {channel} {era} using {pattern!r}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def load_split_halves(channel: str, grooming_mode: str, eras, split_root: Path):
    """Sum the two producer halves over eras; returns ({0: half, 1: half}, sources)."""
    keys = LEGACY_HISTOGRAM_KEYS[grooming_mode]
    fine = PAIR_SPLIT_FINE_AXES[grooming_mode]
    halves = {0: _empty_half(), 1: _empty_half()}
    sources = []
    for era in eras:
        path = discover_split_file(channel, era, split_root)
        sources.append(PairSplitSourceFiles(channel=channel, era=era, mc=path, data=path))
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        for role in ROLES:
            context = f"{path} ({grooming_mode} {role})"
            histogram = payload[keys[role]][{"dataset": sum, "systematic": "nominal"}]
            categories = sorted(int(x) for x in histogram.axes["jk"])
            if categories != [0, 1]:
                raise ValueError(f"{context}: jk categories {categories}, expected [0, 1]")
            for axis_name in _ARRAY_AXES[role]:
                expected = {
                    "ptreco": fine.pt_edges,
                    "ptgen": fine.pt_edges,
                    "mpt_reco": fine.two_log10_rho_reco_edges,
                    "mpt_gen": fine.two_log10_rho_gen_edges,
                }[axis_name]
                _check_edges(histogram, axis_name, expected, context)
            for half in (0, 1):
                projected = histogram[{"jk": hist.loc(half)}].project(*_ARRAY_AXES[role])
                variances = projected.variances(flow=False)
                if variances is None:
                    raise ValueError(f"{context}: no sumw2 storage")
                _accumulate(
                    halves[half], role,
                    np.asarray(projected.values(flow=False), dtype=float),
                    np.asarray(variances, dtype=float),
                )
        del payload
    for half in (0, 1):
        _validate_half(halves[half], f"{channel} {grooming_mode} half {HALF_LABELS[half]}")
    return halves, tuple(sources)


def _validate_half(half, context):
    """Sanity checks on one half; fakes and misses may be slightly negative.

    The producer's gen and matched fills disagree at the level of a few
    hundredths of a weighted event in near-empty edge bins (the nominal
    production carries the same feature and the unfolder tolerates it), so
    negative fake or miss content is only an error beyond 1e-6 of the total.
    """
    matched_reco = half["response"].sum(axis=(2, 3))
    matched_gen = half["response"].sum(axis=(0, 1))
    fakes = half["reco"] - matched_reco
    misses = half["gen"] - matched_gen
    tol = 1e-6 * float(half["gen"].sum())
    for name, values in (("fake", fakes), ("miss", misses)):
        worst = float(values.min())
        if worst < -tol:
            raise ValueError(f"{context}: negative {name} yield {worst:.4g} in a fine bin (tolerance {tol:.3g})")
        if worst < -1e-9:
            print(f"  note: {context}: smallest {name} content {worst:.3g} (sub-event, tolerated)")
    for role in ROLES:
        if np.min(half[role]) < 0 or np.min(half[f"{role}_var"]) < 0:
            raise ValueError(f"{context}: negative {role} content")


def synthetic_halves(channel: str, grooming_mode: str, eras, seed: int = 20260903):
    """Thin the nominal production into two halves (code check only).

    Matched response, fakes, and misses are thinned separately with a
    binomial draw on the effective event count of every fine bin, then the
    reco and gen marginals are recomposed so each half is internally
    consistent (non-negative fakes and misses).
    """
    inputs = load_pairsplit_run2_inputs(channel, eras=eras)
    source = inputs.modes[grooming_mode]
    rng = np.random.default_rng(seed)

    def thin(values, variances):
        values = np.asarray(values, dtype=float)
        variances = np.asarray(variances, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff = np.where(variances > 0, values**2 / variances, 0.0)
            weight = np.where(n_eff > 0, values / n_eff, 0.0)
        n_int = np.rint(n_eff).astype(int)
        n_a = rng.binomial(np.clip(n_int, 0, None), 0.5)
        n_b = n_int - n_a
        return (n_a * weight, n_a * weight**2), (n_b * weight, n_b * weight**2)

    response = np.asarray(source.response_by_systematic["nominal"], dtype=float)
    response_var = np.asarray(source.response_variance_by_systematic["nominal"], dtype=float)
    reco = np.asarray(source.reco_by_systematic["nominal"], dtype=float)
    reco_var = np.asarray(source.reco_variance_by_systematic["nominal"], dtype=float)
    gen = np.asarray(source.gen_by_systematic["nominal"], dtype=float)
    gen_var = np.asarray(source.gen_variance_by_systematic["nominal"], dtype=float)
    fakes = np.clip(reco - response.sum(axis=(2, 3)), 0.0, None)
    fakes_var = np.clip(reco_var - response_var.sum(axis=(2, 3)), 0.0, None)
    misses = np.clip(gen - response.sum(axis=(0, 1)), 0.0, None)
    misses_var = np.clip(gen_var - response_var.sum(axis=(0, 1)), 0.0, None)

    resp_halves = thin(response, response_var)
    fake_halves = thin(fakes, fakes_var)
    miss_halves = thin(misses, misses_var)
    halves = {}
    for half in (0, 1):
        r, rv = resp_halves[half]
        f, fv = fake_halves[half]
        m, mv = miss_halves[half]
        halves[half] = {
            "response": r, "response_var": rv,
            "reco": r.sum(axis=(2, 3)) + f, "reco_var": rv.sum(axis=(2, 3)) + fv,
            "gen": r.sum(axis=(0, 1)) + m, "gen_var": rv.sum(axis=(0, 1)) + mv,
        }
        _validate_half(halves[half], f"{channel} {grooming_mode} synthetic half {HALF_LABELS[half]}")
    return halves, inputs.source_files


def make_inputs(channel, grooming_mode, eras, sources, mc_half, data_half) -> PairSplitRun2Inputs:
    """Assemble ``PairSplitRun2Inputs`` with one half as MC and one as data."""
    reco = np.asarray(data_half["reco"], dtype=float)
    reco_var = np.asarray(data_half["reco_var"], dtype=float)
    covariance = np.diag(reco_var.reshape(-1)).reshape(reco.shape + reco.shape)
    mode_arrays = PairSplitModeArrays(
        grooming_mode=grooming_mode,
        fine_axes=PAIR_SPLIT_FINE_AXES[grooming_mode],
        systematics=("nominal",),
        response_by_systematic={"nominal": mc_half["response"]},
        response_variance_by_systematic={"nominal": mc_half["response_var"]},
        reco_by_systematic={"nominal": mc_half["reco"]},
        reco_variance_by_systematic={"nominal": mc_half["reco_var"]},
        gen_by_systematic={"nominal": mc_half["gen"]},
        gen_variance_by_systematic={"nominal": mc_half["gen_var"]},
        nominal_data=reco,
        nominal_data_variance=reco_var,
        nominal_data_covariance=covariance,
    )
    return PairSplitRun2Inputs(
        channel=channel,
        eras=tuple(eras),
        modes={grooming_mode: mode_arrays},
        source_files=tuple(sources),
        observable_metadata={
            "jet_radius": JET_RADIUS,
            "rho_definition": PHYSICAL_RHO_DEFINITION,
            "transformed_coordinate_name": TRANSFORMED_COORDINATE_NAME,
            "transformed_coordinate_definition": TRANSFORMED_COORDINATE_DEFINITION,
            "data_covariance_source_by_mode": {grooming_mode: "diagonal_reco_sumw2"},
            "systematic_exclusions": dict(PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS),
        },
    )


# ---------------------------------------------------------------------------
# unfolding
# ---------------------------------------------------------------------------
def build_unfolder(channel, grooming_mode, inputs, out_dir: Path):
    """Runner-identical construction (spec, binning, normalization), nominal only."""
    import ROOT

    from unfold.tools.unfolder_core import Unfolder

    argv = ["--channel", channel, "--grooming-mode", grooming_mode]
    if channel == "dijet" and grooming_mode == "groomed":
        argv += ["--normalization-window", "peak"]
    args = runner.parse_args(argv)
    args = runner.channel_resolved_args(args, channel)
    prepared = prepare_pairsplit_inputs(
        inputs, args.binning, ("nominal",), grooming_mode=grooming_mode
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = runner.build_pairsplit_spec(channel, out_dir, args, grooming_mode=grooming_mode)
    #### nominal-only test: no model envelope (mirrors run_nominal_mc_self_closure)
    updates = {}
    if hasattr(spec, "model_envelope"):
        updates["model_envelope"] = False
    if hasattr(spec, "model_envelope_source"):
        updates["model_envelope_source"] = "zjet_offline"
    if updates:
        spec = replace(spec, **updates)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    unfolder = Unfolder.from_prepared_inputs(
        spec,
        grooming_mode == "groomed",
        mc_inputs=prepared.mc_inputs,
        data_inputs=prepared.data_inputs,
        analysis_binning=prepared.analysis_binning,
        systematics=("nominal",),
        measured_covariance=prepared.measured_covariance,
        first_reported_pt_bin=prepared.first_reported_pt_bin,
        cms_label=args.cms_label,
        lumi=args.lumi,
        com=args.com,
    )
    reported_minimum = float(prepared.metadata["reported_two_log10_rho_minimum"])
    return unfolder, reported_minimum


def nominal_stat_fraction(channel, grooming_mode, unfolder):
    """Nominal measurement stat band (fraction of the normalized result) per pT slice."""
    artifact = canonical_run_dir(channel, grooming_mode) / "artifacts" / f"{grooming_mode}_results.npz"
    payload = np.load(artifact, allow_pickle=True)
    normalized = np.asarray(payload["normalized_result"], dtype=float)
    cov_stat = np.asarray(payload["norm_cov_stat"], dtype=float)
    gen_edges = tuple(float(x) for x in payload["two_log10_rho_gen_edges"])
    if tuple(float(x) for x in unfolder.gen_edges_by_pt[0]) != gen_edges:
        raise ValueError(
            f"Canonical artifact gen edges {gen_edges} differ from the split test's "
            f"{tuple(unfolder.gen_edges_by_pt[0])}"
        )
    if normalized.size != sum(len(e) - 1 for e in unfolder.gen_edges_by_pt):
        raise ValueError("Canonical artifact bin count differs from the split test layout")
    stat = np.sqrt(np.clip(np.diag(cov_stat), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(normalized != 0, stat / np.abs(normalized), 0.0)
    return unflatten_gen_by_pt(frac, unfolder.gen_edges_by_pt), artifact


# ---------------------------------------------------------------------------
# plots and metrics
# ---------------------------------------------------------------------------
def _pt_title(unfolder, i):
    lo = unfolder.pt_edges[i]
    hi = unfolder.pt_edges[i + 1] if i + 2 < len(unfolder.pt_edges) else float("inf")
    hi_s = r"\infty" if not np.isfinite(hi) else f"{hi:.0f}"
    return rf"$p_{{T}}$  ${lo:.0f}$–${hi_s}$ GeV"


def plot_slice(unfolder, out_dir, grooming_mode, i, truth, unfolded, own_frac, nominal_frac,
               synthetic=False):
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.gridspec import GridSpec

    hep.style.use("CMS")
    edges = np.asarray(unfolder.gen_edges_by_pt[i], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xlo, xhi = unfolder._observable_xlim(i)
    visible = edges[:-1] >= xlo - 1e-9

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.4], hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    hep.histplot(
        truth, edges, ax=ax_top, histtype="step", color="black", lw=2.0,
        label="Truth, half B",
    )
    ax_top.errorbar(
        centers, unfolded, yerr=unfolded * own_frac, fmt="o", color="#e42536",
        markersize=7, elinewidth=1.4, capsize=2,
        label="Unfolded, half B through half A",
    )
    ymax = float(np.max((truth * 1.0)[visible])) if visible.any() else 1.0
    ymax = max(ymax, float(np.max((unfolded * (1 + own_frac))[visible])) if visible.any() else 0.0)
    ax_top.set_ylim(0, 1.6 * ymax)
    ax_top.set_ylabel(unfolder._normalized_ylabel())
    ax_top.legend(title=_pt_title(unfolder, i), fontsize=15, loc="upper left")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.tick_params(axis="y", pad=8)

    band = np.asarray(nominal_frac, dtype=float)
    band_step = np.append(band, band[-1])
    ax_bot.fill_between(
        edges, -band_step, band_step, step="post", color="0.8", alpha=0.7, lw=0,
        label="Stat. unc. of the measurement",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(truth != 0, unfolded / truth - 1.0, 0.0)
    ax_bot.errorbar(
        centers, resid, yerr=own_frac, fmt="o", color="#e42536", markersize=7,
        elinewidth=1.4, capsize=2, label="Split-sample test (own stat. unc.)",
    )
    ax_bot.axhline(0, color="gray", lw=1)
    lim = 0.0
    if visible.any():
        lim = max(np.max(band[visible]), np.max(np.abs(resid[visible]) + own_frac[visible]))
    lim = max(0.02, 1.6 * lim)
    ax_bot.set_ylim(-lim, lim)
    ax_bot.set_xlim(xlo, xhi)
    ax_bot.set_xlabel(unfolder._observable_label())
    ax_bot.set_ylabel("Unfolded / Truth $-$ 1")
    ax_bot.legend(fontsize=13, loc="upper left", ncol=2)
    ax_bot.tick_params(axis="x", pad=8)
    ax_bot.tick_params(axis="y", pad=8)
    label = unfolder.cms_label if not synthetic else "Synthetic"
    hep.cms.label(
        label, data=False, lumi=unfolder._lumi_label(), com=unfolder._com_label(),
        fontsize=20, ax=ax_top,
    )
    fig.subplots_adjust(left=0.16, bottom=0.11, top=0.93)
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"split_closure_{grooming_mode}_pt{i}.{ext}",
            bbox_inches="tight", pad_inches=0.1,
        )
    plt.close(fig)


def run_mode(channel, grooming_mode, *, split_root: Path, eras, synthetic: bool):
    tag = f"{channel}_{grooming_mode}" + ("_synthetic" if synthetic else "")
    print(f"\n=== Pair-split split-sample closure: {tag} ===")
    if synthetic:
        halves, sources = synthetic_halves(channel, grooming_mode, eras)
    else:
        halves, sources = load_split_halves(channel, grooming_mode, eras, split_root)
    yields = {HALF_LABELS[h]: float(halves[h]["reco"].sum()) for h in (0, 1)}
    print(f"  reco yields: A = {yields['A']:.6g}, B = {yields['B']:.6g} "
          f"(B/A = {yields['B'] / yields['A']:.4f})")

    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    #### response/fakes/misses from A, measured spectrum from B: the data path
    inputs_ab = make_inputs(channel, grooming_mode, eras, sources, halves[0], halves[1])
    u_ab, reported_minimum = build_unfolder(channel, grooming_mode, inputs_ab, out_dir / "unfold_A_response_B_input")
    #### B through B only supplies the half-B truth in the runner's normalization
    inputs_bb = make_inputs(channel, grooming_mode, eras, sources, halves[1], halves[1])
    u_bb, _ = build_unfolder(channel, grooming_mode, inputs_bb, out_dir / "unfold_B_response_B_input")
    nominal_frac_by_pt, artifact = nominal_stat_fraction(channel, grooming_mode, u_ab)
    print(f"  nominal stat band from {artifact.relative_to(REPO_ROOT)}")

    rows = []
    for i in u_ab._reported_pt_indices():
        edges = np.asarray(u_ab.gen_edges_by_pt[i], dtype=float)
        shown = edges[:-1] >= reported_minimum - 1e-9
        truth = np.asarray(u_bb.normalized_results[i]["true"], dtype=float)
        unfolded = np.asarray(u_ab.normalized_results[i]["unfolded"], dtype=float)
        own_frac = np.asarray(u_ab.normalized_results[i]["stat_unc_frac"], dtype=float)
        nominal_frac = np.asarray(nominal_frac_by_pt[i], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where(truth != 0, unfolded / truth - 1.0, 0.0)
            pull_nominal = np.where(nominal_frac > 0, np.abs(resid) / nominal_frac, 0.0)
            pull_own = np.where(own_frac > 0, np.abs(resid) / own_frac, 0.0)
        rows.append({
            "pt_index": int(i),
            "pt_low": float(u_ab.pt_edges[i]),
            "pt_high": float(u_ab.pt_edges[i + 1]) if i + 2 < len(u_ab.pt_edges) else None,
            "n_shown_bins": int(shown.sum()),
            "max_abs_residual": float(np.max(np.abs(resid[shown]))),
            "rms_residual": float(np.sqrt(np.mean(resid[shown] ** 2))),
            "max_pull_vs_nominal_stat": float(np.max(pull_nominal[shown])),
            "max_pull_vs_own_stat": float(np.max(pull_own[shown])),
            "residual": [float(x) for x in resid],
            "nominal_stat_frac": [float(x) for x in nominal_frac],
            "own_stat_frac": [float(x) for x in own_frac],
            "shown": [bool(x) for x in shown],
        })
        plot_slice(u_ab, out_dir, grooming_mode, i, truth, unfolded, own_frac, nominal_frac,
                   synthetic=synthetic)

    worst = max(rows, key=lambda r: r["max_pull_vs_nominal_stat"])
    print(f"  SUMMARY: max |unf/truth-1| = {max(r['max_abs_residual'] for r in rows):.4f}; "
          f"max pull vs nominal stat = {worst['max_pull_vs_nominal_stat']:.2f} "
          f"(pT slice {worst['pt_index']}); "
          f"max pull vs own stat = {max(r['max_pull_vs_own_stat'] for r in rows):.2f}")
    for r in rows:
        hi = "inf" if r["pt_high"] is None else f"{r['pt_high']:.0f}"
        print(f"    pt{r['pt_index']} [{r['pt_low']:.0f},{hi}) n={r['n_shown_bins']:2d} "
              f"max|res|={r['max_abs_residual']:.4f} rms={r['rms_residual']:.4f} "
              f"pull_nom={r['max_pull_vs_nominal_stat']:.2f} pull_own={r['max_pull_vs_own_stat']:.2f}")

    with open(out_dir / "split_closure_metrics.txt", "w") as handle:
        handle.write(
            f"# pair-split split-sample closure ({channel}, {grooming_mode}"
            f"{', SYNTHETIC thinning: code check only' if synthetic else ''}); shown-window bins only\n"
            f"# response/fakes/misses: half A; input: half B reco; truth: half B gen\n"
            f"# nominal stat band: {artifact}\n"
            f"# reco yields: A={yields['A']:.6g} B={yields['B']:.6g}\n"
            "# pt_index pt_low pt_high n_shown_bins max|unf/truth-1| rms_residual "
            "max_pull_vs_nominal_stat max_pull_vs_own_stat\n"
        )
        for r in rows:
            hi = "inf" if r["pt_high"] is None else f"{r['pt_high']:.0f}"
            handle.write(
                f"{r['pt_index']} {r['pt_low']:.0f} {hi} {r['n_shown_bins']} "
                f"{r['max_abs_residual']:.6f} {r['rms_residual']:.6f} "
                f"{r['max_pull_vs_nominal_stat']:.4f} {r['max_pull_vs_own_stat']:.4f}\n"
            )
    with open(out_dir / "split_closure_metrics.json", "w") as handle:
        json.dump(
            {
                "channel": channel,
                "grooming_mode": grooming_mode,
                "synthetic": synthetic,
                "eras": list(eras),
                "sources": [str(s.mc) for s in sources],
                "nominal_artifact": str(artifact),
                "reco_yields": yields,
                "slices": rows,
            },
            handle, indent=2,
        )
    print(f"  wrote {out_dir}/split_closure_metrics.{{txt,json}} and split_closure_{grooming_mode}_pt*.pdf")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", nargs="+", default=["dijet", "trijet"])
    parser.add_argument("--grooming-mode", nargs="+", default=["groomed", "ungroomed"])
    parser.add_argument("--split-root", type=Path, default=SPLIT_INPUT_ROOT)
    parser.add_argument("--eras", nargs="+", default=list(PAIR_SPLIT_ERAS))
    parser.add_argument("--synthetic", action="store_true",
                        help="thin the nominal production instead of reading the rho_split pickles (code check only)")
    args = parser.parse_args()
    for channel in args.channel:
        for mode in args.grooming_mode:
            run_mode(channel, mode, split_root=args.split_root, eras=args.eras, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
