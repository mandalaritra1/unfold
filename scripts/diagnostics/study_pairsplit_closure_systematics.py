#!/usr/bin/env python3
"""Per-systematic and model closure for the pair-split channels.

For every response-matrix variation, the reconstruction-level projection of
the *varied* response is unfolded with the *nominal* response and compared
with that variation's own generator-level truth (matched gen + varied
misses).  For detector variations the gen spectrum is unchanged and the
truth reduces to the nominal one; for the theory-weight and ``model_*``
reweight variations the gen spectrum genuinely moves, and scoring them
against the nominal truth would count the intended gen-level reweighting as
non-closure.

Per channel and grooming mode this writes, under
``outputs/pairsplit_run2/validation_closure/<channel>_<mode>/``:

  - ``closure_scan.txt``           full per-source table (max residual, max
                                   pull vs the variation's own stat, max
                                   residual over the data stat band)
  - ``closure_detector_pt<i>.pdf`` representative detector sources
                                   (JES FlavorQCD, JER, pileup)
  - ``closure_model_pt<i>.pdf``    the five model reweight sources

Usage:
  source scripts/setup_root.sh
  .venv/bin/python scripts/diagnostics/study_pairsplit_closure_systematics.py \
      [--channel dijet trijet] [--grooming-mode groomed ungroomed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")

import numpy as np

import run_pairsplit_unfolding as runner
from unfold.tools.pairsplit_model_envelope import (
    derive_pairsplit_model_envelope_inputs,
)
from unfold.tools.pairsplit_run2_inputs import (
    load_pairsplit_run2_inputs,
    prepare_pairsplit_inputs,
    resolve_pairsplit_systematics,
)
from unfold.tools.pairsplit_vincia import load_pairsplit_vincia_source
from unfold.utils.merge_helpers import unflatten_gen_by_pt

OUT_ROOT = REPO_ROOT / "outputs" / "pairsplit_run2" / "validation_closure"

# Representative detector sources for the figures; the actually-varied
# response key (``*_corr``/``*_uncorr_<era>`` leg) is picked automatically.
DETECTOR_BASES = ["JES_FlavorQCD", "JER", "pu"]
MODEL_KEYS = ["model_vincia", "model_cr1", "model_cr2", "model_fraghard", "model_fragsoft"]

LABELS = {
    "nominal": "Nominal",
    "JES_FlavorQCD": "JES (FlavorQCD)",
    "JER": "JER",
    "pu": "Pileup",
    "model_vincia": "MESS+Vincia",
    "model_cr1": "CR mode 1",
    "model_cr2": "CR mode 2",
    "model_fraghard": "Frag. hard",
    "model_fragsoft": "Frag. soft",
}
COLORS = {
    "nominal": "black",
    "JES_FlavorQCD": "#e42536",
    "JER": "#f89c20",
    "pu": "#5790fc",
    "model_vincia": "#964a8b",
    "model_cr1": "#e42536",
    "model_cr2": "#f89c20",
    "model_fraghard": "#5790fc",
    "model_fragsoft": "#7a21dd",
}


def _norm_jacobian(u, x_abs):
    """Per-pT-slice unit-area normalization Jacobian for an absolute spectrum."""
    n = len(x_abs)
    jac = np.zeros((n, n))
    offset = 0
    for edges in u.gen_edges_by_pt:
        edges = np.asarray(edges, dtype=float)
        nbins = len(edges) - 1
        widths = np.diff(edges)
        block = np.asarray(x_abs[offset : offset + nbins], dtype=float)
        total = block.sum()
        if total == 0:
            total = 1.0
        for i in range(nbins):
            for j in range(nbins):
                jac[offset + i, offset + j] = ((1.0 if i == j else 0.0) - block[i] / total) / (
                    widths[i] * total
                )
        offset += nbins
    return jac


def normalize(u, flat):
    """Unit area per pT slice, bin-width divided."""
    flat = np.asarray(flat, dtype=float)
    out = np.zeros_like(flat)
    offset = 0
    for edges in u.gen_edges_by_pt:
        edges = np.asarray(edges, dtype=float)
        nbins = len(edges) - 1
        widths = np.diff(edges)
        block = flat[offset : offset + nbins]
        denom = block.sum()
        out[offset : offset + nbins] = block / widths / (denom if denom else 1.0)
        offset += nbins
    return out


def closure(u, key):
    """Closure for one variation: varied INPUT, NOMINAL response.

    Returns (norm_unfolded, norm_truth, frac_stat_err) with frac_stat_err the
    per-bin statistical uncertainty of this closure point (input + matrix,
    Jacobian-propagated) as a fraction of the truth.
    """
    resp_nom = np.asarray(u.mosaic_dict["nominal"], dtype=float)
    misses_nom = np.asarray(u.misses_2d, dtype=float)
    misses_key = np.asarray(u.misses_2d_dict.get(key, misses_nom), dtype=float)
    truth = np.asarray(u.mosaic_dict[key], dtype=float).sum(axis=0) + misses_key
    meas = np.asarray(u.mosaic_dict[key], dtype=float).sum(axis=1)
    # Honest MC-stat input errors for the pseudo-data: the stored sumw2 of the
    # varied matched response, not sqrt of the weighted contents.
    meas_var = np.asarray(u.mosaic_var_dict[key], dtype=float).sum(axis=1)

    u._perform_unfold(
        systematic=key, closure=True, meas_flat=meas, resp_np=resp_nom,
        meas_var=meas_var,
    )
    x_abs = np.asarray(u.y_unf if key == "nominal" else u.y_unf_dict[key], dtype=float)

    cov_abs = u.closure_cov_input[key] + u.closure_cov_matrix[key]
    jac = _norm_jacobian(u, x_abs)
    norm_cov = jac @ cov_abs @ jac.T
    norm_stat = np.sqrt(np.clip(np.diag(norm_cov), 0.0, None))
    norm_truth = normalize(u, truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_err = np.where(norm_truth != 0, norm_stat / norm_truth, 0.0)
    return normalize(u, x_abs), norm_truth, frac_err


def _reported_mask(u, reported_minimum):
    """Flat gen-level mask: bins whose lower edge is inside the shown window."""
    blocks = []
    for edges in u.gen_edges_by_pt:
        edges = np.asarray(edges, dtype=float)
        blocks.append(edges[:-1] >= reported_minimum - 1e-9)
    return np.concatenate(blocks)


def build_unfolder(channel: str, grooming_mode: str):
    """Mirror the runner's construction, with closure-covariance capture."""

    import ROOT

    from unfold.tools.unfolder_core import Unfolder

    class ClosureScanUnfolder(Unfolder):
        def _store_covariances(self, unfold, systematic):
            super()._store_covariances(unfold, systematic)
            _, n_true = self.mosaic.shape

            def _to_np(h):
                return np.array(
                    [[h.GetBinContent(i, j) for j in range(1, n_true + 1)]
                     for i in range(1, n_true + 1)]
                )

            if not hasattr(self, "closure_cov_input"):
                self.closure_cov_input = {}
                self.closure_cov_matrix = {}
            self.closure_cov_input[systematic] = _to_np(
                unfold.GetEmatrixInput(f"cs_ein_{systematic}", "input stat")
            )
            self.closure_cov_matrix[systematic] = _to_np(
                unfold.GetEmatrixSysUncorr(f"cs_eun_{systematic}", "matrix stat")
            )

    argv = ["--channel", channel, "--grooming-mode", grooming_mode]
    if channel == "dijet" and grooming_mode == "groomed":
        argv += ["--normalization-window", "peak"]
    args = runner.parse_args(argv)
    args = runner.channel_resolved_args(args, channel)

    inputs = load_pairsplit_run2_inputs(channel, input_root=args.input_root)
    resolved_systematics = resolve_pairsplit_systematics(
        inputs.modes[grooming_mode].systematics, args.systematics
    )
    _, normalization_window = runner.resolved_normalization(args, grooming_mode)
    vincia_source = load_pairsplit_vincia_source(channel)
    model_envelope = derive_pairsplit_model_envelope_inputs(
        inputs,
        vincia_source,
        variant=args.binning,
        normalization_window=normalization_window,
        grooming_mode=grooming_mode,
    )
    prepared = prepare_pairsplit_inputs(
        inputs,
        args.binning,
        resolved_systematics,
        grooming_mode=grooming_mode,
        model_variations=model_envelope.prepared_variations(),
        model_metadata=model_envelope.provenance_payload(),
    )
    out_dir = OUT_ROOT / f"{channel}_{grooming_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = runner.build_pairsplit_spec(
        channel, out_dir, args, grooming_mode=grooming_mode
    )
    ROOT.gErrorIgnoreLevel = ROOT.kError
    u = ClosureScanUnfolder.from_prepared_inputs(
        spec,
        grooming_mode == "groomed",
        mc_inputs=prepared.mc_inputs,
        data_inputs=prepared.data_inputs,
        analysis_binning=prepared.analysis_binning,
        systematics=prepared.systematics,
        measured_covariance=prepared.measured_covariance,
        first_reported_pt_bin=prepared.first_reported_pt_bin,
        cms_label=args.cms_label,
        lumi=args.lumi,
        com=args.com,
    )
    reported_minimum = float(
        getattr(
            prepared.analysis_binning,
            "reported_two_log10_rho_minimum",
            -3.5 if grooming_mode == "groomed" else -2.5,
        )
    )
    return u, out_dir, reported_minimum


def _pick_representative(u, base):
    """Pick the Up-leg response key deviating most from nominal for a base name."""
    nominal = np.asarray(u.mosaic_dict["nominal"], dtype=float)
    candidates = [
        k for k in u.mosaic_dict
        if k != "nominal"
        and k.endswith("Up")
        and (k == base + "Up" or k.startswith(base + "_"))
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda k: float(np.sum(np.abs(np.asarray(u.mosaic_dict[k], float) - nominal))),
    )


def _plot_group(u, out_dir, mode_tag, group_name, keys_by_base, results, errors, stat_frac):
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")

    for i in u._reported_pt_indices():
        edges = np.asarray(u.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xlo, xhi = u._observable_xlim(i)
        visible = edges[:-1] >= xlo
        ymax = 0.0
        fig = plt.figure()

        band = np.asarray(stat_frac[i], dtype=float)
        band_step = np.append(band, band[-1])
        plt.fill_between(
            edges, -band_step, band_step, step="post",
            color="0.8", alpha=0.6, lw=0, label="Stat. unc. (total)",
        )
        if visible.any():
            ymax = max(ymax, np.max(band[visible]))

        for base, key in keys_by_base.items():
            vals = unflatten_gen_by_pt(results[base], u.gen_edges_by_pt)[i]
            errs = unflatten_gen_by_pt(errors[base], u.gen_edges_by_pt)[i]
            color = COLORS.get(base)
            if visible.any():
                ymax = max(ymax, np.max(np.abs(vals[visible]) + errs[visible]))
            hep.histplot(vals, edges, label=LABELS.get(base, base), color=color, lw=2.0)
            plt.errorbar(
                centers, vals, yerr=errs, fmt="none", ecolor=color,
                elinewidth=1.4, capsize=2, alpha=0.9,
            )
        plt.axhline(0, color="gray", lw=1)
        ylim = max(0.01, 1.25 * ymax)
        plt.ylim(-ylim, ylim)
        plt.xlim(xlo, xhi)
        plt.xlabel(u._observable_label())
        plt.ylabel("Unfolded / Truth $-$ 1")
        lo = u.pt_edges[i]
        hi = u.pt_edges[i + 1] if i + 2 < len(u.pt_edges) else float("inf")
        hi_s = r"\infty" if not np.isfinite(hi) else f"{hi:.0f}"
        plt.legend(
            title=rf"$p_{{T}}$  ${lo:.0f}$–${hi_s}$ GeV", fontsize=13
        )
        hep.cms.label(
            u.cms_label, data=False, lumi=u._lumi_label(), com=u._com_label(), fontsize=20
        )
        ax = plt.gca()
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)
        plt.subplots_adjust(left=0.16, bottom=0.15)
        for ext in ("pdf", "png"):
            fig.savefig(
                out_dir / f"closure_{group_name}_pt{i}.{ext}",
                bbox_inches="tight", pad_inches=0.1,
            )
        plt.close(fig)


def run_mode(channel: str, grooming_mode: str):
    print(f"\n=== Pair-split closure scan: {channel} {grooming_mode} ===")
    u, out_dir, reported_minimum = build_unfolder(channel, grooming_mode)

    mask = _reported_mask(u, reported_minimum)
    stat_frac = [
        np.asarray(u.normalized_results[i]["stat_unc_frac"], dtype=float)
        for i in range(len(u.normalized_results))
    ]
    stat_frac_flat = np.concatenate(stat_frac)

    # Assigned normalized systematic shifts (relative), captured BEFORE the
    # closure loop overwrites the unfolder's per-systematic state.  For each
    # source the closure residual is, to first order, the same
    # response-difference effect as the assigned shift, so their ratio is the
    # self-consistency check of the propagation.
    norm_nom_flat = np.concatenate(
        [np.asarray(r["unfolded"], dtype=float) for r in u.normalized_results]
    )
    assigned_rel = {}
    for name in u.systematics:
        if name == "nominal":
            continue
        try:
            shifted = np.concatenate(
                [
                    np.asarray(per_pt["unfolded"][name], dtype=float)
                    for per_pt in u.normalized_systematics
                ]
            )
        except (KeyError, TypeError):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            assigned_rel[name] = np.where(
                norm_nom_flat != 0,
                np.abs(shifted - norm_nom_flat) / np.abs(norm_nom_flat),
                0.0,
            )

    all_keys = ["nominal"] + sorted(k for k in u.mosaic_dict if k != "nominal")
    rows = []
    resid_by_key = {}
    err_by_key = {}
    for key in all_keys:
        norm_unf, norm_truth, frac_err = closure(u, key)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where(norm_truth != 0, norm_unf / norm_truth - 1.0, 0.0)
            pull = np.where(frac_err > 0, np.abs(resid) / frac_err, 0.0)
            rel_band = np.where(stat_frac_flat > 0, np.abs(resid) / stat_frac_flat, 0.0)
        resid_by_key[key] = resid
        err_by_key[key] = frac_err
        max_resid = float(np.max(np.abs(resid[mask])))
        max_assigned = (
            float(np.max(assigned_rel[key][mask])) if key in assigned_rel else float("nan")
        )
        rows.append((
            key,
            max_resid,
            float(np.max(pull[mask])),
            float(np.max(rel_band[mask])),
            max_assigned,
            max_resid / max_assigned if max_assigned and np.isfinite(max_assigned) and max_assigned > 0 else float("nan"),
        ))

    rows_var = [r for r in rows if r[0] != "nominal"]
    nominal = next(r for r in rows if r[0] == "nominal")
    worst_res = max(r[1] for r in rows_var)
    finite_ratio = [r[5] for r in rows_var if np.isfinite(r[5])]
    print(f"  nominal self-closure: max |unf/truth-1| = {nominal[1]:.2e}")
    print(
        f"  SUMMARY: {len(rows_var)} variations, max |residual| = {worst_res:.4f}; "
        f"residual/assigned-shift ratio: median = {np.median(finite_ratio):.2f}, "
        f"max = {np.max(finite_ratio):.2f}"
    )
    for row in sorted(rows_var, key=lambda r: -r[1])[:10]:
        key, res, pull, rel, assig, ratio = row
        print(
            f"    {key:38s} res={res:8.4f} assigned={assig:8.4f} "
            f"res/assigned={ratio:6.2f} vs_stat_band={rel:6.2f}"
        )

    scan_path = out_dir / "closure_scan.txt"
    with open(scan_path, "w") as f:
        f.write(
            f"# pair-split per-systematic closure ({channel}, {grooming_mode}); "
            "shown-window bins only\n"
            "# source  max|unf/truth-1|  max_pull_own_MCstat  max_over_data_stat_band"
            "  max_assigned_rel_shift  resid_over_assigned\n"
        )
        f.write(f"nominal {nominal[1]:.6e} - - - -\n")
        for key, res, pull, rel, assig, ratio in sorted(rows_var, key=lambda r: -r[1]):
            f.write(f"{key} {res:.6f} {pull:.4f} {rel:.4f} {assig:.6f} {ratio:.4f}\n")
    print(f"  wrote {scan_path}")

    detector_keys = {"nominal": "nominal"}
    for base in DETECTOR_BASES:
        key = _pick_representative(u, base)
        if key is not None:
            detector_keys[base] = key
            print(f"  detector figure: {base} -> {key}")
    _plot_group(
        u, out_dir, grooming_mode, "detector",
        detector_keys,
        {b: resid_by_key[k] for b, k in detector_keys.items()},
        {b: err_by_key[k] for b, k in detector_keys.items()},
        stat_frac,
    )

    model_keys = {"nominal": "nominal"}
    for key in MODEL_KEYS:
        if key in resid_by_key:
            model_keys[key] = key
    _plot_group(
        u, out_dir, grooming_mode, "model",
        model_keys,
        {b: resid_by_key[k] for b, k in model_keys.items()},
        {b: err_by_key[k] for b, k in model_keys.items()},
        stat_frac,
    )
    print(f"  figures -> {out_dir}/closure_{{detector,model}}_pt*.pdf")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", nargs="+", default=["dijet", "trijet"])
    parser.add_argument("--grooming-mode", nargs="+", default=["groomed", "ungroomed"])
    args = parser.parse_args()
    for channel in args.channel:
        for mode in args.grooming_mode:
            run_mode(channel, mode)


if __name__ == "__main__":
    main()
