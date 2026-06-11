#!/usr/bin/env python3
"""Regularization study for the zjet rho unfolding.

Compares three setups on the groomed rho unfolding:
  1. tau = 0 (current analysis default)
  2. TUnfold curvature regularization on the density, tau from an L-curve scan
  3. "L-matrix trick": custom curvature conditions on x_j / x_j^MC within each
     pT slice (kRegModeNone + AddRegularisationCondition), so any spectrum
     proportional to the MC prior -- including the steeply falling pT shape --
     carries zero penalty. tau from an L-curve scan.

For each setup it records the unfolded data result with its propagated
input-stat errors, and a Herwig-closure bias test (unfold Herwig pseudo-data
with the Pythia response at the same tau, compare to Herwig truth) so the
regularization bias can be judged against the model-uncertainty baseline.

Usage: source scripts/setup_root.sh && python scripts/study_regularization_rho.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import ROOT

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.utils.merge_helpers import (
    merge_mass_flat,
    reorder_to_expected_2d,
    unflatten_gen_by_pt,
)

OUT_DIR = REPO_ROOT / "outputs" / "zjet" / "validation"

# Shared C++ shim: public AddRegularisationCondition + L-curve/SURE scan
# helpers (PyROOT cannot pass the TGraph** output arguments directly).
from unfold.tools.unfolder_core import _declare_open_l, _graph_to_arrays

_declare_open_l()


def build_unfold(u, resp, misses, true_flat, meas_flat, *, setup, name):
    """Construct a TUnfoldDensity mirroring Unfolder._perform_unfold options."""
    truth_root, reco_root = u._build_root_binning()
    h_meas = reco_root.CreateHistogram(f"hMeas_{name}")
    h_resp = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(
        truth_root, reco_root, f"hResp_{name}"
    )
    u._fill_response_histogram(
        h_resp, resp, misses,
        include_bin_errors=True,
        resp_var=getattr(u, "mosaic_var_dict", {}).get("nominal"),
        misses_var=getattr(u, "misses_var_dict", {}).get("nominal"),
    )
    u._fill_root_histogram(h_meas, meas_flat)

    regmode = (
        ROOT.TUnfold.kRegModeNone if setup == "trick" else ROOT.TUnfold.kRegModeCurvature
    )
    unfold = ROOT.TUnfoldDensityOpenL(
        h_resp,
        ROOT.TUnfold.kHistMapOutputHoriz,
        regmode,
        ROOT.TUnfold.kEConstraintArea,
        ROOT.TUnfoldDensity.kDensityModeBinWidth,
        truth_root,
        reco_root,
        # NB: "signal" (as used in unfolder_core) names a node with no axes, so
        # it yields ZERO regularization conditions. Omitting the argument uses
        # the nullptr default, which regularizes each pt-slice child
        # distribution along its mass axis (steering default is "*[UOB]").
    )
    if setup == "trick":
        # Curvature of the ratio x_j / m_j, per pT slice only (no conditions
        # across slice boundaries). A result proportional to the MC prior has
        # zero penalty, so the falling pT spectrum itself is not regularized
        # against -- only shape deviations from the prior are smoothed.
        offset = 0
        for edges in u.gen_edges_by_pt:
            nbins = len(edges) - 1
            for k in range(1, nbins - 1):
                j0, j1, j2 = offset + k - 1, offset + k, offset + k + 1
                m0, m1, m2 = true_flat[j0], true_flat[j1], true_flat[j2]
                if min(m0, m1, m2) <= 0:
                    continue
                unfold.AddRegularisationCondition(
                    j0 + 1, 1.0 / m0, j1 + 1, -2.0 / m1, j2 + 1, 1.0 / m2
                )
            offset += nbins

    status = unfold.SetInput(h_meas)
    if status >= 10000:
        raise RuntimeError("TUnfold input had overflow/underflow")
    # TUnfold holds raw pointers to the binning trees and histograms; return
    # them so the caller keeps them alive for the lifetime of `unfold`.
    return unfold, (truth_root, reco_root, h_meas, h_resp)


def results_from(unfold, n_true, name):
    h_out = unfold.GetOutput(f"out_{name}")
    y_unf = np.array([h_out.GetBinContent(i) for i in range(1, n_true + 1)])
    e_in = unfold.GetEmatrixInput(f"ein_{name}")
    cov_in = np.array(
        [[e_in.GetBinContent(i, j) for j in range(1, n_true + 1)] for i in range(1, n_true + 1)]
    )
    return y_unf, cov_in


def run_setup(u, resp, misses, true_flat, meas_flat, *, setup, tau=None,
              scan="lcurve", name="x"):
    """Unfold once. tau=None -> scan ('lcurve' or 'sure').

    Returns (y, cov_input, tau, curves) where curves holds the scan graph
    arrays (or None for fixed tau).
    """
    unfold, keepalive = build_unfold(u, resp, misses, true_flat, meas_flat, setup=setup, name=name)
    curves = None
    if tau is None:
        if scan == "sure":
            # NB: ScanSURE's automatic range can stop while SURE is still
            # falling; use an explicit wide range instead.
            result = ROOT.RunUnfoldSureScan(unfold, 60, 1e-4, 1e3)
            tau = unfold.GetTau()
            curves = {
                "kind": "sure",
                "logtau_sure": _graph_to_arrays(result.logTauSURE),
                "df_chi2A": _graph_to_arrays(result.df_chi2A),
            }
            # make sure the object is left exactly at the chosen tau
            unfold.DoUnfold(tau)
        else:
            result = ROOT.RunUnfoldLcurveScan(unfold, 40)
            tau = unfold.GetTau()
            curves = {"kind": "lcurve", "lcurve": _graph_to_arrays(result.lcurve)}
    else:
        unfold.DoUnfold(tau)
    y, cov = results_from(unfold, resp.shape[1], name)
    return y, cov, tau, curves


def normalize(u, flat):
    out = np.empty_like(flat, dtype=float)
    offset = 0
    for edges in u.gen_edges_by_pt:
        nbins = len(edges) - 1
        widths = np.diff(edges)
        block = flat[offset : offset + nbins]
        out[offset : offset + nbins] = block / widths / block.sum()
        offset += nbins
    return out


def main():
    spec = get_spec("zjet", "rho", "original")
    u = Unfolder(spec, groomed=True, do_syst=False)

    resp = u.mosaic_dict["nominal"]
    misses = u.misses_2d
    true_flat = resp.sum(axis=0) + misses
    n_true = resp.shape[1]

    meas_data = u._apply_fake_correction(np.array(u.mosaic_2d, float), "nominal", False, False)
    # closure pseudo-data: matched-reco projection of the HERWIG response,
    # exactly as the herwig_closure path builds mosaic_herwig_2d (that
    # attribute only exists when herwig systematics are enabled)
    herwig = u._load_pickle(spec.input_dir + spec.herwig_file)
    resp4d = herwig[u._histogram_keys()["response"]]
    h2d_her = resp4d[{"systematic": "nominal"}].project("ptreco", u.reco_axis).values(flow=False)
    h2d_her, _ = reorder_to_expected_2d(h2d_her, u.edges, u.pt_edges)
    meas_herwig = np.asarray(merge_mass_flat(h2d_her, u.edges, u.reco_edges_by_pt), dtype=float)
    herwig_truth = np.array(u.herwig_gen_val_flat, dtype=float)

    # self-closure pseudo-data: matched reco from the nominal response itself
    meas_self = np.asarray(u.mosaic.sum(axis=1), dtype=float)

    setups = {}
    for setup, kind, tau, scan in (
        ("tau0", "curvature", 0.0, None),
        ("curvature", "curvature", None, "lcurve"),
        ("trick", "trick", None, "lcurve"),
        ("trick_sure", "trick", None, "sure"),
    ):
        y, cov, tau, curves = run_setup(
            u, resp, misses, true_flat, meas_data, setup=kind, tau=tau, scan=scan,
            name=f"data_{setup}",
        )
        # Herwig closure and self-closure at the SAME tau for the bias tests
        yh, _, _, _ = run_setup(
            u, resp, misses, true_flat, meas_herwig, setup=kind, tau=tau, name=f"her_{setup}"
        )
        ys, _, _, _ = run_setup(
            u, resp, misses, true_flat, meas_self, setup=kind, tau=tau, name=f"self_{setup}"
        )
        frac = np.sqrt(np.clip(np.diag(cov), 0, None)) / np.where(y != 0, np.abs(y), 1.0)
        closure = normalize(u, yh) / normalize(u, herwig_truth) - 1.0
        closure_self = normalize(u, ys) / normalize(u, true_flat) - 1.0
        setups[setup] = dict(y=y, cov=cov, tau=tau, frac=frac, closure=closure,
                             closure_self=closure_self, norm=normalize(u, y),
                             curves=curves)
        print(f"[{setup}] tau = {tau:.4g}")
        for label, arr in (("rel stat unc", frac), ("closure bias", np.abs(closure)),
                           ("self-closure", np.abs(closure_self))):
            per_pt = unflatten_gen_by_pt(arr, u.gen_edges_by_pt)
            print(f"  {label:14s} medians:", [f"{100*np.median(b):.2f}%" for b in per_pt],
                  f"max: {100*np.max(arr):.2f}%")

    np.savez(
        OUT_DIR / "regularization_study_groomed.npz",
        **{f"{k}_{q}": v[q] for k, v in setups.items() for q in ("y", "cov", "frac", "closure", "closure_self", "norm")},
        **{f"{k}_tau": v["tau"] for k, v in setups.items()},
        true_flat=true_flat,
        herwig_truth=herwig_truth,
        gen_bins_per_pt=np.asarray([len(e) - 1 for e in u.gen_edges_by_pt]),
    )

    # ---- plots -----------------------------------------------------------
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    colors = {"tau0": "black", "curvature": "#f89c20", "trick": "#e42536",
              "trick_sure": "#964a8b"}
    labels = {
        "tau0": r"$\tau = 0$ (current)",
        "curvature": rf"curvature, $\tau$={setups['curvature']['tau']:.2g}",
        "trick": rf"curvature on $x/x_{{MC}}$, L-curve $\tau$={setups['trick']['tau']:.2g}",
        "trick_sure": rf"curvature on $x/x_{{MC}}$, SURE $\tau$={setups['trick_sure']['tau']:.2g}",
    }

    # ---- scan curves: L-curve and SURE ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    lx, ly = setups["trick"]["curves"]["lcurve"]
    ax1.plot(lx, ly, "-o", ms=3, lw=1.5, color="#e42536", label="L-curve scan")
    ax1.set_xlabel(r"$\log_{10}\,\chi^2_{\rm data}$")
    ax1.set_ylabel(r"$\log_{10}\,(Lx)^2$")
    ax1.set_title(rf"L-curve (trick), chosen $\tau$ = {setups['trick']['tau']:.3g}", fontsize=14)
    ax1.legend(fontsize=12)
    sx, sy = setups["trick_sure"]["curves"]["logtau_sure"]
    ax2.plot(sx, sy, "-o", ms=3, lw=1.5, color="#964a8b", label="SURE scan")
    ax2.axvline(np.log10(setups["trick_sure"]["tau"]), color="#964a8b", ls="--",
                label=rf"SURE $\tau$ = {setups['trick_sure']['tau']:.3g}")
    ax2.axvline(np.log10(setups["trick"]["tau"]), color="#e42536", ls=":",
                label=rf"L-curve $\tau$ = {setups['trick']['tau']:.3g}")
    ax2.set_xlabel(r"$\log_{10}\,\tau$")
    ax2.set_ylabel("SURE")
    ax2.set_title("Stein's Unbiased Risk Estimate (trick)", fontsize=14)
    ax2.legend(fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"regularization_scan_curves_groomed.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for i in range(4):
        ax = axes.flat[i]
        edges = np.asarray(u.gen_edges_by_pt[i], float)
        for key, st in setups.items():
            per_pt = unflatten_gen_by_pt(st["frac"], u.gen_edges_by_pt)
            ax.stairs(100 * per_pt[i], edges, label=labels[key], color=colors[key], lw=2.5)
        hi = u.pt_edges[i + 1] if i + 2 < len(u.pt_edges) else float("inf")
        ax.set_title(f"${u.pt_edges[i]:.0f} < p_T < {hi:.0f}$ GeV (groomed)", fontsize=15)
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        ax.set_ylabel("Rel. input-stat unc (%)")
        ax.set_yscale("log")
        if i == 0:
            ax.legend(fontsize=13)
    fig.suptitle("Input-stat uncertainty vs regularization", fontsize=16)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"regularization_stat_unc_groomed.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for i in range(4):
        ax = axes.flat[i]
        edges = np.asarray(u.gen_edges_by_pt[i], float)
        for key, st in setups.items():
            per_pt = unflatten_gen_by_pt(st["closure"], u.gen_edges_by_pt)
            ax.stairs(100 * per_pt[i], edges, label=labels[key], color=colors[key], lw=2.5)
        per_pt = unflatten_gen_by_pt(setups["tau0"]["frac"], u.gen_edges_by_pt)
        ax.stairs(100 * per_pt[i], edges, color="gray", ls=":", lw=2,
                  label=r"$\tau=0$ stat unc" if i == 0 else None)
        ax.stairs(-100 * per_pt[i], edges, color="gray", ls=":", lw=2)
        ax.axhline(0, color="gray", lw=1)
        hi = u.pt_edges[i + 1] if i + 2 < len(u.pt_edges) else float("inf")
        ax.set_title(f"${u.pt_edges[i]:.0f} < p_T < {hi:.0f}$ GeV (groomed)", fontsize=15)
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        ax.set_ylabel("HERWIG closure: unfolded/truth - 1 (%)")
        if i == 0:
            ax.legend(fontsize=13)
    fig.suptitle("Regularization bias test (HERWIG pseudo-data, normalized shapes)", fontsize=16)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"regularization_bias_groomed.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for i in range(4):
        ax = axes.flat[i]
        edges = np.asarray(u.gen_edges_by_pt[i], float)
        for key, st in setups.items():
            per_pt = unflatten_gen_by_pt(st["closure_self"], u.gen_edges_by_pt)
            ax.stairs(100 * per_pt[i], edges, label=labels[key], color=colors[key], lw=2.5)
        ax.axhline(0, color="gray", lw=1)
        hi = u.pt_edges[i + 1] if i + 2 < len(u.pt_edges) else float("inf")
        ax.set_title(f"${u.pt_edges[i]:.0f} < p_T < {hi:.0f}$ GeV (groomed)", fontsize=15)
        ax.set_xlabel(r"$\log_{10}(\rho^2)$")
        ax.set_ylabel("Self-closure: unfolded/truth - 1 (%)")
        if i == 0:
            ax.legend(fontsize=13)
    fig.suptitle("Self-closure (PYTHIA pseudo-data, normalized shapes)", fontsize=16)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"regularization_selfclosure_groomed.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved plots in", OUT_DIR)


if __name__ == "__main__":
    main()
