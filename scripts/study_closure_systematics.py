#!/usr/bin/env python3
"""Per-systematic closure test for the zjet rho unfolding (ARC Fig. 44 follow-up).

The ARC asked us to confirm that the unfolding closes not only for the nominal
sample but for the systematic variations one by one. The nominal self-closure
(identical samples) is trivially perfect. Unfolding a variation's reco with that
*same* variation's response would also be trivially perfect and tests nothing.

The meaningful test shifts only the INPUT: we build the pseudo-data from a
systematically-varied reco (the matched-reco projection of the varied response),
but unfold it with the NOMINAL response matrix and nominal misses -- exactly the
response that is applied to real data. For the detector variations shown here
(JES, JER, pileup) the gen-level truth is unchanged, so a closing analysis still
recovers the nominal gen truth. The residual (unfolded / truth - 1) is the
genuine per-systematic non-closure: it should be small and well within the
assigned uncertainties.

We show a representative JES source (FlavorQCD), the JER smearing, and the pileup
reweighting, alongside the nominal self-closure (which closes to rounding) as a
reference. For each requested source we automatically pick the response key
(corr / uncorr_<era>) whose reco deviates most from nominal, so the test uses the
largest input shift available for that source.

Usage:
  source scripts/setup_root.sh && python scripts/study_closure_systematics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from unfold.tools.unfolder_core import Unfolder, get_spec
from unfold.utils.merge_helpers import unflatten_gen_by_pt

OUT_DIR = REPO_ROOT / "outputs" / "zjet" / "validation"

# Representative variations to demonstrate: one JES source, JER, and pileup.
# Keys are the *base* systematic names; the genuinely-varied response key
# (e.g. *_corr or *_uncorr_2018) is selected automatically below.
BASE_SYSTEMATICS = ["JES_FlavorQCDUp", "JERUp", "puUp"]

# Display labels and colors keyed by base systematic name.
LABELS = {
    "JES_FlavorQCDUp": "JES (FlavorQCD)",
    "JERUp": "JER",
    "puUp": "Pileup",
    "nominal": "Nominal",
}
COLORS = {
    "nominal": "black",
    "JES_FlavorQCDUp": "#e42536",
    "JERUp": "#f89c20",
    "puUp": "#5790fc",
}


class ClosureUnfolder(Unfolder):
    """Unfolder that loads only the nominal + chosen-variation responses.

    Overriding _configure_systematics keeps the (expensive) constructor from
    looping over the full ~200-key systematic list: only the responses needed
    for the closure demonstration are reordered, mosaicked, and unfolded.
    """

    base_systematics = BASE_SYSTEMATICS

    def _configure_systematics(self, do_syst):
        available = self.sys_matrix_dic
        nominal = np.asarray(available["nominal"], dtype=float)
        chosen = ["nominal"]
        self.closure_keys = {}
        for base in self.base_systematics:
            candidates = [
                k for k in available if k == base or k.startswith(base + "_")
            ]
            if not candidates:
                print(f"  WARNING: no response key found for {base!r}; skipping")
                continue
            # Pick the variation whose response deviates most from nominal so
            # the closure actually exercises a varied matrix (rho=0 sources such
            # as JER have m_corr == nominal; the variation lives in *_uncorr_*).
            best = max(
                candidates,
                key=lambda k: float(np.sum(np.abs(np.asarray(available[k], float) - nominal))),
            )
            chosen.append(best)
            self.closure_keys[base] = best
            print(f"  {base:18s} -> closure response key {best!r}")
        self.systematics = chosen

    def _store_covariances(self, unfold, systematic):
        """Also stash the per-variation input + matrix stat covariances.

        The base method only keeps these for the nominal/herwig unfolds; we need
        them for every closure unfold to draw a per-variation stat error bar.
        """
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
            unfold.GetEmatrixInput(f"ein_{systematic}", "input stat")
        )
        self.closure_cov_matrix[systematic] = _to_np(
            unfold.GetEmatrixSysUncorr(f"eun_{systematic}", "matrix stat")
        )


def _norm_jacobian(u, x_abs):
    """Per-pT-slice normalization Jacobian for an arbitrary absolute spectrum."""
    n = len(x_abs)
    jac = np.zeros((n, n))
    offset = 0
    for edges in u.gen_edges_by_pt:
        nbins = len(edges) - 1
        sl = slice(offset, offset + nbins)
        x = np.asarray(x_abs[sl], dtype=float)
        widths = np.diff(edges)
        total = x.sum()
        if total != 0:
            jac[sl, sl] = (
                np.eye(nbins) - np.outer(x, np.ones(nbins)) / total
            ) / (widths[:, None] * total)
        offset += nbins
    return jac


def normalize(u, flat):
    """Per-pT-slice normalized shape (matches the analysis normalization)."""
    out = np.empty_like(flat, dtype=float)
    offset = 0
    for edges in u.gen_edges_by_pt:
        nbins = len(edges) - 1
        widths = np.diff(edges)
        block = flat[offset : offset + nbins]
        denom = block.sum()
        out[offset : offset + nbins] = block / widths / (denom if denom else 1.0)
        offset += nbins
    return out


def closure(u, key):
    """Closure for one variation: varied INPUT, NOMINAL response.

    The pseudo-data is the matched-reco projection of the *varied* response, but
    the unfolding uses the nominal response and nominal misses (the response
    applied to real data).

    The reference truth is the gen spectrum *of that variation*,
    ``mosaic_dict[key].sum(axis=0) + misses_2d_dict[key]``. For detector
    variations (JES/JER/pileup/JMS/JMR) the gen spectrum is unchanged and this
    reduces identically to the nominal gen truth, since the varied misses absorb
    exactly the matched-migration delta. For the theory-weight variations
    (ISR/FSR/q2/PDF) the gen spectrum genuinely moves, and comparing those
    against the *nominal* truth would score the intended gen-level reweighting as
    non-closure.

    Returns (norm_unfolded, norm_truth, frac_stat_err), where frac_stat_err is the
    per-bin statistical uncertainty of this closure point (input + matrix stat,
    jacobian-propagated through the normalization) expressed as a fraction of the
    truth -- i.e. the error bar on (unfolded/truth - 1).
    """
    resp_nom = np.asarray(u.mosaic_dict["nominal"], dtype=float)
    misses_nom = np.asarray(u.misses_2d, dtype=float)
    misses_key = np.asarray(
        getattr(u, "misses_2d_dict", {}).get(key, misses_nom), dtype=float
    )
    # gen truth of this variation (== nominal truth for detector systematics)
    truth = np.asarray(u.mosaic_dict[key], dtype=float).sum(axis=0) + misses_key

    # Pseudo-data: matched reco of the (possibly varied) response.
    meas = np.asarray(u.mosaic_dict[key], dtype=float).sum(axis=1)

    # Unfold the varied input with the NOMINAL response.
    u._perform_unfold(systematic=key, closure=True, meas_flat=meas, resp_np=resp_nom)
    x_abs = np.asarray(u.y_unf if key == "nominal" else u.y_unf_dict[key], dtype=float)

    # Statistical uncertainty of this closure point: input (pseudo-data) + matrix
    # (response MC) covariance, propagated through the per-slice normalization.
    cov_abs = u.closure_cov_input[key] + u.closure_cov_matrix[key]
    jac = _norm_jacobian(u, x_abs)
    norm_cov = jac @ cov_abs @ jac.T
    norm_stat = np.sqrt(np.clip(np.diag(norm_cov), 0.0, None))
    norm_truth = normalize(u, truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_err = np.where(norm_truth != 0, norm_stat / norm_truth, 0.0)

    return normalize(u, x_abs), norm_truth, frac_err


def run_mode(groomed, tag):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n=== Closure test ({mode}) ===")
    spec = get_spec("zjet", "rho", tag)
    # match the CMS label flavor of the neighbouring closure figures in the AN
    # (the split-sample closure figure carries "Simulation Preliminary")
    u = ClosureUnfolder(spec, groomed=groomed, do_syst=True,
                        compute_jackknife_stat=False, cms_label="Preliminary")
    # The lowest pT slice (0-200 GeV) is not a reported bin -- skip it so only the
    # analysis pT bins (200-290, 290-400, 400-inf) get a figure.
    u.first_reported_pt_bin = 1

    # base name -> response key actually used (nominal maps to itself)
    keys = {"nominal": "nominal", **u.closure_keys}

    # Total statistical-uncertainty band of the unfolded data result =
    # sqrt(input^2 + matrix^2), jacobian-propagated through the regularized
    # covariance during construction (input = data stat, matrix = response MC
    # stat / "uncorr"). Read before any closure re-unfold overwrites the
    # stored covariances.
    stat_frac = [
        np.asarray(u.normalized_results[i]["stat_unc_frac"], dtype=float)
        for i in range(len(u.normalized_results))
    ]

    results = {}
    errors = {}
    for base, key in keys.items():
        norm_unf, norm_truth, frac_err = closure(u, key)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where(norm_truth != 0, norm_unf / norm_truth - 1.0, 0.0)
        results[base] = resid
        errors[base] = frac_err
        per_pt = unflatten_gen_by_pt(resid, u.gen_edges_by_pt)
        print(
            f"  {LABELS[base]:18s} ({key:>22s})  "
            f"max |unf/truth-1| = {np.max(np.abs(resid)):.3e}  "
            f"per-pT max: " + ", ".join(f"{np.max(np.abs(b)):.1e}" for b in per_pt)
        )

    _plot(u, results, errors, keys, stat_frac, mode)
    return results


def _pt_bin_label(u, i):
    lo = u.pt_edges[i]
    hi = u.pt_edges[i + 1] if i + 2 < len(u.pt_edges) else float("inf")
    hi_s = "∞" if not np.isfinite(hi) else f"{hi:.0f}"
    return f"{lo:.0f}–{hi_s}"


def _plot(u, results, errors, keys, stat_frac, mode):
    """One CMS-style figure per pT slice (x cut to the spec's reported range).

    Residuals are shown as fractions (unfolded/truth - 1). The gray band is the
    jacobian-propagated, regularized TOTAL statistical uncertainty of the unfolded
    data result (input data stat + response-matrix MC stat); each variation
    additionally carries its own per-bin total-stat error bar (input + matrix of
    that closure unfold), so one can read off whether it is compatible with
    closure within uncertainty.
    """
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xlo, _ = u._observable_xlim()
    for i in u._reported_pt_indices():
        edges = np.asarray(u.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        visible = edges[:-1] >= xlo  # bins inside the displayed x-window
        ymax = 0.0
        fig = plt.figure()

        # statistical-uncertainty band of the unfolded (data) result, around zero
        band = np.asarray(stat_frac[i], dtype=float)
        band_step = np.append(band, band[-1])
        plt.fill_between(
            edges, -band_step, band_step, step="post",
            color="0.8", alpha=0.6, lw=0, label="Stat. unc. (total)",
        )
        if visible.any():
            ymax = max(ymax, np.max(band[visible]))

        for base in keys:
            vals = unflatten_gen_by_pt(results[base], u.gen_edges_by_pt)[i]
            errs = unflatten_gen_by_pt(errors[base], u.gen_edges_by_pt)[i]
            color = COLORS.get(base, None)
            if visible.any():
                ymax = max(ymax, np.max(np.abs(vals[visible]) + errs[visible]))
            hep.histplot(vals, edges, label=LABELS[base], color=color, lw=2.0)
            # per-variation stat error bar at the bin centers
            plt.errorbar(
                centers, vals, yerr=errs, fmt="none", ecolor=color,
                elinewidth=1.4, capsize=2, alpha=0.9,
            )
        plt.axhline(0, color="gray", lw=1)
        # Scale to the content of the *displayed* window only (`visible` already
        # drops the hidden buffer bins whose low-stat corners blow up to O(100%)).
        # No hard cap: ymax includes the stat band and every per-variation error
        # bar, so nothing is clipped -- a clipped error bar would hide exactly the
        # information the reader needs to judge whether an edge bin closes.
        ylim = max(0.01, 1.25 * ymax)
        plt.ylim(-ylim, ylim)
        plt.xlim(*u._observable_xlim(i))
        plt.xlabel(u._observable_label())
        plt.ylabel("Unfolded / Truth $-$ 1")
        plt.legend(title=rf"$p_{{T}}$  {_pt_bin_label(u, i)} GeV", fontsize=14)
        hep.cms.label(
            u.cms_label, data=False, lumi=u._lumi_label(), com=u._com_label(), fontsize=20
        )
        ax = plt.gca()
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)
        plt.subplots_adjust(left=0.16, bottom=0.15)
        for ext in ("pdf", "png"):
            path = OUT_DIR / f"closure_systematics_{mode}_{i}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    print(f"  saved {len(list(u._reported_pt_indices()))} per-pT figures "
          f"-> {OUT_DIR}/closure_systematics_{mode}_*.pdf")


def main():
    # jacobian + ratio-curvature regularization on, so the stat band is the
    # jacobian-propagated regularized statistical uncertainty.
    tag = sys.argv[1] if len(sys.argv) > 1 else "original_jacobian_reg"
    run_mode(groomed=True, tag=tag)
    run_mode(groomed=False, tag=tag)


if __name__ == "__main__":
    main()
