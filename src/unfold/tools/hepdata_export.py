"""Reproducible export of unfolded rho results into a HEPData-ready intermediate.

This module reads a *live* :class:`unfold.tools.unfolder_core.Unfolder` instance
(after ``run_all_plots`` / the full ``run`` has populated ``normalized_results``,
``normalized_systematics`` and the TUnfold covariances) and writes a stable
``.npz`` payload plus a JSON manifest.  The ``.npz`` is the single source of
truth for the downstream HEPData YAML generator, so nothing is ever re-read from
plots.

What it captures, per *published* pT bin (the 0--200 GeV control/underflow bin is
dropped by default), for one grooming state:

* gen rho-bin edges (``log10(rho^2)``, rho = m / (pT R));
* central normalized differential value ``(1/sigma) dsigma/d log10(rho^2)``;
* statistical uncertainty (10-way jackknife, absolute);
* pure systematic uncertainty (quadrature of per-source shifts, *stat removed*),
  up/down, reconstructed with the same bucketing as the unfolder;
* the full signed per-source systematic shift table;
* the PYTHIA8 and (if available) HERWIG7 truth spectra for model comparison.

It also propagates the flat global TUnfold covariances through the per-pT-bin
normalization Jacobian to obtain normalized covariance / correlation matrices
over the published bins:

    n_k = c_k / (w_k * S_b)        (S_b = sum of raw counts in pT block b)
    J[k,l] = (1/(w_k S_b)) (delta_kl - c_k / S_b)
    Cov_norm = J Cov_raw J^T       (J is block-diagonal across pT blocks)

Both the data-statistical covariance (``GetEmatrixInput``) and the total
covariance (``GetEmatrixTotal``) are propagated and exported, together with the
correlation matrices.

Note on the statistical uncertainty: the published per-bin stat error uses the
10-way jackknife, whereas the propagated stat covariance diagonal is TUnfold's
analytic input-stat estimate.  Both are exported; the correlation *structure*
comes from TUnfold, the diagonal *magnitude* from the jackknife.  The YAML step
decides how to combine them (default: scale the TUnfold stat correlation to the
jackknife diagonal).  Nothing here silently rescales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Observable / phase-space constants for the rho measurement.  These describe the
# particle-level definition that the HEPData qualifiers must state; they are not
# read from the unfolder (the binning lives there, the physics definition here).
RHO_JET_R = 0.8
SQRT_S_TEV = 13.0


@dataclass
class _PtBlock:
    """Slice bookkeeping for one pT bin in the flat global gen vector."""

    index: int                 # original index into unfolder.pt_edges
    pt_low: float
    pt_high: float             # may be inf
    start: int                 # offset into the flat gen vector
    stop: int                  # exclusive
    edges: np.ndarray          # gen rho edges, len = (stop-start)+1


def _pt_blocks(unfolder):
    """Return the flat-vector slice for every pT bin, in flat order."""
    blocks = []
    offset = 0
    n_pt = len(unfolder.pt_edges) - 1
    for i in range(n_pt):
        edges = np.asarray(unfolder.gen_edges_by_pt[i], dtype=float)
        n = len(edges) - 1
        pt_low = float(unfolder.pt_edges[i])
        # mirror the (slightly unusual) upper-edge convention in the core code:
        # the last pT bin is open-ended.
        pt_high = float(unfolder.pt_edges[i + 1]) if i + 1 < n_pt else float("inf")
        blocks.append(_PtBlock(i, pt_low, pt_high, offset, offset + n, edges))
        offset += n
    if offset != len(unfolder.y_unf):
        raise ValueError(
            f"flat gen length mismatch: blocks cover {offset}, y_unf has {len(unfolder.y_unf)}"
        )
    return blocks


def _normalization_jacobian(unfolder, blocks):
    """Block-diagonal Jacobian d(normalized density)/d(raw count) over all bins."""
    n = len(unfolder.y_unf)
    jac = np.zeros((n, n))
    c_all = np.asarray(unfolder.y_unf, dtype=float)
    for blk in blocks:
        c = c_all[blk.start:blk.stop]
        w = np.diff(blk.edges)
        S = c.sum()
        if S == 0:
            continue
        p = c / S
        # J[k,l] = (1/(w_k S)) (delta_kl - p_k)
        inv = 1.0 / (w * S)
        block = inv[:, None] * (np.eye(len(c)) - p[None, :])
        jac[blk.start:blk.stop, blk.start:blk.stop] = block
    return jac


def _correlation(cov):
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = np.outer(d, d)
        corr = np.where(outer > 0, cov / outer, 0.0)
    return corr


def _pure_systematic(unfolder, pt_index):
    """Up/down pure-systematic (stat removed) for one pT bin.

    Replicates the bucketing in ``Unfolder._compute_total_systematic`` exactly,
    but stops before stat is folded in.
    """
    nominal = np.asarray(unfolder.normalized_results[pt_index]["unfolded"], dtype=float)
    shifts = unfolder.normalized_systematics[pt_index]["unfolded"]
    up_sq = np.zeros_like(nominal)
    down_sq = np.zeros_like(nominal)
    for syst, value in shifts.items():
        diff = np.abs(np.asarray(value, dtype=float) - nominal)
        if "Down" in syst:
            down_sq += diff ** 2
        else:
            up_sq += diff ** 2
    return np.sqrt(up_sq), np.sqrt(down_sq)


def export_unfolder(unfolder, npz_path, json_path=None, *, exclude_control_bin=True):
    """Export one grooming state of a finished :class:`Unfolder` to ``npz_path``.

    Parameters
    ----------
    unfolder : Unfolder
        A fully-run instance (``do_syst=True`` for a real submission).
    npz_path : str or Path
        Destination ``.npz``.  Parent dirs are created.
    json_path : str or Path, optional
        Manifest path; defaults to ``npz_path`` with a ``.json`` suffix.
    exclude_control_bin : bool, default True
        Drop the leading 0--200 GeV pT bin (unfolding control/underflow).
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(json_path) if json_path is not None else npz_path.with_suffix(".json")

    blocks = _pt_blocks(unfolder)
    jac = _normalization_jacobian(unfolder, blocks)

    # Propagate the flat-global covariances into normalized density space.
    cov_stat_norm_full = jac @ np.asarray(unfolder.cov_data_np) @ jac.T
    cov_total_norm_full = jac @ np.asarray(unfolder.cov_np) @ jac.T

    published = [b for b in blocks if not (exclude_control_bin and b.index == 0)]
    if not published:
        raise ValueError("no published pT bins remain after exclusion")

    # Indices, in flat order, of the published gen bins (for slicing the matrices).
    pub_idx = np.concatenate([np.arange(b.start, b.stop) for b in published])
    cov_stat_pub = cov_stat_norm_full[np.ix_(pub_idx, pub_idx)]
    cov_total_pub = cov_total_norm_full[np.ix_(pub_idx, pub_idx)]

    grooming = "groomed" if unfolder.groomed else "ungroomed"

    payload = {
        # --- combined published-bin matrices (flat order = concatenated pT bins) ---
        "cov_stat_norm": cov_stat_pub,
        "corr_stat_norm": _correlation(cov_stat_pub),
        "cov_total_norm": cov_total_pub,
        "corr_total_norm": _correlation(cov_total_pub),
        # block layout so the YAML step can split the combined matrix back per pT bin
        "block_lengths": np.array([b.stop - b.start for b in published], dtype=int),
    }

    systematics = [s for s in unfolder.systematics if s != "nominal"]
    manifest_bins = []

    for out_i, blk in enumerate(published):
        i = blk.index
        res = unfolder.normalized_results[i]
        nominal = np.asarray(res["unfolded"], dtype=float)
        stat = np.asarray(res["stat_unc"], dtype=float)
        syst_up_pure, syst_down_pure = _pure_systematic(unfolder, i)
        prefix = f"pt{out_i}"

        payload[f"{prefix}__edges"] = blk.edges
        payload[f"{prefix}__value"] = nominal
        payload[f"{prefix}__stat"] = stat
        payload[f"{prefix}__syst_up"] = syst_up_pure
        payload[f"{prefix}__syst_down"] = syst_down_pure
        payload[f"{prefix}__total_up"] = np.sqrt(syst_up_pure ** 2 + stat ** 2)
        payload[f"{prefix}__total_down"] = np.sqrt(syst_down_pure ** 2 + stat ** 2)
        payload[f"{prefix}__true_pythia"] = np.asarray(res["true"], dtype=float)

        # per-source signed shifts (syst - nominal), one array per nuisance key
        shifts = unfolder.normalized_systematics[i]["unfolded"]
        for syst in systematics:
            if syst in shifts:
                payload[f"{prefix}__shift__{syst}"] = (
                    np.asarray(shifts[syst], dtype=float) - nominal
                )

        # consistency check vs the value the unfolder itself stored
        stored_up = np.asarray(res["syst_unc"]["up"], dtype=float)
        max_dev = float(np.max(np.abs(payload[f"{prefix}__total_up"] - stored_up))) if stored_up.size else 0.0

        manifest_bins.append({
            "out_index": out_i,
            "src_index": i,
            "pt_low": blk.pt_low,
            "pt_high": None if np.isinf(blk.pt_high) else blk.pt_high,
            "n_bins": int(blk.stop - blk.start),
            "recomputed_vs_stored_total_max_abs_dev": max_dev,
        })

    np.savez(npz_path, **payload)

    spec = unfolder.spec
    manifest = {
        "observable": "log10(rho^2)",
        "rho_definition": "rho = m / (pT * R)",
        "jet_R": RHO_JET_R,
        "sqrt_s_TeV": SQRT_S_TEV,
        "grooming": grooming,
        "normalization": "per-pT-bin unit-area density: (1/sigma) dsigma/d log10(rho^2)",
        "stat_method": "10-way jackknife (input + response), absolute",
        "syst_convention": "quadrature of per-source |shift - nominal|, up/down buckets, stat excluded",
        "control_bin_excluded": exclude_control_bin,
        "pt_edges_all": [float(x) for x in unfolder.pt_edges],
        "published_pt_bins": manifest_bins,
        "systematics": systematics,
        "spec_name": getattr(spec, "name", None),
        "input_dir": getattr(spec, "input_dir", None),
        "output_dir": getattr(spec, "output_dir", None),
        "covariance_note": (
            "cov/corr matrices are in normalized density space over the concatenated "
            "published pT bins; split with 'block_lengths'. Stat from GetEmatrixInput, "
            "total from GetEmatrixTotal, propagated via the normalization Jacobian."
        ),
        "npz_keys": sorted(payload.keys()),
    }
    json_path.write_text(json.dumps(manifest, indent=2))
    return {"npz": str(npz_path), "json": str(json_path), "manifest": manifest}


def export_all(unfolders, out_dir, *, exclude_control_bin=True):
    """Export a {mode_name: Unfolder} mapping (e.g. groomed/ungroomed).

    Returns a dict of per-mode export info.
    """
    out_dir = Path(out_dir)
    results = {}
    for mode_name, unfolder in unfolders.items():
        npz_path = out_dir / f"hepdata_export_{mode_name}.npz"
        results[mode_name] = export_unfolder(
            unfolder, npz_path, exclude_control_bin=exclude_control_bin
        )
    return results
