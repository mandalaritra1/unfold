#!/usr/bin/env python3
"""ARC round-3 WS6: what the per-pT normalization does to the uncertainties.

The ARC asked us to (a) describe the normalization procedure, (b) show the
correlation matrix before and after it, and (c) verify their expectation that
JES shrinks and the pT-bin statistical correlations grow.

Everything here comes from a *single* Unfolder instance per grooming state:
the absolute (pre-normalization) quantities are TUnfold's own output
(``y_unf``, ``y_unf_dict``, ``cov_data_np`` + ``cov_uncorr_np``) and the
normalized ones are the analytic transform of exactly those objects
(``normalized_results``, ``normalized_systematics``, ``norm_cov_stat``).  No
second unfolding is run, so any difference is the normalization and nothing
else.

Outputs (per grooming state), under ``<tag>/unfold/``:

  * ``normcheck_correlation_before_after_<mode>.pdf`` -- stat correlation of
    the absolute spectrum next to that of the normalized one, reported bins.
  * ``normcheck_jes_<mode>_pt<i>.pdf`` -- total JES fractional uncertainty
    before and after normalization, one square panel per reported pT bin.
  * ``normcheck_summary_<mode>.json`` -- the numbers quoted in the reply.

Luminosity is deliberately absent from the systematics list: it is a coherent
multiplicative factor common to every bin of a pT slice, so the normalization
Jacobian annihilates it exactly.  That is checked numerically here rather than
carried as a nuisance.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplhep as hep

from unfold.tools.unfolder_core import Unfolder, get_spec

hep.style.use(hep.style.CMS)

DEFAULT_TAG = "jmsjmr_unity_groomed400_floor3"


# --------------------------------------------------------------------------
# bookkeeping helpers
# --------------------------------------------------------------------------
def flat_offsets(unfolder):
    counts = [len(edges) - 1 for edges in unfolder.gen_edges_by_pt]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def reported_blocks(unfolder):
    """Flat indices of the *reported* gen bins, grouped by pT slice.

    Same definition the published figures use: pT slices from
    ``first_reported_pt_bin`` on, and within each slice only the rho bins at or
    above that slice's shown floor (the window the normalization runs over,
    since the spec sets ``normalize_over_shown``).
    """
    starts, counts = flat_offsets(unfolder)
    floors = unfolder._bl_shown_floors()
    # Published figures report from index 1 on: the 185-200 GeV slice is the
    # migration sink, not a measurement (cf. max(1, first_reported_pt_bin) in
    # the plotting paths).
    first = max(1, getattr(unfolder, "first_reported_pt_bin", 0))
    blocks, labels = [], []
    for i, edges in enumerate(unfolder.gen_edges_by_pt):
        if i < first:
            continue
        lower = np.asarray(edges[:-1], float)
        block = [
            starts[i] + j for j in range(counts[i]) if lower[j] >= floors[i] - 1e-9
        ]
        if not block:
            continue
        blocks.append(block)
        hi = unfolder.pt_edges[i + 1] if i + 1 < len(unfolder.pt_edges) - 1 else None
        labels.append(
            f"{int(unfolder.pt_edges[i])}-{int(hi)}" if hi else
            f"{int(unfolder.pt_edges[i])}-inf"
        )
    return blocks, labels


# Keys are era-decorrelated: "JES_<Source>Up_corr" and
# "JES_<Source>Up_uncorr_<year>" (the same split the total uncertainty uses),
# so each of those is an independent nuisance and all of them enter the sum.
JES_KEY = re.compile(r"^(JES_.+?)(Up|Down)((?:_corr|_uncorr_\d+)?)$")


def jes_source_pairs(available):
    """(source, up_name, down_name) for every JES nuisance present in ``available``.

    Discovered from the unfolder's own systematic keys rather than from the
    ``JES_SYSTEMATICS`` literal, because the loaded pkls decide which sources
    exist and how the era decorrelation spells them.
    """
    names = set(available)
    pairs = []
    for name in sorted(names):
        match = JES_KEY.match(name)
        if not match or match.group(2) != "Up":
            continue
        source, _, suffix = match.groups()
        down = f"{source}Down{suffix}"
        if down in names:
            pairs.append((source + suffix, name, down))
    return pairs


def normalized_flat(unfolder, key=None):
    """Flat normalized vector: nominal (key=None) or one systematic."""
    if key is None:
        return np.concatenate(
            [np.asarray(r["unfolded"], float) for r in unfolder.normalized_results]
        )
    return np.concatenate(
        [
            np.asarray(per_pt["unfolded"][key], float)
            for per_pt in unfolder.normalized_systematics
        ]
    )


def jes_fractional(unfolder):
    """Total JES fractional uncertainty, absolute and normalized, flat vectors.

    Symmetrized per source, (up - down) / 2, then summed in quadrature over
    sources.  The *same* definition is used on both sides so the ratio between
    them is a clean statement about the normalization.
    """
    x_abs = np.asarray(unfolder.y_unf, float)
    y_norm = normalized_flat(unfolder)

    var_abs = np.zeros_like(x_abs)
    var_norm = np.zeros_like(y_norm)
    used = []
    pairs = jes_source_pairs(unfolder.y_unf_dict.keys())
    if not pairs:
        print("  !! no JES up/down pairs found; available keys:")
        print("     " + ", ".join(sorted(unfolder.y_unf_dict.keys())))
    for source, up, down in pairs:
        used.append(source)
        d_abs = 0.5 * (
            np.asarray(unfolder.y_unf_dict[up], float)
            - np.asarray(unfolder.y_unf_dict[down], float)
        )
        d_norm = 0.5 * (
            normalized_flat(unfolder, up) - normalized_flat(unfolder, down)
        )
        var_abs += d_abs ** 2
        var_norm += d_norm ** 2

    frac_abs = np.divide(
        np.sqrt(var_abs), np.abs(x_abs),
        out=np.zeros_like(x_abs), where=x_abs != 0,
    )
    frac_norm = np.divide(
        np.sqrt(var_norm), np.abs(y_norm),
        out=np.zeros_like(y_norm), where=y_norm != 0,
    )
    return frac_abs, frac_norm, used


def lumi_cancellation_check(unfolder, scale=0.016):
    """Scale the absolute spectrum coherently and renormalize by hand.

    A luminosity variation multiplies every bin of every pT slice by the same
    factor.  Since each slice is normalized by its own sum, the factor cancels
    identically; this returns the largest residual over the reported bins,
    which should be at machine precision.
    """
    blocks, _ = reported_blocks(unfolder)
    worst = 0.0
    for block in blocks:
        x = np.asarray(unfolder.y_unf, float)[block]
        scaled = x * (1.0 + scale)
        base = x / x.sum()
        varied = scaled / scaled.sum()
        worst = max(worst, float(np.max(np.abs(varied - base))))
    return worst


def correlation(cov):
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    d[d == 0] = 1e-30
    return cov / np.outer(d, d)


def cross_pt_correlation_summary(corr, blocks):
    """Mean |corr| within pT blocks (off-diagonal) and across pT blocks."""
    offsets, size = [], 0
    for block in blocks:
        offsets.append((size, size + len(block)))
        size += len(block)
    within, across = [], []
    for a, (i0, i1) in enumerate(offsets):
        for b, (j0, j1) in enumerate(offsets):
            sub = corr[i0:i1, j0:j1]
            if a == b:
                mask = ~np.eye(sub.shape[0], dtype=bool)
                within.append(np.abs(sub[mask]))
            elif b > a:
                across.append(np.abs(sub).ravel())
    return (
        float(np.mean(np.concatenate(within))) if within else float("nan"),
        float(np.mean(np.concatenate(across))) if across else float("nan"),
    )


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def discrete_seismic(nbins=20):
    bounds = np.linspace(-1, 1, nbins + 1)
    colors = plt.get_cmap("seismic", nbins)(np.linspace(0, 1, nbins))
    for i in range(len(bounds) - 1):
        if -0.1 <= bounds[i] <= 0.1:
            colors[i] = [1, 1, 1, 1]
    cmap = mcolors.ListedColormap(colors)
    return cmap, mcolors.BoundaryNorm(bounds, cmap.N), bounds


def plot_correlation_before_after(unfolder, corr_abs, corr_norm, labels,
                                  blocks, out_path):
    cmap, norm, bounds = discrete_seismic()
    fig, axes = plt.subplots(1, 2, figsize=(21, 10.6), layout="constrained")
    edges = np.r_[0, np.cumsum([len(b) for b in blocks])]
    centers = (edges[:-1] + edges[1:] - 1) / 2.0

    panels = (
        (axes[0], corr_abs, "before normalization"),
        (axes[1], corr_norm, "after normalization"),
    )
    for ax, corr, title in panels:
        img = ax.imshow(corr, cmap=cmap, norm=norm, origin="lower")
        for x in edges[1:-1]:
            ax.axvline(x - 0.5, color="r", ls="--", lw=2, alpha=0.6)
            ax.axhline(x - 0.5, color="r", ls="--", lw=2, alpha=0.6)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)
        ax.set_yticks(centers)
        ax.set_yticklabels(labels, rotation=90, va="center")
        ax.set_xlabel(r"GEN $p_{T}$ (GeV)")
        ax.set_ylabel(r"GEN $p_{T}$ (GeV)")
        cbar = fig.colorbar(img, ax=ax, ticks=bounds, boundaries=bounds,
                            fraction=0.046, pad=0.04)
        cbar.set_label("Statistical correlation")

    # mplhep freezes the CMS <-> "Internal" gap as an *axes fraction* when the
    # label is stamped, and every colorbar added afterwards renarrows the axes
    # under constrained layout -- which squeezes that frozen gap shut. So let
    # the layout settle, freeze it, and only then stamp both labels.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, _, title in panels:
        hep.cms.label(
            unfolder.cms_label, data=True, lumi=unfolder._lumi_label(),
            com=unfolder._com_label(), ax=ax, fontsize=20,
            # Which side of the normalization this panel is goes in the
            # rlabel; an in-frame text box would sit on the matrix cells.
            rlabel=("Groomed" if unfolder.groomed else "Ungroomed") + f", {title}",
        )

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_jes_panel(unfolder, edges, frac_abs, frac_norm, label, out_path):
    fig, ax = plt.subplots(layout="constrained")
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.step(edges, np.r_[frac_abs, frac_abs[-1]], where="post",
            color="#5790fc", lw=2.5, label="Absolute (before normalization)")
    ax.step(edges, np.r_[frac_norm, frac_norm[-1]], where="post",
            color="#e42536", lw=2.5, ls="--",
            label="Normalized (after)")
    ax.plot(centers, frac_abs, "o", color="#5790fc", ms=6)
    ax.plot(centers, frac_norm, "s", color="#e42536", ms=6)

    ax.set_xlabel(r"$\log_{10}(\rho^{2})$")
    ax.set_ylabel("Fractional JES uncertainty")
    top = max(frac_abs.max(), frac_norm.max())
    ax.set_ylim(0, top * 1.5 if top > 0 else 1.0)
    ax.set_xlim(edges[0], edges[-1])
    ax.legend(loc="upper left", fontsize=18)
    hep.cms.label(
        unfolder.cms_label, data=True, lumi=unfolder._lumi_label(),
        com=unfolder._com_label(), fontsize=20, ax=ax,
        rlabel=(("Groomed" if unfolder.groomed else "Ungroomed")
                + rf",  $p_{{T}}$ {label} GeV"),
    )
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def run(groomed, tag, out_dir):
    mode = "groomed" if groomed else "ungroomed"
    print(f"[{mode}] building unfolder (tag={tag}) ...")
    spec = get_spec("zjet", "rho", tag)
    unfolder = Unfolder(spec, groomed, do_syst=True, cms_label="Internal")

    blocks, labels = reported_blocks(unfolder)
    idx = [i for block in blocks for i in block]

    # --- correlations -----------------------------------------------------
    cov_abs = np.asarray(unfolder.cov_data_np) + np.asarray(unfolder.cov_uncorr_np)
    cov_norm = np.asarray(unfolder.norm_cov_stat)
    corr_abs = correlation(cov_abs[np.ix_(idx, idx)])
    corr_norm = correlation(cov_norm[np.ix_(idx, idx)])

    plot_correlation_before_after(
        unfolder, corr_abs, corr_norm, labels, blocks,
        out_dir / f"normcheck_correlation_before_after_{mode}.pdf",
    )

    within_abs, across_abs = cross_pt_correlation_summary(corr_abs, blocks)
    within_norm, across_norm = cross_pt_correlation_summary(corr_norm, blocks)

    # Legacy vs Jacobian *diagonal*: legacy carries the relative error of the
    # absolute spectrum onto the normalized value, i.e. it keeps the common
    # rate fluctuation that the normalization actually removes.  If the two
    # diagonals agree, the published error bars are unaffected by which
    # correlation matrix is shown; if they do not, the difference is the
    # over-coverage of the legacy stat band.
    x_abs = np.asarray(unfolder.y_unf, float)
    y_norm = normalized_flat(unfolder)
    rel_abs = np.divide(
        np.sqrt(np.clip(np.diag(cov_abs), 0.0, None)), np.abs(x_abs),
        out=np.zeros_like(x_abs), where=x_abs != 0,
    )
    legacy_err = rel_abs * y_norm
    jacobian_err = np.sqrt(np.clip(np.diag(cov_norm), 0.0, None))
    diag_rows = []
    for block, label in zip(blocks, labels):
        ratio = np.divide(
            jacobian_err[block], legacy_err[block],
            out=np.full(len(block), np.nan), where=legacy_err[block] > 0,
        )
        diag_rows.append(
            {
                "pt_bin": label,
                "mean_legacy_frac": float(
                    np.mean(legacy_err[block] / np.abs(y_norm[block]))
                ),
                "mean_jacobian_frac": float(
                    np.mean(jacobian_err[block] / np.abs(y_norm[block]))
                ),
                "mean_ratio_jacobian_over_legacy": float(np.nanmean(ratio)),
            }
        )

    # Rank of each normalized pT block: the sum constraint removes one dof.
    ranks = []
    offset = 0
    for block in blocks:
        n = len(block)
        sub = cov_norm[np.ix_(block, block)]
        eig = np.linalg.eigvalsh(sub)
        ranks.append(
            {
                "n_bins": n,
                "rank": int(np.sum(eig > eig.max() * 1e-10)),
                "min_over_max_eigenvalue": float(eig.min() / eig.max()),
            }
        )
        offset += n

    # --- JES --------------------------------------------------------------
    frac_abs, frac_norm, used = jes_fractional(unfolder)
    jes_rows = []
    for block, label in zip(blocks, labels):
        edges = None
        # gen edges of this slice, cropped to the reported window
        for i, e in enumerate(unfolder.gen_edges_by_pt):
            starts, counts = flat_offsets(unfolder)
            if starts[i] <= block[0] < starts[i] + counts[i]:
                local = [j - starts[i] for j in block]
                edges = np.asarray(e, float)[local[0]:local[-1] + 2]
                break
        a = frac_abs[block]
        n = frac_norm[block]
        plot_jes_panel(
            unfolder, edges, a, n, label,
            out_dir / f"normcheck_jes_{mode}_pt{labels.index(label)}.pdf",
        )
        jes_rows.append(
            {
                "pt_bin": label,
                "mean_abs": float(np.mean(a)),
                "mean_norm": float(np.mean(n)),
                "max_abs": float(np.max(a)),
                "max_norm": float(np.max(n)),
                "mean_ratio_norm_over_abs": float(np.mean(n) / np.mean(a))
                if np.mean(a) > 0 else float("nan"),
            }
        )

    summary = {
        "tag": tag,
        "mode": mode,
        "n_jes_sources": len(used),
        "lumi_residual_after_normalization": lumi_cancellation_check(unfolder),
        "stat_correlation": {
            "mean_abs_within_pt_before": within_abs,
            "mean_abs_within_pt_after": within_norm,
            "mean_abs_across_pt_before": across_abs,
            "mean_abs_across_pt_after": across_norm,
        },
        "normalized_block_rank": ranks,
        "stat_diagonal_legacy_vs_jacobian": diag_rows,
        "jes": jes_rows,
    }
    path = out_dir / f"normcheck_summary_{mode}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--mode", choices=["groomed", "ungroomed", "both"],
                        default="both")
    args = parser.parse_args()

    spec = get_spec("zjet", "rho", args.tag)
    out_dir = Path(spec.output_dir) / "unfold"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [True, False] if args.mode == "both" else [args.mode == "groomed"]
    for groomed in modes:
        summary = run(groomed, args.tag, out_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
