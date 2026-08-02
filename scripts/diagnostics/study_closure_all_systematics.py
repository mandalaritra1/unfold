#!/usr/bin/env python3
"""Per-systematic closure over *every* systematic source (ARC Fig. 44 follow-up).

`study_closure_systematics.py` produces the AN figures for three representative
sources (a JES source, JER, pileup). The ARC asked us to *check* that the
unfolding closes for all systematic variations one by one, even if only a few
are shown. This driver runs the same test for every response-matrix variation
in `sys_matrix_dic` and prints a table, so the statement in the AN is backed by
an actual scan rather than by the three plotted sources.

The test is the same one: the reco projection of the *varied* response is
unfolded with the *nominal* response + nominal misses and compared with the
nominal gen truth. `herwig*` is excluded -- it is a different generator, so the
gen truth changes and "unfolded == nominal truth" is not the expectation (that
case is covered by the model-closure / bias tests instead).

Residuals are evaluated only over the reported pT bins and inside the displayed
observable window, i.e. the region that enters the published result; the hidden
buffer bins and the unreported low-pT sink bin are excluded.

Usage:
  source scripts/setup_root.sh
  .venv/bin/python scripts/diagnostics/study_closure_all_systematics.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from unfold.tools.unfolder_core import get_spec
from unfold.utils.merge_helpers import unflatten_gen_by_pt

from study_closure_systematics import ClosureUnfolder, closure

OUT_DIR = REPO_ROOT / "outputs" / "zjet" / "validation"

# Different generator -> the gen truth is not the nominal one, so this closure
# definition does not apply (covered by the model-closure test instead).
EXCLUDE_PREFIXES = ("herwig",)


class AllClosureUnfolder(ClosureUnfolder):
    """Closure unfolder that keeps every detector-level response variation."""

    def _configure_systematics(self, do_syst):
        keys = [
            k for k in self.sys_matrix_dic
            if k == "nominal" or not k.lower().startswith(EXCLUDE_PREFIXES)
        ]
        self.closure_keys = {k: k for k in keys if k != "nominal"}
        self.systematics = keys
        print(f"  scanning {len(keys) - 1} systematic variations "
              f"(+ nominal); excluded {len(self.sys_matrix_dic) - len(keys)}")


def _reported_mask(u):
    """Flat gen-level boolean mask: reported pT bins, inside the shown x-window."""
    blocks = []
    reported = set(u._reported_pt_indices())
    for i, edges in enumerate(u.gen_edges_by_pt):
        edges = np.asarray(edges, dtype=float)
        nbins = len(edges) - 1
        if i not in reported:
            blocks.append(np.zeros(nbins, dtype=bool))
            continue
        xlo, xhi = u._observable_xlim(i)
        # a bin counts as shown when it lies inside the displayed window
        blocks.append((edges[:-1] >= xlo) & (edges[1:] <= xhi + 1e-9))
    return np.concatenate(blocks)


def run_mode(groomed, tag):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n=== Closure scan over all systematics ({mode}) ===")
    spec = get_spec("zjet", "rho", tag)
    u = AllClosureUnfolder(spec, groomed=groomed, do_syst=True,
                           compute_jackknife_stat=False)
    u.first_reported_pt_bin = 1

    mask = _reported_mask(u)
    stat_frac = np.concatenate([
        np.asarray(u.normalized_results[i]["stat_unc_frac"], dtype=float)
        for i in range(len(u.normalized_results))
    ])

    rows = []
    for key in ["nominal"] + sorted(u.closure_keys):
        norm_unf, norm_truth, frac_err = closure(u, key)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where(norm_truth != 0, norm_unf / norm_truth - 1.0, 0.0)
            # pull against the variation's own statistical uncertainty
            pull = np.where(frac_err > 0, np.abs(resid) / frac_err, 0.0)
            # size relative to the data result's total statistical uncertainty
            rel_band = np.where(stat_frac > 0, np.abs(resid) / stat_frac, 0.0)
        m = mask
        rows.append((
            key,
            float(np.max(np.abs(resid[m]))),
            float(np.max(pull[m])),
            float(np.max(rel_band[m])),
        ))

    rows_var = [r for r in rows if r[0] != "nominal"]
    nominal = next(r for r in rows if r[0] == "nominal")

    print(f"\n  nominal self-closure: max |unf/truth-1| = {nominal[1]:.2e}")
    print(f"  {len(rows_var)} variations scanned over the reported region\n")
    print(f"  {'source':38s} {'max|res|':>10s} {'max pull':>9s} {'max/band':>9s}")
    for key, res, pull, rel in sorted(rows_var, key=lambda r: -r[2])[:15]:
        print(f"  {key:38s} {res:10.4f} {pull:9.2f} {rel:9.2f}")
    print("  ... (worst 15 by pull shown)")

    worst_pull = max(r[2] for r in rows_var)
    worst_res = max(r[1] for r in rows_var)
    worst_band = max(r[3] for r in rows_var)
    n_over_1sig = sum(1 for r in rows_var if r[2] > 1.0)
    n_over_band = sum(1 for r in rows_var if r[3] > 1.0)
    print(f"\n  SUMMARY ({mode}): max |residual| = {worst_res:.4f}, "
          f"max pull vs own stat = {worst_pull:.2f}, "
          f"max |residual| / data stat band = {worst_band:.2f}")
    print(f"  variations with max pull > 1: {n_over_1sig}/{len(rows_var)}; "
          f"exceeding the data stat band: {n_over_band}/{len(rows_var)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"closure_all_systematics_{mode}.txt"
    with open(out, "w") as f:
        f.write("# per-systematic closure, reported region only "
                f"(tag={tag}, mode={mode})\n")
        f.write("# source  max|unf/truth-1|  max_pull_own_stat  max_over_data_stat_band\n")
        f.write(f"nominal {nominal[1]:.6e} - -\n")
        for key, res, pull, rel in sorted(rows_var, key=lambda r: -r[2]):
            f.write(f"{key} {res:.6f} {pull:.4f} {rel:.4f}\n")
    print(f"  wrote {out}")
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "arc_r2"
    run_mode(groomed=True, tag=tag)
    run_mode(groomed=False, tag=tag)


if __name__ == "__main__":
    main()
