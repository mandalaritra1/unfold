# Salvaged from smp_jetmass_run2, 2026-08-13

Everything unfolding-related was removed from `smp_jetmass_run2` — that work
belongs in this repo. Kept here as raw material only; Aritra is rebuilding
from scratch, so treat these as reference, not as a package to import.

`studies_unfold/` — the 17 files that were `scripts/studies/unfold/`:

- **11 were tracked** in smp_jetmass_run2 and are also recoverable from its git
  history (up to commit b4e8e37). Three of them — `rho_profiled_refold.py`,
  `rho_unfold_crosschecks.py`, `rho_unfold_systematics.py` — are the
  **working-tree** versions, carrying ~250 lines of uncommitted work that
  added a `--binning` option (`default="coarse_tail"`, described in-file as
  the production choice). That work exists nowhere else.
- **6 were untracked**, written 2026-08-11, and exist nowhere else:
  `combine_era_rho_hists.py`, `rho_config_comparison.py`, `rho_era_chi2.py`,
  `rho_iterative_bayes.py`, `rho_jes_era_correlated.py`, `rho_phaseb_report.py`.

`appendix_rerun_pairsplit_run2.py` — untracked smp plotting script that imported
four of the above (`rho_unfold_systematics`, `rho_unfold_crosschecks`,
`rho_profiled_refold`, `rho_iterative_bayes`). It cannot run in smp_jetmass_run2
any more, which is why it came along.

`test_rho_profiled_refold.py` — the smp test covering `rho_profiled_refold`.

`UNFOLD_STATUS.md`, `UNFOLDING_REPORT_2026-08-11.md` — the 2026-08-11 status and
report docs from the smp repo root.

Note: the scripts still carry smp-style `sys.path` bootstraps and
`from scripts.studies.unfold.…` imports; both are meaningless here.
