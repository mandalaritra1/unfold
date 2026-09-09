# legacy

The repository as it was before the 2026-09-09 restructure, moved here
unchanged (git history is intact; the tag `snapshot-2026-09-09` marks the
last commit of the old layout).  Nothing in `src/unfold/` imports from this
directory.

* `src/unfold/tools/unfolder_core.py` was the single 8,600-line class that the
  new `engine.py`, `plots.py`, `inputs.py` and `zjet_inputs.py` were carved out
  of.
* `scripts/` holds the ARC-round studies (bottom-line variants, Combine
  cross-checks, model-closure and regularization scans, binning studies).
  They import `unfold.tools.*` and therefore only run against the old tree:

  ```bash
  git worktree add ../unfold-snapshot snapshot-2026-09-09
  cd ../unfold-snapshot && ln -s ../unfold/inputs inputs && ln -s ../unfold/.venv .venv
  source scripts/setup_root.sh && .venv/bin/python scripts/studies/<study>.py
  ```

* `notebooks/` are the interactive drivers and the stale ones
  (`data_mc_rho.ipynb` imports modules that no longer existed even before the
  move).  The data/MC validation CLIs that were in here moved to
  `scripts/datamc/`.
* `tests/` are the old tests, including the two large ones that exercised
  `Unfolder` internals through `__new__`; the regression is now the golden
  comparison in `tests/test_golden.py`.
* `_from_smp_20260813/` is raw material salvaged from `smp_jetmass_run2`.
* `data/` holds the mass and rho spline files used by the `derive_h_over_p`
  notebooks.
* `tools/image_grid_composer/` is the browser-only image grid tool.
