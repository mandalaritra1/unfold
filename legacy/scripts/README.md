# Script map

Run commands from the repository root.  The supported production entry points
are kept at this level:

```bash
source scripts/setup_root.sh
python scripts/run_unfolding.py --channel zjet --observable rho --tag original
python scripts/run_rho_unfolding.py --channel dijet --tag <tag>
python scripts/run_pairsplit_unfolding.py --channel dijet   # pair-split Run-2 inputs
```

The remaining scripts are grouped by purpose.  They are opt-in tools rather
than part of the default production workflow.

| Directory | Purpose |
| --- | --- |
| `staging/` | Prepare, combine, and inspect external input pickles. |
| `diagnostics/` | Purity, closure, normalization, covariance, and response checks. |
| `plotting/` | Re-render figures and build image grids from existing outputs. |
| `studies/` | Explicit alternate-unfolding, regularization, model-closure, and Combine studies. |
| `release/` | Export validated results and assemble a HEPData submission. |
| `_superseded/` | Quarantined scripts replaced by current tooling; see its README before using anything here. |

The pair-split plot book and slide deck builders live in `plotting/`
(`build_pairsplit_all_modes_plot_book.py`, `build_pairsplit_slide_deck.py`);
they read finished runs from `outputs/pairsplit_run2/` via their manifests.

Each script exposes its options with `--help` when it has a command-line
interface.  Scripts that write results use `outputs/`, which is intentionally
ignored by Git; retain important generated products in the configured CERNBox
archive rather than committing them.
