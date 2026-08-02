# Script map

Run commands from the repository root.  The supported production entry points
are kept at this level:

```bash
source scripts/setup_root.sh
python scripts/run_unfolding.py --channel zjet --observable rho --tag original
python scripts/run_rho_unfolding.py --channel dijet --tag <tag>
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

Each script exposes its options with `--help` when it has a command-line
interface.  Scripts that write results use `outputs/`, which is intentionally
ignored by Git; retain important generated products in the configured CERNBox
archive rather than committing them.
