# docs

Method notes written during the analysis.  They predate the 2026-09-09
restructure, so their file references use the old layout; the mapping is:

| old | now |
|---|---|
| `scripts/run_unfolding.py --channel zjet --tag T` | `unfold run --channel zjet --tag T` |
| `scripts/run_rho_unfolding.py --channel dijet` | `unfold run --channel dijet --tag 2018` |
| `scripts/run_pairsplit_unfolding.py --channel dijet` | `unfold run --channel dijet` |
| zjet tag `jmsjmr_unity_groomed400_floor3` | zjet tag `original` |
| `outputs/pairsplit_run2/<channel>/aligned/<fingerprint>/` | `outputs/<channel>/rho/original/<mode>/` |
| `src/unfold/tools/unfolder_core.py` (`Unfolder`, `RHO_SPECS`, `ObservableSpec`) | `src/unfold/engine.py`, `config.py` (`TAGS`) |
| `src/unfold/tools/binning.py` (`bin_edges`) | `src/unfold/binning.py` (`ZJET_BINNINGS`) |
| `src/unfold/tools/pairsplit_*.py` | `src/unfold/pairsplit/` |
| `src/unfold/tools/model_envelope.py`, `model_covariance.py`, `prediction_statistics.py` | `src/unfold/model.py` |
| `notebooks/*.py` data/MC figures | `scripts/datamc/` |
| `scripts/studies/`, `scripts/diagnostics/` | `legacy/scripts/` (run against the snapshot tag, see `legacy/README.md`) |

`Unfolder_core_class_reference.md` describes the old class; the method names
survived the move (numerics in `engine.py`, `plot_*` as functions in
`plots.py`).
