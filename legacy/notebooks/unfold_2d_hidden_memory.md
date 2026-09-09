# Context: `unfold_2d_hidden.ipynb`

## Purpose
2D (pT × jet mass) unfolding for the Z+jet analysis using **TUnfoldDensity**.
Unfolds CMS Run 2 data (2016+2017+2018, combined by summing per-year histograms) against
Pythia8 response matrices, optionally cross-checked with Herwig response matrices.

---

## Key Configuration Flags (Cell 2 — edit here)

| Variable | Default | Options / Notes |
|---|---|---|
| `OBS` | `'g'` | `'g'` = groomed (soft-drop) jet mass; `'u'` = ungroomed |
| `USE_TWO_MC` | `True` | Unfold with Pythia **and** Herwig matrices, overlay both |
| `TAU` | `0.0` | `0` = auto-scan τ; `>0` = use fixed τ |
| `TAU_METHOD` | `'lcurve'` | `'lcurve'` = TUnfold `ScanLcurve`; `'sure'` = min. avg. ρ̄ scan |
| `REBIN_MGEN` | `None` | Pre-unfold rebin of gen mass axis: `None` = off, `int` = merge factor, `list` = new edges |
| `REBIN_MRECO` | `None` | Pre-unfold rebin of reco mass axis: same options as above |

Input pickle files live in `inputs/massInputs/fine_bin/`.  
Keys inside each pkl: `response_matrix_{u/g}`, `ptjet_mjet_{u/g}_{reco/gen}`.  
Data pkl keys: `ptjet_mjet_{u/g}_reco` (background-subtracted).

---

## Global Bin Convention
Bins are flattened **mass-fast within each pT block**:
```
global_bin = pt_idx * n_mass_bins + mass_idx
```
This applies to both reco and gen spaces. The response matrix shape is
`(n_reco_global, n_gen_global)` = `A[reco, gen]`.

---

## Cell-by-Cell Summary

### Cell 0 — Markdown header
Documents axis convention (X=gen, Y=reco in TH2), global bin ordering,
fakes/misses definitions, and the no-regularisation default.

### Cell 2 — Configuration
All user-facing flags: `OBS`, `USE_TWO_MC`, `TAU`, `TAU_METHOD`, file path dicts.

### Cell 4 — Imports
`pickle`, `numpy`, `matplotlib`, `mplhep` (CMS style), `ROOT` (batch mode).

### Cell 4 — Imports
Includes `from unfold.utils.integrate_and_rebin import rebin_hist`.

### Cell 6 — Load & sum histograms
`load_pkl` + `sum_hist_over_years` helpers load and sum per-year boost-histogram
objects. Loads: `response_matrix`, `mc_reco`, `mc_gen`, `data_reco`, and optionally
the alt (Herwig) set when `USE_TWO_MC=True`.

At the end of this cell, if `REBIN_MGEN` or `REBIN_MRECO` is set, the `_apply_prebin`
helper calls `rebin_hist` on all loaded histograms before any numpy extraction:
- `response_hist` — both `mgen` and `mreco` axes
- `mc_gen_hist` — `mgen` only
- `mc_reco_hist`, `data_reco_hist` — `mreco` only
- Alt (Herwig) equivalents rebinned identically

### Cell 8 — Extract numpy arrays
Projects 4D response hist `(ptgen, mgen, ptreco, mreco)`, transposes to
`(ptreco, mreco, ptgen, mgen)`, reshapes to `response_2d (n_reco_global, n_gen_global)`.
Derives bin edges: `ptgen_edges`, `mgen_edges`, `ptreco_edges`, `mreco_edges`.

### Cell 9 — Fakes and Misses
- **Fakes** = `mc_reco_flat − response_2d.sum(axis=1)` (reco events with no gen match)
- **Misses** = `mc_gen_flat − response_2d.sum(axis=0)` (gen events not reconstructed)
- **Efficiency** = `gen_matched / mc_gen_flat` per gen bin

### Cell 11 — ROOT histogram builders
Three helper functions:
- `make_th1(name, values)` — flat 1D array → `TH1D` (bins 1..N, Poisson errors)
- `make_th2_response(name, matrix, fakes)` — 2D numpy → `TH2D` with gen underflow bin 0 for fakes  
- `make_tunfold_binning(name, pt_edges, m_edges)` — `TUnfoldBinning` with one sub-node per pT slice and physical mass edges, so derivative regularisation uses correct widths and no cross-pT penalty

Builds `h_response`, `h_data`, `h_mc_reco`, `gen_binning`, `reco_binning`,
and their `_alt` counterparts.

### Cell 12 — Sanity check
Spot-checks ~12 near-diagonal entries between `response_2d` (numpy) and `h_response`
(ROOT TH2) to confirm axis mapping is correct.

### Cell 14 — TUnfoldDensity setup + τ selection ← **core cell**
Creates `TUnfoldDensity` with:
- `kHistMapOutputHoriz` (gen on X)
- `kRegModeDerivative` (derivative regularisation)
- `kEConstraintArea` (area constraint)
- `kDensityModeBinWidth` (bin-width normalised regularisation)

**τ selection logic:**
```
if TAU == 0:
    if TAU_METHOD == 'lcurve':   → ScanLcurve (50 pts, τ ∈ [1e-6, 1e-1])
                                    → plots L-curve with best-τ star
    elif TAU_METHOD == 'sure':   → manual loop: DoUnfold + GetRhoI per τ
                                    → minimise mean |ρ|, plots ρ̄ vs log10(τ)
else:
    DoUnfold(TAU)                → fixed τ, no plot
```
After τ selection, `t_best` is set. Alt (Herwig) `tunfold_alt` always uses the
same `t_best` via `DoUnfold(t_best)`.

### Cell 16 — Extract results
- `th1_to_numpy(h)` → `(values, errors)` arrays from TH1
- `get_covariance(tunfold_obj, name)` → full `N×N` covariance from `GetEmatrixTotal`
- Results reshaped to `unfolded_2d (n_ptgen, n_mgen)`, `errs_2d`, `cov_matrix`
- Same for alt: `unfolded_2d_alt`, `errs_2d_alt`, `cov_matrix_alt`

### Cell 19 — Response matrix plot (7a)
Probability matrix P(reco|gen) shown as a log-scale imshow with pT-block grid lines.

### Cell 21 — Purity & Stability (7b)
Diagonal-based purity and stability per global bin, plotted vs global gen bin.
Bins below 0.5 are flagged. Threshold line at 0.5.

### Cell 23 — Rebin config (7c setup)
```python
N_REBIN = 1
CUSTOM_REBIN_EDGES = [0, 1, 5, 10, 20, 30, 50, 70, 90, 110, 130, 150]
```
`rebin_1d(vals, errs, edges, n, custom_edges)` handles both uniform and custom rebinning.

### Cell 24 — Unfolded mass distributions (7c)
One panel per pT bin. Plots:
- Unfolded data (Pythia matrix) — black circles
- Unfolded data (Herwig matrix, if `USE_TWO_MC`) — red squares
- MC gen truth (Pythia) — blue step
- MC gen truth (Herwig, if `USE_TWO_MC`) — orange step
- Ratio panels: Herwig/Pythia gen (middle), Herwig/Pythia unfolded (bottom)
All distributions normalised to unit area in displayed bins.

### Cell 26 — τ-scan closure test (commented out)
A commented loop to scan τ and compute χ²/bin between unfolded MC reco and MC gen truth.

### Cell 27 — Closure test (7d)
Runs a separate `TUnfoldDensity` instance with `kRegModeNone` + `kDensityModeNone`,
unfolds `h_mc_reco` at fixed `TAU`. Plots unfolded vs gen truth with ratio panel.
Note: uses `TAU` (not `t_best`) — the fixed-reg version for closure, not scan result.

### Cell 29 — Efficiency & fake rate (7e)
Per-pT-bin step plots of efficiency (gen) and fake rate (reco).

### Cell 31 — Covariance / correlation matrix (7f)
Full correlation matrix from `GetEmatrixTotal`, displayed as RdBu imshow with
pT-block boundary lines.

### Cell 33 — Summary printout (Section 8)
Prints TUnfold status, τ, bin counts, mean efficiency, bins below P/S=0.5,
and per-pT-bin integral of unfolded result.

---

## Important Implementation Notes

- **TH2 bin index**: TUnfold expects fakes in the ROOT underflow (binx=0) of the gen axis.
  Physical gen bins occupy ROOT bins 1..n_gen. So `response_2d[ireco, igen]` →
  `h_response.SetBinContent(igen+1, ireco+1, ...)` and fakes →
  `h_response.SetBinContent(0, ireco+1, fakes[ireco])`.
  The TH2 X-axis is defined as `n_gen, 0.5, n_gen+0.5` (no extra bin needed).
- **TUnfoldBinning sub-nodes**: one per pT slice, each with a mass axis carrying
  physical edges. This prevents cross-pT regularisation and ensures correct
  derivative penalties.
- **SURE scan implementation**: done via an explicit loop (not `ScanTau`) to avoid
  `TGraph **` double-pointer issues in PyROOT. Calls `DoUnfold(τ)` + `GetRhoI(name, '')`
  at each τ point, averages `|ρ|` over non-zero bins, then re-runs `DoUnfold(t_best)`.
- **Alt unfolding**: `tunfold_alt` is always run at `t_best` (same τ as nominal),
  so comparisons between Pythia and Herwig matrices are at identical regularisation.
- **Normalisation**: all mass distributions are normalised to unit area over the
  displayed (rebinned) range before plotting.
- **Closure test (Cell 27)** uses `TAU` not `t_best` and `kRegModeNone` —
  it's a zero-regularisation closure to check the matrix inversion, not a τ-scan closure.
