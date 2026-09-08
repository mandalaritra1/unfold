# Pair-split prediction statistics

The pair-split runner now propagates MC prediction statistics through its
existing per-pT normalization. This change was authorized on 2026-09-08.
Central predictions, normalization windows, and unfolding outputs are unchanged.
Z+jet retains its previous prediction-statistics default.

For raw counts n, bin widths d, and normalization mask w, the density is
p_i = n_i / (d_i N), where N = sum_k w_k n_k. The Jacobian is
J_ik = delta_ik/(d_i N) - n_i w_k/(d_i N^2). The prediction statistical
covariance is J C J^T, including all bins, even outside the normalization window.
The diagonal supplies the statistical error bars; the full matrix enters the
existing data-prediction chi-square calculation. This is first-order error
propagation, using the same fixed normalization-window definition as the curve.

The available C is diagonal sumw2. This fixes denominator fluctuations but
cannot reconstruct missing event-level correlations, such as two jets from the
same dijet event, or cross-pT correlations. Those limitations remain explicit.
It preserves the normalization null direction without diagonal jitter. Bins
with zero nominal content can retain nonzero absolute statistical variance.

The prepared PYTHIA path already stored GEN values and sumw2 but did not connect
them to the prediction-statistics functions. The new pair-split mode consumes
these arrays. Vincia now retains full per-pT covariance alongside density and
errors. Existing theory uncertainty definitions and the assumption of adding
measurement and prediction covariance are unchanged.

Implementation:

- `src/unfold/tools/prediction_statistics.py`: common normalization Jacobian.
- `Unfolder._prediction_stat_covariance`: PYTHIA/HERWIG integration, with the
  existing fixed-denominator default retained for Z+jet.
- `src/unfold/tools/pairsplit_vincia.py`: Vincia covariance, error bars, attachment
  and artifact export.
- `scripts/run_pairsplit_unfolding.py`: opts into the method, changes the run
  identity, and saves PYTHIA raw values/sumw2 and normalized statistical covariance.

Validation: 84 tests passed, including finite-difference checks with correlated
raw inputs, partial windows, unequal widths, normalization null modes, zero
nominal bins, shared band/chi-square covariance, and legacy behavior.

Saved-array Vincia comparison:

```bash
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python \
  scripts/diagnostics/compare_prediction_statistics.py \
  --output outputs/pairsplit_run2/prediction_statistics_new
```

The 2026-09-08 output is in
`outputs/pairsplit_run2/prediction_statistics_2026-09-08_v2/`, with 16 PNG/PDF
panels, `comparison.pdf`, numerical covariance arrays, and a hashed manifest.
It holds the accepted enclosing-model measurement covariance fixed, verifies
unchanged Vincia central values, and changes only Vincia statistics. The largest
absolute chi-square change is about 0.33. These are not fresh production plots,
and the numerical comparison does not quantify the newly connected PYTHIA
statistical contribution because the historical artifacts lack its GEN sumw2.
