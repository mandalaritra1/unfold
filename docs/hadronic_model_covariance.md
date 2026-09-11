# Hadronic model covariance

Accepted by the user on 2026-09-05 after reviewing the before/after comparisons.

The hadronic runner defaults to `--model-covariance enclosing_ellipsoid`.
`--model-covariance selected_variation` reproduces the previous prescription.
Z+jet retains its existing default.

Two groups remain: parton shower (Vincia, FSR up/down) and hadronization
(CR1, CR2, fragmentation hard/soft). Within each group, the signed normalized
variation-minus-nominal vectors define a minimum-volume, origin-centered
ellipsoid in their linear span. Its shape matrix is the group covariance.
Each supplied template and its negative lie inside that group's unit ellipsoid.
The two group matrices are added, retaining their cross-bin and cross-pT terms.
This is a geometric modelling convention, not a calibrated confidence region.
Adding the group covariances does not enclose every simultaneous sum of two
unit-boundary group shifts within the combined unit ellipsoid.

The same matrices supply the displayed model-band variances and covariance
used for chi-square calculations. Other systematic and statistical terms keep
their existing definitions. Signed absolute normalized shifts are retained even
in bins with zero nominal content; the previous model-only 0.5-percent slice
content filter is removed for this method. Normalization constraints are checked
and only floating-point leakage is projected away. No diagonal jitter is added.
Fractional model uncertainties are undefined where the nominal content is zero;
the absolute covariance remains defined there.

For all four saved Run-2 channel/grooming combinations checked on 2026-09-05,
the PS and HAD template matrices have full column ranks 3 and 4. In this case
the enclosing covariance equals the sum of the template outer products.
This algebraic result does not assign independent Gaussian priors to alternative
models. Compared with the previous binwise envelope, its band can be larger;
compared with the previous two selected covariance directions, it retains
additional shape directions.

Implementation: `src/unfold/tools/model_covariance.py`, with integration in
`Unfolder` and `scripts/run_hadronic_unfolding.py`. The artifact stores both
group matrices, the method, and template-containment diagnostics. The raw
binwise envelopes remain diagnostic quantities, not the new displayed model
band. Historical selected-vector diagnostics require legacy artifacts.

Reproduce the saved-output comparison without rerunning unfolding:

```bash
source scripts/setup_root.sh
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python \
  scripts/diagnostics/compare_hadronic_model_covariance.py \
  --output outputs/hadronic/model_covariance_comparison_new
```

The comparison verifies canonical input hashes, reproduces the previous saved
band and total covariance, checks unchanged central values and non-model
covariance, and verifies positive semidefiniteness and normalization null modes.
The 2026-09-05 comparison is in
`outputs/hadronic/model_covariance_comparison_2026-09-05_v2/`, including
24-page `comparison.pdf`, PNG/PDF panels, numerical arrays, and `manifest.json`.
Canonical production outputs and the analysis note were not overwritten.
