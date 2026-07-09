# study2 — smoothed / parametrized response for dijet rho

Three response matrices per mode: **NOMINAL** (raw MC), **SMOOTH_I** (nonparametric Gaussian-kernel smoothing of the reco-rho shape, sigma=1.2 median-bin-widths), **PARAM_II** (single Gaussian kernel fitted per gen bin, mu/sigma smoothed with deg-2 polynomials per ptgen->ptreco pair, re-integrated over reco bins). Both smoothings preserve per-column pT-migration and efficiency to machine precision.

## TL;DR findings
1. **Smoothing does NOT stabilize inversion — it makes it worse.** cond(P) rises NOMINAL->SMOOTH_I->PARAM_II (groomed 1449->3900->78k; ungroomed populated 1396->35k->15k). Widening the migration kernel increases collinearity between neighbouring gen columns, so lstsq / low-tau Tikhonov get MORE negative bins and MORE oscillation, not less.
2. **The D'Agostini central value is barely moved by nonparametric smoothing** (median shift ~1.6% groomed, ~5% ungroomed over reported bins) and the matrix-stat toy component is already tiny (~0.1% groomed) and is NOT meaningfully shrunk by smoothing — the data is systematics-dominated, matrix MC-stat is negligible either way.
3. **A Gaussian parametrization is inadequate.** The reco-rho kernel is skewed (|skew| up to ~2) and heavy-tailed (excess kurtosis up to ~10), so PARAM_II self-closes to only ~20% (groomed) and shifts the Bayes result by ~4%; it is not usable as a drop-in response.
**Conclusion: keep the nominal matrix + D'Agostini n=4.** Smoothing buys no stability for inversion methods and no toy-noise reduction for Bayes, while parametrization injects a real bias. Stability comes from *binning* and *iterative regularization*, not from massaging the matrix.

Study binning: pt 0-200,200-290,290-400,400-570,570-760,760+; 12 gen rho bins, 25 reco rho bins. D'Agostini n_iter=4, 200 toys.


## groomed

### Response conditioning cond(P)

`populated` drops empty gen columns (the pT<200 sink block has unmatched gen bins -> exact-zero singular value -> full cond is inf regardless of smoothing).

| matrix | cond full | cond populated |
|---|---|---|
| NOMINAL | 1449 | 1449 |
| SMOOTH_I | 3900 | 3900 |
| PARAM_II | 7.824e+04 | 7.824e+04 |

### Matrix validation

| matrix | fold median dev | fold p90 | self-closure median bias | self-closure max | herwig model median bias |
|---|---|---|---|---|---|
| NOMINAL | 0.00e+00 | 0.00e+00 | 6.66e-16 | 2.89e-15 | 5.99e-02 |
| SMOOTH_I | 4.93e-02 | 2.29e+00 | 5.50e-02 | 8.79e-01 | 6.56e-02 |
| PARAM_II | 2.19e-01 | 5.19e+00 | 2.08e-01 | 9.85e-01 | 6.57e-02 |

### Money comparison (unfolding DATA)

Metrics over reported pT bins (200 GeV+). osc = median |2nd diff of u/prior|; shift = median|ratio-1| vs nominal-matrix D'Agostini n=4.

| method | matrix | neg bins | oscillation | foldback median | shift vs nom-dago | matrix-stat rel |
|---|---|---|---|---|---|---|
| dagostini_n4 | NOMINAL | 0 | 0.0846 | 0.035 | 0.0 | 0.001062462989746599 |
| dagostini_n4 | SMOOTH_I | 0 | 0.0849 | 0.0455 | 0.0157378419487037 | 0.0011173234187691393 |
| dagostini_n4 | PARAM_II | 0 | 0.0615 | 0.0507 | 0.04187005580908454 | 0.0009521251860976146 |
| lstsq | NOMINAL | 6 | 0.469 | 0.0143 | 0.1145763600835139 |  |
| lstsq | SMOOTH_I | 9 | 1.97 | 0.0446 | 0.44794221164046777 |  |
| lstsq | PARAM_II | 19 | 11.6 | 2.08 | 2.712173731568764 |  |
| tikhonov_tau0 | NOMINAL | 6 | 0.469 | 0.0143 | 0.1145763600835139 |  |
| tikhonov_tau0 | SMOOTH_I | 9 | 1.97 | 0.0446 | 0.44794221164046777 |  |
| tikhonov_tau0 | PARAM_II | 19 | 11.6 | 2.08 | 2.712173731568764 |  |
| tikhonov_tau0.001 | NOMINAL | 3 | 0.383 | 0.00978 | 0.07707052502150408 |  |
| tikhonov_tau0.001 | SMOOTH_I | 5 | 1.32 | 0.011 | 0.3308965009998813 |  |
| tikhonov_tau0.001 | PARAM_II | 14 | 3.22 | 0.118 | 0.9076310489609674 |  |
| tikhonov_tau0.01 | NOMINAL | 2 | 0.271 | 0.0101 | 0.1113195029547927 |  |
| tikhonov_tau0.01 | SMOOTH_I | 1 | 0.345 | 0.0111 | 0.15660959603904046 |  |
| tikhonov_tau0.01 | PARAM_II | 7 | 0.783 | 0.0233 | 0.31650247811409776 |  |
| tikhonov_tau0.1 | NOMINAL | 1 | 0.149 | 0.0237 | 0.08038713204608189 |  |
| tikhonov_tau0.1 | SMOOTH_I | 2 | 0.192 | 0.0247 | 0.09362244717276513 |  |
| tikhonov_tau0.1 | PARAM_II | 2 | 0.177 | 0.0301 | 0.13220531968730886 |  |

## ungroomed

### Response conditioning cond(P)

`populated` drops empty gen columns (the pT<200 sink block has unmatched gen bins -> exact-zero singular value -> full cond is inf regardless of smoothing).

| matrix | cond full | cond populated |
|---|---|---|
| NOMINAL | inf | 1396 |
| SMOOTH_I | inf | 3.485e+04 |
| PARAM_II | inf | 1.497e+04 |

### Matrix validation

| matrix | fold median dev | fold p90 | self-closure median bias | self-closure max | herwig model median bias |
|---|---|---|---|---|---|
| NOMINAL | 0.00e+00 | 0.00e+00 | 4.44e-16 | 1.00e+00 | 2.84e-01 |
| SMOOTH_I | 2.38e+00 | 1.11e+01 | 8.49e-01 | 1.00e+00 | 3.87e-01 |
| PARAM_II | 3.91e-01 | 1.00e+00 | 9.79e-02 | 6.65e+02 | 3.36e-01 |

### Money comparison (unfolding DATA)

Metrics over reported pT bins (200 GeV+). osc = median |2nd diff of u/prior|; shift = median|ratio-1| vs nominal-matrix D'Agostini n=4.

| method | matrix | neg bins | oscillation | foldback median | shift vs nom-dago | matrix-stat rel |
|---|---|---|---|---|---|---|
| dagostini_n4 | NOMINAL | 0 | 0.0869 | 0.0463 | 0.0 | 0.004469113187801952 |
| dagostini_n4 | SMOOTH_I | 0 | 0.0887 | 0.0589 | 0.050793080502898325 | 0.033867823019172655 |
| dagostini_n4 | PARAM_II | 0 | 0.0863 | 0.0414 | 0.00961254753321572 | 0.003551626857995467 |
| lstsq | NOMINAL | 15 | 820 | 0.431 | 4.735942131166803 |  |
| lstsq | SMOOTH_I | 21 | 1.15e+04 | 15.9 | 372.5860239567457 |  |
| lstsq | PARAM_II | 19 | 2.56e+03 | 0.983 | 66.92379163892723 |  |
| tikhonov_tau0 | NOMINAL | 15 | 820 | 0.431 | 4.735942131166803 |  |
| tikhonov_tau0 | SMOOTH_I | 21 | 1.15e+04 | 15.9 | 372.5860239567457 |  |
| tikhonov_tau0 | PARAM_II | 19 | 2.56e+03 | 0.983 | 66.92379163892723 |  |
| tikhonov_tau0.001 | NOMINAL | 14 | 615 | 0.437 | 4.394251366742542 |  |
| tikhonov_tau0.001 | SMOOTH_I | 17 | 1.48e+03 | 0.559 | 12.35328162942238 |  |
| tikhonov_tau0.001 | PARAM_II | 19 | 2.01e+03 | 0.57 | 11.942081448627798 |  |
| tikhonov_tau0.01 | NOMINAL | 10 | 105 | 0.268 | 3.5958313611938992 |  |
| tikhonov_tau0.01 | SMOOTH_I | 17 | 112 | 0.207 | 8.538341609022874 |  |
| tikhonov_tau0.01 | PARAM_II | 13 | 27.6 | 0.201 | 1.3812899768291835 |  |
| tikhonov_tau0.1 | NOMINAL | 11 | 4.56 | 0.115 | 0.5336796584561843 |  |
| tikhonov_tau0.1 | SMOOTH_I | 14 | 56.5 | 0.0846 | 2.7400633010706485 |  |
| tikhonov_tau0.1 | PARAM_II | 10 | 12 | 0.139 | 1.6090483562257782 |  |

## Files
- overlay_dagostini_{groomed,ungroomed}.png — per-pT D'Agostini shape, nominal vs smoothed vs parametric
- response_maps_{groomed,ungroomed}.png — folding-prob block for pT 200-290, three matrices
- metrics.json — full metric table

