# Study 2 (v2): prior-free least-squares vs the Bayes family, dijet rho

x = log10(rho^2); pt blocks [200,290,400,570,760,inf] + pt<200 sink.
Foundation: scripts/dijet_methods/{loader,methods,result}.py. Data-term
inverse-variance weighted (Poisson-floored); ratio-curvature L matches the
production `ratio_curvature`. Reported = pt slices 1..5. Toys: 100,
data-stat only.

**Bias metric (v2): data-driven pseudo-truth closure.** The HERWIG
yardstick is DEPRECATED (low MC stats). truth* = smooth iterative
reweight of the Pythia gen (per pt slice, 4 passes, [0.25,0.5,0.25]
smoothing, w clipped to [0.2,5], anchored on populated bins) such that
P @ truth* matches the data matched-reco shape; pseudo-data = P @ truth*
is unfolded with the NOMINAL Pythia prior/anchor; bias = median
|u/truth* - 1| over reported bins.

## groomed

pseudo-truth reweight: w in [0.71, 1.25], foldback median relres 0.057. Weighted-lstsq foldback chi2/ndof (study binning) = 52 -- the data is grossly incompatible with the nominal response model at these statistics; every metric below is shaped by that, not by conditioning alone.

D'Agostini n=4: neg=0 osc=0.0846 closure bias=0.0209 spread=0.00407
Tikhonov ratio-curv, production tau~0.79: neg=4 osc=0.636 bias=6.58e-05 (still ~unregularized).
Tikhonov ratio-curv, plateau tau=56.2341: neg=6 osc=0.0847 bias=0.00838 spread=0.0048.

### prior-dependence of the central value (real data, reported bins)

Priors/anchors: nominal Pythia gen, data-reweighted truth*, +-40%
linear tilt per pt slice. Spread = max-over-priors |u_p - u_nom| /
max(|u_nom|, 5% MC truth).

| method | median | max |
|---|---|---|
| dagostini_n4_study | 0.0336 | 0.224 |
| tikhonov_ratio_plateau_study (tau=56.2341) | 0.0353 | 3.600 |
| dagostini_n4_coarse | 0.0095 | 0.128 |
| unreg_lstsq_coarse | 0.0000 | 0.000 |

Unregularized lstsq takes NO prior input: zero dependence by
construction (verified: repeated solves bit-identical).

### binning scan (unregularized weighted lstsq on data)

| binning | #gen | cond(P) | purity med/min | stab med/min | eff | neg | osc | chi2/ndof | vs dago4 med |
|---|---|---|---|---|---|---|---|---|---|
| study | 72 | 1.45e+03 | 0.41/0.06 | 0.33/0.03 | 0.95 | 7 | 0.752 | 52 | 0.163 |
| coarse_A | 60 | 802 | 0.52/0.13 | 0.43/0.25 | 0.96 | 2 | 0.479 | 68 | 0.137 |
| coarse_B | 54 | 693 | 0.60/0.13 | 0.49/0.25 | 0.96 | 1 | 0.368 | 143 | 0.0653 |
| coarse_C | 48 | 521 | 0.64/0.13 | 0.52/0.25 | 0.96 | 1 | 0.378 | 173 | 0.0731 |
| push_A | 48 | 754 | 0.65/0.34 | 0.56/0.28 | 0.96 | 1 | 0.319 | 144 | 0.0803 |
| push_B | 42 | 669 | 0.72/0.34 | 0.61/0.32 | 0.96 | 1 | 0.255 | 948 | 0.0754 |
| push_C | 36 | 598 | 0.78/0.43 | 0.63/0.32 | 0.97 | 1 | 0.313 | 902 | 0.0495 |

**Winner: push_B** (gen edges [-10.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]).
NO candidate meets the criteria: the remaining negative bins are deterministic (toy pulls of O(-100)), a data/model incompatibility (weighted foldback chi2/ndof >> 1), not a conditioning problem; this is the least-bad candidate by (neg_bins, osc).
Unreg toy spread (price of prior freedom): median 0.00699, p90 0.0287.
Deterministic negative bins (data/model incompatibility, not noise):
- pt bin 1, rho [-3.0, -2.5]: u/gen = -2.11, toy pull = -131

Per-bin purity/stability/efficiency (reported pt slices):

| pt bin | rho | purity | stability | efficiency |
|---|---|---|---|---|
| 1 | [-10.0, -3.0] | 0.70 | 0.64 | 0.92 |
| 1 | [-3.0, -2.5] | 0.44 | 0.33 | 0.92 |
| 1 | [-2.5, -2.0] | 0.48 | 0.43 | 0.92 |
| 1 | [-2.0, -1.5] | 0.63 | 0.55 | 0.93 |
| 1 | [-1.5, -1.0] | 0.74 | 0.63 | 0.93 |
| 1 | [-1.0, -0.5] | 0.86 | 0.73 | 0.92 |
| 1 | [-0.5, 0.0] | 0.72 | 0.40 | 0.83 |
| 2 | [-10.0, -3.0] | 0.80 | 0.67 | 0.95 |
| 2 | [-3.0, -2.5] | 0.43 | 0.32 | 0.97 |
| 2 | [-2.5, -2.0] | 0.54 | 0.46 | 0.97 |
| 2 | [-2.0, -1.5] | 0.72 | 0.56 | 0.97 |
| 2 | [-1.5, -1.0] | 0.79 | 0.63 | 0.97 |
| 2 | [-1.0, -0.5] | 0.89 | 0.69 | 0.97 |
| 2 | [-0.5, 0.0] | 0.64 | 0.38 | 0.96 |
| 3 | [-10.0, -3.0] | 0.85 | 0.73 | 0.94 |
| 3 | [-3.0, -2.5] | 0.44 | 0.38 | 0.96 |
| 3 | [-2.5, -2.0] | 0.61 | 0.51 | 0.96 |
| 3 | [-2.0, -1.5] | 0.77 | 0.61 | 0.97 |
| 3 | [-1.5, -1.0] | 0.82 | 0.68 | 0.97 |
| 3 | [-1.0, -0.5] | 0.90 | 0.74 | 0.96 |
| 3 | [-0.5, 0.0] | 0.56 | 0.44 | 0.95 |
| 4 | [-10.0, -3.0] | 0.88 | 0.72 | 0.94 |
| 4 | [-3.0, -2.5] | 0.50 | 0.43 | 0.96 |
| 4 | [-2.5, -2.0] | 0.66 | 0.54 | 0.97 |
| 4 | [-2.0, -1.5] | 0.80 | 0.62 | 0.97 |
| 4 | [-1.5, -1.0] | 0.84 | 0.69 | 0.97 |
| 4 | [-1.0, -0.5] | 0.91 | 0.74 | 0.97 |
| 4 | [-0.5, 0.0] | 0.47 | 0.49 | 0.96 |
| 5 | [-10.0, -3.0] | 0.91 | 0.82 | 0.94 |
| 5 | [-3.0, -2.5] | 0.56 | 0.52 | 0.97 |
| 5 | [-2.5, -2.0] | 0.70 | 0.61 | 0.97 |
| 5 | [-2.0, -1.5] | 0.83 | 0.69 | 0.97 |
| 5 | [-1.5, -1.0] | 0.86 | 0.76 | 0.97 |
| 5 | [-1.0, -0.5] | 0.91 | 0.80 | 0.97 |
| 5 | [-0.5, 0.0] | 0.34 | 0.57 | 0.96 |

### production confirmation (unfolding_coarse run)

`run_rho_unfolding.py --binning coarse` (TUnfold, tau resolved to 0,
area constraint on, merged reco sink below the first interior gen
edge; outputs/dijet/2018/rho/unfolding_coarse/):

- production result: **0 negative reported bins**, osc = 0.143, stat err median 0.00836 (p90 0.0349).
- numpy unreg lstsq with the SAME merged reco sink: 0 negative bins, osc 0.251; matches production to 0.0064 median (0.160 p90).
- production vs D'Agostini n=4 (independent method, same gen edges): 0.030 median, 0.091 p90.
- production coarse run: TUnfold kRegModeDerivative path with tau resolved to 0 (effectively unregularized) + area constraint (kEConstraintArea) + merged reco sink below the first interior gen edge. The merged reco sink removes the incompatible low-rho reco constraints and eliminates the deterministic negative bins seen with the native fine reco.

## ungroomed

pseudo-truth reweight: w in [0.78, 1.72], foldback median relres 0.047. Weighted-lstsq foldback chi2/ndof (study binning) = 237 -- the data is grossly incompatible with the nominal response model at these statistics; every metric below is shaped by that, not by conditioning alone.

D'Agostini n=4: neg=0 osc=0.0988 closure bias=0.0297 spread=0.00903
Tikhonov ratio-curv, production tau~0.79: neg=17 osc=0.448 bias=0.00124 (still ~unregularized).
Tikhonov ratio-curv, plateau tau=10: neg=15 osc=0.0968 bias=0.00637 spread=0.00866.

### prior-dependence of the central value (real data, reported bins)

Priors/anchors: nominal Pythia gen, data-reweighted truth*, +-40%
linear tilt per pt slice. Spread = max-over-priors |u_p - u_nom| /
max(|u_nom|, 5% MC truth).

| method | median | max |
|---|---|---|
| dagostini_n4_study | 0.0366 | 0.480 |
| tikhonov_ratio_plateau_study (tau=10) | 0.1129 | 5.090 |
| dagostini_n4_coarse | 0.0189 | 0.135 |
| unreg_lstsq_coarse | 0.0000 | 0.000 |

Unregularized lstsq takes NO prior input: zero dependence by
construction (verified: repeated solves bit-identical).

### binning scan (unregularized weighted lstsq on data)

| binning | #gen | cond(P) | purity med/min | stab med/min | eff | neg | osc | chi2/ndof | vs dago4 med |
|---|---|---|---|---|---|---|---|---|---|
| study | 72 | 1.4e+03 | 0.16/0.00 | 0.10/0.00 | 0.95 | 19 | 29.1 | 237 | 1.04 |
| coarse_A | 60 | 1.07e+03 | 0.27/0.00 | 0.27/0.00 | 0.95 | 12 | 2.77 | 185 | 0.487 |
| coarse_B | 54 | 916 | 0.40/0.00 | 0.41/0.00 | 0.95 | 9 | 2.75 | 184 | 0.299 |
| coarse_C | 48 | 701 | 0.58/0.00 | 0.47/0.00 | 0.95 | 7 | 1.75 | 406 | 0.24 |
| push_A | 36 | 455 | 0.79/0.06 | 0.60/0.06 | 0.96 | 2 | 0.21 | 547 | 0.0504 |
| push_B | 30 | 416 | 0.83/0.21 | 0.64/0.28 | 0.96 | 2 | 0.141 | 837 | 0.0679 |
| push_C | 24 | 324 | 0.79/0.21 | 0.66/0.28 | 0.96 | 2 | 0.258 | 791 | 0.107 |

**Winner: push_B** (gen edges [-10.0, -2.0, -1.5, -1.0, -0.5, 0.0]).
NO candidate meets the criteria: the remaining negative bins are deterministic (toy pulls of O(-100)), a data/model incompatibility (weighted foldback chi2/ndof >> 1), not a conditioning problem; this is the least-bad candidate by (neg_bins, osc).
Unreg toy spread (price of prior freedom): median 0.00406, p90 0.0343.
Deterministic negative bins (data/model incompatibility, not noise):
- pt bin 1, rho [-10.0, -2.0]: u/gen = -0.55, toy pull = -359
- pt bin 2, rho [-10.0, -2.0]: u/gen = -1.87, toy pull = -334

Per-bin purity/stability/efficiency (reported pt slices):

| pt bin | rho | purity | stability | efficiency |
|---|---|---|---|---|
| 1 | [-10.0, -2.0] | 0.21 | 0.28 | 0.93 |
| 1 | [-2.0, -1.5] | 0.45 | 0.60 | 0.93 |
| 1 | [-1.5, -1.0] | 0.80 | 0.64 | 0.93 |
| 1 | [-1.0, -0.5] | 0.92 | 0.66 | 0.92 |
| 1 | [-0.5, 0.0] | 0.83 | 0.43 | 0.83 |
| 2 | [-10.0, -2.0] | 0.29 | 0.35 | 0.95 |
| 2 | [-2.0, -1.5] | 0.58 | 0.62 | 0.96 |
| 2 | [-1.5, -1.0] | 0.84 | 0.60 | 0.97 |
| 2 | [-1.0, -0.5] | 0.93 | 0.66 | 0.97 |
| 2 | [-0.5, 0.0] | 0.83 | 0.41 | 0.96 |
| 3 | [-10.0, -2.0] | 0.40 | 0.44 | 0.95 |
| 3 | [-2.0, -1.5] | 0.70 | 0.66 | 0.96 |
| 3 | [-1.5, -1.0] | 0.86 | 0.65 | 0.96 |
| 3 | [-1.0, -0.5] | 0.93 | 0.72 | 0.96 |
| 3 | [-0.5, 0.0] | 0.84 | 0.47 | 0.95 |
| 4 | [-10.0, -2.0] | 0.52 | 0.49 | 0.95 |
| 4 | [-2.0, -1.5] | 0.77 | 0.65 | 0.96 |
| 4 | [-1.5, -1.0] | 0.88 | 0.68 | 0.97 |
| 4 | [-1.0, -0.5] | 0.94 | 0.73 | 0.97 |
| 4 | [-0.5, 0.0] | 0.84 | 0.51 | 0.96 |
| 5 | [-10.0, -2.0] | 0.69 | 0.66 | 0.96 |
| 5 | [-2.0, -1.5] | 0.79 | 0.71 | 0.96 |
| 5 | [-1.5, -1.0] | 0.88 | 0.77 | 0.97 |
| 5 | [-1.0, -0.5] | 0.94 | 0.80 | 0.97 |
| 5 | [-0.5, 0.0] | 0.83 | 0.58 | 0.96 |

### production confirmation (unfolding_coarse run)

`run_rho_unfolding.py --binning coarse` (TUnfold, tau resolved to 0,
area constraint on, merged reco sink below the first interior gen
edge; outputs/dijet/2018/rho/unfolding_coarse/):

- production result: **0 negative reported bins**, osc = 0.138, stat err median 0.00892 (p90 0.0562).
- numpy unreg lstsq with the SAME merged reco sink: 0 negative bins, osc 0.138; matches production to 0.0010 median (0.016 p90).
- production vs D'Agostini n=4 (independent method, same gen edges): 0.034 median, 0.141 p90.
- production coarse run: TUnfold kRegModeDerivative path with tau resolved to 0 (effectively unregularized) + area constraint (kEConstraintArea) + merged reco sink below the first interior gen edge. The merged reco sink removes the incompatible low-rho reco constraints and eliminates the deterministic negative bins seen with the native fine reco.

## Takeaways

- **Model incompatibility floor**: the weighted foldback chi2/ndof of the
  best-fit lstsq is 50-950 depending on binning/mode. With 5.1e9 events
  the 10-35% reco-level data/MC mismatch is thousands of sigma; with the
  NATIVE fine reco binning no gen binning alone removes the last 1-2
  deterministic negative bins (toy pulls O(-100)). NNLS parks the same
  bins at 0 with worse foldback. This is a response-modeling problem
  (resolution/migration mismatch), not conditioning.
- **The fix is coarse gen binning + a merged reco sink**: merging the
  reco bins below the first interior gen edge (the production 'coarse'
  convention) removes the incompatible low-rho reco constraints, and the
  fully unregularized inversion becomes sane: 0 negative reported bins,
  osc ~0.14, in BOTH modes -- confirmed in the production TUnfold run
  (see production confirmation sections above).
- push_B reduces the unregularized oscillation to 0.14-0.26 (from
  0.75-29 on the study binning), agrees with D'Agostini n=4 to 3-8%
  (median), and costs only ~0.4-0.9% median stat spread (p90 3-6%).
- **Prior dependence** (the boss-facing number): D'Agostini n=4 moves by
  3-4% (median) and up to 22-48% (max) across reasonable priors on the
  study binning; on the coarse binning 1-2% (median), 13-14% (max).
  Plateau-tau ratio-curvature Tikhonov is worse (median up to 11%, max
  ~4-5 in near-zero bins). Unregularized lstsq: exactly zero.
- **Pseudo-truth closure**: the prior-free candidate recovers truth*
  exactly (bias ~1e-13, by construction on noiseless pseudo-data);
  D'Agostini n=4 carries 2-3% median closure bias, plateau-tau
  ratio-curvature Tikhonov 0.6-0.8% (rising to ~5-10% at very large tau).
  The production tau~0.79 is confirmed to sit at the unregularized
  plateau (far too weak to matter).
- Herwig-based numbers from v1 of this study are deprecated and removed.

Files: money_bias_variance.png, metric_vs_tau.png, bayes_vs_niter.png,
binning_scan.png, metrics.json.
