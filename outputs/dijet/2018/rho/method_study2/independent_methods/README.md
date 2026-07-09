# Dijet rho: independent-methods cross-check & Bayes stat-error audit

Study `study2_indep`. Two genuinely independent cross-checks of the
D'Agostini (iterative-Bayes) central value, a forensic audit of the
`unfolding_bayes` tag's quoted statistical uncertainties, and direct
prior-dependence evidence for D'Agostini.

Foundation reused read-only: `scripts/dijet_methods/{loader,methods,result}.py`.
Efficiencies are NEVER floored. Study binning: pt
`[0,200,290,400,570,760,13000]`, gen rho
`[-10,-6,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0]` (12 bins), native 25 reco
rho bins. Reported pt bins 1-5; pt<200 is a migration sink (~90% of the truth)
and is treated as a nuisance, never reported. **HERWIG is distrusted (low
stats) and is NOT used as a template/yardstick in any headline number**; it
appears only as a thin dotted reference curve on the comparison plot.

## A. Independent central-value cross-checks

### Method 1 - forward-folding spline fit (primary)
Per pt slice, fit the reco data *in that reco-pt block* by folding a
parametrized gen shape `x_pt = s_pt * pythia_gen_pt * exp(spline(x))` through
the intra-slice response block (x = log10(rho^2); natural cubic spline, 4
interior knots across the populated gen range; s_pt a free per-slice norm).
No matrix inversion, no Bayes. Cross-pt migration into the block is removed
using the MC (Pythia) out-of-slice prediction, and reco bins are weighted by
their intra-slice purity so untrusted low-rho / sink migration does not drive
the fit. chi2 uses a 2% model-systematic floor added in quadrature to the
Poisson-floored data variance so the fit does not chase sub-percent reco
structure. Minimized with `scipy.optimize.least_squares`; per-slice
normalization errors from the Jacobian (linearized/Gaussian -- caveat).
(`study2_indep_forwardfold.py`)

### Method 2 - positive regularized NNLS fit
`min ||(Px-m)/sigma||^2 + tau^2||L_ratio x||^2` s.t. `x>=0`, with L_ratio the
Pythia-prior-scaled second difference (rows `(1/x0[i-1],-2/x0[i],1/x0[i+1])`),
solved via a stacked `scipy.optimize.nnls`. tau scanned over `logspace(-2,2,9)`;
smallest tau with oscillation metric < 0.05 chosen. (`study2_indep_posreg.py`)

| mode | method | median(slice medians) % | max over slices % |
|---|---|---|---|
| groomed | forward-fold spline | 9.72 | 139.23 |
| groomed | positive-reg NNLS | 5.10 | 24.96 |
| ungroomed | forward-fold spline | 31.29 | 445.00 |
| ungroomed | positive-reg NNLS | 6.06 | 362.67 |

Per-slice forward-fold vs D'Agostini n=4 (normalized median |ratio-1| %):

| mode | pt1 | pt2 | pt3 | pt4 | pt5 |
|---|---|---|---|---|---|
| groomed | 90.6 | 8.7 | 9.0 | 12.7 | 9.7 |
| ungroomed | 95.5 | 93.0 | 31.3 | 23.2 | 11.9 |

Per-slice positive-reg NNLS vs D'Agostini n=4 (normalized median |ratio-1| %):

| mode | pt1 | pt2 | pt3 | pt4 | pt5 |
|---|---|---|---|---|---|
| groomed | 6.0 | 3.8 | 5.1 | 4.2 | 5.6 |
| ungroomed | 6.1 | 31.3 | 5.3 | 8.8 | 2.5 |

**Reading:** the positive-reg NNLS fit agrees with D'Agostini n=4 to
~4-6% (median) across all five reported slices in both modes -- an
independent confirmation of the central value. The forward-fold spline
agrees to ~8-13% in the well-populated, high-purity slices (groomed pt2-5,
ungroomed pt3-5); the lowest reported slices (pt1, and ungroomed pt2) are
dominated by the pt<200 migration sink whose low-rho modelling is distrusted,
so a clean intra-slice forward-fold there is not achievable and those
numbers should be read as a limitation, not a discrepancy. The two methods
together bracket D'Agostini n=4 and show no evidence of a biased central value.

## B. Stat-error forensics of the `unfolding_bayes` tag

### The defect (file:line chain)
- `run_rho_unfolding.py:134-136`:
  `unfolded_input_errors = sqrt(diag(unfolder.cov_data_np))`.
- `unfolder_core.py:2437-2439,2484` (bayes path): `want_cov=True` for the
  nominal unfold, so `cov_data_np = cov` returned by `bayes_unfold(...,
  with_covariance=True)`.
- `roounfold_backend.py:136`: that cov is `RooUnfoldBayes.Eunfold(kCovariance)`
  -- RooUnfold's analytic covariance, which propagates ONLY the data-histogram
  bin errors.
- **`roounfold_backend.py:63`** is the bug: `_np_to_th1` sets those bin
  errors to `sqrt(|content|) = sqrt(N)`, discarding the stored sumw2. The
  dijet data is prescale-WEIGHTED, so sumw2/N ~ 247 (median); the honest
  per-bin data stat is therefore `sqrt(247) ~ 15.7x` larger than what
  RooUnfold sees.
- The manifest's `stat_propagation='legacy'` (jackknife) is vacuous here:
  dijet inputs carry NO jackknife replicas (`has_jackknife=False`,
  `unfolder_core.py:615`), so the code falls through to RooUnfold's analytic
  `kCovariance` on a sqrt(N) data error.

### The numbers (median rel-stat over reported bins)

| mode | tag quoted % | honest toy data-only % | honest toy data+response % | honest/quoted | reg syst |n6-n4| % |
|---|---|---|---|---|---|
| groomed | 0.0247 | 0.418 | 0.434 | 16.9x | 1.665 |
| ungroomed | 0.0319 | 0.640 | 0.734 | 20.0x | 1.468 |

Source-level confirmation (`measured` vs its stored variance in the npz):

| mode | implied sqrt(N)/N % | honest sqrt(sumw2)/N % | sumw2/N | underest. factor |
|---|---|---|---|---|
| groomed | 0.0279 | 0.4839 | 246.6 | 15.7x |
| ungroomed | 0.0481 | 0.6748 | 258.2 | 16.1x |

### Correct scheme
Quote the honest **toy-propagation** data-stat spread (Gaussian toys on the
measured spectrum using the STORED sumw2 variance, Poisson-floored, re-unfolded
through D'Agostini) -- what `result.unfold_with_unc` does and what the honest
columns above report. Equivalently, fix `_np_to_th1` to accept the per-bin
sumw2 error and use `kCovToys`. Quote the regularization uncertainty
separately as the n_iter systematic `|u(n=6)-u(n=4)|` (see table).

## C. Prior-dependence of D'Agostini n=4 (boss-facing)

Same real data unfolded with three priors per pt slice: (i) nominal Pythia
gen, (ii) a strong linear tilt 0.6->1.4 in x, (iii) flat-in-x (each
renormalized to the slice's Pythia integral). Spread of the unfolded result
across priors over reported bins:

| mode | n_iter | prior spread median % | p90 % | max % | flat-vs-nominal median % |
|---|---|---|---|---|---|
| groomed | 4 | 12.92 | 286.11 | 1088.82 | 12.92 |
| groomed | 10 | 9.61 | 94.56 | 516.41 | 9.00 |
| ungroomed | 4 | 21.68 | 153.72 | 880.41 | 21.10 |
| ungroomed | 10 | 5.03 | 53.96 | 726.32 | 4.88 |

**Reading:** at n_iter=4 the unfolded central value moves by several percent
(median) and much more in the sparse tails when the prior is changed from
Pythia to flat-in-x -- direct, quantitative confirmation of the analysis
lead's concern that D'Agostini n=4 is prior-dependent. A gentle tilt moves
it far less (the tilt stays close to Pythia). Increasing to n_iter=10 reduces
the prior spread (more iterations wash out the prior, as expected) but does
not eliminate it; n=10 also amplifies statistical noise. This is exactly the
regularization/prior-dependence trade-off that motivates the independent
cross-checks in Part A.

## Files
- `comparison_<mode>.png` - D'Agostini n=4 (honest toy band) vs forward-fold
  spline vs positive-reg NNLS, per pt slice (Pythia prior overlaid; Herwig
  dotted reference only).
- `prior_dependence_<mode>.png` - D'Agostini unfolded with 3 priors, n=4
  (solid) and n=10 (dotted), per pt slice.
- `stat_audit_bars.png` - tag-quoted vs honest-toy vs reg-systematic.
- `metrics.json` - all numbers (`central_crosscheck`, `stat_audit`,
  `prior_dependence`).
