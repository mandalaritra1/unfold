# Dijet rho unfolding — numerical method study (2018)

**Goal:** a dijet QCD rho unfolded distribution that matches the generator
shape within uncertainty, where plain TUnfold produced inconsistent shapes.

**Result:** iterative Bayesian unfolding (D'Agostini) at native rho resolution
with a lightly-merged low-rho tail gives a smooth, positive, physically sensible
result that tracks PYTHIA8 gen within uncertainty and sits between PYTHIA and
HERWIG — for both groomed and ungroomed, all reported pT bins.

## Why plain TUnfold / matrix inversion failed

The response is **rank-deficient**: the low-rho gen bins (`log10(rho^2) < -6`)
are nearly empty at the populated pT, so the migration matrix has zero/near-zero
columns (`cond = inf`). Any method that inverts it (TUnfold, SVD, NNLS)
oscillates or goes negative there — see `methods_*` (SVD crashes to 0 / spikes to
2x; NNLS floors at 0 then spikes). The committed result hid this by collapsing
the whole low-rho tail into one merged bin, and still showed ~30% bottom-line
swings.

Diagnosis was not a huge miss-rate: efficiency in the reported pT bins is
0.90-0.96; the 9x gen/reco ratio is almost entirely the `pT<200` sink bin
(kept only to absorb feed-up migration, which matters only for the 200-290 bin,
17.6%).

## Method: D'Agostini iterative Bayes

- Never inverts the response, so robust to the empty/ill-conditioned bins.
- Folding probabilities `P[reco,gen] = R[reco,gen] / gen_total[gen]`
  (column sums = efficiency, so efficiency and misses are built in).
- Data fakes subtracted with the MC fake fraction; `pT<200` sink kept as a
  truth bin to absorb feed-up; **no efficiency flooring** (a floor breaks the
  fixed point and was an early bug — see below).
- **n_iter = 4** (data converged to ~1%/iteration; reweighted-MC bias <1%).

## Validation (the numbers that matter)

| Test | Meaning | groomed (n=4) |
|---|---|---|
| Self-closure (MC reco -> MC truth, MC prior) | bookkeeping | **exact, 1e-15** at every iteration |
| Reweighted-MC (+-15% rho tilt, *nominal* prior) | prior independence | recovered to **<1% bulk** (p90 ~5%) |
| HERWIG-as-data (PYTHIA response) | model bias | **~5% bulk**, p90 ~17% |
| Response systematics (scale/PS/JER/JMS/...) | exp+theory | **~18%** (Q2muF 9.7%, Q2muR 8.9%, ISR 6.9%, FSR 4.6%; JES absent from inputs) |

The reweighted-MC test is decisive: a +-15% shape distortion is recovered to
<1%, so the result follows the data shape, **not** the prior.

### Statistical robustness (toy closure) -- the noiseless 1e-15 is NOT the point
The exact closure above is noiseless (unfold `P@truth`), so it only tests
bookkeeping. The honest test injects realistic statistical noise into the
pseudo-data and unfolds 500 toys (`toy_closure.py`, `noise_amplification.py`):

| | dijet groomed | trijet groomed |
|---|---|---|
| bias (mean unfolded/truth - 1) | <0.25% (unbiased) | <2% worst bin (unbiased) |
| stat spread (std/truth), median | 0.4% | 4% |
| stat spread, p90 (tail) | 1.5% | 14% |

So the method is **unbiased under noise** and the spread is commensurate with the
input data statistics -- no pathological amplification. Pushing the *same* toys
through matrix inversion instead (Tikhonov/SVD/near-inverse) blows the spread up
to **30-40% median and 2000-10000% in the tail** (`noise_amplification_*.png`);
D'Agostini stays at the input-noise level. That 2-3 order-of-magnitude gap in the
low-rho tail is exactly the rank-deficiency pathology, and why iterative Bayes is
the trustworthy choice. (Dijet stat is tiny because of 5e9 events -> the overlay
error bars are systematic-dominated; trijet stat ~4% is visible.)

### Binning
- pT (reported): `[200,290,400,570,760,inf]` + a `[0,200]` migration sink.
- rho gen: **adaptive low-rho merge** — collapse the lowest gen bins into one
  until every reported (pT>=200) bin has efficiency >= 0.5. Dijet groomed lands
  at low-edge `-6` (worst eff 0.68); dijet ungroomed at `-4` (0.57). Native
  14-bin gen has `[-10,-8]`/`[-8,-7]` with efficiency -> 0 (uninformative).
- rho reco kept ~native (finer than gen -> over-constrained).

### Trijet
The same channel-agnostic pipeline runs on trijet
(`outputs/trijet/2018/rho/method_study/`, `python -m scripts.dijet_methods.plots
trijet`). It works equally well; trijet has ~150x lower stats (larger tail
errors), no miss sink (eff ~0.95, ~18% fakes instead), and no HERWIG sample.

## Files
- `overlay_{mode}_merged.png` — **headline**: unfolded data vs PYTHIA/HERWIG gen,
  per pT slice (shape-normalized), with total stat(+)syst error bars.
- `closure_{mode}_merged.png` — self-closure / reweighted-MC / HERWIG closure.
- `methods_{mode}_merged.png` — D'Agostini vs Tikhonov / SVD / NNLS on data.
- `*_native.png` — same at native rho (shows the tail bins blow up).

## Reproduce
```bash
source .venv/bin/activate
python -m scripts.dijet_methods.plots          # all figures
python -m scripts.dijet_methods.diagnose       # conditioning / purity / eff
python -m scripts.dijet_methods.run_closure    # quick closure tables
```
Code: `scripts/dijet_methods/{loader,methods,result,plots,diagnose}.py`
(standalone numpy/scipy, independent of ROOT/TUnfold).

## Caveats / follow-ups
- JES is **not** in the supplied response variations; the ~18% syst is therefore
  incomplete (missing the usually-dominant JES). Regenerate inputs with JES to
  close the budget.
- Stat band uses data + MC-response toys; no jackknife response replicas here.
- A `pT<200` sink-as-truth choice (vs background-subtracting feed-up) only
  affects the 200-290 bin; both were checked, sink-as-truth closes exactly.
