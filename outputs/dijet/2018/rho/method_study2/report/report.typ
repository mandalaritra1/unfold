#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set heading(numbering: "1.1")
#show link: set text(fill: blue.darken(20%))

#align(center)[
  #text(17pt, weight: "bold")[Stabilizing the Dijet #box[$rho = m^2\/(p_T R)^2$] Unfolding]

  #text(12pt)[Diagnosis and method bake-off — CMS dijet 2018, groomed & ungroomed #box[$log_10 rho^2$]]

  #text(10pt, style: "italic")[Claude (Fable) investigation, 2026-07-08 — repo `unfold`, worktree `objective-swanson-22e275`]
]

#v(0.5em)
#outline(depth: 2)
#v(1em)

= Executive summary

*The noisy TUnfold result was never a statistics or rank problem — it was an
effectively unregularized least-squares chasing a genuine 10–35% data/MC shape
mismatch through an ill-conditioned (cond $approx$ 500–1450, full-rank)
response, in a regime where $5.1 times 10^9$ events make statistical errors
negligible.* The L-curve $tau$ ($approx 0.79$) was self-defeating: balanced
against negligible stat errors it lands on the unregularized plateau.

*The fix that survives all constraints* (no HERWIG yardstick; no
prior-dependent primary method): *resolution-matched coarse gen binning +
merged reco sink + $tau = 0$ TUnfold* — completely prior-free — now
implemented and run in the production pipeline
(`outputs/dijet/2018/rho/unfolding_coarse/`, `--binning coarse`). Result:
0 negative bins, smooth, generator-discriminating, honest ~0.9% median stat
errors, and confirmed by three independent methods (D'Agostini, NNLS-positive
fit, forward-folding spline fit) at the 3–13% level. *The groomed second-peak
region keeps its 0.5-wide bins* — an injection test shows the dip is trivially
resolvable at data statistics (hundreds of $sigma$) and even a dipless-prior
D'Agostini recovers it; only the low-$rho$ tail ($x lt.approx -3$), where the
kernel RMS reaches 0.7–0.8 and purity collapses, must be merged.

Also found and documented: *the committed Bayes tag's statistical errors are
$approx 17 times$ too small* (a $sqrt(N)$ hard-coded in the RooUnfold backend
discards the prescale-weighted sumw2, $"sumw2"\/N approx 247$); D'Agostini
prior dependence is real (13–22% median under a flat-prior swap — the
analysis lead's concern is quantitatively confirmed); response-matrix
smoothing / parametric toy-MC responses make things *worse*, not better; and
two latent crashes in rarely-exercised code paths were fixed. The one input
gap requiring upstream (skimmer) action eventually: *JES variations are absent
from the dijet response systematics.* No reskim is needed for the
stabilization itself.

= The symptom

The committed TUnfold results (`outputs/dijet/2018/rho/unfolding`,
`unfolding_jacobian_reg`) oscillate wildly: negative bins, deviations from the
PYTHIA prior up to a factor 21 (groomed) / 45 (ungroomed), bin-to-bin swings far
beyond any uncertainty band. The visual suspicion was the "smeared bottom-left of
each $p_T$ block" of the response matrix.

= Diagnosis

Quantitative findings on the committed binning ($p_T$ blocks
$[200,290,400,570,760,infinity]$ GeV + a $p_T<200$ sink; groomed: 100 reco
$times$ 50 gen bins):

1. *The response is full-rank, not rank-deficient.* The unrolled probability
   matrix has cond $approx$ 575 (groomed) / 455 (ungroomed). The singular
   spectrum decays gently down to mode $approx 42$, then falls off a cliff — the
   trailing $approx 8$ modes (factor 15–60 weaker) are exactly the
   low-efficiency, low-$rho$ directions (8 gen bins with $epsilon < 0.5$).

2. *The regime is systematics-dominated, not statistics-dominated.* With
   $5.1 times 10^9$ data events the median reco-level relative statistical error
   is 0.5%, while the reco-level data/PYTHIA *shape* disagreement is 10–35%.

3. *Why TUnfold explodes:* an (effectively) unregularized least-squares fit
   must chase that genuine 10–35% model mismatch through the ill-conditioned
   trailing modes; wild gen-space excursions along the weak directions buy tiny
   reco-space $chi^2$ improvements. The wildness is a bias–variance problem,
   not a bug.

4. *The L-curve chose a useless $tau$.* The `_jacobian_reg` tag genuinely
   applied ratio-curvature regularization with L-curve-scanned
   $tau approx 0.70 "/" 0.79$ — and produced a result nearly identical to
   $tau = 0$. The L-curve balances the penalty against *statistical* residuals;
   at these statistics the corner sits at negligible $tau$. The criterion — not
   the implementation — is unfit for this regime.

5. *Fold-back test:* neither method describes the data at statistical
   precision. Median fold-back residuals: TUnfold 2.2%, Bayes $n=4$ 4.0%
   (groomed), versus 0.5% median data stat. TUnfold buys its factor-21 gen-space
   oscillation with a fold-back $chi^2$ of 30k/95 vs Bayes' 49k/95 — a classic
   overfit through weak modes. Conversely the Bayes gap quantifies its
   residual prior memory.

6. *The "bottom-left smear" quantified:* each gen bin's migration kernel spans
   9–13 reco bins (column perplexity) at low $rho$ — the kernel width is
   comparable to half a $p_T$ block. No unfolding method can resolve gen
   structure much finer than this kernel; the information simply is not in the
   data.

#figure(
  image("../diagnosis/diagnosis_groomed.png", width: 100%),
  caption: [Groomed diagnosis. Left: singular spectrum of the probability
  matrix — note the cliff after mode $approx 42$. Middle: reco-level
  data/PYTHIA shape ratio (10–35% mismatch). Right: fold-back residuals of the
  committed TUnfold and Bayes results vs the data statistical band (gray).],
)

= Strategy A — response-matrix manipulation (smoothing, toy-MC parametric response)

Tested on the study binning (12 gen $rho$ bins $times$ 6 $p_T$, 25 reco bins;
`method_study2/smoothed_response/`): (i) nonparametric Gaussian-kernel smoothing
of each gen column's reco-$rho$ shape and (ii) a full parametric "toy-MC"
response — per-gen-bin Gaussian smearing kernels with $mu(x_"gen")$,
$sigma(x_"gen")$ smoothed by low-order polynomials per $p_T$-migration pair.
Both preserve per-column $p_T$-migration fractions and efficiency to machine
precision.

*Result: a clean negative.*

- Smoothing *destabilizes* inversion methods: cond(P) rises
  1449 → 3900 (smoothed) → 78k (parametric) for groomed — widening the kernel
  makes neighbouring gen columns *more* collinear. Unregularized least-squares
  gets more negative bins (6 → 9 → 19) and more oscillation (0.47 → 2.0 → 12).
- The migration kernel is *not Gaussian*: skewness up to $|s| approx 2$, excess
  kurtosis up to $approx 10$ at low $rho$. The parametric matrix self-closes to
  only $approx 20%$ — unusable as a drop-in response.
- Matrix MC-stat is a non-issue: the response-stat toy component is
  $approx 0.1%$ and smoothing does not shrink it; the D'Agostini central value
  moves by only $approx 1.6%$ (groomed) under kernel smoothing.

The useful by-product: the Bayes result is demonstrably robust against
response-matrix noise — stability must come from binning and the regularization
scheme, not from massaging the matrix.

= Strategy B — rescuing least-squares: $tau$ scans, prior dependence, binning

(`method_study2/tunfold_rescue/`.) A faithful numpy replica of the production
ratio-curvature Tikhonov (identical L rows, inverse-variance-weighted,
Poisson-floored data term) scanned over $tau in [10^(-3), 10^3]$ against a
D'Agostini $n_"iter" in {1 dots 32}$ scan. Constraint honored: HERWIG is *not*
used as a yardstick; the bias metric is a *data-driven pseudo-truth closure*
(PYTHIA gen reweighted — smooth, iterated, clipped — until its folding matches
the data reco shape; unfold that pseudo-data with the nominal prior).

- *The production $tau approx 0.79$ sits on the unregularized plateau* —
  materially indistinguishable from $tau = 0$; the L-curve criterion, balanced
  against negligible statistical errors, under-regularized by ~2 orders of
  magnitude. Matching D'Agostini-$n=4$ smoothness needs $tau approx 56$
  (groomed) / $tau approx 10$ (ungroomed), and even there least-squares keeps
  6/15 negative reported bins.
- *Plain curvature-of-deviation regularization is disqualified* (saturates
  into a biased shape); only ratio-curvature is acceptable — consistent with
  the zjet finding.
- *Prior-dependence table on real data* (priors: nominal / data-reweighted /
  $plus.minus 40%$ tilt; median / max spread over reported bins):

  #table(
    columns: 3, align: (left, right, right), stroke: 0.4pt,
    [*method*], [*groomed*], [*ungroomed*],
    [D'Agostini $n=4$, fine binning], [3.4% / 22%], [3.7% / 48%],
    [Tikhonov plateau $tau$, fine binning], [3.5% / 360%], [11% / 509%],
    [D'Agostini $n=4$, coarse binning], [1.0% / 13%], [1.9% / 14%],
    [unregularized lstsq, coarse binning], [*0 / 0 (exact)*], [*0 / 0 (exact)*],
  )

  Strong regularization does *not* cure prior dependence — it relocates it
  into the L-anchor (the plateau-$tau$ maxima are the worst in the table).
  The only prior-free option is unregularized least-squares.
- *Binning scan* (unregularized weighted lstsq on data, groomed):
  72 → 42 gen bins (winner `push_B`, gen edges
  $[-10, -3, -2.5, -2, -1.5, -1, -0.5, 0]$ — *the 0.5-wide bins across the
  second peak survive*) improves cond(P) 1450 → 669, median purity
  0.41 → 0.72, negative bins 7 → 1, oscillation 0.75 → 0.26. Ungroomed winner:
  $[-10, -2, -1.5, -1, -0.5, 0]$, purity 0.83 median.
- *A model-incompatibility floor remains at fine reco binning*: the best-fit
  weighted fold-back $chi^2\/"ndof"$ is 50–950 — at $5 times 10^9$ events the
  10–35% data/MC mismatch is thousands of $sigma$, and 1–2 residual negative
  bins are *deterministic* (toy pulls of $O(-100)$), not noise. The cure came
  from the production convention (next section): merging the reco bins below
  the first interior gen edge removes exactly those incompatible constraints.
- Two latent code bugs were exposed and fixed along the way:
  `ye_unf_dict` uninitialized on the dijet prepared-inputs constructor path
  (crash on any explicit-$tau$ dijet run), and the HERWIG-closure export
  writing to a hardcoded zjet path from dijet runs.

= The publishable recipe — production run `unfolding_coarse`

The endgame combines the findings: *coarse (resolution-matched) gen binning +
merged reco sink below the first interior gen edge + $tau = 0$* — completely
prior-free — implemented in the production pipeline
(`--binning coarse`, `rho_channel_inputs.py` variant; default untouched) and
run end-to-end through ROOT TUnfold
(`outputs/dijet/2018/rho/unfolding_coarse/`):

Adopted binning (v3, after the tail-split study below): groomed gen
$[-10,-5,-3,-2.5,-2,-1.5,-1,-0.5,0]$ — the lowest bin is an *unpublished
edge/buffer bin* (like the $p_T<200$ sink) and the published range starts at
$[-5,-3]$; ungroomed $[-10,-2,-1.5,-1,-0.5,0]$, published from $[-2,-1.5]$.

- *0 negative bins in the published range* (the only negatives sit inside the
  hidden buffer bins, where they belong); published $u\/"prior"$ within
  $[0.72, 1.64]$;
- statistical errors 0.83% median in the published range (honest scale);
- systematic band (v4, HERWIG excluded per policy): few-% and smooth —
  FSR 1.2%, JER 0.8%, q2 0.5% medians; larger only where physical (JMS
  $plus.minus 20$–29% in the kinematic-endpoint bin $[-0.5,0]$). NB the
  generator $chi^2$ labels harden accordingly (PYTHIA 118/7 in the 290–400
  GeV groomed slice) — the band still lacks JES and a trusted model
  uncertainty, so face-value $chi^2$ overstates the discrimination;
- matches the standalone numpy least-squares on the same binning
  (implementation cross-check, sub-percent) and D'Agostini $n=4$ to ~3–4%
  median (independent-method consistency);
- the groomed dip/second-peak structure remains fully binned (0.5-wide bins
  across $[-3, 0]$) and the unfolded data now *discriminate generators*:
  e.g. 570–760 GeV groomed $chi^2\/6$ = 7.5 (PYTHIA8) vs 182 (HERWIG7).

One more cross-channel bug surfaced while reviewing these plots (caught by
Aritra): the shared plotting overlaid a *VINCIA* curve that is actually the
*Z+jet-selection* Vincia gen cache (`CMS_ZJET_JETMASS`, 60k events) — invalid
on dijet. The overlay is now gated to the zjet channel and the dijet plots
regenerated without it.

All five reported $p_T$ slices of the production result:

#figure(
  grid(columns: 2, gutter: 6pt,
    image("figs/coarse_groomed_pt0.png"), image("figs/coarse_groomed_pt1.png"),
    image("figs/coarse_groomed_pt2.png"), image("figs/coarse_groomed_pt3.png"),
    image("figs/coarse_groomed_pt4.png"),
  ),
  caption: [Production `unfolding_coarse` (v3 binning), *groomed*, all
  reported $p_T$ slices (200–290, 290–400, 400–570, 570–760, 760+ GeV):
  prior-free TUnfold ($tau=0$). The $[-10,-5]$ bin is the unpublished
  edge/buffer bin; the published range starts at $[-5,-3]$. Smooth, positive,
  dip structure preserved.],
)

#figure(
  grid(columns: 2, gutter: 6pt,
    image("figs/coarse_ungroomed_pt0.png"), image("figs/coarse_ungroomed_pt1.png"),
    image("figs/coarse_ungroomed_pt2.png"), image("figs/coarse_ungroomed_pt3.png"),
    image("figs/coarse_ungroomed_pt4.png"),
  ),
  caption: [Production `unfolding_coarse`, *ungroomed*, all reported $p_T$
  slices. The merged low bin spans $[-10, -2]$ (purity 0.2–0.5 at low $p_T$ —
  candidate for dropping from the reported range).],
)

== How far down can the published range go? (tail-split study)

Follow-up to Aritra's point that the lowest bin is an unpublished edge/buffer
bin (like the $p_T < 200$ sink), so the real question is where the *published*
range can start (`method_study2/tail_split/`; metrics over published bins only,
production reco-sink convention, 100 data-stat toys):

#table(
  columns: 6, align: (left, right, right, right, right, left), stroke: 0.4pt,
  [*groomed candidate (buffer + published)*], [*cond*], [*neg*], [*spread p90*], [*vs Bayes*], [*verdict*],
  [$[-10,-3]$ + publish from $-3$ (v2)], [936], [0], [3.4%], [3.4%], [OK],
  [$[-10,-5]$ + *publish $[-5,-3]$,…* ], [670], [0], [3.2%], [4.4%], [*adopted*],
  [$[-10,-4]$ + publish $[-4,-3]$,…], [698], [0], [3.0%], [5.1%], [method-dep. at high $p_T$],
  [… + any $0.5$-wide bin below $-3$], [—], [0], [3.6–6%], [5%+], [rejected],
)

- The *adopted* groomed binning $[-10,-5,-3,-2.5,-2,-1.5,-1,-0.5,0]$ publishes
  one extra bin, $[-5,-3]$, and that bin is *good*: purity 0.53–0.62,
  stability 0.62–0.69, toy spread 0.2–1.6%, cross-method agreement 3–11% in
  every $p_T$ slice. It even improves the $[-3,-2.5]$ bin at low $p_T$
  (method agreement 24% → 6% in the lowest slice).
- Publishing $[-4,-3]$ instead is fine at low $p_T$ (2–8%) but becomes
  method-dependent at high $p_T$ (31–47% vs Bayes) — at fixed $rho$, higher
  $p_T$ means lower mass, so the $[-4,-3]$ window migrates worse as $p_T$
  grows. Splitting $[-5,-4]$ off (purity 0.20–0.43, 28–65% vs Bayes) or any
  0.5-wide bin below $-3$ (purity $lt.approx 0.32$, up to 23% toy spread,
  12–59% vs Bayes) is method-dependent noise — rejected.
- *Ungroomed cannot publish below $-2$*: every candidate that tries
  ($[-3,-2]$, $[-4,-2]$, finer) collapses (purity 0.04–0.36 in the new bins,
  method disagreement 20% to $times 20$, and negative published bins return in
  two of them). The current $[-10,-2]$ buffer + publish-from-$-2$ stands.

== Systematic-band forensics (why the v3 bands looked horrible)

The first buffered-binning plots carried huge, bin-to-bin fluctuating
systematic bands. Decomposition of the per-source shifts (published bins):

- *The dominant component was a HERWIG-response systematic* (8.3% median, up
  to 40%, 30% sign-flip rate between adjacent bins) that had silently entered
  the band: the original dijet runs excluded HERWIG deliberately, but the
  machinery re-enables it whenever the herwig pickle exists. Its shift is
  dominated by the low-stat HERWIG sample's amplified MC noise — exactly the
  reason HERWIG was ruled out as a yardstick. *Fixed as policy:* the runner
  now has an explicit `--herwig-band` opt-in (default off); HERWIG remains a
  gen overlay and bias-test sample only.
- *The JMS $plus.minus 20$–29% values are real physics but live in exactly one
  bin*: the endpoint bin $[-0.5, 0]$, where $m -> p_T R\/2$ and the spectrum
  plummets — a $plus.minus 1%$ mass-scale shift moves that nearly-empty bin's
  content by $plus.minus 20%$, cleanly anti-symmetric in Up/Down (a genuine
  scale derivative, not noise). Everywhere else JMS is $plus.minus 1$–2%.
- Everything else is small and smooth: FSR 1.2%, JER 0.7–0.8%, q2 0.5%, pu,
  pdf, lumi $lt.approx 0.1%$ (weight-based sources have $approx 0$ sign-flip
  rate — the machinery itself is sound).
- Cross-check: D'Agostini sees nearly the same JMS endpoint shifts (18–20% vs
  20–30%), confirming they are properties of the varied response, not
  least-squares amplification.

== Buffer-edge scan (was $-5$ the right edge?)

The $[-10,-5,-3]$ scheme was proposed as an *example*; scanning the buffer
edge (published wide bin $[-4.5,-3]$ vs $[-5,-3]$ vs $[-6,-3]$, all metrics on
published bins): $[-5,-3]$ is the only choice with consistent cross-method
agreement in *all* $p_T$ slices (3–11%; purity 0.53–0.62). $[-4.5,-3]$ is a
close second but degrades at high $p_T$ (26% method disagreement, purity
0.495); $[-6,-3]$ is method-dependent everywhere (14–40% — the wider window
swallows more of the unresolvable region). The $-5$ edge is confirmed, now by
scan rather than by example.

#figure(
  image("../tunfold_rescue/money_bias_variance.png", width: 100%),
  caption: [Bias–variance plane with the *data-driven pseudo-truth closure*
  bias (HERWIG-free). The $tau$ and $n_"iter"$ families trade bias for spread;
  the prior-free least-squares point at the coarse binning sits at exactly
  zero closure bias by construction.],
) <money>

= Strategy C — independent cross-checks and uncertainty forensics

(`method_study2/independent_methods/`.) Constraints applied: HERWIG excluded
as a yardstick (low statistics, per Aritra); prior dependence treated as a
first-class concern (per the analysis lead).

== Independent central-value cross-checks

- *Forward-folding spline fit* (per-$p_T$-slice smooth multiplicative spline
  correction on the PYTHIA gen shape, fitted to reco data through the response;
  no unfolding, no Herwig): agrees with the D'Agostini result to *9–13% median*
  in the well-populated slices, and describes the reco data better than the
  Bayes fold-back. It disagrees only in the lowest $p_T$ slice, which is
  dominated (~90%) by the $p_T < 200$ migration sink.
- *Positivity-constrained regularized fit* (NNLS with ratio-curvature penalty):
  agrees with D'Agostini $n=4$ to *5–6% median* across all reported slices.

Three methods with different failure modes land on the same answer at the
5–13% level — the unfolded central value is real; only TUnfold-at-fine-binning
is the outlier.

== The statistical uncertainties of the Bayes tag are wrong by $approx 17 times$

Forensics chain (file:line in the study README): `unfolded_input_errors` comes
from `RooUnfoldBayes::Eunfold(kCovariance)`, which propagates the *input
histogram bin errors* — and the numpy→TH1 converter
(`roounfold_backend.py:63`) hard-codes those to $sqrt(N)$, *discarding the
stored sumw2*. The dijet data is prescale-weighted with
$"sumw2"/N approx 247$, so honest errors are $sqrt(247) approx 15.7 times$
larger than Poisson. Honest toy propagation gives *0.42% / 0.64%* median
(groomed/ungroomed) vs the tag's quoted 0.025% / 0.032%. In addition the
$n_"iter"$ regularization systematic ($|u_(n=6) - u_(n=4)| approx 1.5%$
median) is the single largest uncertainty and is absent from the tag.

== Prior dependence of D'Agostini, quantified

Swapping the prior from PYTHIA to flat-in-$x$ moves the $n=4$ central value by
*12.9% (groomed) / 21.7% (ungroomed) median* over reported bins; a gentle
$0.6 -> 1.4$ tilt moves it only ~1.4%. More iterations wash the prior out
($n=10$: 9.6% / 5.0%) at the cost of amplified noise. The prior-dependence
concern is quantitatively confirmed — D'Agostini cannot be the *primary*
result without a defended prior-uncertainty band.

= Is the groomed second peak resolvable? (binning defense)

The fine 0.5-wide gen bins across $x in [-3,-1]$ were chosen deliberately to
resolve the groomed dip/second-peak structure. The detector kernel there has
RMS 0.45–0.65 — comparable to the bin width — so the question is whether the
structure survives unfolding. A dedicated injection test
(`method_study2/dip_resolvability/`): truth with the PYTHIA dip vs a smoothed
dipless truth, folded, fluctuated with *data-level statistics* (300 toys),
unfolded, and the two hypotheses discriminated with the full toy covariance.

*Result: the dip is trivially resolvable — keep the peak binning.*

- Hypothesis separation is effectively hundreds of $sigma$ at data statistics
  for *every* method and binning tried; statistics and resolution are not the
  limitation in $[-3,-1]$.
- Prior-free unregularized least-squares (tail-merged binning) recovers the
  injected dip *exactly* (bias 0, toy spread 1.45% median).
- Even D'Agostini with a deliberately *dipless* prior recovers the dip to 0.7%
  median (worst dip bin 16% at $n=4$, 10% at $n=10$) — bounding the prior-pull
  on this specific structure.
- The instability of the committed result does *not* come from the peak
  region: it comes from the low-$rho$ tail ($x < approx -3.5$), where kernel
  RMS reaches 0.7–0.8, efficiency drops below 0.5, and the trailing singular
  modes live.

*The defensible binning: merge the tail below $approx -3.5 " to " -4$ into one
bin per $p_T$ slice; keep the 0.5-wide bins across the second peak.* The
physics goal and the numerical stability are not in conflict.

#figure(
  image("../dip_resolvability/dip_recovery_groomed_tailmerged_0p5.png", width: 100%),
  caption: [Dip-recovery injection test (groomed, tail-merged binning, 0.5-wide
  peak bins). Black: injected truth with the PYTHIA dip; gray dashed: dipless
  alternative; red: prior-free least-squares (sits exactly on the dip truth);
  green: D'Agostini $n=4$ run with the *dipless* prior — the dip is still
  recovered.],
)

= Conclusions and recommendation

*Recommended primary result*: the prior-free production run
`unfolding_coarse` (coarse gen binning `push_B`, merged reco sink,
TUnfold $tau = 0$, area constraint). Quote as uncertainties: honest input-stat
(toy-propagated on stored sumw2, ~0.9% median), response systematics, a
cross-method envelope vs D'Agostini $n=4$ (~3%), and — until the skimmer adds
them — an explicit caveat that *JES response variations are missing*.

*Recommended cross-check (not primary)*: D'Agostini $n=4$ on the same binning,
with a quoted prior-variation band (1–2% median on the coarse binning) and the
$n_"iter" plus.minus 2$ band (~1.5%).

*Do not*: use L-curve-chosen $tau$ in this statistics regime; use plain
curvature-of-deviation regularization; smooth or parametrize the response
matrix; floor efficiencies; trust the committed `unfolding_bayes` error bars.

*Follow-ups*: (1) fix `roounfold_backend.py:63` to propagate sumw2 bin errors
(and regenerate any result quoting them); (2) add JES-varied response matrices
in the next skimmer production (`smp_jetmass_run2`); (3) decide whether the
ungroomed low-$rho$ merged bin ($[-10, -2]$, purity 0.2–0.5 at low $p_T$)
should be dropped from the reported range instead; (4) port the honest
stat/systematic bands into the `unfolding_coarse` plots and re-generate the
comparison gallery.

= Artifacts

All under `outputs/dijet/2018/rho/`: `method_study2/{diagnosis, smoothed_response,
tunfold_rescue, independent_methods, dip_resolvability}/` (each with README.md,
metrics.json, PNGs), the production run `unfolding_coarse/`, and study code
`scripts/dijet_methods/study2_*.py`. Code changes (uncommitted, on worktree
branch `claude/objective-swanson-22e275`): `unfolder_core.py` (`ye_unf_dict`
init; guarded HERWIG-closure export), `rho_channel_inputs.py`
(`channel_rho_binning(variant="coarse")`), `run_rho_unfolding.py`
(`--binning` flag), `scripts/dijet_methods/` (imported from
`experiment/dijet-numerical-unfolding` + new `study2_*` studies).
