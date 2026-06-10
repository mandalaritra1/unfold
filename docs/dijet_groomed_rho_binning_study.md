# Dijet Groomed Low-Rho Binning Study

This study uses the nominal 2018 dijet MG+PYTHIA8 gen histogram and response
matrix, plus the measured-data variances. It changes no Z+jet or trijet
binning.

## Producer Bins

The groomed gen-level rho axis contains:

```text
[-10, -8, -7, -6, -5, -4.5, -4, ..., 0]
```

The `[-10,-8]` bin is empty in every reported pT interval. The `[-8,-7]`
bin is populated but has low efficiency and reconstructs broadly at larger
rho, so it cannot be unfolded as a separate truth bin reliably.

## Schemes Tested

- Existing: one truth bin `[-10,-4.5]`.
- Two-bin tail: `[-10,-5]`, `[-5,-4.5]`.
- Three-bin tail: `[-10,-6]`, `[-6,-5]`, `[-5,-4.5]`.
- Finer tail with a separate boundary at `-7`.

The three-bin and finer schemes have adequate nominal MC counts, but the
measured-data unfolding oscillates and produces negative low-rho bins. They
are therefore rejected without changing the established TUnfold
regularization.

The two-bin tail was tested independently in every reported pT interval.
It is stable in 400-570 and 570-760 GeV:

| pT (GeV) | Gen low-rho bins | Normalized data/PYTHIA8 | Input stat. frac. |
| --- | --- | --- | --- |
| 400-570 | `[-10,-5]`, `[-5,-4.5]` | 0.769, 0.873 | 0.107, 0.098 |
| 570-760 | `[-10,-5]`, `[-5,-4.5]` | 0.912, 0.813 | 0.060, 0.075 |

Nominal MG+PYTHIA8 closure is 1.000 in both bins for both pT intervals. The
column-normalized same-pT low-rho response condition numbers are 4.2 and 4.9,
respectively.

## Selected Binning

- Keep `[-10,-4.5]` merged for 0-200, 200-290, 290-400, and
  760-infinity GeV.
- Use `[-10,-5]`, `[-5,-4.5]` for 400-570 and 570-760 GeV.
- In those two intervals, retain finer reco edges
  `[-10,-6,-5.5,-5,-4.75,-4.5]` before the standard 0.25-wide bins.

The 0-200 GeV interval remains internal for migrations. Dijet groomed plots
show the complete selected bin range down to `-10`.
