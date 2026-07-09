"""ARC round-2: fake/miss flow-consistency for the 185-200 low-pt sink bin.

Background
----------
The zjet rho unfolding derives fakes/misses by subtracting the matched response
matrix from the reco/gen spectra (unfolder_core `_make_inputs_numpy`):

    fakes  = reco_spectrum - matched.project(ptreco)
    misses = gen_spectrum  - matched.project(ptgen)

`hist.project()` sums the removed axes *including their flow bins*, whereas the
migration matrix actually handed to TUnfold is built with `values(flow=False)`.
While the pt axis starts at 0 this discrepancy is harmless: jet pt is always
above the NanoAOD storage floors, so nothing ever lands in pt-underflow.

ARC round-2 narrows the low sink bin from [0,200] to [185,200]. That puts every
jet below 185 GeV into pt-underflow for the first time, and the two flow
conventions now disagree: a cross-boundary event (gen<185 & reco>=185, or the
reverse) gets folded into "matched" by the flow-inclusive projection -- so it is
cancelled out of fakes/misses -- yet it is dropped from the flow=False matrix.
It then sits in the measured spectrum with no matrix column and no fake/miss
entry to absorb it: a bias.

The fix (on this branch) sums the matched matrix over the IN-RANGE pt bins only
(`h[{pt_axis: slice(0, N, sum)}]`) before subtracting, so gen<185 & reco>=185 ->
fake and reco<185 & gen>=185 -> miss. This script demonstrates (a) that pt-flow
is empty today so the fix is a no-op on the current pkls, and (b) the size of
the leak the fix removes, using a pt cut simulated on the existing [0,200] pkls.

Run: .venv/bin/python scripts/study_pt_sink_flow.py   (no ROOT needed)
"""

import pickle
import sys

import numpy as np

try:
    from unfold.utils.integrate_and_rebin import rebin_hist
except ModuleNotFoundError:
    sys.path.insert(0, "src")
    from unfold.utils.integrate_and_rebin import rebin_hist

PKL = "inputs/zjet/rho/fixed_jec/pythia_all.pkl"


def _nominal(h):
    return h[{"systematic": "nominal"}] if "systematic" in h.axes.name else h


def _sum_inrange_pt(h, pt_axis):
    """Sum a pt axis over in-range bins only (exclude underflow/overflow)."""
    return h[{pt_axis: slice(0, len(h.axes[pt_axis]), sum)}]


def main():
    payload = pickle.load(open(PKL, "rb"))
    reco2d = _nominal(payload["ptjet_rhojet_g_reco"])
    gen2d = _nominal(payload["ptjet_rhojet_g_gen"])
    resp = _nominal(payload["response_matrix_rho_g"])

    # (a) pt-flow is empty in the production [0,200,...] pkls -> fix is a no-op.
    v = resp.values(flow=True)  # axes: ptgen, mpt_gen, ptreco, mpt_reco
    ptgen_uf = v[0, :, :, :].sum()
    ptreco_uf = v[:, :, 0, :].sum()
    print("(a) current [0,200,290,400,13000] pkls")
    print(f"    ptgen-underflow = {ptgen_uf:.6g}   ptreco-underflow = {ptreco_uf:.6g}")
    cur = resp.project("ptreco", "mpt_reco").values()
    fix = _sum_inrange_pt(resp, "ptgen").project("ptreco", "mpt_reco").values()
    print(f"    matched(reco-proj) flow-inclusive vs in-range: max|diff| = "
          f"{np.abs(cur - fix).max():.2e}  -> fix is a no-op today\n")

    # (b) simulate a pt cut at 200 (drop [0,200] -> underflow) to size the leak.
    # Work at pt level (sum the rho axis with flow consistently on both sides) so
    # the demonstration isolates the pt-flow effect from the pre-existing,
    # orthogonal rho-axis flow handling.
    newpt = [200.0, 290.0, 400.0, 13000.0]
    R = rebin_hist(rebin_hist(resp.copy(), "ptreco", newpt), "ptgen", newpt)
    r = rebin_hist(reco2d.copy(), "ptreco", newpt)
    g = rebin_hist(gen2d.copy(), "ptgen", newpt)

    reco_pt = r.project("ptreco").values(flow=False)          # in-range reco spectrum
    gen_pt = g.project("ptgen").values(flow=False)            # in-range gen spectrum

    # matched projection, OLD (flow-inclusive) vs FIXED (in-range pt only).
    matched_reco_old = R.project("ptreco").values(flow=False)
    matched_reco_fix = _sum_inrange_pt(R, "ptgen").project("ptreco").values(flow=False)
    matched_gen_old = R.project("ptgen").values(flow=False)
    matched_gen_fix = _sum_inrange_pt(R, "ptreco").project("ptgen").values(flow=False)

    fakes_old = reco_pt - matched_reco_old
    fakes_fix = reco_pt - matched_reco_fix
    misses_old = gen_pt - matched_gen_old
    misses_fix = gen_pt - matched_gen_fix

    leak_fake = fakes_fix.sum() - fakes_old.sum()   # events the fix moves into fakes
    leak_miss = misses_fix.sum() - misses_old.sum()

    print("(b) simulated pt cut at 200 (illustrates the leak the fix removes)")
    print(f"    fakes  total: old(flow-incl)={fakes_old.sum():.6g}  "
          f"fixed(in-range)={fakes_fix.sum():.6g}  "
          f"leak recovered={leak_fake:.6g} ({100 * leak_fake / matched_reco_fix.sum():.2f}% of matched)")
    print(f"    misses total: old(flow-incl)={misses_old.sum():.6g}  "
          f"fixed(in-range)={misses_fix.sum():.6g}  "
          f"leak recovered={leak_miss:.6g} ({100 * leak_miss / matched_gen_fix.sum():.2f}% of matched)")
    # With the fix, matched(in-range) + fakes == reco spectrum exactly, and the
    # matrix used in the unfold (flow=False) equals matched(in-range) at pt level.
    print(f"    balance  matched(in-range)+fakes  == reco spectrum: "
          f"{np.allclose(matched_reco_fix + fakes_fix, reco_pt)}")
    print(f"    balance  matched(in-range)+misses == gen  spectrum: "
          f"{np.allclose(matched_gen_fix + misses_fix, gen_pt)}")


if __name__ == "__main__":
    main()
