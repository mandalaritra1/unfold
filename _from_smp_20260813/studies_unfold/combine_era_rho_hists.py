#!/usr/bin/env python
"""Sum the four Run 2 era pickles into ONE combined pickle per channel/kind.

Every era file is already fully scaled to its own luminosity (MC xs*L/sumw,
data prescale weights), so the Run 2 combination is a plain sum of values AND
variances, done AFTER the per-file dataset sum (the dataset StrCategory differs
era to era) and WITH flow bins.

Output keeps the exact schema rho_unfold_inputs_from_hists.py expects -- a dict
of Hist with a (single-entry) `dataset` axis, the `systematic` axis and the
untouched numeric axes -- so that script runs on the combined file unmodified.

Eras are statistically independent, so covariances add too.

    python scripts/studies/unfold/combine_era_rho_hists.py            # all four
    python scripts/studies/unfold/combine_era_rho_hists.py dijet_mc   # one job

then feed the result to the input-maker unchanged:

    python scripts/studies/unfold/rho_unfold_inputs_from_hists.py --channel dijet \
        --mc  outputs/pairsplit_unfold/combined/combined_run2_dijet_mc.pkl \
        --data outputs/pairsplit_unfold/combined/combined_run2_dijet_data.pkl \
        --groom g --out outputs/pairsplit_unfold/npz/dijet_run2_nominal.npz
"""
import gc
import pickle
import resource
import sys
import time
from pathlib import Path

import hist
import numpy as np

BASE = Path("/Users/aritra/cernbox (2)/hadronic_minimal_rho_pairsplit")
ERAS = ["2016APV", "2016", "2017", "2018"]
OUT = Path("/Users/aritra/Projects/smp_jetmass_run2/outputs/pairsplit_unfold/combined")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = {
    "mc": ["ptjet_rhojet_g_reco", "ptjet_rhojet_g_gen", "response_matrix_rho_g",
           "ptjet_rhojet_u_reco", "ptjet_rhojet_u_gen", "response_matrix_rho_u"],
    "data": ["ptjet_rhojet_g_reco", "ptjet_rhojet_u_reco",
             "reco_cov_rho_g", "reco_cov_rho_u"],
}


def era_path(era, channel, kind):
    stem = ("minimal_rho_%s_mg_pythia8_%s.pkl" % (channel, era) if kind == "mc"
            else "minimal_rho_%s_data_%s.pkl" % (channel, era))
    return BASE / era / f"{channel}_{kind}" / stem


def combine(channel, kind):
    acc, names, axes, edges_ref, absent = {}, {}, {}, {}, set()
    for era in ERAS:
        p = era_path(era, channel, kind)
        t = time.time()
        d = pickle.load(open(p, "rb"))
        for key in KEYS[kind]:
            if key in absent:
                if key in d:
                    raise SystemExit(f"{key} present in {era}, absent earlier")
                continue
            if key not in d:
                if era == ERAS[0]:
                    print(f"    {key}: absent in {era}, skipped everywhere")
                    absent.add(key)
                    continue
                raise SystemExit(f"{key} missing in {era} but present in {ERAS[0]}")
            h = d[key][{"dataset": sum}]
            ax = [a for a in h.axes if a.name != "systematic"]
            ed = [(a.name, np.asarray(a.edges)) for a in ax]
            syst = list(h.axes["systematic"])
            if key not in acc:
                names[key] = syst
                axes[key] = ax
                edges_ref[key] = ed
                acc[key] = {s: [np.zeros(0), np.zeros(0)] for s in syst}
            else:
                if set(syst) != set(names[key]):
                    miss = set(names[key]) ^ set(syst)
                    raise SystemExit(f"{key}: systematic set differs in {era}: {miss}")
                for (n0, e0), (n1, e1) in zip(edges_ref[key], ed):
                    if n0 != n1 or not np.allclose(e0, e1):
                        raise SystemExit(f"{key}: axis {n0} differs in {era}")
            for s in names[key]:
                hs = h[{"systematic": s}]
                v, w = hs.values(flow=True), hs.variances(flow=True)
                a = acc[key][s]
                a[0] = v if a[0].size == 0 else a[0] + v
                a[1] = w if a[1].size == 0 else a[1] + w
            del h
        del d
        gc.collect()
        print(f"  {era}: {time.time()-t:.1f}s, maxrss "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")

    out = {}
    for key in acc:
        ns = names[key]
        H = hist.Hist(
            hist.axis.StrCategory(["run2"], growth=True, name="dataset",
                                  label="Primary dataset"),
            hist.axis.StrCategory(ns, growth=True, name="systematic",
                                  label="Systematic Uncertainty"),
            *axes[key], storage=hist.storage.Weight())
        view = H.view(flow=True)
        for i, s in enumerate(ns):
            view["value"][0, i] = acc[key][s][0]
            view["variance"][0, i] = acc[key][s][1]
        out[key] = H
        print(f"    {key}: {len(ns)} systs, sum(nominal) = "
              f"{acc[key][ns[0] if 'nominal' not in ns else 'nominal'][0].sum():.6g}")
    p = OUT / f"combined_run2_{channel}_{kind}.pkl"
    pickle.dump(out, open(p, "wb"), protocol=4)
    print(f"  wrote {p}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["trijet_data", "trijet_mc", "dijet_data", "dijet_mc"]
    for job in todo:
        ch, kind = job.rsplit("_", 1)
        print(f"\n===== {ch} {kind}")
        combine(ch, kind)
