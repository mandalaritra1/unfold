"""Analysis binnings as data.

A ``Binning`` carries everything the unfolding needs to know about the axes:
the pT edges, the fine observable edges the input histograms are stored with
(reco and gen), and the per-pT-slice analysis edges the unfold actually runs
on (a subset of the fine edges, possibly different per slice).

Z+jet binnings are listed in ``ZJET_BINNINGS`` by name; the dijet/trijet and
hadronic loaders build their ``Binning`` from their own tables (see
``channel_inputs.py`` and ``hadronic/inputs.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Binning:
    pt_edges: tuple[float, ...]
    reco_edges: tuple[float, ...]            # fine reco edges of the input histograms
    gen_edges: tuple[float, ...]             # fine gen edges of the input histograms
    reco_edges_by_pt: tuple[tuple[float, ...], ...]   # analysis reco edges, one list per pT slice
    gen_edges_by_pt: tuple[tuple[float, ...], ...]    # analysis gen edges, one list per pT slice

    def __post_init__(self):
        n_pt = len(self.pt_edges) - 1
        if len(self.reco_edges_by_pt) != n_pt or len(self.gen_edges_by_pt) != n_pt:
            raise ValueError("need one reco and one gen edge list per pT slice")
        for by_pt, fine in ((self.reco_edges_by_pt, self.reco_edges), (self.gen_edges_by_pt, self.gen_edges)):
            for edges in by_pt:
                missing = [e for e in edges if not np.isclose(e, fine).any()]
                if missing:
                    raise ValueError(f"analysis edges {missing} are not in the fine edges {fine}")

    @property
    def n_pt(self):
        return len(self.pt_edges) - 1

    def merge_gen_below(self, threshold):
        """Collapse every per-pT gen bin below ``threshold`` into one bin."""
        return replace(
            self,
            gen_edges_by_pt=tuple(merge_edges_below(e, threshold) for e in self.gen_edges_by_pt),
        )


def merge_edges_below(edges, threshold, *, tol=1e-9):
    """Keep edges[0]; drop interior edges strictly below ``threshold``."""
    edges = [float(e) for e in edges]
    kept = [edges[0]] + [e for e in edges[1:] if e >= threshold - tol]
    return tuple(kept)


def _same(edges, n_pt):
    return tuple(tuple(float(e) for e in edges) for _ in range(n_pt))


# ---------------------------------------------------------------------------
# Z+jet.  The pT sink is [185, 200] since ARC round 2 (the arc_r2 reskims); the
# earlier [0, 200] sink inputs cannot be unfolded against these tables.
# ---------------------------------------------------------------------------
ZJET_PT_EDGES = (185.0, 200.0, 290.0, 400.0, 13000.0)

# rho = m / (pT R), stored as log10(rho^2).  These are the skimmer axes.
_RHO_RECO_V2 = (-10, -8, -6, -5.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3,
                -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0)
_RHO_GEN_V2 = (-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0)

# ARC round-2 groomed buffer binning: the shown region stays 0.5 wide, the tail
# below -3.5 is resolved at 0.25 down to -5 as a hidden migration buffer, and
# reco is a strict 2:1 refinement of gen.  Requires the arc_r2 reskims.
_RHO_GEN_ARCR2_GROOMED = (-10, -6, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5,
                          -3, -2.5, -2, -1.5, -1, 0)
_RHO_RECO_ARCR2_GROOMED = (-10, -8, -6, -5.5, -5, -4.875, -4.75, -4.625, -4.5, -4.375,
                           -4.25, -4.125, -4, -3.875, -3.75, -3.625, -3.5, -3.25, -3,
                           -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.5, 0)

_MASS_RECO = (0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,
              140, 150, 160, 170, 185, 200, 215, 1000)
_MASS_GEN = (0, 10, 20, 30, 50, 70, 90, 110, 130, 150, 170, 200, 1000)
_MASS_GEN_BY_PT = (
    (0, 10, 20, 30, 50, 70, 90, 110, 1000),
    (0, 10, 20, 30, 50, 70, 90, 110, 1000),
    (0, 10, 20, 30, 50, 70, 90, 110, 130, 150, 1000),
    (0, 10, 20, 30, 50, 70, 90, 110, 130, 150, 170, 200, 1000),
)

ZJET_BINNINGS = {
    # groomed rho, pre-arc_r2 skims: merge everything below -3 into one low
    # bin (shown from -4.5), last two bins merged into [-1, 0].
    "rho_groomed_v2": Binning(
        pt_edges=ZJET_PT_EDGES,
        reco_edges=_RHO_RECO_V2,
        gen_edges=_RHO_GEN_V2,
        reco_edges_by_pt=_same((-10, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3, -2.75,
                                -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0), 4),
        gen_edges_by_pt=_same((-10, -3, -2.5, -2, -1.5, -1, 0), 4),
    ),
    # ungroomed rho: everything below -2.5 is one catch-all bin, strict 2:1
    # reco:gen (the merged gen [-1, 0] bin gets two 0.5-wide reco bins).
    "rho_ungroomed_v2": Binning(
        pt_edges=ZJET_PT_EDGES,
        reco_edges=_RHO_RECO_V2,
        gen_edges=_RHO_GEN_V2,
        reco_edges_by_pt=_same((-10, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.5, 0), 4),
        gen_edges_by_pt=_same((-10, -2.5, -2, -1.5, -1, 0), 4),
    ),
    "rho_groomed_arcr2": Binning(
        pt_edges=ZJET_PT_EDGES,
        reco_edges=_RHO_RECO_ARCR2_GROOMED,
        gen_edges=_RHO_GEN_ARCR2_GROOMED,
        reco_edges_by_pt=_same(_RHO_RECO_ARCR2_GROOMED, 4),
        gen_edges_by_pt=_same(_RHO_GEN_ARCR2_GROOMED, 4),
    ),
    "mass_groomed": Binning(
        pt_edges=ZJET_PT_EDGES,
        reco_edges=_MASS_RECO,
        gen_edges=_MASS_GEN,
        reco_edges_by_pt=(
            (0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1000),
            (0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1000),
            (0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 1000),
            (0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
             170, 185, 200, 215, 1000),
        ),
        gen_edges_by_pt=_MASS_GEN_BY_PT,
    ),
    "mass_ungroomed": Binning(
        pt_edges=ZJET_PT_EDGES,
        reco_edges=_MASS_RECO,
        gen_edges=_MASS_GEN,
        reco_edges_by_pt=(
            (0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1000),
            (0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1000),
            (0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 1000),
            (0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
             170, 185, 200, 215, 1000),
        ),
        gen_edges_by_pt=_MASS_GEN_BY_PT,
    ),
}
# The ungroomed skimmer axes were not changed by the ARC round-2 rebinning.
ZJET_BINNINGS["rho_ungroomed_arcr2"] = ZJET_BINNINGS["rho_ungroomed_v2"]


def compress_open_ended_last_bin(edges, *, match_to_second_last=True):
    """Replace the open-ended last upper edge by a finite one, for display only.

    ``[.., 200, 500, 13000] -> [.., 200, 500, 800]``: the last bin gets the
    width of the second-last one.  Bin contents are untouched.
    """
    e = np.asarray(edges, dtype=float).copy()
    if len(e) < 3:
        return e
    w1 = e[-1] - e[-2]
    w2 = e[-2] - e[-3]
    if match_to_second_last or w2 <= 0:
        e[-1] = e[-2] + abs(w2)
    else:
        e[-1] = e[-2] + max(1.0, min(abs(w1), abs(w2)))
    return e


def compress_edges_by_pt(edge_lists):
    return [compress_open_ended_last_bin(edges) for edges in edge_lists]
