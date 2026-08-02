from array import array
from dataclasses import dataclass, field, replace
from pathlib import Path
import re

import hist
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pickle as pkl
import ROOT
from scipy.stats import chi2 as scipy_chi2
from cycler import cycler
from matplotlib.colors import LogNorm

from unfold.tools import *
from unfold.tools import binning
from unfold.utils.integrate_and_rebin import *
from unfold.utils.merge_helpers import *

plt.rcParams["axes.prop_cycle"] = cycler(color=['#5790fc', '#f89c20', '#e42536', '#964a8b', '#9c9ca1', '#7a21dd'])

# Shared with standalone plotting scripts (kept ROOT-free over there).
from unfold.utils.cms_plot import (
    DEFAULT_CMS_LABEL_FLAVORS,
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    save_cms_label_flavors,
    stamp_figure,
)


def _declare_open_l():
    """Expose TUnfold's protected AddRegularisationCondition and scan helpers.

    The subclass is needed for the ratio-curvature regularization (the public
    RegularizeCurvature cannot express the exact (1/m0, -2/m1, 1/m2) row).
    The scan helpers exist because PyROOT cannot pass TGraph**/TSpline**
    output arguments directly.
    """
    if not hasattr(ROOT, "TUnfoldDensityOpenL"):
        ROOT.gInterpreter.Declare(
            """
            class TUnfoldDensityOpenL : public TUnfoldDensity {
            public:
              using TUnfoldDensity::TUnfoldDensity;
              using TUnfoldDensity::AddRegularisationCondition;
            };

            struct UnfoldLcurveScan {
              TGraph* lcurve = nullptr;
              TSpline* logTauX = nullptr;
              TSpline* logTauY = nullptr;
              TSpline* curvature = nullptr;
              Int_t iBest = -1;
            };
            UnfoldLcurveScan RunUnfoldLcurveScan(TUnfold& u, Int_t nScan) {
              UnfoldLcurveScan r;
              r.iBest = u.ScanLcurve(nScan, 0., 0., &r.lcurve,
                                     &r.logTauX, &r.logTauY, &r.curvature);
              return r;
            }

            struct UnfoldSureScan {
              TGraph* logTauSURE = nullptr;
              TGraph* df_chi2A = nullptr;
              TGraph* lCurve = nullptr;
              Int_t iBest = -1;
            };
            UnfoldSureScan RunUnfoldSureScan(TUnfoldDensity& u, Int_t nScan,
                                             Double_t tauMin, Double_t tauMax) {
              UnfoldSureScan r;
              r.iBest = u.ScanSURE(nScan, tauMin, tauMax, &r.logTauSURE,
                                   &r.df_chi2A, &r.lCurve);
              return r;
            }
            """
        )


def _graph_to_arrays(graph):
    n = graph.GetN()
    return (
        np.array([graph.GetPointX(i) for i in range(n)]),
        np.array([graph.GetPointY(i) for i in range(n)]),
    )


def _merge_edges_below(edges, threshold, *, tol=1e-9):
    """Collapse all bins below ``threshold`` into a single low-edge bin.

    Keeps the first (lowest) edge and every edge at or above ``threshold``,
    dropping the interior edges strictly below it. E.g. with threshold -2.5,
    ``[-10, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]`` becomes
    ``[-10, -2.5, -2, -1.5, -1, -0.5, 0]``. Idempotent when the binning is
    already merged at or above the threshold.
    """
    edges = [float(e) for e in edges]
    kept = [edges[0]]
    kept.extend(e for e in edges[1:] if e >= threshold - tol)
    return kept


@dataclass
class ObservableSpec:
    """Holds every observable-specific knob the Unfolder needs.

    Used to parameterize a single Unfolder implementation for different
    observables (jet mass, rho = ln(m^2/pT^2), ...). See MASS_SPEC and
    RHO_SPEC below for concrete instances.
    """
    name: str                       # short tag, e.g. "mass", "rho"
    reco_axis: str                  # histogram axis name on reco side
    gen_axis: str                   # histogram axis name on gen side

    # Input/output path conventions. Per-era files are joined with input_dir.
    input_dir: str                  # e.g. "./inputs/zjet/mass/"
    mc_file: str                    # merged pythia pkl (no dir)
    data_file: str
    herwig_file: str
    jk_data_file: str
    era_mc_files: list              # per-era mc pkl names (no dir)
    era_jk_files: list              # per-era jk mc pkl names (no dir)
    era_prefix: str                 # prefix stripped from per-era stems
    # Fallback list for the reweighted/herwig-model file used in _make_inputs_numpy
    reweighted_fallback_files: list
    output_dir: str                 # e.g. "outputs/zjet/mass/"

    # Bin accessors: names of attributes on binning.bin_edges objects
    edges_reco_attr: str
    edges_gen_attr: str
    reco_edges_by_pt_attr: str
    gen_edges_by_pt_attr: str

    # Histogram key triples {response, reco, gen} for each groomed flag
    hist_keys_groomed: dict
    hist_keys_ungroomed: dict

    # Plot labels and axis limits
    x_label_groomed: str
    x_label_ungroomed: str
    short_label_groomed: str
    short_label_ungroomed: str
    xlim_lower_groomed: float
    xlim_lower_ungroomed: float
    normalized_ylabel: str

    # How normalized-result statistical uncertainties are obtained:
    #   "legacy"   -> per-bin relative errors of the absolute spectrum applied
    #                 to the normalized values (no normalization correlations)
    #   "jacobian" -> full covariance propagated through the per-pT-slice
    #                 normalization Jacobian (errors and correlation matrix)
    stat_propagation: str = "legacy"

    # Unfolding regularization:
    #   "none"            -> DoUnfold(0), the historical behavior
    #   "ratio_curvature" -> custom L rows (1/m0, -2/m1, 1/m2): curvature of
    #                        x / x_MC within each pT slice, so spectra
    #                        proportional to the nominal MC prior carry zero
    #                        penalty (handles the falling pT spectrum). tau is
    #                        L-curve-scanned on the nominal data unfold, then
    #                        frozen for all systematic and jackknife unfolds.
    regularization: str = "none"

    # Fixed regularization strength. None -> L-curve scan on the nominal data
    # unfold. Ignored when regularization == "none".
    tau: float | None = None

    # Unfolding backend:
    #   "tunfold"          -> TUnfoldDensity (default, the production path)
    #   "roounfold_bayes"  -> iterative Bayes / D'Agostini via RooUnfold; the
    #                         jackknife replicas re-unfold through it unchanged,
    #                         so the statistical uncertainty stays jackknife-based.
    method: str = "tunfold"
    # D'Agostini iterations (the Bayes "regularization"). Used when
    # method == "roounfold_bayes"; ignored otherwise.
    n_iter: int = 4

    # Optional gen-binning coarsening: when set, every per-pT gen edge list is
    # collapsed so that all bins below this log10(rho^2) threshold merge into a
    # single low-rho bin (the first edge is kept; interior edges strictly below
    # the threshold are dropped). The fine reco binning and the fine flat
    # ``edges_gen`` intermediate are left untouched, mirroring how the ungroomed
    # binning already merges its low-rho tail. None -> no coarsening.
    gen_merge_below: float | None = None

    # TUnfold area constraint. True -> kEConstraintArea (the production default,
    # one global area constraint). False -> kEConstraintNone (no area
    # constraint), the plain-TUnfold configuration.
    area_constraint: bool = True

    # Shower/hadronization model uncertainty from offline column-scaled
    # responses (unfold.tools.model_envelope): re-unfold the data through
    # Vincia/CR/frag-reweighted responses and take the per-bin envelope,
    # maxed with the FSR PSWeight re-unfold shift. When True, the herwig /
    # fsr / isr entries are *excluded* from the quadrature systematics sum
    # (herwig is superseded as the model term; def-ISR is pathological) and
    # the model term is added instead. False -> legacy behavior.
    model_envelope: bool = False

    # Correlation prescription for the modelling covariance used by the
    # bottom-line test:
    #   "global_shown" -> one coherent PS and one HAD nuisance across all
    #                     published pT/rho bins; source selection also uses
    #                     only that shown phase space (production default).
    #   "global_all"   -> same covariance, but choose the representative
    #                     sources using every internal bin (diagnostic only).
    #   "per_pt"       -> independent coherent PS/HAD nuisances in each
    #                     published pT slice (cross-pT correlation diagnostic).
    model_covariance_scope: str = "global_shown"

    # Bottom-line test: the rho value each pT slice is cut at for display, so
    # the chi2 is evaluated only over the SHOWN space (not the hidden buffer
    # bins). Per-pT list aligned with the pT bins; None -> fall back to the
    # uniform xlim_lower for the grooming mode.
    bl_shown_floors_groomed: list | None = None
    bl_shown_floors_ungroomed: list | None = None
    # Normalize each pT slice over its SHOWN bins only (per-pT floors from
    # _bl_shown_floors), so the published spectra integrate to 1 over the
    # reported fiducial bins instead of including hidden buffer/catch-all
    # bins in the denominator. Off by default: mass and the legacy rho tags
    # keep the full-slice normalization.
    normalize_over_shown: bool = False

    # Merged non-DY background pkl (no dir), e.g. "bkg_all.pkl": ttbar, single
    # top and diboson simulated with the same processor/binning as the data.
    # Its reco spectrum is subtracted bin-by-bin from the data input BEFORE the
    # DY fake correction, and its MC-stat variance is added to the input
    # variance (see _apply_background_subtraction). None -> no subtraction.
    # When set but the file is absent the loader warns and leaves the input
    # unsubtracted, so the run still completes with the legacy behavior.
    bkg_file: str | None = None


MASS_SPEC = ObservableSpec(
    name="mass",
    reco_axis="mreco",
    gen_axis="mgen",
    input_dir="./inputs/zjet/mass/",
    mc_file="mass_pythia_all.pkl",
    data_file="mass_data_all.pkl",
    herwig_file="mass_herwig_all.pkl",
    jk_data_file="mass_jk_data_all.pkl",
    era_mc_files=[
        "mass_pythia_2016.pkl",
        "mass_pythia_2016APV.pkl",
        "mass_pythia_2017.pkl",
        "mass_pythia_2018.pkl",
    ],
    era_jk_files=[
        "mass_jk_pythia_2016.pkl",
        "mass_jk_pythia_2016APV.pkl",
        "mass_jk_pythia_2017.pkl",
        "mass_jk_pythia_2018.pkl",
    ],
    era_prefix="mass_pythia_",
    reweighted_fallback_files=[
        "mass_pythia_reweighted_all.pkl",
        "pythia_reweighted_all.pkl",
        "mass_herwig_all.pkl",
        "herwig_all.pkl",
    ],
    output_dir="outputs/zjet/mass/",
    edges_reco_attr="mass_edges_reco",
    edges_gen_attr="mass_edges_gen",
    reco_edges_by_pt_attr="reco_mass_edges_by_pt",
    gen_edges_by_pt_attr="gen_mass_edges_by_pt",
    hist_keys_groomed={
        "response": "response_matrix_g",
        "reco": "ptjet_mjet_g_reco",
        "gen": "ptjet_mjet_g_gen",
    },
    hist_keys_ungroomed={
        "response": "response_matrix_u",
        "reco": "ptjet_mjet_u_reco",
        "gen": "ptjet_mjet_u_gen",
    },
    x_label_groomed="Groomed Jet Mass (GeV)",
    x_label_ungroomed="Ungroomed Jet Mass (GeV)",
    short_label_groomed="Jet Mass (GeV), Groomed",
    short_label_ungroomed="Jet Mass (GeV), Ungroomed",
    xlim_lower_groomed=10.0,
    xlim_lower_ungroomed=20.0,
    normalized_ylabel=r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{dm\,dp_T} (GeV^{-1})$",
)


RHO_FIXED_JEC_SPEC = ObservableSpec(
    name="rho",
    reco_axis="mpt_reco",
    gen_axis="mpt_gen",
    input_dir="./inputs/zjet/rho/fixed_jec/",
    mc_file="pythia_all.pkl",
    data_file="data_all.pkl",
    herwig_file="herwig_all.pkl",
    jk_data_file="jk_data_all.pkl",
    era_mc_files=[
        "pythia_2016.pkl",
        "pythia_2016APV.pkl",
        "pythia_2017.pkl",
        "pythia_2018.pkl",
    ],
    era_jk_files=[
        "jk_pythia_2016.pkl",
        "jk_pythia_2016APV.pkl",
        "jk_pythia_2017.pkl",
        "jk_pythia_2018.pkl",
    ],
    era_prefix="pythia_",
    reweighted_fallback_files=[
        "pythia_reweighted_all.pkl",
        "herwig_all.pkl",
    ],
    output_dir="outputs/zjet/rho/fixed_jec/",
    edges_reco_attr="rho_edges",
    edges_gen_attr="rho_edges_gen",
    reco_edges_by_pt_attr="reco_rho_edges_by_pt",
    gen_edges_by_pt_attr="gen_rho_edges_by_pt",
    hist_keys_groomed={
        "response": "response_matrix_rho_g",
        "reco": "ptjet_rhojet_g_reco",
        "gen": "ptjet_rhojet_g_gen",
    },
    hist_keys_ungroomed={
        "response": "response_matrix_rho_u",
        "reco": "ptjet_rhojet_u_reco",
        "gen": "ptjet_rhojet_u_gen",
    },
    # ARC round-2: lowercase groomed/ungroomed everywhere
    x_label_groomed=r"$\log_{10}(\rho^2)$, groomed",
    x_label_ungroomed=r"$\log_{10}(\rho^2)$, ungroomed",
    short_label_groomed=r"$\log_{10}(\rho^2)$, Groomed",
    short_label_ungroomed=r"$\log_{10}(\rho^2)$, Ungroomed",
    xlim_lower_groomed=-4.5,
    xlim_lower_ungroomed=-2.5,
    normalized_ylabel=r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d\log_{10}(\rho^2)\,dp_T}$",
)

RHO_ORIGINAL_SPEC = replace(
    RHO_FIXED_JEC_SPEC,
    input_dir="./inputs/zjet/rho/original/",
    output_dir="outputs/zjet/rho/original/",
)

# Same inputs as the fixed-JEC set, but produced with the corrected misses /
# gen-level theory-weight handling (ISR/FSR/q2/PDF now vary the gen spectrum).
RHO_FIXED_MISS_SPEC = replace(
    RHO_FIXED_JEC_SPEC,
    input_dir="./inputs/zjet/rho/fixed_miss/",
    output_dir="outputs/zjet/rho/fixed_miss/",
)

# ARC round-2 production set (2026-07-10): pT sink 185-200 + the groomed-rho
# buffer binning (shown region 0.5-wide, hidden 0.25 buffer over -5..-3.5,
# display cropped at -3.5 via xlim). Inputs are the arc_r2 reskims (per-era
# pythia with full systematics + all-era data + all-era herwig). HERWIG is used
# ONLY for the comparison overlay and the closure/bias diagnostics -- it is NOT
# a modelling-uncertainty leg (that is the two-leg PS/HAD envelope) and, with
# model_envelope=True, the herwig systematic is excluded from the total. The
# herwig-*systematic* still reads reweighted_fallback_files (kept at the pythia
# file -> zero shift), so HERWIG cannot leak into the uncertainty band.
# Jackknife pkls are pending; until they are dropped into input_dir the loader
# falls back to the TUnfold-propagated covariances for the stat error (input
# term from GetEmatrixInput, response-matrix term from GetEmatrixSysUncorr).
RHO_ARC_R2_SPEC = replace(
    RHO_FIXED_JEC_SPEC,
    input_dir="./inputs/zjet/rho/arc_r2/",
    output_dir="outputs/zjet/rho/arc_r2/",
    herwig_file="herwig_all.pkl",
    # ARC round-2 ("describe the handling of the background"): the non-DY
    # background is subtracted from the input rather than neglected. The
    # confirmed current-binning background is staged with the JMS/JMR-unity
    # inputs, so ARC must resolve that exact pkl rather than an obsolete copy.
    bkg_file="../jmsjmr_unity/bkg_all.pkl",
    reweighted_fallback_files=["pythia_all.pkl"],
    edges_reco_attr="rho_edges_arcr2",
    edges_gen_attr="rho_edges_gen_arcr2",
    reco_edges_by_pt_attr="reco_rho_edges_by_pt_arcr2",
    gen_edges_by_pt_attr="gen_rho_edges_by_pt_arcr2",
    xlim_lower_groomed=-3.5,
    # Publication-space groomed rho floors corresponding to the documented
    # 5 GeV mass floor in [185,200,290,400,inf] pT bins. The 185-200 entry is
    # retained for alignment but that migration bin is not reported.
    bl_shown_floors_groomed=[-2.5, -3.0, -3.0, -3.5],
    # Publication choice (approval prep): report AND normalize over the shown
    # per-pT window only; the ungroomed floors fall back to the uniform -2.5
    # xlim, which excludes the [-10, -2.5] catch-all bin from the denominator.
    normalize_over_shown=True,
    # WS2 two-leg modelling uncertainty (PS = Vincia+FSR envelope, HAD =
    # CR/frag envelope, combined in quadrature in _compute_total_systematic).
    # The weighters were derived with the old [0,200] low-pT bin; that only
    # approximates the (non-reported) 185-200 sink bin.
    model_envelope=True,
)

# JMS/JMR-unity comparison (2026-07-22): identical to arc_r2 except the input
# pkls come from the reskim with corrections.py jmssf/jmrsf ported to the UL
# SoftDropJMSJMRULRun2 unity tables (JMS 1.00 +- 1%, JMR 1.00 +- 2% groomed,
# deviations doubled for ungroomed). arc_r2 ran the old per-year values (JMS
# 0.982/0.999 and JMR 1.09/1.108 centrals in 2017/2018, 2016 JMR +- 20%) --
# see ai-wiki bugs/zjet_stale_jmsjmr_tables_arc_r2.md. Compare against arc_r2
# to quantify the nominal-response shift and the JMS/JMR band change.
RHO_JMSJMR_UNITY_SPEC = replace(
    RHO_ARC_R2_SPEC,
    input_dir="./inputs/zjet/rho/jmsjmr_unity/",
    output_dir="outputs/zjet/rho/jmsjmr_unity/",
)

# Snapshot of the ARC round-2 result before the approved removal of the
# groomed 400--infinity GeV [-3.5, -3.0] bin.  Its existing outputs are kept
# under this tag; do not rerun it when comparing to the post-cut result.
RHO_ARC_R2_PRE_GROOMED400_FLOOR_SPEC = replace(
    RHO_ARC_R2_SPEC,
    output_dir="outputs/zjet/rho/arc_r2_pre_groomed400_floor/",
)

# Approved ARC phase-space update: exclude the response-limited first shown
# groomed-rho bin in the 400--infinity GeV pT slice.  The -3.0 floor applies
# to both the per-slice display and the reported-space normalization, while
# the hidden migration buffer remains available internally to the unfold.
RHO_ARC_R2_GROOMED400_FLOOR3_SPEC = replace(
    RHO_ARC_R2_SPEC,
    output_dir="outputs/zjet/rho/arc_r2_groomed400_floor3/",
    xlim_lower_groomed=-3.0,
    bl_shown_floors_groomed=[-2.5, -3.0, -3.0, -3.0],
)

# Approved publication phase space applied to the corrected JMS/JMR reskim.
# This is the candidate replacement for the old PAS plots: it keeps the ARC
# round-2 settings, uses the UL unity JMS/JMR inputs, and omits the rejected
# groomed 400--infinity GeV [-3.5, -3.0] bin from the shown/normalized space.
RHO_JMSJMR_UNITY_GROOMED400_FLOOR3_SPEC = replace(
    RHO_JMSJMR_UNITY_SPEC,
    output_dir="outputs/zjet/rho/jmsjmr_unity_groomed400_floor3/",
    xlim_lower_groomed=-3.0,
    bl_shown_floors_groomed=[-2.5, -3.0, -3.0, -3.0],
)

RHO_SPECS = {
    "original": RHO_ORIGINAL_SPEC,
    "fixed_jec": RHO_FIXED_JEC_SPEC,
    "fixed_miss": RHO_FIXED_MISS_SPEC,
    "arc_r2": RHO_ARC_R2_SPEC,
    "jmsjmr_unity": RHO_JMSJMR_UNITY_SPEC,
    "arc_r2_pre_groomed400_floor": RHO_ARC_R2_PRE_GROOMED400_FLOOR_SPEC,
    "arc_r2_groomed400_floor3": RHO_ARC_R2_GROOMED400_FLOOR3_SPEC,
    "jmsjmr_unity_groomed400_floor3": RHO_JMSJMR_UNITY_GROOMED400_FLOOR3_SPEC,
}

# Default rho spec: the "original" (pre-JEC-fix) input set. Select the
# fixed-JEC set explicitly via RHO_SPECS["fixed_jec"] when needed.
RHO_SPEC = RHO_ORIGINAL_SPEC


# ---------------------------------------------------------------------------
# Channel / observable registry
#
# Makes (channel, observable, tag) a first-class lookup. The zjet specs are
# full ObservableSpec instances usable directly with Unfolder. The dijet and
# trijet rho channels are produced through unfold.tools.rho_channel_inputs
# (prepared inputs) rather than an ObservableSpec; they are registered here as
# available so callers can introspect the full matrix.
# ---------------------------------------------------------------------------

# zjet ObservableSpec instances, keyed by (observable, tag).
ZJET_SPECS = {
    ("rho", "original"): RHO_ORIGINAL_SPEC,
    ("rho", "fixed_jec"): RHO_FIXED_JEC_SPEC,
    ("rho", "fixed_miss"): RHO_FIXED_MISS_SPEC,
    ("rho", "arc_r2"): RHO_ARC_R2_SPEC,
    ("rho", "jmsjmr_unity"): RHO_JMSJMR_UNITY_SPEC,
    ("rho", "arc_r2_pre_groomed400_floor"): RHO_ARC_R2_PRE_GROOMED400_FLOOR_SPEC,
    ("rho", "arc_r2_groomed400_floor3"): RHO_ARC_R2_GROOMED400_FLOOR3_SPEC,
    ("rho", "jmsjmr_unity_groomed400_floor3"): RHO_JMSJMR_UNITY_GROOMED400_FLOOR3_SPEC,
    ("mass", "nominal"): MASS_SPEC,
}


def _with_jacobian_stat(spec):
    """Variant of ``spec`` with Jacobian-propagated normalized statistics.

    Same inputs; the output dir gets a ``_jacobian`` sibling suffix so legacy
    and Jacobian results pair up by relative path in the comparison app.
    """
    return replace(
        spec,
        output_dir=spec.output_dir.rstrip("/") + "_jacobian/",
        stat_propagation="jacobian",
    )


# Every (observable, tag) gains a "<tag>_jacobian" twin, e.g.
# ("rho", "original_jacobian") writing to outputs/zjet/rho/original_jacobian/,
# and a "<tag>_jacobian_reg" twin that additionally turns on the
# ratio-curvature regularization (tau from an L-curve scan).
ZJET_SPECS.update(
    {
        (observable, f"{tag}_jacobian"): _with_jacobian_stat(spec)
        for (observable, tag), spec in list(ZJET_SPECS.items())
    }
)
ZJET_SPECS.update(
    {
        (observable, f"{tag}_reg"): replace(
            spec,
            output_dir=spec.output_dir.rstrip("/") + "_reg/",
            regularization="ratio_curvature",
        )
        for (observable, tag), spec in list(ZJET_SPECS.items())
        if tag.endswith("_jacobian")
    }
)
RHO_SPECS.update(
    {
        f"{tag}_jacobian{suffix}": ZJET_SPECS[("rho", f"{tag}_jacobian{suffix}")]
        for tag in ("original", "fixed_jec", "fixed_miss")
        for suffix in ("", "_reg")
    }
)

# Merged-low-rho-tail tests on the "original" inputs: collapse every per-pT gen
# bin below log10(rho^2) = -2.5 into a single low-rho bin (the bins below -2.5
# all sit at purity/stability ~0.3, well under the 0.5 rule of thumb across
# every pT slice). The groomed binning loses its -4.5..-2.5 fine bins; the
# ungroomed binning is already merged there so it is unchanged. Two variants:
#   original_merge25         -> with the area constraint (production default)
#   original_merge25_noarea  -> without the area constraint (plain TUnfold)
RHO_MERGE25_SPEC = replace(
    RHO_ORIGINAL_SPEC,
    output_dir="outputs/zjet/rho/original_merge25/",
    gen_merge_below=-2.5,
)
RHO_MERGE25_NOAREA_SPEC = replace(
    RHO_MERGE25_SPEC,
    output_dir="outputs/zjet/rho/original_merge25_noarea/",
    area_constraint=False,
)
ZJET_SPECS[("rho", "original_merge25")] = RHO_MERGE25_SPEC
ZJET_SPECS[("rho", "original_merge25_noarea")] = RHO_MERGE25_NOAREA_SPEC
RHO_SPECS["original_merge25"] = RHO_MERGE25_SPEC
RHO_SPECS["original_merge25_noarea"] = RHO_MERGE25_NOAREA_SPEC

# Identical settings/inputs to "original"; separate output dir so the corrected
# jackknife stat-uncertainty scale (sqrt(g-1)=3 in _compute_stat_unc, replacing
# the legacy sqrt(10/9) that under-covered input/matrix stat by ~2.85x) can be
# compared side-by-side against the original tag.
RHO_JKFIXED_SPEC = replace(
    RHO_ORIGINAL_SPEC,
    output_dir="outputs/zjet/rho/original_jkfixed/",
)
ZJET_SPECS[("rho", "original_jkfixed")] = RHO_JKFIXED_SPEC
RHO_SPECS["original_jkfixed"] = RHO_JKFIXED_SPEC

# Full-production candidate (2026-07): inflated ungroomed JMS/JMR inputs
# (ratio-injected onto the production nominal, see inputs/zjet/rho/
# jmsjmr_inflated/) + the offline column-scaled model envelope
# (Vincia/CR/frag re-unfold shifts maxed with FSR) replacing the legacy
# herwig-diff model term and the fsr/isr quadrature entries. tau=0
# ("coarsen, not regularize") inherited from the original spec.
RHO_MODEL_SPEC = replace(
    RHO_ORIGINAL_SPEC,
    input_dir="./inputs/zjet/rho/jmsjmr_inflated/",
    output_dir="outputs/zjet/rho/original_model/",
    model_envelope=True,
)
ZJET_SPECS[("rho", "original_model")] = RHO_MODEL_SPEC
RHO_SPECS["original_model"] = RHO_MODEL_SPEC

# RooUnfold iterative-Bayes (D'Agostini) on the "original" inputs with n_iter=1
# -- the minimal single D'Agostini iteration (the least-regularized real Bayes
# step; RooUnfoldBayes with n_iter=0 never runs the loop and returns zeros).
# A study baseline for the iterative-Bayes convergence scan. Registered after
# the jacobian/_reg twin block so it gets no jacobian twins (the Bayes path uses
# jackknife statistics).
RHO_ROOUNFOLD_1ITER_SPEC = replace(
    RHO_ORIGINAL_SPEC,
    output_dir="outputs/zjet/rho/RooUnfold_1iter/",
    method="roounfold_bayes",
    n_iter=1,
)
ZJET_SPECS[("rho", "RooUnfold_1iter")] = RHO_ROOUNFOLD_1ITER_SPEC
RHO_SPECS["RooUnfold_1iter"] = RHO_ROOUNFOLD_1ITER_SPEC

# Default tag for each (channel, observable).
DEFAULT_TAGS = {
    ("zjet", "rho"): "original",
    ("zjet", "mass"): "nominal",
    ("dijet", "rho"): "2018",
    ("trijet", "rho"): "2018",
}

# How each (channel, observable) is produced:
#   "spec"           -> get_spec(...) returns an ObservableSpec for Unfolder
#   "channel_inputs" -> use unfold.tools.rho_channel_inputs +
#                       scripts/run_rho_unfolding.py (prepared inputs)
#   None             -> not currently available
CHANNEL_OBSERVABLES = {
    ("zjet", "rho"): "spec",
    ("zjet", "mass"): "spec",          # code ready; inputs/zjet/mass/ must be regenerated
    ("dijet", "rho"): "channel_inputs",
    ("trijet", "rho"): "channel_inputs",
    ("dijet", "mass"): None,
    ("trijet", "mass"): None,
}


def get_spec(channel="zjet", observable="rho", tag=None):
    """Return the ObservableSpec for a (channel, observable[, tag]).

    Only zjet currently exposes ObservableSpec instances. dijet/trijet rho are
    produced via unfold.tools.rho_channel_inputs (see CHANNEL_OBSERVABLES), so
    requesting a spec for them raises KeyError with a pointer to the right path.
    """
    if channel != "zjet":
        raise KeyError(
            f"No ObservableSpec for channel {channel!r}; {channel} rho is "
            "produced via unfold.tools.rho_channel_inputs "
            "(scripts/run_rho_unfolding.py). See CHANNEL_OBSERVABLES."
        )
    if tag is None:
        tag = DEFAULT_TAGS.get((channel, observable))
    try:
        return ZJET_SPECS[(observable, tag)]
    except KeyError:
        available = ", ".join(f"{obs}/{t}" for (obs, t) in ZJET_SPECS)
        raise KeyError(
            f"No spec for zjet {observable!r} tag {tag!r}; available: {available}"
        ) from None

JES_SYSTEMATICS = [
    "JES_AbsoluteMPFBiasUp", "JES_AbsoluteMPFBiasDown", "JES_AbsoluteScaleUp", "JES_AbsoluteScaleDown",
    "JES_AbsoluteStatUp", "JES_AbsoluteStatDown", "JES_FlavorQCDUp", "JES_FlavorQCDDown", "JES_FragmentationUp",
    "JES_FragmentationDown", "JES_PileUpDataMCUp", "JES_PileUpDataMCDown", "JES_PileUpPtBBUp", "JES_PileUpPtBBDown",
    "JES_PileUpPtEC1Up", "JES_PileUpPtEC1Down", "JES_PileUpPtEC2Up", "JES_PileUpPtEC2Down", "JES_PileUpPtHFUp",
    "JES_PileUpPtHFDown", "JES_PileUpPtRefUp", "JES_PileUpPtRefDown", "JES_RelativeFSRUp", "JES_RelativeFSRDown",
    "JES_RelativeJEREC1Up", "JES_RelativeJEREC1Down", "JES_RelativeJEREC2Up", "JES_RelativeJEREC2Down",
    "JES_RelativeJERHFUp", "JES_RelativeJERHFDown", "JES_RelativePtBBUp", "JES_RelativePtBBDown",
    "JES_RelativePtEC1Up", "JES_RelativePtEC1Down", "JES_RelativePtEC2Up", "JES_RelativePtEC2Down",
    "JES_RelativePtHFUp", "JES_RelativePtHFDown", "JES_RelativeBalUp", "JES_RelativeBalDown",
    "JES_RelativeSampleUp", "JES_RelativeSampleDown", "JES_RelativeStatECUp", "JES_RelativeStatECDown",
    "JES_RelativeStatFSRUp", "JES_RelativeStatFSRDown", "JES_RelativeStatHFUp", "JES_RelativeStatHFDown",
    "JES_SinglePionECALUp", "JES_SinglePionECALDown", "JES_SinglePionHCALUp", "JES_SinglePionHCALDown",
    "JES_TimePtEtaUp", "JES_TimePtEtaDown",
]

NON_JES_SYSTEMATICS = [
    "nominal", "puUp", "puDown", "elerecoUp", "elerecoDown", "eleidUp", "eleidDown", "eletrigUp", "eletrigDown",
    "murecoUp", "murecoDown", "muidUp", "muidDown", "mutrigUp", "muisoUp", "muisoDown", "mutrigDown", "pdfUp",
    "pdfDown", "q2Up", "q2Down", "l1prefiringUp", "l1prefiringDown", "ISRUp", "ISRDown", "FSRUp", "FSRDown",
    "JERUp", "JERDown", "JMSUp", "JMSDown", "herwigUp", "herwigDown",
]


class Unfolder:
    # Flavors every saved plot is replicated into (see _save_label_flavors);
    # override on an instance to restrict, e.g. u.cms_label_flavors = ()
    cms_label_flavors = DEFAULT_CMS_LABEL_FLAVORS

    def __init__(
        self,
        spec,
        groomed,
        closure=False,
        herwig_closure=False,
        do_syst=False,
        cms_label="Internal",
        compute_jackknife_stat=True,
    ):
        self.spec = spec
        self.reco_axis = spec.reco_axis
        self.gen_axis = spec.gen_axis
        self.groomed = groomed
        self.cms_label = cms_label
        self.lumi = 138.0
        self.com = 13.0
        self.stat_propagation = getattr(spec, "stat_propagation", "legacy")
        self.regularization = getattr(spec, "regularization", "none")
        # None -> L-curve scan on the nominal data unfold sets it
        self.tau = getattr(spec, "tau", None)
        # Unfolding backend: "tunfold" (default) or "roounfold_bayes".
        self.method = getattr(spec, "method", "tunfold")
        self.n_iter = getattr(spec, "n_iter", 4)
        self.has_jackknife = bool(compute_jackknife_stat)
        self.has_herwig = True
        self.has_validation_inputs = True
        self.response_matrix_stat_available = True
        self.first_reported_pt_bin = 0
        self.closure = closure
        self.herwig_closure = herwig_closure
        self.y_unf_dict = {}
        self.ye_unf_dict = {}
        self._ensure_output_dirs()
        self._setup_binning()
        self._make_inputs_numpy()
        self._configure_systematics(do_syst)
        self._load_data(
            filename_mc=spec.input_dir + spec.mc_file,
            filename_data=spec.input_dir + spec.data_file,
            filename_herwig=spec.input_dir + spec.herwig_file,
        )
        self._perform_unfold(closure=self.closure, herwig_closure=self.herwig_closure)
        for syst in self.systematics:
            self._perform_unfold(systematic=syst, closure=self.closure, herwig_closure=self.herwig_closure)
        if compute_jackknife_stat and self.has_jackknife:
            self._compute_stat_unc()
        else:
            # A fast diagnostic run still has the TUnfold-propagated input-data
            # covariance needed by the bottom-line test, but intentionally
            # skips the 20 data/matrix jackknife re-unfolds.
            self._compute_input_stat_unc_from_covariance()
        self._normalize_result()
        self._compute_total_systematic()

    @classmethod
    def from_prepared_inputs(
        cls,
        spec,
        groomed,
        *,
        mc_inputs,
        data_inputs,
        analysis_binning,
        systematics,
        herwig_inputs=None,
        cms_label="Internal",
        lumi=59.7,
        com=13.0,
    ):
        """Run the core unfolding from already adapted in-memory histograms.

        This entry point is intended for producer outputs that do not use the
        legacy merged-era pickle layout. It deliberately excludes jackknife
        and alternate-generator inputs unless a future caller provides a
        separately validated implementation for those uncertainty sources.
        """

        self = cls.__new__(cls)
        self.spec = spec
        self.reco_axis = spec.reco_axis
        self.gen_axis = spec.gen_axis
        self.groomed = groomed
        self.cms_label = cms_label
        self.lumi = float(lumi)
        self.com = float(com)
        self.stat_propagation = getattr(spec, "stat_propagation", "legacy")
        self.regularization = getattr(spec, "regularization", "none")
        self.tau = getattr(spec, "tau", None)
        # Unfolding backend. dijet/trijet have no jackknife inputs, so under
        # method="roounfold_bayes" the statistical uncertainty falls back to
        # RooUnfold's propagated covariance (via _compute_input_stat_unc_from_covariance),
        # mirroring how the TUnfold path uses the input covariance here.
        self.method = getattr(spec, "method", "tunfold")
        self.n_iter = getattr(spec, "n_iter", 4)
        self.closure = False
        self.herwig_closure = False
        self.has_jackknife = False
        self.has_herwig = herwig_inputs is not None
        self.has_validation_inputs = False
        self.response_matrix_stat_available = False
        self.first_reported_pt_bin = 1
        self.stat_uncertainty_method = "TUnfold input covariance"
        self.y_unf_dict = {}
        # _store_unfold_result writes both dicts; the model envelope reaches this
        # path via _compute_total_systematic, so ye_unf_dict must exist here too
        # (the main __init__ already sets it).
        self.ye_unf_dict = {}
        self.y_unf_jk_input_list = []
        self.y_unf_jk_matrix_list = []
        self._ensure_output_dirs()
        self._setup_prepared_binning(analysis_binning)
        self._load_prepared_inputs(mc_inputs, data_inputs, systematics, herwig_inputs)

        self._perform_unfold(systematic="nominal")
        for systematic in self.systematics:
            if systematic != "nominal":
                self._perform_unfold(systematic=systematic)

        self._compute_input_stat_unc_from_covariance()
        self._normalize_result()
        self._compute_total_systematic()
        return self

    def _ensure_output_dirs(self):
        output_dir = Path(self.spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Category subfolders (summary/, inputs/, response/, bottom_line/,
        # unfolded/, validation/, uncertainties/, data/) are created on demand by
        # _categorize_output; only the gallery preview mirror is pre-made here.
        (output_dir / "_previews").mkdir(parents=True, exist_ok=True)

    def _setup_binning(self):
        self.bins = binning.bin_edges(self.groomed)
        self.edges = getattr(self.bins, self.spec.edges_reco_attr)
        self.edges_gen = getattr(self.bins, self.spec.edges_gen_attr)
        self.reco_edges_by_pt = getattr(self.bins, self.spec.reco_edges_by_pt_attr)
        self.gen_edges_by_pt = getattr(self.bins, self.spec.gen_edges_by_pt_attr)
        merge_below = getattr(self.spec, "gen_merge_below", None)
        if merge_below is not None:
            self.gen_edges_by_pt = [
                _merge_edges_below(edges, merge_below)
                for edges in self.gen_edges_by_pt
            ]
        self.pt_edges = self.bins.pt_edges

    def _setup_prepared_binning(self, analysis_binning):
        self.bins = analysis_binning

        # Honour the spec's edge-attribute names, exactly as _setup_binning
        # does. arc_r2 and later select the *_arcr2 axes, which for GROOMED rho
        # differ from the defaults (buffer binning); ungroomed aliases the two,
        # which is why hardcoding the default names only ever broke groomed.
        # Prepared-input binnings that define no such attribute (dijet/trijet
        # via rho_channel_inputs) fall back to the plain names.
        def pick(attr_name, default_name):
            name = getattr(self.spec, attr_name, None) or default_name
            return getattr(analysis_binning, name, None) or getattr(
                analysis_binning, default_name
            )

        self.edges = list(pick("edges_reco_attr", "rho_edges"))
        self.edges_gen = list(pick("edges_gen_attr", "rho_edges_gen"))
        self.reco_edges_by_pt = [
            list(edges) for edges in pick("reco_edges_by_pt_attr", "reco_rho_edges_by_pt")
        ]
        self.gen_edges_by_pt = [
            list(edges) for edges in pick("gen_edges_by_pt_attr", "gen_rho_edges_by_pt")
        ]
        self.pt_edges = list(analysis_binning.pt_edges)

    def _configure_systematics(self, do_syst):
        available_systematics = list(self.sys_matrix_dic.keys())
        self.systematics = available_systematics if do_syst else ["nominal"]

    @property
    def _band_is_stat_only(self):
        """True when nothing but the statistical error feeds the total band.

        With no systematic variations unfolded and no model envelope,
        ``syst_unc`` is stat_unc by construction (_compute_total_systematic
        seeds the quadrature sum with it), so the "total" band would paint the
        same box as the stat band under a legend entry that promises
        systematics -- and both would repeat what the markers' error bars
        already show. The self-closure figures run stats-only: they draw NO
        filled band, and the stat uncertainty appears once, on the points.
        """
        return (set(getattr(self, "systematics", ["nominal"])) <= {"nominal"}
                and not getattr(self.spec, "model_envelope", False))

    def _load_pickle(self, filename):
        with open(filename, "rb") as handle:
            return pkl.load(handle)

    def _resolve_input_path(self, filename, *fallbacks):
        candidates = [filename, *fallbacks]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if path.exists():
                return str(path)
        candidate_list = ", ".join(str(Path(candidate)) for candidate in candidates if candidate is not None)
        raise FileNotFoundError(f"Could not find any of: {candidate_list}")

    def _categorize_output(self, stem):
        """Map a plot basename to (category_subfolder, descriptive_stem).

        Single source of truth for the per-mode output layout: every plot lands
        in a semantic subfolder (nothing dumped in the mode root) and the bare
        unfolded-result names (e.g. 'groomed_1') gain a descriptive prefix. The
        trailing '_<mode>_<idx>' panel index is normalized to '_<mode>_pt<idx>'
        ('ptall' for the inclusive -1 panel) so filenames are searchable.
        """
        def _ptnorm(name):
            return re.sub(
                r"_(groomed|ungroomed)_(-?\d+)$",
                lambda m: f"_{m.group(1)}_"
                + ("ptall" if int(m.group(2)) < 0 else f"pt{m.group(2)}"),
                name,
            )

        s = stem
        # bare unfolded result, e.g. 'groomed_1' / 'ungroomed_-1'
        if re.fullmatch(r"(?:un)?groomed_-?\d+", s):
            return "unfolded", _ptnorm("unfolded_" + s)
        # per-mode result summaries, e.g. 'groomed_summary', 'groomed_summary_linear'
        m = re.fullmatch(r"(groomed|ungroomed)_summary(_linear|_ratio)?", s)
        if m:
            return "summary", f"unfolded_summary{m.group(2) or ''}_{m.group(1)}"
        # prefix -> (category, rename); first match wins, order matters
        rules = [
            ("bottom_line_chi2",         "summary",       None),
            ("bottom_line",              "bottom_line",   None),
            ("purity_stability",         "response",      None),
            ("response_",                "response",      lambda n: n.replace("response_", "response_matrix_", 1)),
            ("fakerates",                "response",      lambda n: n.replace("fakerates", "fakes_misses", 1)),
            ("input_",                   "inputs",        lambda n: n.replace("input_", "input_data_mc_", 1)),
            ("herwig_pythia_comparison", "unfolded",      None),
            ("herwig_closure_unc",       "validation",    None),
            ("herwig_bias_test",         "validation",    None),
            ("closure_",                 "validation",    None),
            ("folded_",                  "validation",    None),
            ("jackknife_convergence",    "validation",    None),
            ("jk_inputs",                "validation",    None),
            ("jk_outputs",               "validation",    None),
            ("L_matrix",                 "validation",    None),
            ("lcurve",                   "validation",    None),
            ("correlation",              "unfolded",      None),
            ("unfolded_unrolled_2d",     "unfolded",      None),
            ("unfolded_basic",           "unfolded",      None),
            ("unfolded_2d",              "data",          None),
            ("uncertainty_summary_2d",   "data",          None),
            ("normalized_covariance",    "data",          None),
            ("stat_fraction",            "uncertainties", None),
            ("summary_grouped",          "uncertainties", None),
            ("summary_linear",           "uncertainties", None),
            ("summary_",                 "uncertainties", None),
            ("nominal_minus_",           "uncertainties", None),
            ("heatmap",                  "uncertainties", None),
        ]
        for prefix, cat, fn in rules:
            if s.startswith(prefix):
                return cat, _ptnorm(fn(s) if fn else s)
        # remaining '<SYST>_<mode>_<idx>' plots (JES/JMS/ElectronSF/herwig/...)
        if re.search(r"_(groomed|ungroomed)_-?\d+$", s):
            return "uncertainties", _ptnorm(s)
        return "misc", _ptnorm(s)

    def _relocate_output(self, save_path):
        """Rewrite a legacy save path into the categorized layout."""
        p = Path(save_path)
        cat, stem = self._categorize_output(p.stem)
        return Path(self.spec.output_dir) / cat / f"{stem}{p.suffix}"

    def _finalize_plot(self, save_path=None, show=True, fig=None):
        if save_path is not None:
            path = self._relocate_output(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            target_fig = fig if fig is not None else plt.gcf()
            self._save_label_flavors(target_fig, path,
                                     bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        else:
            plt.close(fig if fig is not None else plt.gcf())

    def _save_label_flavors(self, fig, path, **savefig_kw):
        save_cms_label_flavors(fig, path, self.cms_label,
                               flavors=self.cms_label_flavors, **savefig_kw)

    def _cms_extra_label(self):
        return f" {self.cms_label}" if self.cms_label and not self.cms_label.startswith(" ") else self.cms_label

    @staticmethod
    def _as_int_when_whole(value):
        """138.0 -> 138 so the CMS header shows integers, 59.7 stays 59.7."""
        return int(value) if float(value).is_integer() else value

    @staticmethod
    def _subset_fraction_unc(numer, other, numer_var, other_var):
        """MC-stat error of f = numer/(numer+other) for disjoint subsets."""
        total = numer + other
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(
                total > 0,
                (other**2 * numer_var + numer**2 * other_var) / total**4,
                0.0,
            )
        return np.sqrt(np.clip(var, 0.0, None))

    def _lumi_label(self):
        return self._as_int_when_whole(self.lumi)

    def _com_label(self):
        return self._as_int_when_whole(self.com)

    def _reported_pt_indices(self):
        return range(
            getattr(self, "first_reported_pt_bin", 0),
            len(self.pt_edges) - 1,
        )

    def _reported_matrix_view(self, matrix):
        first_pt_bin = getattr(self, "first_reported_pt_bin", 0)
        reco_offset = sum(
            len(edges) - 1 for edges in self.reco_edges_by_pt[:first_pt_bin]
        )
        gen_offset = sum(
            len(edges) - 1 for edges in self.gen_edges_by_pt[:first_pt_bin]
        )
        return (
            np.asarray(matrix)[reco_offset:, gen_offset:],
            self.reco_edges_by_pt[first_pt_bin:],
            self.gen_edges_by_pt[first_pt_bin:],
            self.pt_edges[first_pt_bin:],
        )

    def _has_systematic(self, *prefixes):
        lowered = tuple(prefix.lower() for prefix in prefixes)
        return any(
            systematic.lower().startswith(lowered)
            for systematic in self.systematics
            if systematic != "nominal"
        )

    def _observable_label(self):
        return self.spec.x_label_groomed if self.groomed else self.spec.x_label_ungroomed

    def _observable_short_label(self):
        return self.spec.short_label_groomed if self.groomed else self.spec.short_label_ungroomed

    def _observable_xlim(self, i_pt=None):
        if i_pt is not None:
            edges = self.gen_edges_by_pt[i_pt]
            upper = float(edges[-1] if self.spec.name == "rho" else edges[-2])
            # Per-pT display floor (falls back to the mode's uniform
            # xlim_lower when the spec sets no per-pT floors).
            return (self._bl_shown_floors()[i_pt], upper)
        edge_index = -1 if self.spec.name == "rho" else -2
        upper = max(float(edges[edge_index]) for edges in self.gen_edges_by_pt)
        lower = self.spec.xlim_lower_groomed if self.groomed else self.spec.xlim_lower_ungroomed
        return (lower, upper)

    def _normalized_ylabel(self):
        return self.spec.normalized_ylabel

    def _histogram_keys(self):
        return self.spec.hist_keys_groomed if self.groomed else self.spec.hist_keys_ungroomed

    def _prepare_jackknife_inputs(self, data2d_jk, mass_edges_reco, pt_edges, reco_mass_edges_by_pt):
        mosaic_2d_jk_list = []
        for i in range(10):
            reco_proj_jk = data2d_jk.project("jk", "ptreco", self.reco_axis)[i, ...]
            h2d_jk = reco_proj_jk.values()
            h2d_jk_reordered, _ = reorder_to_expected_2d(h2d_jk, mass_edges_reco, pt_edges)
            mosaic_2d_jk_list.append(
                merge_mass_flat(h2d_jk_reordered, mass_edges_reco, reco_mass_edges_by_pt)
            )
        return mosaic_2d_jk_list

    def _prepare_truth_spectrum(self, gen2d, mass_edges_gen, pt_edges, gen_mass_edges_by_pt):
        if "systematic" in gen2d.axes.name:
            gen_proj = gen2d.project("ptgen", self.gen_axis, "systematic")[:, :, "nominal"]
        else:
            gen_proj = gen2d.project("ptgen", self.gen_axis)
        h2d_gen, _ = reorder_to_expected_2d(gen_proj.values(), mass_edges_gen, pt_edges)
        return merge_mass_flat(h2d_gen, mass_edges_gen, gen_mass_edges_by_pt)

    def _prepare_jackknife_response_matrices(
        self,
        response4d_jk,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        mosaic_jk_list = []
        nominal_jk = response4d_jk[{"systematic": "nominal"}]
        for i in range(10):
            proj_jk = nominal_jk[{"jk": i}].project("ptgen", self.gen_axis, "ptreco", self.reco_axis)
            resp_np_jk = proj_jk.values(flow=False)
            resp_np_jk_reordered, _ = reorder_to_expected(
                resp_np_jk,
                mass_edges_reco,
                pt_edges,
                mass_edges_gen,
            )
            mosaic_jk, _ = mosaic_no_padding(
                resp_np_jk_reordered,
                mass_edges_reco,
                mass_edges_gen,
                reco_mass_edges_by_pt,
                gen_mass_edges_by_pt,
            )
            mosaic_jk_list.append(mosaic_jk)
        return mosaic_jk_list

    def _prepare_nominal_inputs(
        self,
        input_data,
        fakes,
        misses,
        resp_matrix_4d_gen,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        reco_proj = input_data.project("ptreco", self.reco_axis)
        self.h2d, _ = reorder_to_expected_2d(reco_proj.values(), mass_edges_reco, pt_edges)

        # Preserve the data sumw2 through exactly the same reordering and
        # merging used for the measured spectrum.  The legacy Z+jet path used
        # to leave the TUnfold input-bin errors unset, which is harmless for
        # the central value at tau=0 but prevents a defensible data-stat-only
        # covariance / bottom-line chi-square calculation.
        reco_variances = reco_proj.variances()
        if reco_variances is None:
            self.measured_variances = None
        else:
            reordered_variances, _ = reorder_to_expected_2d(
                np.clip(reco_variances, 0.0, None),
                mass_edges_reco,
                pt_edges,
            )
            self.measured_variances = merge_mass_flat(
                reordered_variances,
                mass_edges_reco,
                reco_mass_edges_by_pt,
            )

        reco_proj_fakes = fakes.project("ptreco", self.reco_axis)
        self.h2d_fakes, _ = reorder_to_expected_2d(reco_proj_fakes.values(), mass_edges_reco, pt_edges)

        reco_proj_misses = misses.project("ptgen", self.gen_axis)
        self.h2d_misses, _ = reorder_to_expected_2d(reco_proj_misses.values(), mass_edges_gen, pt_edges)

        nominal_gen = resp_matrix_4d_gen[{"systematic": "nominal"}]
        proj_gen = nominal_gen.project("ptreco", self.reco_axis, "ptgen", self.gen_axis)
        self.M_np_2d_gen, _ = reorder_to_expected(proj_gen.values(flow=False), mass_edges_gen, pt_edges, mass_edges_gen)
        self.mosaic_gen, _ = mosaic_no_padding(
            self.M_np_2d_gen,
            mass_edges_gen,
            mass_edges_gen,
            gen_mass_edges_by_pt,
            gen_mass_edges_by_pt,
        )

        self.fakes_2d = merge_mass_flat(self.h2d_fakes, mass_edges_reco, reco_mass_edges_by_pt)
        self.misses_2d = merge_mass_flat(self.h2d_misses, mass_edges_gen, gen_mass_edges_by_pt)

    def _prepare_herwig_inputs(
        self,
        resp_matrix_4d_herwig,
        resp_matrix_4d_herwig_gen,
        fakes_herwig,
        misses_herwig,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        resp_matrix_4d_syst = resp_matrix_4d_herwig[{"systematic": "nominal"}]
        h2d_herwig = resp_matrix_4d_syst.project("ptreco", self.reco_axis).values(flow=False)
        self.h2d_herwig, _ = reorder_to_expected_2d(h2d_herwig, mass_edges_reco, pt_edges)

        reco_proj_fakes = fakes_herwig.project("ptreco", self.reco_axis)
        self.h2d_fakes_herwig, _ = reorder_to_expected_2d(reco_proj_fakes.values(), mass_edges_reco, pt_edges)

        reco_proj_misses = misses_herwig.project("ptgen", self.gen_axis)
        self.h2d_misses_herwig, _ = reorder_to_expected_2d(reco_proj_misses.values(), mass_edges_gen, pt_edges)

        self.fakes_2d_herwig = merge_mass_flat(self.h2d_fakes_herwig, mass_edges_reco, reco_mass_edges_by_pt)
        self.misses_2d_herwig = merge_mass_flat(self.h2d_misses_herwig, mass_edges_gen, gen_mass_edges_by_pt)

        nominal_herwig_gen = resp_matrix_4d_herwig_gen[{"systematic": "nominal"}]
        proj_herwig_gen = nominal_herwig_gen.project("ptreco", self.reco_axis, "ptgen", self.gen_axis)
        M_np_2d_herwig_gen, _ = reorder_to_expected(
            proj_herwig_gen.values(flow=False), mass_edges_gen, pt_edges, mass_edges_gen
        )
        self.mosaic_gen_herwig, _ = mosaic_no_padding(
            M_np_2d_herwig_gen,
            mass_edges_gen,
            mass_edges_gen,
            gen_mass_edges_by_pt,
            gen_mass_edges_by_pt,
        )

    def _response_matrix_for_systematic(self, syst, resp_matrix_4d_herwig, sys_matrix_dic):
        if syst in {"herwigUp", "herwigDown"}:
            resp_matrix_4d_syst = resp_matrix_4d_herwig[{"systematic": "nominal"}]
            proj = resp_matrix_4d_syst.project("ptreco", self.reco_axis, "ptgen", self.gen_axis)
            return sys_matrix_dic[syst], proj.variances(flow=False)
        return sys_matrix_dic[syst], None

    def _compute_fake_fraction(self, fakes_flat, matched_flat):
        total_reco = matched_flat + fakes_flat
        with np.errstate(divide="ignore", invalid="ignore"):
            fake_fraction = np.divide(
                fakes_flat,
                total_reco,
                out=np.zeros_like(fakes_flat, dtype=float),
                where=total_reco > 0,
            )
        return np.clip(fake_fraction, 0.0, 1.0)

    def _select_nominal_histogram(self, h_obj):
        if "systematic" in h_obj.axes.name:
            return h_obj[{"systematic": "nominal"}]
        return h_obj

    def _flatten_prepared_2d(self, h_obj, mass_edges, edges_by_pt, axes):
        projected = h_obj.project(*axes)
        values, _ = reorder_to_expected_2d(
            projected.values(flow=False),
            mass_edges,
            self.pt_edges,
        )
        flat_values = merge_mass_flat(values, mass_edges, edges_by_pt)

        variances = projected.variances(flow=False)
        flat_variances = None
        if variances is not None:
            reordered_variances, _ = reorder_to_expected_2d(
                variances,
                mass_edges,
                self.pt_edges,
            )
            flat_variances = merge_mass_flat(
                reordered_variances,
                mass_edges,
                edges_by_pt,
            )
        return flat_values, flat_variances

    def _prepared_response_mosaic(self, response_hist, systematic):
        selected = response_hist
        if "systematic" in selected.axes.name:
            selected = selected[{"systematic": systematic}]
        projected = selected.project(
            "ptgen",
            self.gen_axis,
            "ptreco",
            self.reco_axis,
        )
        response_reordered, _ = reorder_to_expected(
            projected.values(flow=False),
            self.edges,
            self.pt_edges,
            self.edges_gen,
        )
        mosaic, _ = mosaic_no_padding(
            response_reordered,
            self.edges,
            self.edges_gen,
            self.reco_edges_by_pt,
            self.gen_edges_by_pt,
        )
        return response_reordered, mosaic

    def _load_prepared_inputs(self, mc_inputs, data_inputs, systematics, herwig_inputs=None):
        """Populate the state used by the existing TUnfold implementation."""

        keys = self._histogram_keys()
        response_hist = mc_inputs[keys["response"]]
        mc_reco_hist = mc_inputs[keys["reco"]]
        mc_gen_hist = mc_inputs[keys["gen"]]
        data_reco_hist = data_inputs[keys["reco"]]

        available_response_systematics = list(response_hist.axes["systematic"])
        requested_systematics = list(systematics)
        missing = [
            name for name in requested_systematics
            if name not in available_response_systematics
        ]
        if missing:
            raise ValueError(
                f"Prepared response is missing requested systematics: {missing}"
            )
        self.systematics = requested_systematics
        self.mosaic_dict = {}
        self.M_np_2d_dict = {}
        for systematic in self.systematics:
            reordered, mosaic = self._prepared_response_mosaic(
                response_hist,
                systematic,
            )
            self.M_np_2d_dict[systematic] = reordered
            self.mosaic_dict[systematic] = mosaic

        self.M_np_2d = self.M_np_2d_dict["nominal"]
        self.mosaic = self.mosaic_dict["nominal"]
        nominal_data = self._select_nominal_histogram(data_reco_hist)
        nominal_mc_reco = self._select_nominal_histogram(mc_reco_hist)
        nominal_mc_gen = self._select_nominal_histogram(mc_gen_hist)

        self.mosaic_2d, self.measured_variances = self._flatten_prepared_2d(
            nominal_data,
            self.edges,
            self.reco_edges_by_pt,
            ("ptreco", self.reco_axis),
        )
        reco_mc_flat, _ = self._flatten_prepared_2d(
            nominal_mc_reco,
            self.edges,
            self.reco_edges_by_pt,
            ("ptreco", self.reco_axis),
        )
        gen_mc_flat, _ = self._flatten_prepared_2d(
            nominal_mc_gen,
            self.edges_gen,
            self.gen_edges_by_pt,
            ("ptgen", self.gen_axis),
        )

        matched_reco = self.mosaic.sum(axis=1)
        matched_gen = self.mosaic.sum(axis=0)
        self.fakes_2d = reco_mc_flat - matched_reco
        self.misses_2d = gen_mc_flat - matched_gen
        self.fake_fraction_2d = self._compute_fake_fraction(
            self.fakes_2d,
            matched_reco,
        )

        self.input_data = data_reco_hist
        self.data_2d = data_reco_hist
        self.pythia_2d = mc_reco_hist
        self.pythia_4d = response_hist

        if herwig_inputs is not None:
            self._setup_prepared_herwig(herwig_inputs)
        else:
            self.y_true_herwig = None

    def _setup_prepared_herwig(self, herwig_inputs):
        """Build HERWIG state for the prepared-inputs path (dijet/trijet).

        Mirrors the nominal MC setup for the HERWIG sample, then adds the
        alternate-generator (model) uncertainty by registering the HERWIG
        response as the herwigUp/herwigDown systematics: the data is re-unfolded
        through it and the difference from nominal enters the total band (the
        same mechanism the zjet path uses). Also stores the HERWIG gen
        prediction and matched reco used by the HERWIG bias (non-closure) plot.
        """
        keys = self._histogram_keys()
        reordered_herwig, mosaic_herwig = self._prepared_response_mosaic(
            herwig_inputs[keys["response"]], "nominal"
        )
        h_reco = self._select_nominal_histogram(herwig_inputs[keys["reco"]])
        h_gen = self._select_nominal_histogram(herwig_inputs[keys["gen"]])
        herwig_reco_flat, _ = self._flatten_prepared_2d(
            h_reco, self.edges, self.reco_edges_by_pt, ("ptreco", self.reco_axis)
        )
        herwig_gen_flat, herwig_gen_var = self._flatten_prepared_2d(
            h_gen, self.edges_gen, self.gen_edges_by_pt, ("ptgen", self.gen_axis)
        )

        matched_reco = mosaic_herwig.sum(axis=1)
        matched_gen = mosaic_herwig.sum(axis=0)
        self.fakes_2d_herwig = herwig_reco_flat - matched_reco
        self.misses_2d_herwig = herwig_gen_flat - matched_gen
        self.fake_fraction_2d_herwig = self._compute_fake_fraction(
            self.fakes_2d_herwig, matched_reco
        )
        # Matched HERWIG reco = measured input of the bias (non-closure) test;
        # full HERWIG gen = truth/prediction (bias test + 2D summary).
        self.mosaic_herwig_2d = matched_reco
        self.y_true_herwig = herwig_gen_flat
        self.herwig_gen_val_flat = herwig_gen_flat
        self.herwig_gen_var_flat = herwig_gen_var
        self.has_herwig = True

        # Register the HERWIG response as the herwigUp/herwigDown systematics so
        # the existing systematics loop + _compute_total_systematic fold the
        # alternate-generator difference in as "Model Uncertainty" (symmetric:
        # both directions share the HERWIG mosaic).
        self.mosaic_dict["herwigUp"] = mosaic_herwig
        self.mosaic_dict["herwigDown"] = mosaic_herwig
        self.M_np_2d_dict["herwigUp"] = reordered_herwig
        self.M_np_2d_dict["herwigDown"] = reordered_herwig
        for name in ("herwigUp", "herwigDown"):
            if name not in self.systematics:
                self.systematics.append(name)

    def _compute_input_stat_unc_from_covariance(self):
        """Use TUnfold's propagated covariances when JK inputs are absent.

        Input-stat term from GetEmatrixInput (data statistics); matrix-stat
        term from GetEmatrixSysUncorr (uncorrelated statistical uncertainties
        of the response matrix). Both are the analytic TUnfold error
        propagation, standing in for the jackknife replicas.
        """

        input_variance = np.clip(np.diag(self.cov_data_np), 0.0, None)
        input_std = np.sqrt(input_variance)
        cov_uncorr = getattr(self, "cov_uncorr_np", None)
        if cov_uncorr is None:
            cov_uncorr = np.zeros_like(self.cov_data_np)
        matrix_variance = np.clip(np.diag(cov_uncorr), 0.0, None)
        matrix_std = np.sqrt(matrix_variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.input_stat_unc_frac = np.abs(
                np.divide(
                    input_std,
                    self.y_unf,
                    out=np.zeros_like(input_std),
                    where=self.y_unf != 0,
                )
            )
            self.matrix_stat_unc_frac = np.abs(
                np.divide(
                    matrix_std,
                    self.y_unf,
                    out=np.zeros_like(matrix_std),
                    where=self.y_unf != 0,
                )
            )
        self.stat_unc_frac = np.sqrt(
            self.input_stat_unc_frac**2 + self.matrix_stat_unc_frac**2
        )
        self.input_stat_unc_pt_binned = unflatten_gen_by_pt(
            self.input_stat_unc_frac,
            self.gen_edges_by_pt,
        )
        self.matrix_stat_unc_pt_binned = unflatten_gen_by_pt(
            self.matrix_stat_unc_frac,
            self.gen_edges_by_pt,
        )
        self.stat_unc_pt_binned = unflatten_gen_by_pt(
            self.stat_unc_frac,
            self.gen_edges_by_pt,
        )

    def _finalize_reco_views(self, mass_edges_reco, reco_mass_edges_by_pt):
        self.M_np_2d = self.M_np_2d_dict["nominal"]
        self.mosaic = self.mosaic_dict["nominal"]
        self.mosaic_2d = merge_mass_flat(self.h2d, mass_edges_reco, reco_mass_edges_by_pt)
        self.fake_fraction_2d = self._compute_fake_fraction(self.fakes_2d, self.mosaic.sum(axis=1))

        if "herwigUp" in self.systematics or "herwigDown" in self.systematics:
            self.mosaic_herwig_2d = merge_mass_flat(self.h2d_herwig, mass_edges_reco, reco_mass_edges_by_pt)
            self.fake_fraction_2d_herwig = self._compute_fake_fraction(
                self.fakes_2d_herwig,
                self.mosaic_herwig_2d,
            )

    def _flatten_gen_total(self, gen_hist, systematic, mass_edges_gen, pt_edges, gen_mass_edges_by_pt):
        """Flatten the full gen spectrum (matched + misses) for one systematic."""
        proj = gen_hist.project("ptgen", self.gen_axis, "systematic")[:, :, systematic]
        reordered, _ = reorder_to_expected_2d(
            proj.project("ptgen", self.gen_axis).values(),
            mass_edges_gen,
            pt_edges,
        )
        return merge_mass_flat(reordered, mass_edges_gen, gen_mass_edges_by_pt)

    def _flatten_gen_var(self, gen_hist, systematic, mass_edges_gen, pt_edges, gen_mass_edges_by_pt):
        """Flatten the gen-spectrum *variances* for one systematic (MC stat).

        Mirrors ``_flatten_gen_total`` but carries ``variances()``. Merging bins
        sums variances, which is the correct combination for independent bins.
        Returns None when the histogram stores no variances.
        """
        proj = gen_hist.project("ptgen", self.gen_axis, "systematic")[:, :, systematic]
        variances = proj.project("ptgen", self.gen_axis).variances()
        if variances is None:
            return None
        reordered, _ = reorder_to_expected_2d(variances, mass_edges_gen, pt_edges)
        return merge_mass_flat(reordered, mass_edges_gen, gen_mass_edges_by_pt)

    def _build_misses_dict(self, pythia_gen2d, mass_edges_gen, pt_edges, gen_mass_edges_by_pt):
        """Per-systematic misses, kept consistent with the varied response matrix.

        The misses (gen events that fail reco) define the unfolding efficiency, so
        they must vary together with the matched response under every systematic.
        Reusing the nominal misses with a varied matched matrix breaks
        efficiency = matched / (matched + misses) and biases the unfolded result.

        We anchor on the nominal misses and add only the *changes*:

            misses[s] = misses_nom
                      + (gen_total[s]  - gen_total_nom)   # theory reshaping of gen
                      - (matched[s]    - matched_nom)     # migration in/out of reco

        For detector/JES systematics the gen spectrum is unchanged (gen_total delta
        is zero) and only the matched term moves, which captures the acceptance
        change. For ISR/FSR/q2/PDF the gen_total term carries the genuine gen-level
        variation -- provided the producer fills the gen histogram with the theory
        weights (older inputs leave those columns equal to nominal, in which case
        this reduces to the matched-only correction).
        """
        self.misses_2d_dict = {}
        if "nominal" not in self.mosaic_dict:
            return

        gen_syst_labels = set(pythia_gen2d.axes["systematic"])
        matched_nom = self.mosaic_dict["nominal"].sum(axis=0)
        gen_total_nom = self._flatten_gen_total(
            pythia_gen2d, "nominal", mass_edges_gen, pt_edges, gen_mass_edges_by_pt
        )

        for syst in self.systematics:
            if syst in {"nominal", "herwigUp", "herwigDown"}:
                continue
            if syst not in self.mosaic_dict:
                continue

            gen_delta = 0.0
            if syst in gen_syst_labels:
                gen_total_syst = self._flatten_gen_total(
                    pythia_gen2d, syst, mass_edges_gen, pt_edges, gen_mass_edges_by_pt
                )
                gen_delta = gen_total_syst - gen_total_nom

            matched_delta = self.mosaic_dict[syst].sum(axis=0) - matched_nom
            self.misses_2d_dict[syst] = np.clip(
                self.misses_2d + gen_delta - matched_delta, 0.0, None
            )

    def _load_background(self, reco_key, mass_edges_reco, pt_edges, reco_mass_edges_by_pt):
        """Prepare the non-DY background reco spectrum for bin-by-bin subtraction.

        The background (ttbar, single top, WW/WZ/ZZ) is a *different process*,
        not DY signal that failed the generator-level cut, so it is removed from
        the measured spectrum before the DY fake fraction is applied -- see
        _apply_background_subtraction for the ordering argument.

        Sets self.bkg_2d / self.bkg_var_2d (flattened like self.mosaic_2d), or
        leaves them None when no background file is configured or present.
        """
        self.bkg_2d = None
        self.bkg_var_2d = None

        bkg_file = getattr(self.spec, "bkg_file", None)
        if not bkg_file:
            return

        bkg_path = Path(self.spec.input_dir + bkg_file)
        if not bkg_path.exists():
            print(
                "WARNING: background pkl not found -> the non-DY background is "
                "NOT subtracted from the unfolding input (legacy behavior). "
                f"Missing: {bkg_path}"
            )
            return

        output_bkg = self._load_pickle(str(bkg_path))
        if reco_key not in output_bkg:
            raise RuntimeError(
                f"Background pkl {bkg_path} has no '{reco_key}' histogram; it must "
                "be produced with the same processor/histogram set as the data."
            )
        bkg_hist = output_bkg[reco_key]

        # The background is subtracted bin-by-bin, so its binning has to be the
        # binning of the data input -- a background skim left over from an older
        # production silently mis-subtracts. Fail loudly instead.
        for axis_name, expected in (("ptreco", pt_edges), (self.reco_axis, mass_edges_reco)):
            found = np.asarray(bkg_hist.axes[axis_name].edges, dtype=float)
            if not np.allclose(found, np.asarray(expected, dtype=float)):
                raise RuntimeError(
                    f"Background pkl {bkg_path} has '{axis_name}' edges {list(found)} "
                    f"but the unfolding input uses {list(np.asarray(expected, dtype=float))}. "
                    "Re-produce the background samples with the current binning."
                )

        # Sum every background dataset (all processes, all eras). Select the
        # nominal systematic explicitly so a multi-systematic background pkl is
        # not summed over its variations.
        if "systematic" in bkg_hist.axes.name and len(bkg_hist.axes["systematic"]) > 1:
            bkg_hist = bkg_hist[{"systematic": "nominal"}]
        bkg_proj = bkg_hist.project("ptreco", self.reco_axis)

        h2d_bkg, _ = reorder_to_expected_2d(bkg_proj.values(), mass_edges_reco, pt_edges)
        self.bkg_2d = merge_mass_flat(h2d_bkg, mass_edges_reco, reco_mass_edges_by_pt)

        bkg_variances = bkg_proj.variances()
        if bkg_variances is not None:
            reordered_var, _ = reorder_to_expected_2d(
                np.clip(bkg_variances, 0.0, None),
                mass_edges_reco,
                pt_edges,
            )
            self.bkg_var_2d = merge_mass_flat(
                reordered_var,
                mass_edges_reco,
                reco_mass_edges_by_pt,
            )

        print(
            f"Loaded non-DY background from {bkg_path.name}: "
            f"{self.bkg_2d.sum():.1f} events will be subtracted from the input."
        )

    def _apply_background_subtraction(self, meas_flat, closure, herwig_closure):
        """Remove the non-DY background from the measured spectrum.

        Applied BEFORE the fake correction: the fake fraction is derived from DY
        simulation as fakes/(matched+fakes) and describes what fraction of the
        *DY* reco yield originates outside the fiducial gen phase space. ttbar,
        single top and diboson events are not DY at all, so they must leave the
        spectrum first; scaling them by (1-f) instead would attribute a DY
        migration property to a foreign process.

        Closure tests unfold simulation through its own response and contain no
        background by construction, so they are returned unchanged.
        """
        if closure or herwig_closure:
            return meas_flat
        if getattr(self, "bkg_2d", None) is None:
            return meas_flat
        subtracted = np.asarray(meas_flat, dtype=float) - np.asarray(self.bkg_2d, dtype=float)
        return np.clip(subtracted, 0.0, None)

    def _add_background_variance(self, variances, closure, herwig_closure):
        """Add the background MC-stat variance to the input variance.

        Data and simulated background are independent, so the variances add.
        No-op when nothing was subtracted, or when the background pkl carries no
        sumw2 (in which case only the data statistical term is propagated).
        """
        if closure or herwig_closure:
            return variances
        if getattr(self, "bkg_var_2d", None) is None:
            return variances
        return np.asarray(variances, dtype=float) + np.asarray(self.bkg_var_2d, dtype=float)

    def _load_data(self, filename_mc=None, filename_data=None, filename_herwig=None, filename_jk_data=None):
        print("------------- Adding inputs to unfolder -----------------")
        if filename_mc is None:
            filename_mc = self.spec.input_dir + self.spec.mc_file
        if filename_data is None:
            filename_data = self.spec.input_dir + self.spec.data_file
        if filename_herwig is None:
            filename_herwig = self.spec.input_dir + self.spec.herwig_file
        if filename_jk_data is None:
            filename_jk_data = self.spec.input_dir + self.spec.jk_data_file
        self._merge_eras()
        # Jackknife replicas are a separate skim product; when an input set does
        # not ship them yet, fall back to the TUnfold-propagated input-data
        # covariance instead of failing (the matrix-stat jackknife term is then
        # absent). Dropping the jk pkls into input_dir later restores the full
        # production stat treatment with no code change.
        _jk_paths = [Path(self.spec.input_dir + f) for f in self.spec.era_jk_files]
        _jk_paths.append(Path(filename_jk_data))
        if self.has_jackknife and not all(p.exists() for p in _jk_paths):
            missing = [str(p) for p in _jk_paths if not p.exists()]
            print(
                "WARNING: jackknife pkls missing -> statistical uncertainty falls "
                "back to the TUnfold-propagated input covariance (no jackknife, "
                "no matrix-stat term). Missing: " + ", ".join(missing)
            )
            self.has_jackknife = False
        if self.has_jackknife:
            self._merge_eras_jk()
            output_pythia_jk = self.pythia_hists_jk
        output_pythia = self.pythia_hists
        print("Keys in pythia file:", output_pythia.keys())
        filename_data = self._resolve_input_path(filename_data, self.spec.input_dir + "data_all.pkl")
        filename_herwig = self._resolve_input_path(
            filename_herwig,
            self.spec.input_dir + "herwig_all.pkl",
        )

        output_herwig = self._load_pickle(filename_herwig)
        output_data = self._load_pickle(filename_data)
        if self.has_jackknife:
            filename_jk_data = self._resolve_input_path(filename_jk_data, self.spec.input_dir + "jk_data_all.pkl")
            output_jk_data = self._load_pickle(filename_jk_data)
            print("Keys in jk data file:", output_jk_data.keys())

        sys_matrix_dic = self.sys_matrix_dic
        keys = self._histogram_keys()

        pythia4d = output_pythia[keys["response"]]
        pythia4d_jk = output_pythia_jk[keys["response"]] if self.has_jackknife else None
        herwig4d = output_herwig[keys["response"]]
        pythia2d = output_pythia[keys["reco"]]
        herwig2d = output_herwig[keys["reco"]]
        data2d = output_data[keys["reco"]]
        data2d_jk = output_jk_data[keys["reco"]] if self.has_jackknife else None
        pythia_gen2d = output_pythia[keys["gen"]]
        herwig_gen2d = output_herwig[keys["gen"]]

        pythia4d_gen = rebin_hist(pythia4d.copy(), self.reco_axis,self.edges_gen )
        herwig4d_gen = rebin_hist(herwig4d.copy(), self.reco_axis,self.edges_gen )

        resp_matrix_4d_gen = pythia4d_gen

        reco_mass_edges_by_pt = self.reco_edges_by_pt
        gen_mass_edges_by_pt = self.gen_edges_by_pt
        pt_edges = self.bins.pt_edges
        mass_edges_reco = self.edges
        mass_edges_gen = self.edges_gen

        # Fakes/misses are derived by subtracting the matched matrix from the
        # reco/gen spectra. The matched term must sum over the IN-RANGE pt bins
        # only (exclude pt flow): from ARC round-2 the low sink bin is [185,200]
        # rather than [0,200], so the pt axis now has underflow (jets below 185).
        # A flow-inclusive projection would fold a cross-boundary event (gen<185 &
        # reco>=185, or reco<185 & gen>=185) back into "matched", cancelling it
        # out of fakes/misses -- yet the matrix itself is built with flow=False
        # and drops it, so it would leak out of both and bias the result. Summing
        # only the in-range pt bins puts those events where they belong: gen<185 &
        # reco>=185 -> fake, reco<185 & gen>=185 -> miss. No-op while the pt axis
        # starts at 0 (pt flow is empty); see scripts/study_pt_sink_flow.py.
        def _sum_inrange_pt(h, pt_axis):
            return h[{pt_axis: slice(0, len(h.axes[pt_axis]), sum)}]

        fakes = pythia2d.project('ptreco', self.reco_axis, 'systematic')[:, :, 'nominal'] + (-1)*_sum_inrange_pt(pythia4d, 'ptgen').project('ptreco', self.reco_axis, 'systematic')[:, :, 'nominal']
        fakes_herwig = herwig2d.project('ptreco', self.reco_axis, 'systematic') + (-1)*_sum_inrange_pt(herwig4d, 'ptgen').project('ptreco', self.reco_axis, 'systematic')
        self.fakes = fakes
        self.fakes_herwig = fakes_herwig

        misses = pythia_gen2d.project('ptgen', self.gen_axis, 'systematic')[:, :, 'nominal'] + (-1)*_sum_inrange_pt(pythia4d, 'ptreco').project('ptgen', self.gen_axis, 'systematic')[:, :, 'nominal']
        misses_herwig = herwig_gen2d.project('ptgen', self.gen_axis, 'systematic') + (-1)*_sum_inrange_pt(herwig4d, 'ptreco').project('ptgen', self.gen_axis, 'systematic')
        self.misses = misses
        self.misses_herwig = misses_herwig
        self.y_true_herwig = self._prepare_truth_spectrum(
            herwig_gen2d,
            mass_edges_gen,
            pt_edges,
            gen_mass_edges_by_pt,
        )

        self.mosaic_dict = {}
        self.M_np_2d_dict = {}
        resp_matrix_4d = pythia4d
        resp_matrix_4d_herwig = herwig4d
        input_data = data2d


        self.input_data = input_data
        self.pythia_2d = pythia2d
        self.pythia_4d = pythia4d
        self.herwig_2d = herwig2d
        # Kept so the HERWIG bias test (plot_herwig_bias_test) can build the
        # HERWIG reco mosaic on demand even when no herwig systematic was run.
        self.herwig_4d = herwig4d
        self.herwig_4d_gen = herwig4d_gen
        self.data_2d = data2d

        self._load_background(
            keys["reco"],
            mass_edges_reco,
            pt_edges,
            reco_mass_edges_by_pt,
        )

        print("Loaded pkl files and rebinned histograms.")

        if self.has_jackknife:
            print("Processing jk inputs...")
            self.mosaic_2d_jk_list = self._prepare_jackknife_inputs(
                data2d_jk,
                mass_edges_reco,
                pt_edges,
                reco_mass_edges_by_pt,
            )
            self.mosaic_jk_list = self._prepare_jackknife_response_matrices(
                pythia4d_jk,
                mass_edges_reco,
                mass_edges_gen,
                pt_edges,
                reco_mass_edges_by_pt,
                gen_mass_edges_by_pt,
            )
        else:
            self.mosaic_2d_jk_list = None
            self.mosaic_jk_list = None

        for syst in self.systematics:
            if syst == 'nominal':
                #print("Processing nominal systematic:", syst)
                self._prepare_nominal_inputs(
                    input_data,
                    fakes,
                    misses,
                    resp_matrix_4d_gen,
                    mass_edges_reco,
                    mass_edges_gen,
                    pt_edges,
                    reco_mass_edges_by_pt,
                    gen_mass_edges_by_pt,
                )
            if syst in {"herwigUp", "herwigDown"}:
                #print("Processing Herwig systematic:", syst)
                self._prepare_herwig_inputs(
                    resp_matrix_4d_herwig,
                    herwig4d_gen,
                    fakes_herwig,
                    misses_herwig,
                    mass_edges_reco,
                    mass_edges_gen,
                    pt_edges,
                    reco_mass_edges_by_pt,
                    gen_mass_edges_by_pt,
                )

            M_np, _ = self._response_matrix_for_systematic(syst, resp_matrix_4d_herwig, sys_matrix_dic)
            self.M_np_orig = M_np
            self.M_np_2d_dict[syst], _ = reorder_to_expected(M_np, mass_edges_reco, pt_edges, mass_edges_gen)
            self.mosaic_dict[syst], _ = mosaic_no_padding(
                self.M_np_2d_dict[syst], mass_edges_reco, mass_edges_gen,
                reco_mass_edges_by_pt, gen_mass_edges_by_pt
            )
                
        self._build_misses_dict(
            pythia_gen2d,
            mass_edges_gen,
            pt_edges,
            gen_mass_edges_by_pt,
        )

        # Nominal gen-level values + MC-stat variances for the PYTHIA / HERWIG
        # predictions, used to draw their uncertainty error bars on the unfolded
        # plots. Values use the same flattening as the variance so the per-bin
        # relative stat uncertainty sqrt(var)/N is self-consistent.
        self.pythia_gen_val_flat = self._flatten_gen_total(
            pythia_gen2d, "nominal", mass_edges_gen, pt_edges, gen_mass_edges_by_pt
        )
        self.pythia_gen_var_flat = self._flatten_gen_var(
            pythia_gen2d, "nominal", mass_edges_gen, pt_edges, gen_mass_edges_by_pt
        )
        self.herwig_gen_val_flat = self._flatten_gen_total(
            herwig_gen2d, "nominal", mass_edges_gen, pt_edges, gen_mass_edges_by_pt
        )
        self.herwig_gen_var_flat = self._flatten_gen_var(
            herwig_gen2d, "nominal", mass_edges_gen, pt_edges, gen_mass_edges_by_pt
        )

        # ---- MC-stat variances from the pkl hist objects -----------------
        # Response: nominal matched-response sumw2 through the same reorder +
        # mosaic pipeline as the values (both steps only permute or sum bins,
        # so variances stay valid). Used as TUnfold bin errors.
        self.mosaic_var_dict = {}
        self.misses_var_dict = {}
        if getattr(self, "sys_matrix_variance", None) is not None:
            var_2d, _ = reorder_to_expected(
                self.sys_matrix_variance, mass_edges_reco, pt_edges, mass_edges_gen
            )
            mosaic_var, _ = mosaic_no_padding(
                var_2d, mass_edges_reco, mass_edges_gen,
                reco_mass_edges_by_pt, gen_mass_edges_by_pt,
            )
            self.mosaic_var_dict["nominal"] = np.clip(mosaic_var, 0.0, None)

        # Fakes/misses are disjoint event subsets of the totals, so their
        # sumw2 is the *difference* of the variances (hist subtraction would
        # add them and overestimate).
        def _nominal_var(hist_obj, pt_axis, mass_axis):
            proj = hist_obj.project(pt_axis, mass_axis, "systematic")[:, :, "nominal"]
            return proj.variances()

        reco_total_var = _nominal_var(pythia2d, "ptreco", self.reco_axis)
        reco_matched_var = _nominal_var(pythia4d, "ptreco", self.reco_axis)
        gen_total_var = _nominal_var(pythia_gen2d, "ptgen", self.gen_axis)
        gen_matched_var = _nominal_var(pythia4d, "ptgen", self.gen_axis)
        if all(v is not None for v in (reco_total_var, reco_matched_var)):
            fakes_var_2d, _ = reorder_to_expected_2d(
                np.clip(reco_total_var - reco_matched_var, 0.0, None),
                mass_edges_reco, pt_edges,
            )
            self.fakes_2d_var = merge_mass_flat(
                fakes_var_2d, mass_edges_reco, reco_mass_edges_by_pt
            )
        if all(v is not None for v in (gen_total_var, gen_matched_var)):
            misses_var_2d, _ = reorder_to_expected_2d(
                np.clip(gen_total_var - gen_matched_var, 0.0, None),
                mass_edges_gen, pt_edges,
            )
            self.misses_var_dict["nominal"] = merge_mass_flat(
                misses_var_2d, mass_edges_gen, gen_mass_edges_by_pt
            )

        print("Loaded data and prepared response matrices.")
        self._finalize_reco_views(mass_edges_reco, reco_mass_edges_by_pt)
        self.y_unf_jk_input_list = []
        self.y_unf_jk_matrix_list = []
        #print("h2d shape:", self.h2d.shape)
        #print("reco_mass_edges_by_pt:", reco_mass_edges_by_pt)
        #print("len reco_mass_edges_by_pt:", len(reco_mass_edges_by_pt))
    def plot_fakes_misses(self, show=True):
        title_list = [""]
        npt = len(self.pt_edges) - 1
        for i in range(1, npt):
            lo = int(self.pt_edges[i])
            if i + 1 < npt:
                hi = int(self.pt_edges[i + 1])
                title_list.append(rf"{lo} $<$ $p_T$ $<$ {hi} GeV")
            else:
                title_list.append(rf"{lo} $<$ $p_T$ $< \, \infty$ GeV")

        # Use the same MC fake fraction used by the unfolding fake correction.
        # Dividing by data here can produce infinities in sparse rho tail bins.
        fakerate = np.asarray(self.fake_fraction_2d, dtype=float)
        efficiency = 1 - (self.misses_2d/(self.misses_2d + self.mosaic.sum(axis=0)))
        efficiency_pt_binned = unflatten_gen_by_pt(efficiency, self.gen_edges_by_pt)
        fakerate_pt_binned = unflatten_gen_by_pt(fakerate, self.reco_edges_by_pt)

        _ratio_unc = self._subset_fraction_unc

        # MC-stat error bars from the pkl-hist sumw2 (when available).
        fakerate_unc_pt_binned = efficiency_unc_pt_binned = None
        mosaic_var = getattr(self, "mosaic_var_dict", {}).get("nominal")
        misses_var = getattr(self, "misses_var_dict", {}).get("nominal")
        fakes_var = getattr(self, "fakes_2d_var", None)
        if mosaic_var is not None and fakes_var is not None:
            matched_reco = self.mosaic.sum(axis=1)
            fakerate_unc = _ratio_unc(
                self.fakes_2d, matched_reco, fakes_var, mosaic_var.sum(axis=1)
            )
            fakerate_unc_pt_binned = unflatten_gen_by_pt(fakerate_unc, self.reco_edges_by_pt)
        if mosaic_var is not None and misses_var is not None:
            matched_gen = self.mosaic.sum(axis=0)
            efficiency_unc = _ratio_unc(
                self.misses_2d, matched_gen, misses_var, mosaic_var.sum(axis=0)
            )
            efficiency_unc_pt_binned = unflatten_gen_by_pt(efficiency_unc, self.gen_edges_by_pt)

        for i in self._reported_pt_indices():
            hep.histplot(
                1 - fakerate_pt_binned[i],
                self.reco_edges_by_pt[i],
                yerr=fakerate_unc_pt_binned[i] if fakerate_unc_pt_binned is not None else None,
                label=r"$1-$ fake rate $= N_{\mathrm{gen}\wedge\mathrm{reco}}/N_{\mathrm{reco}}$",
                lw=1.5,
                histtype="step",
            )
            hep.histplot(
                efficiency_pt_binned[i],
                self.gen_edges_by_pt[i],
                yerr=efficiency_unc_pt_binned[i] if efficiency_unc_pt_binned is not None else None,
                label=r"Acceptance $\times$ efficiency $= N_{\mathrm{gen}\wedge\mathrm{reco}}/N_{\mathrm{gen}}$",
                lw=1.5,
                histtype="step",
            )
            plt.legend(title = title_list[i])
            plt.xlabel(self._observable_short_label())
            plt.xlim(*self._observable_xlim(i))
            plt.ylim(0,1.05)
            hep.cms.label(
                self.cms_label,
                data=False,
                lumi=self._lumi_label(),
                com=self._com_label(),
                fontsize=20,
            )
            if self.groomed:
                save_path = f"./{self.spec.output_dir}fakerates_groomed_{i-1}.pdf"
            else:
                save_path = f"./{self.spec.output_dir}fakerates_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show)

    def _gen_binned_migration(self, matrix=None):
        """Compress the matched response to gen binning on both axes.

        Returns A with A[j, k] = matched events reconstructed in the gen-bin
        region j and generated in gen bin k (reco mass bins are grouped into
        the gen bins of the same pT slice; pT edges are shared). ``matrix``
        defaults to the nominal mosaic; pass its variance mosaic to compress
        sumw2 the same way (grouping = summing, so variances stay valid).
        """
        mosaic = self.mosaic_dict["nominal"] if matrix is None else matrix
        n_gen = mosaic.shape[1]
        compressed = np.zeros((n_gen, n_gen))
        reco_offset = 0
        gen_offset = 0
        for i, gen_edges in enumerate(self.gen_edges_by_pt):
            reco_edges = np.asarray(self.reco_edges_by_pt[i], dtype=float)
            centers = 0.5 * (reco_edges[:-1] + reco_edges[1:])
            gen_edges = np.asarray(gen_edges, dtype=float)
            for k in range(len(gen_edges) - 1):
                rows = np.flatnonzero(
                    (centers >= gen_edges[k]) & (centers < gen_edges[k + 1])
                ) + reco_offset
                compressed[gen_offset + k, :] += mosaic[rows, :].sum(axis=0)
            reco_offset += len(reco_edges) - 1
            gen_offset += len(gen_edges) - 1
        return compressed

    def plot_purity_stability(self, show=True):
        """Per-gen-bin purity and stability of the matched response.

        purity_j    = fraction of events reconstructed in bin j that were
                      also generated in bin j;
        stability_j = fraction of events generated in bin j that were also
                      reconstructed in bin j (matched events only).
        """
        title_list = [""]
        npt = len(self.pt_edges) - 1
        for i in range(1, npt):
            lo = int(self.pt_edges[i])
            if i + 1 < npt:
                hi = int(self.pt_edges[i + 1])
                title_list.append(rf"{lo} $<$ $p_T$ $<$ {hi} GeV")
            else:
                title_list.append(rf"{lo} $<$ $p_T$ $< \, \infty$ GeV")

        compressed = self._gen_binned_migration()
        diagonal = np.diag(compressed)
        reco_totals = compressed.sum(axis=1)
        gen_totals = compressed.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            purity = np.divide(
                diagonal, reco_totals, out=np.zeros_like(diagonal), where=reco_totals > 0
            )
            stability = np.divide(
                diagonal, gen_totals, out=np.zeros_like(diagonal), where=gen_totals > 0
            )
        purity_pt_binned = unflatten_gen_by_pt(purity, self.gen_edges_by_pt)
        stability_pt_binned = unflatten_gen_by_pt(stability, self.gen_edges_by_pt)

        # MC-stat error bars: compress the sumw2 mosaic the same way, then
        # binomial-style propagation (the diagonal is a disjoint subset of
        # the row/column totals).
        purity_unc_pt_binned = stability_unc_pt_binned = None
        mosaic_var = getattr(self, "mosaic_var_dict", {}).get("nominal")
        if mosaic_var is not None:
            compressed_var = self._gen_binned_migration(mosaic_var)
            diag_var = np.diag(compressed_var)
            purity_unc = self._subset_fraction_unc(
                diagonal, reco_totals - diagonal,
                diag_var, compressed_var.sum(axis=1) - diag_var,
            )
            stability_unc = self._subset_fraction_unc(
                diagonal, gen_totals - diagonal,
                diag_var, compressed_var.sum(axis=0) - diag_var,
            )
            purity_unc_pt_binned = unflatten_gen_by_pt(purity_unc, self.gen_edges_by_pt)
            stability_unc_pt_binned = unflatten_gen_by_pt(stability_unc, self.gen_edges_by_pt)

        for i in self._reported_pt_indices():
            hep.histplot(
                purity_pt_binned[i],
                self.gen_edges_by_pt[i],
                yerr=purity_unc_pt_binned[i] if purity_unc_pt_binned is not None else None,
                label="Purity",
                lw=1.5,
                histtype="step",
            )
            hep.histplot(
                stability_pt_binned[i],
                self.gen_edges_by_pt[i],
                yerr=stability_unc_pt_binned[i] if stability_unc_pt_binned is not None else None,
                label="Stability",
                lw=1.5,
                histtype="step",
            )
            plt.axhline(0.5, color="gray", ls="dotted", lw=1)
            plt.legend(title=title_list[i])
            plt.xlabel(self._observable_short_label())
            plt.ylabel("Purity / Stability")
            plt.xlim(*self._observable_xlim(i))
            plt.ylim(0, 1.05)
            hep.cms.label(
                self.cms_label,
                data=False,
                lumi=self._lumi_label(),
                com=self._com_label(),
                fontsize=20,
            )
            suffix = "groomed" if self.groomed else "ungroomed"
            save_path = f"./{self.spec.output_dir}purity_stability_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show)

    def plot_input_data_mc(self, show=True):
        """Reco-level overlay of Data vs PYTHIA vs HERWIG per pt bin, with a ratio pad."""
        hep.style.use("CMS")
        pt_edges = np.asarray(self.bins.pt_edges, dtype=float)
        mode = "groomed" if self.groomed else "ungroomed"
        x_label = self.spec.x_label_groomed if self.groomed else self.spec.x_label_ungroomed

        def _project_nominal(hist_2d):
            if "systematic" in hist_2d.axes.name:
                return hist_2d.project("ptreco", self.reco_axis, "systematic")[:, :, "nominal"]
            return hist_2d.project("ptreco", self.reco_axis)

        h_data = _project_nominal(self.data_2d)
        h_mc = _project_nominal(self.pythia_2d)
        h_her = _project_nominal(self.herwig_2d)

        pt_axis_edges = np.asarray(h_data.axes['ptreco'].edges, dtype=float)
        reco_axis_edges = np.asarray(h_data.axes[self.reco_axis].edges, dtype=float)

        def _slice(h, lo, hi):
            vals = np.asarray(h.values(), dtype=float)
            raw_var = h.variances()
            if raw_var is None:
                # Fallback for histograms that do not store variances explicitly.
                vars_ = np.clip(vals, 0.0, None)
            else:
                vars_ = np.asarray(raw_var, dtype=float)
            i_lo = int(np.searchsorted(pt_axis_edges, lo - 1e-9, side='left'))
            if np.isfinite(hi):
                i_hi = int(np.searchsorted(pt_axis_edges, hi - 1e-9, side='left'))
            else:
                i_hi = len(pt_axis_edges) - 1
            return vals[i_lo:i_hi, :].sum(axis=0), vars_[i_lo:i_hi, :].sum(axis=0)

        n_pt = len(pt_edges) - 1
        for bin_idx in range(n_pt):
            lo, hi = pt_edges[bin_idx], pt_edges[bin_idx + 1]
            pt_label = (rf"{lo:g}$-\infty$ GeV" if (not np.isfinite(hi) or hi >= 13000)
                        else f"{lo:g}-{hi:g} GeV")

            data_vals, data_var = _slice(h_data, lo, hi)
            mc_vals,   mc_var   = _slice(h_mc,   lo, hi)
            her_vals,  her_var  = _slice(h_her,  lo, hi)
            data_errs = np.sqrt(data_var)
            mc_errs   = np.sqrt(mc_var)
            her_errs  = np.sqrt(her_var)
            edges = reco_axis_edges

            # Normalize over the SHOWN reco window when the spec reports/normalizes
            # over the shown space (production default), so the displayed spectrum
            # integrates to 1 over exactly what is drawn; otherwise full range.
            if getattr(self.spec, "normalize_over_shown", False):
                shown = reco_axis_edges[:-1] >= self._bl_shown_floors()[bin_idx] - 1e-9
            else:
                shown = np.ones(len(reco_axis_edges) - 1, dtype=bool)
            for vals, errs in ((data_vals, data_errs), (mc_vals, mc_errs), (her_vals, her_errs)):
                s = vals[shown].sum()
                if s != 0:
                    vals /= s
                    errs /= s

            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )
            plt.sca(ax_main)
            hep.histplot(data_vals, edges, yerr=data_errs, label="Data",    color="black")
            hep.histplot(mc_vals,   edges, yerr=mc_errs,   label="PYTHIA8", color="red")
            hep.histplot(her_vals,  edges, yerr=her_errs,  label="HERWIG7", color="blue", alpha=0.7)
            ax_main.set_ylabel("Normalized entries")
            ax_main.legend(title=pt_label)
            hep.cms.label(self.cms_label, data=True, lumi=138, com=13, fontsize=20, ax=ax_main)

            safe_d = np.where(data_vals != 0, data_vals, 1.0)
            safe_m = np.where(mc_vals != 0, mc_vals, 1.0)
            safe_h = np.where(her_vals != 0, her_vals, 1.0)
            r_mc = np.divide(mc_vals,  data_vals, out=np.zeros_like(data_vals), where=data_vals != 0)
            r_h  = np.divide(her_vals, data_vals, out=np.zeros_like(data_vals), where=data_vals != 0)
            r_mc_err = r_mc * np.sqrt((data_errs / safe_d) ** 2 + (mc_errs  / safe_m) ** 2)
            r_h_err  = r_h  * np.sqrt((data_errs / safe_d) ** 2 + (her_errs / safe_h) ** 2)

            plt.sca(ax_ratio)
            hep.histplot(r_mc, edges, yerr=r_mc_err, label="PYTHIA/Data", color="red",  ax=ax_ratio)
            hep.histplot(r_h,  edges, yerr=r_h_err,  label="HERWIG/Data", color="blue", ls="--", alpha=0.7, ax=ax_ratio)
            ax_ratio.axhline(1, color="gray", ls="--")
            ax_ratio.set_xlabel(x_label)
            ax_ratio.set_ylabel("Theory / Data")
            ax_ratio.set_ylim(0, 2)

            # Trim the display to the SHOWN rho window for this pT slice (drop the
            # low-rho migration buffer that is only there to catch feed-in). The
            # shared x-axis carries the limit to both pads.
            floor = self._bl_shown_floors()[bin_idx]
            ax_ratio.set_xlim(floor, reco_axis_edges[-1])

            save_path = f"./{self.spec.output_dir}input_{mode}_{bin_idx}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def _compute_stat_unc(self):
        for i in range(10):
            meas_flat = self.mosaic_2d_jk_list[i]
            self._perform_unfold(
                systematic='nominal',
                closure=self.closure,
                herwig_closure=self.herwig_closure,
                meas_flat=meas_flat,
                do_jk=True,
                jk_target="input",
            )

        for i in range(10):
            resp_np = self.mosaic_jk_list[i]
            self._perform_unfold(
                systematic='nominal',
                closure=self.closure,
                herwig_closure=self.herwig_closure,
                resp_np=resp_np,
                do_jk=True,
                jk_target="matrix",
            )

        # Delete-one-tenth (grouped) jackknife with g=10 groups. The correct SE
        # on the population std of the replicas (np.std, ddof=0) is sqrt(g-1);
        # var_jack = (g-1)/g * sum((theta_i - mean)^2) = (g-1) * s_pop^2. The old
        # sqrt(10/9) was a Bessel-style factor that under-covered by ~2.85x.
        jk_scale = np.sqrt(9.0)
        input_std = jk_scale * np.std(self.y_unf_jk_input_list, axis=0)
        matrix_std = jk_scale * np.std(self.y_unf_jk_matrix_list, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.input_stat_unc_frac = np.abs(
                np.divide(input_std, self.y_unf, out=np.zeros_like(input_std), where=self.y_unf != 0)
            )
            self.matrix_stat_unc_frac = np.abs(
                np.divide(matrix_std, self.y_unf, out=np.zeros_like(matrix_std), where=self.y_unf != 0)
            )

        self.stat_unc_frac = np.sqrt(self.input_stat_unc_frac**2 + self.matrix_stat_unc_frac**2)
        self.input_stat_unc_pt_binned = unflatten_gen_by_pt(self.input_stat_unc_frac, self.gen_edges_by_pt)
        self.matrix_stat_unc_pt_binned = unflatten_gen_by_pt(self.matrix_stat_unc_frac, self.gen_edges_by_pt)
        self.stat_unc_pt_binned = unflatten_gen_by_pt(self.stat_unc_frac, self.gen_edges_by_pt)

    def _select_measured_spectrum(self, closure, herwig_closure, meas_flat):
        if meas_flat is None:
            if closure:
                meas_flat = self.mosaic.sum(axis=1)
            else:
                meas_flat = self.mosaic_2d
            if herwig_closure:
                meas_flat = self.mosaic_herwig_2d
        return meas_flat

    def _apply_fake_correction(self, meas_flat, systematic, closure, herwig_closure):
        if closure or herwig_closure:
            return meas_flat

        fake_fraction = self.fake_fraction_2d
        if systematic in {"herwigUp", "herwigDown"} and hasattr(self, "fake_fraction_2d_herwig"):
            fake_fraction = self.fake_fraction_2d_herwig

        corrected = np.asarray(meas_flat, dtype=float) * (1.0 - fake_fraction)
        return np.clip(corrected, 0.0, None)

    def _build_root_binning(self):
        truth_root = ROOT.TUnfoldBinning("truth")
        reco_root = ROOT.TUnfoldBinning("reco")

        truth_signal = truth_root.AddBinning("signal")
        reco_primary = reco_root.AddBinning("primary")

        for i, edges in enumerate(self.gen_edges_by_pt):
            truth_node = truth_signal.AddBinning(f"pt{i}")
            truth_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        for i, edges in enumerate(self.reco_edges_by_pt):
            reco_node = reco_primary.AddBinning(f"pt{i}")
            reco_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        return truth_root, reco_root

    def _fill_root_histogram(self, hist, values, variances=None):
        for index, value in enumerate(values, 1):
            hist.SetBinContent(index, float(value))
            if variances is not None:
                hist.SetBinError(index, float(np.sqrt(max(variances[index - 1], 0.0))))

    def _fill_response_histogram(
        self,
        h_resp,
        resp_np,
        misses,
        *,
        include_bin_errors=True,
        resp_var=None,
        misses_var=None,
    ):
        """Fill the migration matrix + miss row, with optional MC-stat errors.

        When ``resp_var``/``misses_var`` (sumw2 from the pkl hist objects) are
        given, bin errors are set to sqrt(var). Without them ROOT defaults
        GetBinError to sqrt(weighted content), which overestimates the MC-stat
        error of weighted histograms — so pass them whenever available.
        """
        n_reco, n_true = resp_np.shape
        for i_reco in range(n_reco):
            for j_true in range(n_true):
                h_resp.SetBinContent(j_true + 1, i_reco + 1, resp_np[i_reco, j_true])
                if not include_bin_errors:
                    h_resp.SetBinError(j_true + 1, i_reco + 1, 0.0)
                elif resp_var is not None:
                    h_resp.SetBinError(
                        j_true + 1, i_reco + 1,
                        float(np.sqrt(max(resp_var[i_reco, j_true], 0.0))),
                    )
        for j_true in range(n_true):
            h_resp.SetBinContent(j_true + 1, 0, misses[j_true])
            if not include_bin_errors:
                h_resp.SetBinError(j_true + 1, 0, 0.0)
            elif misses_var is not None:
                h_resp.SetBinError(
                    j_true + 1, 0, float(np.sqrt(max(misses_var[j_true], 0.0)))
                )

    def _add_ratio_curvature_conditions(self, unfold, prior_flat):
        """Register curvature-of-ratio regularization rows.

        For interior gen bins of each pT slice adds the row
        (1/m0, -2/m1, 1/m2) so that any spectrum proportional to the prior m
        (the nominal MC truth) has exactly zero penalty; only shape deviations
        from the prior are smoothed. No conditions cross pT-slice boundaries.
        Validated in scripts/study_regularization_rho.py: exact self-closure
        at any tau, <1% added HERWIG-closure bias at the L-curve tau.
        """
        offset = 0
        for edges in self.gen_edges_by_pt:
            nbins = len(edges) - 1
            for k in range(1, nbins - 1):
                j0, j1, j2 = offset + k - 1, offset + k, offset + k + 1
                m0, m1, m2 = prior_flat[j0], prior_flat[j1], prior_flat[j2]
                if min(m0, m1, m2) <= 0:
                    continue
                unfold.AddRegularisationCondition(
                    j0 + 1, 1.0 / m0, j1 + 1, -2.0 / m1, j2 + 1, 1.0 / m2
                )
            offset += nbins

    def _store_covariances(self, unfold, systematic):
        if systematic == "nominal":
            self.cov = unfold.GetEmatrixTotal("cov", "Covariance Matrix")
            self.cov_uncorr = unfold.GetEmatrixSysUncorr(
                "cov_uncorr",
                "Covariance Matrix from Uncorrelated Uncertainties",
            )
            self.cov_uncorr_data = unfold.GetEmatrixInput(
                "cov_uncorr_data",
                "Covariance Matrix from Stat Uncertainties of Input Data",
            )
            self.cov_total = unfold.GetEmatrixTotal("total", "Cov")

            _, n_true = self.mosaic.shape
            self.cov_np = np.zeros((n_true, n_true))
            self.cov_uncorr_np = np.zeros((n_true, n_true))
            self.cov_data_np = np.zeros((n_true, n_true))
            for i in range(1, n_true + 1):
                for j in range(1, n_true + 1):
                    self.cov_np[i - 1, j - 1] = self.cov.GetBinContent(i, j)
                    self.cov_uncorr_np[i - 1, j - 1] = self.cov_uncorr.GetBinContent(i, j)
                    self.cov_data_np[i - 1, j - 1] = self.cov_uncorr_data.GetBinContent(i, j)

            # Reco-space hat matrix H = K J for the bottom-line N_dof (TWiki):
            # K is the smearing/probability matrix (reco x gen), J = dx/dy the
            # error-propagation matrix (gen x reco).  Both are exact TUnfold
            # linear-response outputs; must be read while ``unfold`` is alive
            # (see _store_unfold_result).  useAxisBinning=False -> flat global
            # bins.  The TWiki N_dof is the effective rank of K J J^T K^T; for
            # the oblique reco-space projection K J that rank equals the trace
            # of H (the standard effective number of degrees of freedom), which
            # is what :meth:`_effective_ndof` uses -- see the note there.
            self.hat_reco_np = None
            try:
                K = self._th2_to_np(unfold.GetProbabilityMatrix("bl_K", "K", False))
                J = self._th2_to_np(unfold.GetDXDY("bl_dxdy", "dxdy", False))
                n_reco = self.mosaic.shape[0]
                if K.shape == (n_true, n_reco):
                    K = K.T
                if J.shape == (n_reco, n_true):
                    J = J.T
                if K.shape == (n_reco, n_true) and J.shape == (n_true, n_reco):
                    self.hat_reco_np = K @ J        # reco -> reco hat matrix H
            except Exception:
                self.hat_reco_np = None

        if systematic == "herwigUp":
            self.cov_uncorr_data = unfold.GetEmatrixInput(
                "cov_uncorr_data",
                "Covariance Matrix from Stat Uncertainties of Input Data",
            )
            _, n_true = self.mosaic.shape
            self.cov_data_herwig_np = np.zeros((n_true, n_true))
            for i in range(1, n_true + 1):
                for j in range(1, n_true + 1):
                    self.cov_data_herwig_np[i - 1, j - 1] = self.cov_uncorr_data.GetBinContent(i, j)

    @staticmethod
    def _th2_to_np(h):
        """Dense (nbinsX, nbinsY) array of a ROOT TH2, excluding over/underflow."""
        nx, ny = h.GetNbinsX(), h.GetNbinsY()
        arr = np.empty((nx, ny))
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                arr[i - 1, j - 1] = h.GetBinContent(i, j)
        return arr

    def _effective_ndof(self, reco_idx):
        """Bottom-line N_dof: effective rank of K J J^T K^T over a reco block.

        The CMS Statistics Committee / TUnfold TWiki prescribes, under
        regularization, ``N_dof = effective rank of K J J^T K^T`` for the
        chi-square test in the unfolded space (the raw covariance rank
        over-counts modes the regularization has suppressed).

        The reco-space hat matrix ``H = K J`` is an *oblique* projection
        (n_reco > n_gen), so its singular values differ from its eigenvalues
        and the participation ratio of ``K J J^T K^T = H H^T`` is inflated by a
        few large singular modes -- it does not count degrees of freedom.  The
        robust, scale-free equivalent is the effective number of degrees of
        freedom ``tr(H)`` (Wahba's effective dof): it equals the rank of
        ``K J J^T K^T`` when the unfolding is unregularized and shrinks
        smoothly as regularization suppresses modes.  ``tr(H)`` over a reco
        sub-block (the summed leverages) is the N_dof carried by that block.

        Returns a float, or ``None`` when the TUnfold matrices are unavailable
        (e.g. the RooUnfold backend), so callers fall back to the covariance
        rank.
        """
        H = getattr(self, "hat_reco_np", None)
        if H is None:
            return None
        idx = np.asarray(list(reco_idx), dtype=int)
        return float(np.trace(H[np.ix_(idx, idx)]))

    @staticmethod
    def _chi2_from_covariance(delta, covariance, *, rcond=1e-12, ndof=None):
        """Return a stable correlated chi-square and the covariance rank.

        A per-pT normalization or an exactly constrained unfolding covariance
        can be singular by construction.  Diagonalizing the symmetric part and
        retaining only resolved positive eigenmodes gives the appropriate
        pseudo-inverse statistic and its corresponding number of degrees of
        freedom.

        ``ndof`` overrides the number of degrees of freedom used for the
        p-value (and reported as ``ndof``).  The CMS Statistics Committee /
        TUnfold TWiki bottom-line test prescribes, for the *unfolded* space
        under regularization, the effective rank of ``K J J^T K^T`` rather
        than the raw covariance rank -- see :meth:`_effective_ndof`.  The
        covariance rank is still reported separately as ``cov_rank``.
        """
        delta = np.asarray(delta, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (delta.size, delta.size):
            raise ValueError(
                "Covariance shape does not match the tested residual vector: "
                f"{covariance.shape} versus {delta.size}."
            )

        symmetric_covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        keep = eigenvalues > rcond * scale
        rank = int(np.count_nonzero(keep))
        if rank == 0:
            raise ValueError("Bottom-line covariance has no resolved positive modes.")

        projected_residual = eigenvectors[:, keep].T @ delta
        chi2_value = float(np.sum(projected_residual**2 / eigenvalues[keep]))
        condition = float(np.max(eigenvalues[keep]) / np.min(eigenvalues[keep]))
        dof = rank if ndof is None else float(ndof)
        return {
            "chi2": chi2_value,
            "ndof": dof,
            "cov_rank": rank,
            "pvalue": float(scipy_chi2.sf(chi2_value, dof)) if dof > 0 else float("nan"),
            "condition": condition,
        }

    def _model_cov_unfolded(self, gidx):
        """Correlated PS/HAD modelling covariance on the unfolded data, in
        the un-normalized gen space, restricted to flat gen indices ``gidx``.

        ARC round-2: the bottom-line test folds the modelling uncertainty into
        the *unfolded data* covariance (the detector-level data carries no
        modelling term, so the smeared side is untouched). The stored
        The stored signed PS/HAD shifts are fractional variations of the
        normalized per-pT shape. Multiplying them by the un-normalized
        ``y_unf`` gives the absolute nuisance vectors (the per-pT normalization
        is a common scale, so shape-fraction == count-fraction to the accuracy
        of this test). Their outer products retain the bin-to-bin correlations.
        Returns ``None`` when no modelling envelope is available.
        """
        scope = getattr(
            getattr(self, "spec", None),
            "model_covariance_scope",
            "global_shown",
        )
        ps_frac = getattr(self, "model_ps_shift_flat", None)
        had_frac = getattr(self, "model_had_shift_flat", None)
        if ps_frac is None or had_frac is None:
            return None
        gidx = list(gidx)
        y = np.asarray(self.y_unf, float)[gidx]
        if scope == "per_pt":
            covariance = np.zeros((len(gidx), len(gidx)), dtype=float)
            for ps_full, had_full in zip(
                    self.model_ps_shifts_by_pt_flat,
                    self.model_had_shifts_by_pt_flat):
                ps_abs = np.asarray(ps_full, float)[gidx] * y
                had_abs = np.asarray(had_full, float)[gidx] * y
                covariance += (
                    np.outer(ps_abs, ps_abs) + np.outer(had_abs, had_abs)
                )
            return covariance
        if scope not in ("global_shown", "global_all"):
            raise ValueError(f"Unknown model_covariance_scope: {scope}")
        ps_abs = np.asarray(ps_frac, float)[gidx] * y
        had_abs = np.asarray(had_frac, float)[gidx] * y
        return np.outer(ps_abs, ps_abs) + np.outer(had_abs, had_abs)

    def bottom_line_test(self):
        """Evaluate the bottom-line test for the nominal model.

        The detector-space comparison is between the fake-corrected measured
        data and the matched reconstructed PYTHIA prediction.  The unfolded
        comparison is between the unfolded data and the full PYTHIA truth
        (matched plus misses).  Both use only propagated *data* statistics:
        fake rates, response MC statistics, detector systematics, and any
        model/bias covariance are deliberately excluded here.

        This is the quantitative counterpart to :meth:`plot_bottom_line`.
        For the unregularized production configuration, the expected result is
        ``chi2_unfolded <= chi2_smeared``.  Equality is not required because
        the detector-space fit can contain residual modes that the finite truth
        parameterization cannot represent.
        """
        measured_variances = getattr(self, "corrected_measured_variances", None)
        if measured_variances is None:
            raise RuntimeError(
                "No fake-corrected data sumw2 is available for the bottom-line "
                "test. Re-run with input data histogram variances enabled."
            )
        unfolded_covariance = getattr(self, "cov_data_np", None)
        if unfolded_covariance is None:
            raise RuntimeError(
                "No propagated input-data covariance is available for the "
                "bottom-line test."
            )

        # The response matrix contains matched events only.  The fake
        # correction therefore puts data and model in the same reco space.
        reco_residual = np.asarray(self.y_meas, float) - np.asarray(
            self.mosaic.sum(axis=1), float
        )
        truth_residual = np.asarray(self.y_unf, float) - np.asarray(self.y_true, float)
        smeared = self._chi2_from_covariance(
            reco_residual,
            np.diag(np.asarray(measured_variances, dtype=float)),
        )
        ndof_unf = self._effective_ndof(range(reco_residual.size))
        unfolded_stat = self._chi2_from_covariance(
            truth_residual, unfolded_covariance, ndof=ndof_unf,
        )
        model_cov = self._model_cov_unfolded(range(truth_residual.size))
        if model_cov is not None:
            unfolded = self._chi2_from_covariance(
                truth_residual, unfolded_covariance + model_cov, ndof=ndof_unf,
            )
        else:
            unfolded = unfolded_stat
        return {
            "scope": ("data stat + modelling uncertainty on the unfolded data"
                      if model_cov is not None else "data-stat-only"),
            "model_uncertainty_included": model_cov is not None,
            "regularization": self.regularization,
            "tau": float(self.tau) if self.tau is not None else 0.0,
            "smeared": smeared,
            "unfolded": unfolded,
            "unfolded_stat_only": unfolded_stat,
            "chi2_difference_unfolded_minus_smeared": (
                unfolded["chi2"] - smeared["chi2"]
            ),
            "inequality_holds": bool(unfolded["chi2"] <= smeared["chi2"] + 1e-9),
        }

    def _bl_shown_floors(self):
        """Per-pT rho floor of the SHOWN space (aligned with ``pt_edges``).

        Uses the spec's ``bl_shown_floors_{groomed,ungroomed}`` when set (so the
        chi2 is evaluated only up to the rho value each slice is cut at for
        display), else falls back to the uniform ``xlim_lower`` for the mode.
        """
        floors = (self.spec.bl_shown_floors_groomed if self.groomed
                  else self.spec.bl_shown_floors_ungroomed)
        if floors is not None:
            return list(floors)
        lo = (self.spec.xlim_lower_groomed if self.groomed
              else self.spec.xlim_lower_ungroomed)
        return [lo] * (len(self.pt_edges) - 1)

    def _shown_gen_mask(self, i):
        """Boolean mask over the gen bins of slice ``i`` that are shown/reported.

        All-True unless ``spec.normalize_over_shown`` is set; then a bin is in
        the mask when its lower edge sits at or above the slice's shown floor.
        """
        edges = np.asarray(self.gen_edges_by_pt[i], dtype=float)
        if not getattr(self.spec, "normalize_over_shown", False):
            return np.ones(len(edges) - 1, dtype=bool)
        return edges[:-1] >= self._bl_shown_floors()[i] - 1e-9

    def _shown_reco_mask(self, i):
        """Reco-binning companion of :meth:`_shown_gen_mask`."""
        edges = np.asarray(self.reco_edges_by_pt[i], dtype=float)
        if not getattr(self.spec, "normalize_over_shown", False):
            return np.ones(len(edges) - 1, dtype=bool)
        return edges[:-1] >= self._bl_shown_floors()[i] - 1e-9

    def _shown_norm_total(self, values, i, reco=False):
        """Slice-normalization denominator: sum of ``values`` over shown bins."""
        mask = self._shown_reco_mask(i) if reco else self._shown_gen_mask(i)
        return float(np.asarray(values, dtype=float)[mask].sum())

    @staticmethod
    def _reco_to_gen_rebin_1d(reco_vals, reco_edges, gen_edges):
        """Sum reco-binned values into gen bins (reco edges nest in gen edges).
        Works for counts and for variances (both add)."""
        reco_edges = np.asarray(reco_edges, float)
        gen_edges = np.asarray(gen_edges, float)
        reco_vals = np.asarray(reco_vals, float)
        out = np.zeros(len(gen_edges) - 1)
        for k in range(len(gen_edges) - 1):
            m = ((reco_edges[:-1] >= gen_edges[k] - 1e-9)
                 & (reco_edges[1:] <= gen_edges[k + 1] + 1e-9))
            out[k] = reco_vals[m].sum()
        return out

    def bottom_line_test_by_pt(self, min_edge=None, smeared_binning="reco"):
        """Per-pT-slice bottom-line test, plus a reported-range global row.

        ``smeared_binning="gen"`` rebins the detector level onto the gen binning
        before forming chi2_smeared (the ARC's alternative: with the reco side
        rebinned to gen, K is square and chi2_unfold == chi2_smeared is the
        expectation for an unregularized unfold). Default "reco" keeps the
        native 2x-finer reco binning.

        Same residual / covariance definitions as :meth:`bottom_line_test`,
        sliced per reported pT bin (>= 200 GeV; the 0-200 GeV migration sink is
        excluded). Each entry carries the ``smeared`` and ``unfolded``
        ``_chi2_from_covariance`` dicts. Used by the per-panel annotation and by
        the bottom-line chi2 bar chart. Returns ``(rows, global_row)`` or
        ``([], None)`` when the data-stat inputs are unavailable.

        ``min_edge`` restricts both the gen and reco chi2 sums to bins whose
        LOWER edge is >= the floor. It may be a scalar (uniform floor), a
        per-pT sequence aligned with ``pt_edges``, or the string ``"shown"`` to
        use the per-pT display floors from :meth:`_bl_shown_floors` (evaluate
        each slice only up to the rho value it is cut at).
        """
        if min_edge == "shown":
            min_edge = self._bl_shown_floors()
        var_y = getattr(self, "corrected_measured_variances", None)
        cov_x = getattr(self, "cov_data_np", None)
        if var_y is None or cov_x is None:
            return [], None
        var_y = np.asarray(var_y, dtype=float)
        cov_x = np.asarray(cov_x, dtype=float)
        reco_resid = np.asarray(self.y_meas, float) - np.asarray(self.mosaic.sum(axis=1), float)
        truth_resid = np.asarray(self.y_unf, float) - np.asarray(self.y_true, float)

        def offsets(edges_by_pt):
            counts = [len(e) - 1 for e in edges_by_pt]
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
            return starts, counts

        gstart, gcount = offsets(self.gen_edges_by_pt)
        rstart, rcount = offsets(self.reco_edges_by_pt)

        rows, g_gen, g_reco = [], [], []
        g_sm_resid, g_sm_var = [], []   # gen-rebinned smeared accumulators
        for i in self._reported_pt_indices():
            if self.pt_edges[i] < 200:
                continue
            gidx = list(range(gstart[i], gstart[i] + gcount[i]))
            ridx = list(range(rstart[i], rstart[i] + rcount[i]))
            if min_edge is not None:
                floor_i = (min_edge[i] if isinstance(min_edge, (list, tuple, np.ndarray))
                           else min_edge)
                gedges = np.asarray(self.gen_edges_by_pt[i], float)
                redges = np.asarray(self.reco_edges_by_pt[i], float)
                gidx = [gstart[i] + j for j in range(gcount[i])
                        if gedges[j] >= floor_i - 1e-9]
                ridx = [rstart[i] + j for j in range(rcount[i])
                        if redges[j] >= floor_i - 1e-9]
                if not gidx or not ridx:
                    continue
            hi = int(self.pt_edges[i + 1]) if i + 1 < len(self.pt_edges) - 1 else None
            cov_unf = cov_x[np.ix_(gidx, gidx)]
            unf_stat = self._chi2_from_covariance(
                truth_resid[gidx], cov_unf, ndof=self._effective_ndof(ridx))
            model_cov = self._model_cov_unfolded(gidx)
            unf = self._chi2_from_covariance(
                truth_resid[gidx], cov_unf + model_cov,
                ndof=self._effective_ndof(ridx),
            ) if model_cov is not None else unf_stat
            if smeared_binning == "gen":
                r_full = slice(rstart[i], rstart[i] + rcount[i])
                resid_g = self._reco_to_gen_rebin_1d(
                    reco_resid[r_full], self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                var_g = self._reco_to_gen_rebin_1d(
                    var_y[r_full], self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                gen_local = [gi - gstart[i] for gi in gidx]
                sm_resid_i = resid_g[gen_local]
                sm_var_i = var_g[gen_local]
                smeared_row = self._chi2_from_covariance(sm_resid_i, np.diag(sm_var_i))
                g_sm_resid.append(sm_resid_i)
                g_sm_var.append(sm_var_i)
            else:
                smeared_row = self._chi2_from_covariance(reco_resid[ridx], np.diag(var_y[ridx]))
            rows.append({
                "i": i, "pt_lo": int(self.pt_edges[i]), "pt_hi": hi,
                "smeared": smeared_row,
                "unfolded": unf,
                "unfolded_stat_only": unf_stat,
            })
            g_gen.extend(gidx)
            g_reco.extend(ridx)
        if not rows:
            return [], None
        g_model_cov = self._model_cov_unfolded(g_gen)
        g_cov = cov_x[np.ix_(g_gen, g_gen)]
        g_unf_stat = self._chi2_from_covariance(
            truth_resid[g_gen], g_cov, ndof=self._effective_ndof(g_reco))
        if smeared_binning == "gen":
            g_sm_resid = np.concatenate(g_sm_resid)
            g_sm_var = np.concatenate(g_sm_var)
            glob_smeared = self._chi2_from_covariance(g_sm_resid, np.diag(g_sm_var))
        else:
            glob_smeared = self._chi2_from_covariance(reco_resid[g_reco], np.diag(var_y[g_reco]))
        glob = {
            "smeared": glob_smeared,
            "unfolded": (self._chi2_from_covariance(
                truth_resid[g_gen], g_cov + g_model_cov,
                ndof=self._effective_ndof(g_reco))
                if g_model_cov is not None else g_unf_stat),
            "unfolded_stat_only": g_unf_stat,
            "model_uncertainty_included": g_model_cov is not None,
        }
        return rows, glob

    @staticmethod
    def _fmt_ndof(ndof):
        """Integer bin-count ndof prints cleanly; effective (float) ndof to .1f."""
        if ndof is None or not np.isfinite(ndof):
            return "-"
        return f"{ndof:.0f}" if abs(ndof - round(ndof)) < 1e-6 else f"{ndof:.1f}"

    def _bottom_line_panel_text(self, slice_row, global_row):
        """Annotation text: the global chi2/ndof (the headline number the ARC
        asked for) plus this slice's chi2/ndof, evaluated over the shown space."""
        def rndf(metric):
            n = metric.get("ndof", float("nan"))
            return metric["chi2"] / n if n and np.isfinite(n) and n > 0 else float("nan")

        with_model = bool((global_row or slice_row or {}).get(
            "model_uncertainty_included", False))
        unc = "stat + model" if with_model else "data stat"
        lines = []
        if global_row is not None:
            gsm, gun = global_row["smeared"], global_row["unfolded"]
            grsm, grun = rndf(gsm), rndf(gun)
            gok = grun <= grsm + 1e-9
            lines += [
                fr"Global $\chi^2/n_\mathrm{{dof}}$ ({unc}, shown range):",
                fr"  smeared $={grsm:.2f}$   unfolded $={grun:.2f}$  "
                + (r"$\checkmark$" if gok else r"$\times$"),
            ]
        if slice_row is not None:
            sm, un = slice_row["smeared"], slice_row["unfolded"]
            rsm, run_ = rndf(sm), rndf(un)
            lines += [
                "",
                fr"This slice: $\chi^2/n_\mathrm{{dof}}$ "
                fr"smear $={rsm:.2f}$, unfold $={run_:.2f}$",
                fr"  ($\chi^2_\mathrm{{unf}}={un['chi2']:.0f}/{self._fmt_ndof(un['ndof'])}$, "
                fr"$\chi^2_\mathrm{{sm}}={sm['chi2']:.0f}/{self._fmt_ndof(sm['ndof'])}$)",
            ]
        return "\n".join(lines)

    def _store_unfold_result(self, systematic, do_jk, jk_target, unfold, h_meas, h_true):
        h_unfold = unfold.GetOutput("unfold")
        h_folded = unfold.GetFoldedOutput("folded")

        y_meas, ye_meas = self._th1_to_arrays(h_meas)
        y_true, ye_true = self._th1_to_arrays(h_true)
        x_folded, _ = self._th1_to_arrays(h_folded)
        y_unf, ye_unf = self._th1_to_arrays(h_unfold)

        if do_jk and systematic == "nominal":
            if jk_target == "matrix":
                self.y_unf_jk_matrix_list.append(y_unf)
            else:
                self.y_unf_jk_input_list.append(y_unf)
            return

        if systematic == "nominal":
            self.y_meas = y_meas
            self.ye_meas = ye_meas
            self.y_unf = y_unf
            self.ye_unf = ye_unf
            self.y_true = y_true
            self.x_folded = x_folded
            self.L = unfold.GetL("Lmatrix", "Lmatrix")
        else:
            self.y_unf_dict[systematic] = y_unf
            self.ye_unf_dict[systematic] = ye_unf

    def _perform_unfold(self, systematic = 'nominal', closure = False, herwig_closure = False, meas_flat = None, do_jk = False, resp_np = None, jk_target = "input", meas_var = None, true_flat_override = None):
        if getattr(self, "method", "tunfold") == "roounfold_bayes":
            return self._perform_unfold_bayes(
                systematic=systematic, closure=closure, herwig_closure=herwig_closure,
                meas_flat=meas_flat, do_jk=do_jk, resp_np=resp_np, jk_target=jk_target,
            )
        uses_default_measurement = meas_flat is None and not closure and not herwig_closure
        uses_stored_matrix = resp_np is None
        if resp_np is None:
            resp_np = self.mosaic_dict[systematic]
        meas_flat = self._select_measured_spectrum(closure, herwig_closure, meas_flat)
        meas_flat = self._apply_background_subtraction(meas_flat, closure, herwig_closure)
        meas_flat = self._apply_fake_correction(meas_flat, systematic, closure, herwig_closure)

        # The truth prior doubles as the regularization bias (ratio-curvature
        # conditions + bookkeeping). A caller unfolding through a reweighted
        # response passes the matching reweighted truth so the bias is consistent.
        if true_flat_override is not None:
            true_flat = np.asarray(true_flat_override, dtype=float)
        else:
            true_flat = self.mosaic.sum(axis = 0) + self.misses_2d
        n_reco, n_true = resp_np.shape
        assert len(meas_flat) == n_reco, "measured spectrum must have n_reco bins"
        truth_root, reco_root = self._build_root_binning()
        h_meas = reco_root.CreateHistogram("hRecoData")
        h_true = truth_root.CreateHistogram("hTruthPrior")
        h_resp = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(truth_root, reco_root, "hResponse")

        if systematic in {"herwigUp", "herwigDown"}:
            misses = self.misses_2d_herwig
        else:
            # Per-systematic misses keep the efficiency consistent with the varied
            # response matrix; falls back to nominal when not available.
            misses = getattr(self, "misses_2d_dict", {}).get(systematic, self.misses_2d)
        # Proper MC-stat errors (sumw2 from the pkl hists) exist for the
        # stored nominal matrix; JK replica matrices (resp_np override) and
        # systematic variations fall back to the previous behavior.
        resp_var = (
            getattr(self, "mosaic_var_dict", {}).get(systematic)
            if uses_stored_matrix
            else None
        )
        misses_var = (
            getattr(self, "misses_var_dict", {}).get(systematic)
            if uses_stored_matrix
            else None
        )
        self._fill_response_histogram(
            h_resp,
            resp_np,
            misses,
            include_bin_errors=getattr(
                self,
                "response_matrix_stat_available",
                True,
            ),
            resp_var=resp_var,
            misses_var=misses_var,
        )
        measured_variances = (
            self.measured_variances
            if uses_default_measurement and hasattr(self, "measured_variances")
            else None
        )
        if measured_variances is not None:
            # Same order as the central value: the MC-stat variance of the
            # subtracted background adds to the data variance, then both are
            # scaled by the fake survival factor.
            measured_variances = self._add_background_variance(
                measured_variances, closure, herwig_closure
            )
            fake_survival = 1.0 - np.asarray(self.fake_fraction_2d, dtype=float)
            measured_variances = measured_variances * np.square(fake_survival)
            if systematic == "nominal":
                self.corrected_measured_variances = np.array(
                    measured_variances,
                    copy=True,
                )
        # Explicit measured variance (e.g. unfolding the HERWIG sample with its
        # own MC-stat in herwig_closure mode, where the default path feeds none).
        # Caller supplies the variance already matched to ``meas_flat``; the fake
        # correction (when applicable) is applied here for consistency with data.
        if meas_var is not None:
            mv = np.asarray(meas_var, dtype=float)
            if not (closure or herwig_closure):
                mv = self._add_background_variance(mv, closure, herwig_closure)
                mv = mv * np.square(1.0 - np.asarray(self.fake_fraction_2d, dtype=float))
            measured_variances = mv
            if systematic == "nominal":
                self.corrected_measured_variances = np.array(mv, copy=True)
        self._fill_root_histogram(h_meas, meas_flat, measured_variances)
        self._fill_root_histogram(h_true, true_flat)
        self.h_resp = h_resp

        # Area constraint: kEConstraintArea (default) keeps one global area
        # constraint; kEConstraintNone disables it (plain TUnfold config).
        e_constraint = (
            ROOT.TUnfold.kEConstraintArea
            if getattr(self.spec, "area_constraint", True)
            else ROOT.TUnfold.kEConstraintNone
        )

        if self.regularization == "ratio_curvature":
            _declare_open_l()
            unfold = ROOT.TUnfoldDensityOpenL(
                h_resp,
                ROOT.TUnfold.kHistMapOutputHoriz,
                ROOT.TUnfold.kRegModeNone,             # L built by hand below
                e_constraint,
                ROOT.TUnfoldDensity.kDensityModeBinWidth,
                truth_root,
                reco_root,
            )
            # The L matrix uses the *nominal* truth prior for every systematic
            # and jackknife variation: the regularization is part of the
            # measurement definition and must not vary with the response.
            self._add_ratio_curvature_conditions(unfold, true_flat)
        else:
            unfold = ROOT.TUnfoldDensity(
                h_resp,
                ROOT.TUnfold.kHistMapOutputHoriz,          # mapping of TH2 axes
                ROOT.TUnfold.kRegModeDerivative,            # curvature regularisation
                e_constraint,                              # area constraint (see above)
                ROOT.TUnfoldDensity.kDensityModeBinWidth,  # bin-width aware scaling
                truth_root,                              # output (truth) binning tree
                reco_root,                               # input  (reco)  binning tree
                "signal",                                # regularisationDistributionName
                "*[UOB]"                                 # regularisationAxisSteering
            )

        # feed measured spectrum
        status = unfold.SetInput(h_meas)
        if status >= 10000:
            raise RuntimeError("TUnfold input had overflow/underflow – check your hist.")
        if self.regularization == "none":
            unfold.DoUnfold(0.0) # No regularization
        elif self.tau is None:
            # Nominal data unfold (always the first call): scan once, then
            # freeze tau for every systematic / jackknife re-unfold.
            _declare_open_l()
            scan = ROOT.RunUnfoldLcurveScan(unfold, 40)
            self.tau = float(unfold.GetTau())
            x, y = _graph_to_arrays(scan.lcurve)
            self.lcurve_scan = {
                "x": x,
                "y": y,
                "best_x": np.log10(max(unfold.GetChi2A(), 1e-300)),
                "best_y": np.log10(max(unfold.GetChi2L() / max(self.tau, 1e-300) ** 2, 1e-300)),
                "tau": self.tau,
            }
            print(f"L-curve scan: tau = {self.tau:.4g}")
        else:
            unfold.DoUnfold(self.tau)
        # JK replica unfolds also run with systematic == "nominal"; without
        # this guard the last replica (a 90% matrix with no sumw2 errors)
        # silently overwrites the nominal covariances.
        if not do_jk:
            self._store_covariances(unfold, systematic)
        self._store_unfold_result(systematic, do_jk, jk_target, unfold, h_meas, h_true)


    def _perform_unfold_bayes(self, systematic="nominal", closure=False, herwig_closure=False,
                              meas_flat=None, do_jk=False, resp_np=None, jk_target="input"):
        """RooUnfoldBayes (D'Agostini) backend, dispatched from _perform_unfold.

        Mirrors the TUnfold path's input prep (fake-corrected measured spectrum,
        truth-with-misses prior) and result storage, but unfolds with iterative
        Bayes. The jackknife and systematics loops call _perform_unfold
        unchanged, so the same replicas/variations flow through here -- the
        statistical uncertainty stays jackknife-based.
        """
        from .roounfold_backend import bayes_unfold

        uses_default_measurement = meas_flat is None and not closure and not herwig_closure
        if resp_np is None:
            resp_np = self.mosaic_dict[systematic]
        resp_np = np.asarray(resp_np, dtype=float)
        meas_flat = self._select_measured_spectrum(closure, herwig_closure, meas_flat)
        meas_flat = self._apply_fake_correction(meas_flat, systematic, closure, herwig_closure)

        if systematic in {"herwigUp", "herwigDown"}:
            misses = self.misses_2d_herwig
        else:
            misses = getattr(self, "misses_2d_dict", {}).get(systematic, self.misses_2d)
        truth_flat = resp_np.sum(axis=0) + np.asarray(misses, dtype=float)

        want_cov = (not do_jk) and systematic == "nominal" and not herwig_closure
        out = bayes_unfold(resp_np, meas_flat, truth_flat, n_iter=self.n_iter,
                           with_covariance=want_cov, tag=str(systematic))
        if want_cov:
            y_unf, ye_unf, cov = out
        else:
            y_unf, ye_unf = out
            cov = None

        # Jackknife replicas only need the central unfolded vector.
        if do_jk and systematic == "nominal":
            if jk_target == "matrix":
                self.y_unf_jk_matrix_list.append(y_unf)
            else:
                self.y_unf_jk_input_list.append(y_unf)
            return

        if systematic != "nominal":
            self.y_unf_dict[systematic] = y_unf
            if systematic == "herwigUp":
                self.cov_data_herwig_np = np.diag(ye_unf ** 2)
            return

        # Nominal (or closure/herwig-closure): populate the same attributes the
        # TUnfold store path sets. Folded prediction = column-normalized response
        # (including efficiency) applied to the unfolded truth.
        with np.errstate(divide="ignore", invalid="ignore"):
            col = np.divide(resp_np, truth_flat[None, :],
                            out=np.zeros_like(resp_np), where=truth_flat[None, :] != 0)
        self.y_meas = np.asarray(meas_flat, dtype=float)
        self.ye_meas = np.sqrt(np.abs(self.y_meas))
        self.y_unf = y_unf
        self.ye_unf = ye_unf
        self.y_true = truth_flat
        self.x_folded = col @ y_unf
        self.L = None
        # Fake-corrected measured variances (consumed by the dijet artifact
        # writer and normalized-covariance code), mirroring the TUnfold path.
        if uses_default_measurement and hasattr(self, "measured_variances"):
            fake_survival = 1.0 - np.asarray(self.fake_fraction_2d, dtype=float)
            self.corrected_measured_variances = (
                self._add_background_variance(
                    np.asarray(self.measured_variances, dtype=float),
                    closure,
                    herwig_closure,
                )
                * np.square(fake_survival)
            )
        if cov is None:
            cov = np.diag(ye_unf ** 2)
        self.cov_np = cov
        self.cov_uncorr_np = cov.copy()
        self.cov_data_np = cov.copy()

    def plot_L(self, show=True):
        lMatrix = self.L
        #try plotting the L matrix root way
        c = ROOT.TCanvas("c", "L-curve Matrix", 800, 600)
        lMatrix.Draw("colz")
        l_root_path = self._relocate_output("L_matrix_root.png")
        l_root_path.parent.mkdir(parents=True, exist_ok=True)
        c.SaveAs(str(l_root_path))
        nx, ny = lMatrix.GetNbinsX(), lMatrix.GetNbinsY() 
        l_np = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                l_np[i, j] = lMatrix.GetBinContent(i + 1, j + 1)
        #mask zeros for better visualization
        l_np_masked = np.ma.masked_where(l_np == 0, l_np)
        plt.imshow(l_np_masked, origin='lower', aspect='auto')
        plt.colorbar(label='L-curve Matrix Value')
        hep.cms.label(self.cms_label, data = False, lumi = 138 , com = 13, fontsize = 20)
        self._finalize_plot(save_path=f"{self.spec.output_dir}unfold/L_matrix_matplotlib.png", show=show)
        
        


    
    def _th1_to_arrays(self,h):
        nb = h.GetNbinsX()                       # bin numbers
        x  = np.arange(1, nb + 1)
        y  = np.array([h.GetBinContent(int(i)) for i in x])
        ye = np.array([h.GetBinError(int(i))   for i in x])
        return  y, ye
    def plot_folded(self, show=True):
        folded_pt_binned = unflatten_gen_by_pt(self.x_folded, self.reco_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.reco_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.reco_edges_by_pt)
        for i in self._reported_pt_indices():
            bin_widths_reco = np.diff(self.reco_edges_by_pt[i])
            # two-panel plot: main + ratio
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            plt.sca(ax_top)
            hep.histplot(folded_pt_binned[i]/bin_widths_reco/folded_pt_binned[i].sum(), self.reco_edges_by_pt[i], label='Folded', color='#e42536', alpha=0.8, ls='dotted', lw=3, ax=ax_top)
            hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.reco_edges_by_pt[i], color='k', ls='--', alpha=1, label='Measured Data', ax=ax_top)

            # ratio (Measured / Folded)
            edges = np.array(self.reco_edges_by_pt[i], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            folded = folded_pt_binned[i]/bin_widths_reco/folded_pt_binned[i].sum()
            meas = measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum()
            ratio = np.divide(meas, folded, out=np.full_like(meas, np.nan), where=folded != 0)

            plt.sca(ax_bot)
            ax_bot.axhline(1.0, color='gray', ls='--')
            ax_bot.errorbar(centers, ratio, yerr=None, fmt='o', color='k')
            ax_bot.set_ylabel('Data / Folded')
            ax_bot.set_xlim(edges[0], edges[-1])
            ax_bot.set_ylim(0.5, 1.5)
            plt.xlabel(self._observable_label())

            # switch back to top axes so subsequent plotting (reco_mc, legend, labels) goes to the main panel
            plt.sca(ax_top)
            #hep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.reco_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title) 
            
            if self.groomed:
                #plt.xlim(0,250)
                plt.xlim(*self._observable_xlim(i))
                hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(*self._observable_xlim(i))
                hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
            save_path = f"./{self.spec.output_dir}unfold/folded_groomed_{i-1}.pdf" if self.groomed else f"./{self.spec.output_dir}unfold/folded_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
    
    def plot_jk(self, show= True):
        # Outputs
        n_pt_bins = len(self.pt_edges) - 1
        jk_pt_binned = [
            unflatten_gen_by_pt(sample, self.gen_edges_by_pt)
            for sample in self.y_unf_jk_input_list
        ]

        for pt_index in range(n_pt_bins):
            fig, ax = plt.subplots()
            for jk_index, unfolded_pt_binned in enumerate(jk_pt_binned):
                ax.stairs(
                    unfolded_pt_binned[pt_index],
                    self.gen_edges_by_pt[pt_index],
                    label=f"JK sample {jk_index}",
                    alpha=0.6,
                )

            pt_low = int(self.pt_edges[pt_index])
            pt_high = self.pt_edges[pt_index + 1]
            pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

            ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
            
            ax.set_ylabel("Unfolded entries")
            if self.groomed:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./{self.spec.output_dir}unfold/jk_outputs_groomed_pt{pt_index-1}.pdf"
            else:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./{self.spec.output_dir}unfold/jk_outputs_ungroomed_pt{pt_index-1}.pdf"

            plt.sca(ax)
            hep.cms.label(self.cms_label, data=True, lumi=138, com=13, fontsize=20)
            plt.tight_layout()
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        # Inputs
        for pt_index in range(n_pt_bins):
            fig, ax = plt.subplots()
            for jk_index, mosaic_2d_jk in enumerate(self.mosaic_2d_jk_list):
                reco_pt_binned_jk = unflatten_gen_by_pt(self.mosaic_2d_jk_list[jk_index], self.reco_edges_by_pt)
                ax.stairs(
                    reco_pt_binned_jk[pt_index],
                    self.reco_edges_by_pt[pt_index],
                    label=f"JK sample {jk_index}",
                    alpha=0.6,
                )

            pt_low = int(self.pt_edges[pt_index])
            pt_high = self.pt_edges[pt_index + 1]
            pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

            ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
            
            ax.set_ylabel("Entries")
            if self.groomed:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./{self.spec.output_dir}unfold/jk_inputs_groomed_pt{pt_index-1}.pdf"
            else:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./{self.spec.output_dir}unfold/jk_inputs_ungroomed_pt{pt_index-1}.pdf"

            plt.sca(ax)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20)
            self._finalize_plot(save_path=save_path, show=show, fig=fig)



    def _model_reco_signed_shifts(self):
        """Signed per-source model fracs on the normalized reco-level MC.

        Column-scaled Vincia/CR/frag responses projected to reco (see
        ``model_envelope.compute_model_reco_shifts``) plus the FSR PSWeight
        response projections from ``mosaic_dict``. Returns
        ``{source: {i: signed frac array on reco binning}}``, or None when the
        spec does not use the model envelope. Cached after the first call.
        """
        if not getattr(self.spec, "model_envelope", False):
            return None
        cached = getattr(self, "_model_reco_shifts_cache", None)
        if cached is not None:
            return cached
        from unfold.tools.model_envelope import compute_model_reco_shifts
        print("Computing reco-level model shifts (column-scaled projections)...")
        shifts = compute_model_reco_shifts(self)
        # Reco-level alternate-model and shower-scale responses. FSR and ISR are
        # the stored PSWeight variations; herwigUp is the HERWIG response mosaic
        # registered by _prepare_herwig_inputs. These are analysis mosaics, so
        # the baseline is the analysis nominal. Used by the bottom-line test,
        # which draws its band from the HERWIG difference and the ISR/FSR
        # envelope rather than from the gen-level PS/HAD envelope.
        nom_proj = unflatten_gen_by_pt(
            self.mosaic_dict["nominal"].sum(axis=1), self.reco_edges_by_pt)
        for var in ("fsrUp", "fsrDown", "isrUp", "isrDown"):
            mosaic = self.mosaic_dict.get(var)
            if mosaic is None:
                continue
            var_proj = unflatten_gen_by_pt(
                np.asarray(mosaic, float).sum(axis=1), self.reco_edges_by_pt)
            shifts[var] = {}
            for i in range(len(self.reco_edges_by_pt)):
                nom = nom_proj[i] / max(nom_proj[i].sum(), 1e-300)
                alt = var_proj[i] / max(var_proj[i].sum(), 1e-300)
                shifts[var][i] = np.divide(
                    alt, nom, out=np.ones_like(nom), where=nom != 0) - 1.0
        self._model_reco_shifts_cache = shifts
        return shifts

    def plot_bottom_line(self, show=True, rebin_reco_to_gen=False,
                         annotate_chi2=True, include_herwig_band=False,
                         ratio_ylim=(0.0, 2.0),
                         show_isr_fsr_envelope=False):
        # This remains a visual diagnostic.  The quantitative test is evaluated
        # in the full unnormalised space below and printed on every panel so the
        # plot cannot be mistaken for a bin-by-bin chi-square test.
        # Evaluate the test only over the SHOWN space (each slice up to the rho
        # value it is cut at for display), matching what the panel displays.
        # ``rebin_reco_to_gen`` puts the detector level on the gen binning (the
        # ARC's alternative: square K, so chi2_smeared and chi2_unfold are
        # directly comparable). Both ratio curves then share the gen axis.
        hep.style.use("CMS")
        smeared_binning = "gen" if rebin_reco_to_gen else "reco"
        try:
            bl_rows, bl_global = self.bottom_line_test_by_pt(
                min_edge="shown", smeared_binning=smeared_binning)
        except RuntimeError:
            bl_rows, bl_global = [], None
        bl_by_i = {r["i"]: r for r in bl_rows}

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.gen_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.gen_edges_by_pt)

        model_reco_shifts = self._model_reco_signed_shifts()

        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.reco_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.reco_edges_by_pt)

        # Optional: reco-level HERWIG spectrum per pT bin, projected exactly like
        # the input-distribution overlay (same reco binning as reco_mc). Used to
        # add the reco-level HERWIG-vs-PYTHIA difference to the model band.
        herwig_reco_pt_binned = None
        if include_herwig_band and getattr(self, "herwig_2d", None) is not None:
            _hh = self.herwig_2d
            if "systematic" in _hh.axes.name:
                _hh = _hh.project("ptreco", self.reco_axis, "systematic")[:, :, "nominal"]
            else:
                _hh = _hh.project("ptreco", self.reco_axis)
            _hpt = np.asarray(_hh.axes['ptreco'].edges, float)
            _hvals = np.asarray(_hh.values(), float)
            _hedges = np.asarray(_hh.axes[self.reco_axis].edges, float)
            _pt_edges = np.asarray(self.bins.pt_edges, float)
            herwig_reco_pt_binned = []
            for _k in range(len(_pt_edges) - 1):
                _a = int(np.searchsorted(_hpt, _pt_edges[_k] - 1e-9))
                _b = (len(_hpt) - 1 if not np.isfinite(_pt_edges[_k + 1])
                      else int(np.searchsorted(_hpt, _pt_edges[_k + 1] - 1e-9)))
                _global = _hvals[_a:_b, :].sum(axis=0)
                # Map the global fine reco axis onto this pT slice's reco binning
                # (identity for groomed where they coincide; a coarsening for the
                # nested ungroomed per-pT binning) so it matches reco_mc.
                herwig_reco_pt_binned.append(self._reco_to_gen_rebin_1d(
                    _global, _hedges, self.reco_edges_by_pt[_k]))

        #now plot the ratio of unfolded to true and measured to reco mc in the same axis, just the ratio plot (no main panel)
        gen_offset = sum(
            len(edges) - 1
            for edges in self.gen_edges_by_pt[:getattr(self, "first_reported_pt_bin", 0)]
        )
        for i in self._reported_pt_indices():
            n_gen_bins = len(self.gen_edges_by_pt[i]) - 1
            gen_slice = slice(gen_offset, gen_offset + n_gen_bins)
            gen_offset += n_gen_bins
            fig, ax = plt.subplots(figsize=(12, 9))
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            unfolded = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
            true = true_pt_binned[i]/bin_widths/true_pt_binned[i].sum()
            
            ratio_unf_true = np.divide(unfolded, true, out=np.full_like(unfolded, np.nan), where=true != 0)

            # Propagate the data-stat covariance through the same per-pT
            # normalisation used for the displayed ratio.  This is deliberately
            # not the total uncertainty band: the bottom-line test is a
            # data-stat-only diagnostic at this stage.
            ratio_unf_true_err = None
            covariance = getattr(self, "cov_data_np", None)
            if covariance is not None:
                raw_unfolded = np.asarray(self.y_unf[gen_slice], dtype=float)
                total_unfolded = raw_unfolded.sum()
                if total_unfolded > 0:
                    normalisation_jacobian = (
                        np.diag(1.0 / (bin_widths * total_unfolded))
                        - np.outer(
                            raw_unfolded / (bin_widths * total_unfolded**2),
                            np.ones(n_gen_bins),
                        )
                    )
                    normalised_covariance = (
                        normalisation_jacobian
                        @ np.asarray(covariance)[gen_slice, gen_slice]
                        @ normalisation_jacobian.T
                    )
                    ratio_unf_true_err = np.divide(
                        np.sqrt(np.clip(np.diag(normalised_covariance), 0.0, None)),
                        np.abs(true),
                        out=np.zeros_like(true),
                        where=true != 0,
                    )
            
            if rebin_reco_to_gen:
                # Detector level rebinned onto the gen binning: both curves share
                # the gen axis and the smeared side is a square-K comparison.
                meas_counts = self._reco_to_gen_rebin_1d(
                    measured_pt_binned[i], self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                recomc_counts = self._reco_to_gen_rebin_1d(
                    reco_mc_pt_binned[i], self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                measured = meas_counts / bin_widths / meas_counts.sum()
                reco_mc = recomc_counts / bin_widths / recomc_counts.sum()
                meas_edges = self.gen_edges_by_pt[i]
            else:
                bin_widths_reco = np.diff(self.reco_edges_by_pt[i])
                measured = measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum()
                reco_mc = reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum()
                meas_edges = self.reco_edges_by_pt[i]
            ratio_meas_reco = np.divide(measured, reco_mc, out=np.full_like(measured, np.nan), where=reco_mc != 0)

            # Model uncertainty on the Reco_MC denominator. This is a
            # reconstruction-level band and is deliberately NOT the gen-level
            # PS/HAD envelope used for the unfolded result: it is the envelope
            # of the ISR and FSR shower-scale PSWeight variations, taken
            # directly at reco level.
            model_band = None
            if model_reco_shifts is not None:
                def _frac(source):
                    frac = model_reco_shifts.get(source, {}).get(i)
                    if frac is None:
                        return np.zeros(len(meas_edges) - 1)
                    if rebin_reco_to_gen:
                        # Rebin the signed shift as a nominal-count-weighted
                        # mean: rebin nom*(1+frac) against rebinned nom.
                        nom_reb = self._reco_to_gen_rebin_1d(
                            reco_mc_pt_binned[i],
                            self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                        var_reb = self._reco_to_gen_rebin_1d(
                            reco_mc_pt_binned[i] * (1.0 + frac),
                            self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                        return np.divide(
                            var_reb, nom_reb,
                            out=np.ones_like(nom_reb), where=nom_reb != 0) - 1.0
                    return frac
                isr_frac = np.maximum(
                    np.abs(_frac("isrUp")), np.abs(_frac("isrDown")))
                fsr_frac = np.maximum(
                    np.abs(_frac("fsrUp")), np.abs(_frac("fsrDown")))
                band_frac = np.maximum(isr_frac, fsr_frac)
                herwig_frac = None
                if include_herwig_band and herwig_reco_pt_binned is not None:
                    her = herwig_reco_pt_binned[i]
                    pyt = reco_mc_pt_binned[i]
                    hn = her / max(her.sum(), 1e-300)
                    pn = pyt / max(pyt.sum(), 1e-300)
                    hfrac_reco = np.divide(hn, pn, out=np.ones_like(pn),
                                           where=pn != 0) - 1.0
                    if rebin_reco_to_gen:
                        nom_reb = self._reco_to_gen_rebin_1d(
                            pyt, self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                        var_reb = self._reco_to_gen_rebin_1d(
                            pyt * (1.0 + hfrac_reco),
                            self.reco_edges_by_pt[i], self.gen_edges_by_pt[i])
                        herwig_frac = np.divide(
                            var_reb, nom_reb, out=np.ones_like(nom_reb),
                            where=nom_reb != 0) - 1.0
                    else:
                        herwig_frac = hfrac_reco
                    band_frac = np.maximum(band_frac, np.abs(herwig_frac))
                model_band = band_frac * np.abs(ratio_meas_reco)
                print(f"  BLT band pt{i}: max|ISR|={isr_frac.max():.3f} "
                      f"max|FSR|={fsr_frac.max():.3f}"
                      + (f" max|HERWIG|={np.abs(herwig_frac).max():.3f}"
                         if herwig_frac is not None else ""))

            # Data statistical uncertainty on the normalized Measured/Reco_MC
            # ratio: Poisson on the reported data counts propagated through the
            # per-slice unit-area normalization, rel = sqrt((1 - p) / N).
            if rebin_reco_to_gen:
                raw_meas = self._reco_to_gen_rebin_1d(
                    measured_pt_binned[i], self.reco_edges_by_pt[i],
                    self.gen_edges_by_pt[i])
            else:
                raw_meas = np.asarray(measured_pt_binned[i], float)
            _Mtot = raw_meas.sum()
            _p = np.divide(raw_meas, _Mtot, out=np.zeros_like(raw_meas), where=_Mtot > 0)
            stat_frac = np.sqrt(np.clip(np.divide(
                1.0 - _p, raw_meas, out=np.zeros_like(raw_meas), where=raw_meas > 0),
                0.0, None))
            stat_band = stat_frac * np.abs(ratio_meas_reco)

            ax.axhline(1.0, color='gray', ls='--')
            x_steps = np.repeat(np.asarray(meas_edges, float), 2)[1:-1]
            if model_band is not None:
                total_band = np.sqrt(model_band ** 2 + stat_band ** 2)
                band_label = ('Model $\\oplus$ stat unc. (Reco MC, incl. HERWIG)'
                              if include_herwig_band and herwig_frac is not None
                              else 'Model $\\oplus$ stat unc. (Reco MC)')
            else:
                total_band = stat_band
                band_label = 'Stat unc. (Reco MC)'
            ax.fill_between(
                x_steps,
                np.repeat(ratio_meas_reco - total_band, 2),
                np.repeat(ratio_meas_reco + total_band, 2),
                color='#e42536', alpha=0.20, linewidth=0, label=band_label)
            # Shower-scale diagnostics are separate from the filled total
            # band.  The band takes their bin-wise maximum, while the two
            # outlines retain the individual ISR and FSR excursions.  This is
            # optional because the ISR weight is not result-ready.
            if model_band is not None and show_isr_fsr_envelope:
                for frac, color, label in (
                    (isr_frac, '#ff7f0e', 'ISR envelope'),
                    (fsr_frac, '#2ca02c', 'FSR envelope'),
                ):
                    shower_band = frac * np.abs(ratio_meas_reco)
                    for sign in (+1, -1):
                        ax.plot(
                            x_steps,
                            np.repeat(ratio_meas_reco + sign * shower_band, 2),
                            color=color, ls='-', lw=1.2, alpha=0.9,
                            label=(label if sign == +1 else None),
                        )
            if model_band is not None and include_herwig_band and herwig_frac is not None:
                herwig_band = np.abs(herwig_frac) * np.abs(ratio_meas_reco)
                for sign in (+1, -1):
                    ax.plot(x_steps, np.repeat(ratio_meas_reco + sign * herwig_band, 2),
                            color='#3f90da', ls='--', lw=1.3,
                            label=('HERWIG reco diff' if sign == +1 else None))
            hep.histplot(ratio_unf_true, self.gen_edges_by_pt[i], yerr=ratio_unf_true_err,
                         label='Unfolded / True', color='k', ls='--')
            reco_label = 'Measured / Reco_MC (rebinned to gen)' if rebin_reco_to_gen else 'Measured / Reco_MC'
            hep.histplot(ratio_meas_reco, meas_edges, yerr=stat_band,
                         label=reco_label, color='#e42536', ls=':')
            ax.set_ylabel('Ratio')
            ax.set_xlim(self.gen_edges_by_pt[i][0], self.gen_edges_by_pt[i][-1])
            # Keep the full ratio excursion visible by default.  Pass ``None``
            # for Matplotlib's automatic limits when inspecting outliers.
            if ratio_ylim is not None:
                ax.set_ylim(*ratio_ylim)
            plt.xlim(*self._observable_xlim(i))
            plt.xlabel(self._observable_label())
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            # Pinned lower-left with an opaque frame (matching the chi2 text
            # box, which owns the top-left): 'best' placement can land on the
            # model band or the chi2 text in the busier panels.
            plt.legend(title=title, loc='lower left', frameon=True,
                       framealpha=0.85, facecolor='white', edgecolor='0.7')
            slice_row = bl_by_i.get(i)
            # The quantitative test lives in the chi2 summary bar charts; the
            # panels stamp it only when asked, so the ratio curves stay readable.
            if annotate_chi2 and (slice_row is not None or bl_global is not None):
                ax.text(
                    0.03,
                    0.97,
                    self._bottom_line_panel_text(slice_row, bl_global),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=PUB_ANNOTATION_FONTSIZE,
                    bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
                )
            hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
            mode = "groomed" if self.groomed else "ungroomed"
            tag = "_rebinned" if rebin_reco_to_gen else ""
            if include_herwig_band and herwig_reco_pt_binned is not None:
                tag += "_herwig"
            save_path = Path(self.spec.output_dir) / f"bottom_line{tag}_{mode}_{i - 1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_bottom_line_chi2_summary(self, show=True, normalized=False,
                                      min_edge=None):
        r"""Grouped bar chart of chi2_smeared vs chi2_unfold per pT slice.

        Direct visual of the bottom-line inequality: the (blue) unfolded bar
        must not exceed the (red) smeared bar. Raw chi2 -- not chi2/ndf -- is the
        quantity the test constrains; the ndf is annotated on each bar. The
        smeared ndf is the reco bin count; the unfolded ndf is the effective
        rank of K J J^T K^T (CMS Stat Committee / TUnfold TWiki), which under
        regularization is below the gen bin count. Data-stat-only, per
        bottom_line_test_by_pt.

        ``normalized=True`` plots chi2/ndf instead -- the quantity the ARC asked
        to compare (they no longer want the absolute chi2). The smeared ndf is
        the reco bin count and the unfolded ndf is the effective rank of
        K J J^T K^T (CMS Stat Committee / TUnfold TWiki); the single global
        chi2/ndf for each is annotated on the plot. The raw-chi2 inequality
        (the underlying bottom-line theorem) is still evaluated and printed to
        the log, but is no longer stamped on the figure.
        """
        rows, glob = self.bottom_line_test_by_pt(min_edge=min_edge)
        if not rows:
            return
        labels, groups = [], []
        for r in rows:
            labels.append(f"{r['pt_lo']}-{r['pt_hi']}" if r["pt_hi"] else f"{r['pt_lo']}-∞")
            groups.append(r)
        labels.append("Global")
        groups.append(glob)

        def val(g, key):
            m = g[key]
            return m["chi2"] / m["ndof"] if normalized and m["ndof"] > 0 else m["chi2"]

        x = np.arange(len(labels))
        w = 0.38
        c_sm = [val(g, "smeared") for g in groups]
        c_unf = [val(g, "unfolded") for g in groups]
        with_model = bool(glob.get("model_uncertainty_included", False))
        unf_label = (r"$\chi^2_\mathrm{unfold}$  (truth space, unfolded"
                     + (r"$\oplus$model" if with_model else "")
                     + " vs PYTHIA gen)")
        fig, ax = plt.subplots(figsize=(12, 9))
        b1 = ax.bar(x - w / 2, c_sm, w, color="#e42536", alpha=0.85,
                    label=r"$\chi^2_\mathrm{smeared}$  (reco space, data vs folded PYTHIA)")
        b2 = ax.bar(x + w / 2, c_unf, w, color="#5790fc", alpha=0.9,
                    label=unf_label)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\chi^2/n_\mathrm{dof}$ vs PYTHIA8" if normalized
                      else r"$\chi^2$ vs PYTHIA8")
        if with_model:
            ax.text(0.03, 0.03,
                    r"smeared: data stat  ·  unfolded: stat$\oplus$model",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=11, color="#555555")
        ax.set_xlabel(r"$p_T$ slice (GeV)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for bars, groups_ in ((b1, groups), (b2, groups)):
            key = "smeared" if bars is b1 else "unfolded"
            for rect, g in zip(bars, groups_):
                note = (fr"{g[key]['chi2']:.0f}/{self._fmt_ndof(g[key]['ndof'])}"
                        if normalized else f"ndf={self._fmt_ndof(g[key]['ndof'])}")
                ax.annotate(note,
                            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            ha="center", va="bottom", fontsize=9)
        # Resolve a human/file label for the cut region.
        is_shown = (min_edge == "shown")
        is_seq = isinstance(min_edge, (list, tuple, np.ndarray))
        if min_edge is None:
            cut_word, cut_tag = "full range", ""
        elif is_shown or is_seq:
            cut_word, cut_tag = "shown range", "_shown"
        else:
            cut_word = rf"$\log_{{10}}(\rho^2)\geq{min_edge:g}$"
            cut_tag = f"_above{abs(min_edge):g}".replace(".", "")

        passed = all(g["unfolded"]["chi2"] <= g["smeared"]["chi2"] + 1e-9 for g in groups)
        mode = "Groomed" if self.groomed else "Ungroomed"
        gsm_r = glob["smeared"]["chi2"] / max(glob["smeared"]["ndof"], 1e-9)
        gun_r = glob["unfolded"]["chi2"] / max(glob["unfolded"]["ndof"], 1e-9)
        print(f"[bottom-line {mode.lower()} ({cut_word})] "
              + ("PASS" if passed else "FAIL")
              + f" | GLOBAL chi2/ndf: smeared={gsm_r:.2f} unfolded={gun_r:.2f}")
        ax.text(0.03, 0.97, f"{mode}, {cut_word}", transform=ax.transAxes,
                ha="left", va="top", fontsize=18)
        # The single global chi2/ndof number the ARC asked to compare.
        ax.text(0.03, 0.88,
                fr"Global $\chi^2/n_\mathrm{{dof}}$:  smeared $={gsm_r:.2f}$,  "
                fr"unfolded $={gun_r:.2f}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=14, color="#333333")
        ax.legend(loc="upper right", fontsize=16)
        ax.set_ylim(top=max(max(c_sm), max(c_unf)) * 4)
        hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
        suffix = "groomed" if self.groomed else "ungroomed"
        stem = f"bottom_line_chi2{'_perndf' if normalized else ''}_summary{cut_tag}_{suffix}.pdf"
        self._finalize_plot(
            save_path=Path(self.spec.output_dir) / stem,
            show=show, fig=fig,
        )

    def _pythia_gen_theory_band(self, i):
        """PYTHIA gen-level theory uncertainty (ISR/FSR/q2/PDF) for pt bin ``i``.

        Returns ``(up_frac, down_frac)`` arrays in the same normalized,
        per-bin-width units as ``normalized_results[i]['true']`` (the plotted
        PYTHIA8 shape). Each theory source is symmetrized into an up/down
        deviation of the *normalized* gen shape and the sources are summed in
        quadrature. Detector systematics are excluded -- they do not change the
        gen-level prediction. Returns zeros when no theory variation is
        available (e.g. do_syst=False, or inputs without varied gen columns).
        """
        theory_bases = ("isr", "fsr", "q2", "pdf")
        mosaic_dict = getattr(self, "mosaic_dict", {})
        misses_dict = getattr(self, "misses_2d_dict", {})
        nominal_misses = getattr(self, "misses_2d", None)
        if "nominal" not in mosaic_dict or nominal_misses is None:
            return None, None

        def norm_shape(full_gen_flat):
            pt_binned = unflatten_gen_by_pt(full_gen_flat, self.gen_edges_by_pt)[i]
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            total = self._shown_norm_total(pt_binned, i)
            if total == 0:
                return np.zeros_like(pt_binned, dtype=float)
            return pt_binned / bin_widths / total

        def full_gen(key):
            return mosaic_dict[key].sum(axis=0) + misses_dict.get(key, nominal_misses)

        nominal = norm_shape(full_gen("nominal"))
        up_sq = np.zeros_like(nominal)
        down_sq = np.zeros_like(nominal)
        found = False
        for base in theory_bases:
            deviations = []
            for key in (f"{base}Up", f"{base}Down"):
                if key in mosaic_dict:
                    deviations.append(norm_shape(full_gen(key)) - nominal)
            if not deviations:
                continue
            found = True
            deviations = np.array(deviations)
            up_sq += np.maximum(np.max(deviations, axis=0), 0.0) ** 2
            down_sq += np.maximum(-np.min(deviations, axis=0), 0.0) ** 2

        if not found:
            return None, None
        return np.sqrt(up_sq), np.sqrt(down_sq)

    def _prediction_uncertainty(self, i, kind="pythia"):
        """Total (up, down) uncertainty on a gen prediction curve in pt bin ``i``.

        Returned in the same normalized, per-bin-width units as the plotted
        PYTHIA8 / HERWIG7 curve, so it can be dropped straight into errorbar().
        PYTHIA = gen theory envelope (ISR/FSR/q2/PDF) ⊕ MC stat; HERWIG = MC
        stat only (no theory-weight variations available yet). Returns
        (None, None) when nothing is available.
        """
        bin_widths = np.diff(self.gen_edges_by_pt[i])

        if kind == "pythia":
            theory_up, theory_down = self._pythia_gen_theory_band(i)
            val_flat = getattr(self, "pythia_gen_val_flat", None)
            var_flat = getattr(self, "pythia_gen_var_flat", None)
        else:
            theory_up = theory_down = None
            val_flat = getattr(self, "herwig_gen_val_flat", None)
            var_flat = getattr(self, "herwig_gen_var_flat", None)

        # MC-stat band in normalized units: per-bin relative stat = sqrt(var)/N,
        # applied to the normalized shape value.
        stat = None
        if val_flat is not None and var_flat is not None:
            counts = unflatten_gen_by_pt(np.asarray(val_flat, float), self.gen_edges_by_pt)[i]
            variance = unflatten_gen_by_pt(np.asarray(var_flat, float), self.gen_edges_by_pt)[i]
            total = self._shown_norm_total(counts, i)
            if total > 0:
                norm_shape = counts / bin_widths / total
                rel = np.divide(
                    np.sqrt(np.clip(variance, 0.0, None)), counts,
                    out=np.zeros_like(norm_shape), where=counts > 0,
                )
                stat = norm_shape * rel

        if theory_up is None and stat is None:
            return None, None
        if theory_up is None:
            return stat, stat
        if stat is None:
            return theory_up, theory_down
        return np.sqrt(theory_up**2 + stat**2), np.sqrt(theory_down**2 + stat**2)

    def _prediction_chi2_covariance(self, i, kind="pythia"):
        """Slice-local covariance of a gen prediction for the quoted chi2.

        PYTHIA: coherent symmetrized ISR/FSR/q2/PDF shift vectors of the
        normalized gen shape (rank-1 each; every variation is separately
        normalized, so the sum-constraint null space is preserved) plus the
        MC-stat diagonal. HERWIG: MC-stat diagonal only -- no theory weights
        are available, an asymmetry the figure caption must state. Returns
        None when nothing is available (the chi2 then uses the measurement
        covariance alone, as for closure runs).
        """
        n_bins = len(self.gen_edges_by_pt[i]) - 1
        covariance = None

        if kind == "pythia":
            theory_bases = ("isr", "fsr", "q2", "pdf")
            mosaic_dict = getattr(self, "mosaic_dict", {})
            misses_dict = getattr(self, "misses_2d_dict", {})
            nominal_misses = getattr(self, "misses_2d", None)
            if "nominal" in mosaic_dict and nominal_misses is not None:
                def norm_shape(key):
                    flat = mosaic_dict[key].sum(axis=0) + misses_dict.get(
                        key, nominal_misses)
                    pt_binned = unflatten_gen_by_pt(flat, self.gen_edges_by_pt)[i]
                    widths = np.diff(self.gen_edges_by_pt[i])
                    total = self._shown_norm_total(pt_binned, i)
                    if total == 0:
                        return np.zeros_like(pt_binned, dtype=float)
                    return pt_binned / widths / total

                nominal = norm_shape("nominal")
                for base in theory_bases:
                    up_key, down_key = f"{base}Up", f"{base}Down"
                    have_up = up_key in mosaic_dict
                    have_down = down_key in mosaic_dict
                    if not (have_up or have_down):
                        continue
                    if have_up and have_down:
                        shift = 0.5 * (norm_shape(up_key) - norm_shape(down_key))
                    else:
                        shift = norm_shape(up_key if have_up else down_key) - nominal
                    if covariance is None:
                        covariance = np.zeros((n_bins, n_bins))
                    covariance += np.outer(shift, shift)

        val_flat = getattr(self, f"{kind}_gen_val_flat", None)
        var_flat = getattr(self, f"{kind}_gen_var_flat", None)
        if val_flat is not None and var_flat is not None:
            counts = unflatten_gen_by_pt(
                np.asarray(val_flat, float), self.gen_edges_by_pt)[i]
            variance = unflatten_gen_by_pt(
                np.asarray(var_flat, float), self.gen_edges_by_pt)[i]
            total = self._shown_norm_total(counts, i)
            if total > 0:
                widths = np.diff(self.gen_edges_by_pt[i])
                shape = counts / widths / total
                rel = np.divide(
                    np.sqrt(np.clip(variance, 0.0, None)), counts,
                    out=np.zeros_like(shape), where=counts > 0,
                )
                if covariance is None:
                    covariance = np.zeros((n_bins, n_bins))
                covariance += np.diag((shape * rel) ** 2)

        return covariance

    def plot_unfolded_fancy(self, log=False, show=True):
        markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
        npt = len(self.pt_edges)-1
        has_herwig = getattr(self, "has_herwig", True)
        # Self-closure figures compare unfolded MC to its own truth: the
        # alternate-generator overlays are noise there, and the points are not
        # data.  Keep the PYTHIA truth curve (its chi2 is the closure metric).
        is_closure = bool(self.closure or self.herwig_closure)
        if is_closure:
            has_herwig = False
        points_label = 'Unfolded' if is_closure else 'Data'
        # Stats-only run (self-closure): the total band would just repeat the
        # stat band, so only the stat band is drawn.
        stat_only = self._band_is_stat_only
        stat_label = "Stat. Unc." if self.response_matrix_stat_available else "Input Stat. Unc."
        total_label = (
            r"Syst. $\oplus$ Stat. Unc."
            if self.response_matrix_stat_available and has_herwig
            else r"Partial Syst. $\oplus$ Stat. Unc."
        )
        title_list = []
        for i in range(npt):
            lo = int(self.pt_edges[i])
            if i + 1 < npt:
                hi = int(self.pt_edges[i + 1])
                title_list.append(rf"${lo} < p_{{\mathrm{{T}}}} < {hi}$ GeV")
            else:
                title_list.append(rf"${lo} < p_{{\mathrm{{T}}}} < \infty$ GeV")
        true_herwig_pt_binned = (
            unflatten_gen_by_pt(self.y_true_herwig, self.gen_edges_by_pt)
            if has_herwig
            else None
        )
        # True standalone-Vincia gen prediction (rho only; None if unavailable).
        vincia_truth = None
        if self.spec.name == "rho" and not is_closure:
            try:
                from unfold.tools.model_envelope import vincia_truth_by_pt
                vincia_truth = vincia_truth_by_pt(self)
            except Exception:
                vincia_truth = None
        for i in self._reported_pt_indices():
            # CMS-default canvas (forced figsize shrinks fonts in the scaled
            # PDF); ratio panel close to the main panel (ARC round-2).
            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07},
            )
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            if has_herwig:
                herwig_norm = true_herwig_pt_binned[i] / bin_widths / self._shown_norm_total(true_herwig_pt_binned[i], i)
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            rho_edges = np.array(self.gen_edges_by_pt[i], dtype=float)
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            plt.sca(ax_main)
            if not stat_only:
                plt.stairs( unfolded + syst_up,
                    self.gen_edges_by_pt[i],
                    baseline = unfolded - syst_down,
                    fill = True, color = "yellowgreen" , label = total_label)
                plt.stairs( unfolded + stat_unc,
                    self.gen_edges_by_pt[i],
                    baseline = unfolded - stat_unc,
                    fill = True, color = "darkgreen" , label = stat_label)
            pythia = np.array(self.normalized_results[i]['true'], dtype=float)
            # Track the tallest drawn curve so the legend headroom clears the
            # predictions too, not just the data band.
            curve_max = float(np.max(unfolded + syst_up))
            curve_max = max(curve_max, float(np.max(pythia)))
            if has_herwig:
                curve_max = max(curve_max, float(np.max(herwig_norm)))
            if vincia_truth is not None:
                curve_max = max(curve_max, float(np.max(vincia_truth[i][0])))
            # Data-vs-MC chi2/ndf per panel using the full covariance of the
            # normalized result (ARC round-3: stat through the normalization
            # Jacobian + rank-1 systematics + projected PS/HAD model term)
            # PLUS each prediction's own uncertainty (Pythia: theory + MC
            # stat; Herwig: MC stat; Vincia: sample stat). The per-pT sum
            # constraint makes the block singular by one, so chi2 uses the
            # pseudo-inverse and ndf = n_shown - 1. Restricted to the shown
            # bins so the quoted agreement covers exactly what the panel
            # displays.
            shown = self._shown_gen_mask(i)
            chi2_offset = sum(
                len(edges) - 1 for edges in self.gen_edges_by_pt[:i]
            )
            chi2_block = np.arange(chi2_offset, chi2_offset + len(shown))
            results_cov = self._results_chi2_covariance()
            def _mc_chi2_label(name, prediction, pred_cov=None):
                sigma = np.maximum(syst_up, syst_down)
                good = (sigma > 0) & np.isfinite(prediction) & shown
                ndf = max(int(good.sum()) - 1, 1)
                idx = chi2_block[good]
                residual = unfolded[good] - prediction[good]
                cov_sub = results_cov[np.ix_(idx, idx)]
                if pred_cov is not None:
                    cov_sub = cov_sub + pred_cov[np.ix_(good, good)]
                chi2 = float(
                    residual @ np.linalg.pinv(cov_sub, rcond=1e-10) @ residual
                )
                return rf"{name} ($\chi^2$/ndf = {chi2:.1f}/{ndf})"
            plt.stairs(pythia, self.gen_edges_by_pt[i],
                       label=_mc_chi2_label(
                           'Pythia8', pythia,
                           self._prediction_chi2_covariance(i, "pythia")),
                       color='#5790fc', ls='dotted', lw=3, baseline=None)
            py_unc_up, py_unc_down = self._prediction_uncertainty(i, "pythia")
            if py_unc_up is not None:
                plt.errorbar(centers, pythia, yerr=[py_unc_down, py_unc_up], fmt='none',
                             ecolor='#5790fc', elinewidth=1.5, capsize=3)
            if has_herwig:
                plt.stairs(herwig_norm, self.gen_edges_by_pt[i],
                           label=_mc_chi2_label(
                               'Herwig7', herwig_norm,
                               self._prediction_chi2_covariance(i, "herwig")),
                           color='#e42536', ls='dashdot', lw=2, baseline=None)
                hw_unc_up, hw_unc_down = self._prediction_uncertainty(i, "herwig")
                if hw_unc_up is not None:
                    plt.errorbar(centers, herwig_norm, yerr=[hw_unc_down, hw_unc_up], fmt='none',
                                 ecolor='#e42536', elinewidth=1.5, capsize=3)
            if vincia_truth is not None:
                vincia_norm, vincia_err = vincia_truth[i]
                plt.stairs(vincia_norm, self.gen_edges_by_pt[i],
                           label=_mc_chi2_label(
                               'Vincia', vincia_norm,
                               np.diag(np.asarray(vincia_err, float) ** 2)),
                           color='#964a8b', ls='dashed', lw=2, baseline=None)
                plt.errorbar(centers, vincia_norm, yerr=vincia_err, fmt='none',
                             ecolor='#964a8b', elinewidth=1.5, capsize=3)
            # ARC round-2: data as black points (not per-panel markers), with
            # vertical (stat) AND horizontal (bin-width) error bars; the
            # errorbar handle carries the bars into the legend.
            plt.errorbar(centers, unfolded, yerr=stat_unc, xerr=np.diff(rho_edges) / 2,
                         fmt='o', color='k', markersize=7, label=points_label)

            # Legend headroom, then the pT-bin info below the legend and
            # left-aligned to it (ARC round-2).
            ax_main.set_ylim(0, 1.9 * curve_max)
            # Approval comments (A. Meyer): Data at the top of the legend.
            handles, labels = ax_main.get_legend_handles_labels()
            order = sorted(range(len(labels)), key=lambda k: labels[k] != points_label)
            legend = ax_main.legend([handles[k] for k in order],
                                    [labels[k] for k in order],
                                    loc="upper right", fontsize=PUB_LEGEND_FONTSIZE)
            # In-frame CMS tag (loc=2), with the pT range right under it
            # (Aritra's preference).
            fig.canvas.draw()
            label_art = hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(),
                                      com=self._com_label(), loc=2, ax=ax_main)
            fig.canvas.draw()
            inv = ax_main.transAxes.inverted()
            boxes = [t.get_window_extent().transformed(inv) for t in label_art
                     if getattr(t, "get_text", lambda: "")()]
            boxes = [b for b in boxes if b.y0 < 1.0]  # skip the lumi text above the frame
            cms_x0 = min(b.x0 for b in boxes)
            cms_y0 = min(b.y0 for b in boxes)
            pt_text = ax_main.text(cms_x0, cms_y0 - 0.04, title_list[i],
                                   transform=ax_main.transAxes, ha="left", va="top",
                                   fontsize=PUB_ANNOTATION_FONTSIZE)
            # Final headroom: every curve stays below the CMS+pT block and the
            # legend (all live in axes fractions, unaffected by ylim).
            fig.canvas.draw()
            tbox = pt_text.get_window_extent().transformed(inv)
            lbox = legend.get_window_extent().transformed(inv)
            clearance = min(tbox.y0, lbox.y0)
            ax_main.set_ylim(0, curve_max / max(0.35, clearance - 0.04))
            plt.ylabel(self._normalized_ylabel(), fontsize=PUB_LABEL_FONTSIZE)
            ax_main.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)

            # Ratio Plot
            plt.sca(ax_ratio)
            plt.axhline(1.0, color='gray', ls='--')
            ratio_pythia = np.divide(unfolded, self.normalized_results[i]['true'])
            stat_frac = np.divide(stat_unc, unfolded, out=np.zeros_like(stat_unc), where=unfolded != 0)
            total_frac_up = np.divide(syst_up, unfolded, out=np.zeros_like(syst_up), where=unfolded != 0)
            total_frac_down = np.divide(syst_down, unfolded, out=np.zeros_like(syst_down), where=unfolded != 0)

            if stat_only:
                # No band: the ratio carries the same stat uncertainty as the
                # points above, drawn once as error bars on the ratio markers.
                # No x error bars here: at bin width they merge into a solid
                # line that buries the unity reference.
                plt.errorbar(centers, ratio_pythia, yerr=stat_frac, fmt='o',
                             color='k', markersize=5)
            else:
                plt.stairs(1.0 + total_frac_up, self.gen_edges_by_pt[i], baseline=1.0 - total_frac_down, fill=True, color="yellowgreen", label=total_label)
                plt.stairs(1.0 + stat_frac, self.gen_edges_by_pt[i], baseline=1.0 - stat_frac, fill=True, color="darkgreen", label=stat_label)
                plt.stairs(ratio_pythia, self.gen_edges_by_pt[i], color='#5790fc', ls='dotted', lw=2,
                           label='Unfolded / Pythia8' if is_closure else 'Data / Pythia8', baseline=None)
            # PYTHIA8 uncertainty propagated onto Data/PYTHIA8 (ratio ~ 1/PYTHIA,
            # so a +sigma on PYTHIA pulls the ratio down by ratio*sigma/PYTHIA).
            if py_unc_up is not None:
                rel_py = np.divide(np.abs(ratio_pythia), pythia, out=np.zeros_like(pythia), where=pythia != 0)
                plt.errorbar(
                    centers, ratio_pythia, yerr=[rel_py * py_unc_down, rel_py * py_unc_up],
                    fmt='none', ecolor='#5790fc', elinewidth=1.2, capsize=2,
                )
            if has_herwig:
                ratio_herwig = np.divide(unfolded, herwig_norm)
                plt.stairs(ratio_herwig, self.gen_edges_by_pt[i], color='#e42536', ls='dashdot', lw=2, label='Data / Herwig7', baseline=None)
                if hw_unc_up is not None:
                    rel_hw = np.divide(np.abs(ratio_herwig), herwig_norm, out=np.zeros_like(herwig_norm), where=herwig_norm != 0)
                    plt.errorbar(
                        centers, ratio_herwig, yerr=[rel_hw * hw_unc_down, rel_hw * hw_unc_up],
                        fmt='none', ecolor='#e42536', elinewidth=1.2, capsize=2,
                    )
            if vincia_truth is not None:
                ratio_vincia = np.divide(
                    unfolded, vincia_norm,
                    out=np.zeros_like(unfolded), where=vincia_norm != 0)
                plt.stairs(ratio_vincia, self.gen_edges_by_pt[i], color='#964a8b', ls='dashed', lw=2, label='Data / Vincia', baseline=None)
            plt.ylim(0, 2)
            # No tick label at 0 or 2 (as in the data/MC figure): the corner "0"
            # collides with the first x tick label and the top "2" crowds the
            # main panel's "0.0" above.
            ax_ratio.set_yticks([0.5, 1.0, 1.5])
            plt.xlabel(self._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
            plt.ylabel(r"$\frac{Unfolded}{Truth}$" if is_closure else r"$\frac{Data}{Simulation}$",
                       fontsize=PUB_LABEL_FONTSIZE)
            ax_ratio.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
            plt.xlim(*self._observable_xlim(i))
            if self.closure:
                save_path = f"./{self.spec.output_dir}closure_groomed_{i-1}.pdf" if self.groomed else f"./{self.spec.output_dir}closure_ungroomed_{i-1}.pdf"
            else:
                save_path = f"./{self.spec.output_dir}unfold/groomed_{i-1}.pdf" if self.groomed else f"./{self.spec.output_dir}unfold/ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        
        # Summary plot: all pT slices together, offset by 10^n. ARC round-2:
        # ratio panel added (Data / Pythia8 per slice) "to better understand
        # the level of agreement"; CMS-default canvas; Petroff colors;
        # Pythia8/Herwig7/Vincia casing; data as black markers with vertical
        # + horizontal error bars carried into the legend.
        fig, ax_main = plt.subplots()
        ratio_inputs = []  # per-slice payload for the separate ratio figure
        pt_key_entries = []  # (handle, label) rows for the SMP-24-010-style pT key
        for i in range(max(1, self.first_reported_pt_bin), npt):
            exponent = 2 * i - 1
            scale = 10 ** exponent
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            if has_herwig:
                herwig_norm = true_herwig_pt_binned[i] / bin_widths / self._shown_norm_total(true_herwig_pt_binned[i], i)

            # Crop every per-slice array to the SHOWN bins: on the shared
            # x-axis (deepest floor for the mode) each slice's curves then
            # span exactly its reported window instead of leaking below its
            # own floor. k0 = 0 when no per-pT floors are set.
            k0 = int(np.argmax(self._shown_gen_mask(i)))
            rho_edges = np.array(self.gen_edges_by_pt[i], dtype=float)[k0:]
            unfolded = unfolded[k0:]
            syst_up = syst_up[k0:]
            syst_down = syst_down[k0:]
            stat_unc = stat_unc[k0:]
            bin_widths = bin_widths[k0:]
            if has_herwig:
                herwig_norm = herwig_norm[k0:]

            y_syst_up = scale * (unfolded + syst_up)
            y_syst_down = scale * (unfolded - syst_down)
            y_syst_down = np.maximum(y_syst_down, scale * unfolded * 1e-1)
            y_stat_up = scale * (unfolded + stat_unc)
            y_stat_down = scale * (unfolded - stat_unc)
            pythia = np.array(self.normalized_results[i]['true'], dtype=float)[k0:]
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            summary_max = max(getattr(self, "_summary_max", 0.0), float(np.max(y_syst_up)), float(np.max(scale * pythia)))
            self._summary_max = summary_max
            positive = y_syst_down[y_syst_down > 0]
            if positive.size:
                self._summary_min = min(getattr(self, "_summary_min", np.inf), float(np.min(positive)))
            ax_main.stairs(scale * pythia, rho_edges, label='Pythia8', color='#5790fc', ls='dotted', lw=3, baseline=None)
            py_unc_up, py_unc_down = self._prediction_uncertainty(i, "pythia")
            if py_unc_up is not None:
                py_unc_up, py_unc_down = py_unc_up[k0:], py_unc_down[k0:]
                ax_main.errorbar(centers, scale * pythia, yerr=[scale * py_unc_down, scale * py_unc_up],
                                 fmt='none', ecolor='#5790fc', elinewidth=1.2, capsize=2)
            if has_herwig:
                ax_main.stairs(scale * herwig_norm, rho_edges, label='Herwig7', color='#e42536', ls='dashdot', lw=2, baseline=None)
                hw_unc_up, hw_unc_down = self._prediction_uncertainty(i, "herwig")
                if hw_unc_up is not None:
                    hw_unc_up, hw_unc_down = hw_unc_up[k0:], hw_unc_down[k0:]
                    ax_main.errorbar(centers, scale * herwig_norm, yerr=[scale * hw_unc_down, scale * hw_unc_up],
                                     fmt='none', ecolor='#e42536', elinewidth=1.2, capsize=2)
            if vincia_truth is not None:
                vincia_norm, vincia_err = vincia_truth[i]
                vincia_norm, vincia_err = vincia_norm[k0:], vincia_err[k0:]
                ax_main.stairs(scale * vincia_norm, rho_edges, label='Vincia', color='#964a8b', ls='dashed', lw=2, baseline=None)
                ax_main.errorbar(centers, scale * vincia_norm, yerr=scale * vincia_err,
                                 fmt='none', ecolor='#964a8b', elinewidth=1.2, capsize=2)
            if not stat_only:
                ax_main.stairs(y_syst_up, rho_edges, baseline=y_syst_down, fill=True, color="yellowgreen", label=total_label, alpha = 0.8)
                ax_main.stairs(y_stat_up, rho_edges, baseline=y_stat_down, fill=True, color="darkgreen", label=stat_label)
            # Approval comments (A. Meyer, SMP-24-010 style): the main legend
            # keeps only the series identity -- one "Data" row with a
            # representative marker -- while a separate compact key maps the
            # per-slice marker to its pT interval and 10^n offset factor,
            # mirroring SMP-24-010 Fig. 10 (marker key with the offsets
            # printed in the key, nothing written next to the curves).
            h_data = ax_main.errorbar(centers, scale * unfolded, yerr=scale * stat_unc,
                                      xerr=np.diff(rho_edges) / 2, fmt=markers[i],
                                      color='k', markersize=7, label='Data')
            pt_key_entries.append(
                (h_data, rf'{title_list[i]} ($\times 10^{{{exponent}}}$)'))

            ratio_inputs.append({
                "i": i, "edges": rho_edges, "centers": centers,
                "widths": np.diff(rho_edges), "unfolded": unfolded,
                "pythia": pythia, "syst_up": syst_up, "syst_down": syst_down,
                "stat_unc": stat_unc,
                "herwig": herwig_norm if has_herwig else None,
                "vincia": vincia_norm if vincia_truth is not None else None,
                "pythia_unc": (py_unc_up, py_unc_down),
                "herwig_unc": (hw_unc_up, hw_unc_down) if has_herwig else (None, None),
                "vincia_err": vincia_err if vincia_truth is not None else None,
            })

        ax_main.set_yscale('log')
        # Group duplicate legend entries: keep first occurrence, hide subsequent ones
        handles, labels = ax_main.get_legend_handles_labels()
        seen = set()
        for h, l in zip(handles, labels):
            if l == "" or l in seen:
                try:
                    h.set_label('_nolegend_')  # matplotlib ignores labels starting with '_'
                except Exception:
                    pass
            else:
                seen.add(l)
        # Legend headroom on the log scale (ARC round-2 "legend must not
        # touch the plots"): measure the legend box, then solve the decade
        # span so the tallest histogram sits just below it.
        # Approval comments (A. Meyer): Data at the top of the legend.
        handles = [h for h in handles if h.get_label() and not h.get_label().startswith('_')]
        handles.sort(key=lambda h: h.get_label() != 'Data')
        legend = ax_main.legend(handles=handles, loc="upper right",
                                fontsize=PUB_LEGEND_FONTSIZE)
        fig.canvas.draw()
        lbox = legend.get_window_extent().transformed(ax_main.transAxes.inverted())
        # SMP-24-010-style pT key: a second, frameless legend right below the
        # series legend, one row per pT slice with its marker and 10^n factor.
        ax_main.add_artist(legend)
        key_legend = ax_main.legend(
            [h for h, _ in pt_key_entries], [l for _, l in pt_key_entries],
            loc="upper right", bbox_to_anchor=(lbox.x1, lbox.y0 - 0.005),
            bbox_transform=ax_main.transAxes, fontsize=PUB_LEGEND_FONTSIZE - 2,
            frameon=False, handletextpad=0.4, labelspacing=0.35)
        fig.canvas.draw()
        kbox = key_legend.get_window_extent().transformed(ax_main.transAxes.inverted())
        lbox = kbox  # headroom solve clears the lower of the two blocks
        # Floor just below the lowest drawn band (the autoscale bottom can be
        # decades lower from a near-empty bin, bloating the axis range).
        ybot = getattr(self, "_summary_min", ax_main.get_ylim()[0]) / 3.0
        need = np.log10(self._summary_max / ybot) / max(0.2, lbox.y0 - 0.05)
        ax_main.set_ylim(ybot, ybot * 10 ** need)
        self._summary_max = self._summary_min = None
        del self._summary_max, self._summary_min
        ax_main.set_ylabel(self._normalized_ylabel(), fontsize=PUB_LABEL_FONTSIZE)

        ax_main.set_xlabel(self._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
        ax_main.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        ax_main.set_xlim(*self._observable_xlim())

        # In-frame CMS tag: the clearance solve keeps the top-left free.
        fig.canvas.draw()
        hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), loc=2, ax=ax_main)

        save_path = f"./{self.spec.output_dir}groomed_summary.pdf" if self.groomed else f"./{self.spec.output_dir}ungroomed_summary.pdf"
        self._finalize_plot(save_path=save_path, show=show, fig=fig)

        # ARC round-2: separate RATIO figure -- a grid of panels, one per pT
        # slice (highest pT on top, like the summary). Ratio TO DATA: the
        # data uncertainty bands sit at 1, generator curves float around it.
        n = len(ratio_inputs)
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 4 * n),
                                 gridspec_kw={"hspace": 0.1})
        axes = np.atleast_1d(axes)
        for k, d in enumerate(ratio_inputs):
            ax = axes[n - 1 - k]
            edges, centers, widths = d["edges"], d["centers"], d["widths"]
            unfolded = d["unfolded"]

            def _r(arr):
                return np.divide(arr, unfolded, out=np.zeros_like(unfolded), where=unfolded != 0)

            # Data uncertainty bands hugging unity; Data/Theory curves on top.
            if not stat_only:
                ax.stairs(1.0 + _r(d["syst_up"]), edges,
                          baseline=1.0 - _r(d["syst_down"]), fill=True,
                          color="yellowgreen", alpha=0.8, label=total_label)
                ax.stairs(1.0 + _r(d["stat_unc"]), edges,
                          baseline=1.0 - _r(d["stat_unc"]), fill=True,
                          color="darkgreen", label=stat_label)
            ax.axhline(1.0, color='gray', ls='--', lw=1)

            def _data_over(arr):
                return np.divide(unfolded, arr, out=np.zeros_like(unfolded), where=arr != 0)

            def _theory_band(theory, unc_up, unc_down, color):
                # Theory uncertainty propagated onto Data/Theory, drawn as
                # error-bar "sticks" at the bin centers (band was too faint).
                if unc_up is None:
                    return
                ratio = _data_over(theory)
                lo = np.divide(unfolded, theory + unc_up,
                               out=np.zeros_like(unfolded), where=(theory + unc_up) != 0)
                hi_den = np.clip(theory - unc_down, 1e-12, None)
                hi = np.divide(unfolded, hi_den,
                               out=np.zeros_like(unfolded), where=hi_den > 0)
                ax.errorbar(centers, ratio,
                            yerr=[np.clip(ratio - lo, 0, None),
                                  np.clip(hi - ratio, 0, None)],
                            fmt='none', ecolor=color, elinewidth=1.5, capsize=3)

            _theory_band(d["pythia"], *d["pythia_unc"], '#5790fc')
            ax.stairs(_data_over(d["pythia"]), edges, color='#5790fc',
                      ls='dotted', lw=3, label='Data / Pythia8', baseline=None)
            if d["herwig"] is not None:
                _theory_band(d["herwig"], *d["herwig_unc"], '#e42536')
                ax.stairs(_data_over(d["herwig"]), edges, color='#e42536',
                          ls='dashdot', lw=2, label='Data / Herwig7', baseline=None)
            if d["vincia"] is not None:
                if d["vincia_err"] is not None:
                    _theory_band(d["vincia"], d["vincia_err"], d["vincia_err"], '#964a8b')
                ax.stairs(_data_over(d["vincia"]), edges, color='#964a8b',
                          ls='dashed', lw=2, label='Data / Vincia', baseline=None)
            ax.set_ylim(0, 2)
            ax.set_yticks([0.5, 1.0, 1.5])
            ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
            ax.text(0.97, 0.95, title_list[d["i"]], transform=ax.transAxes,
                    ha="right", va="top", fontsize=PUB_ANNOTATION_FONTSIZE)
        # Legend outside the frame: one row spanning the top panel's width,
        # with the CMS label stamped above it (pad from the measured box).
        handles, leg_labels = axes[0].get_legend_handles_labels()
        # Two columns, sized to their content rather than stretched to the axes
        # width (mode="expand"): at this type size three columns pushed the last
        # entry past the right frame edge, and equal thirds were narrower than
        # "Syst. + Stat. Unc." so its text ran under the next column's handle.
        # The uncertainty bands go in the left column and the Data/Theory ratios
        # in the right one. matplotlib fills columns top-to-bottom, so padding
        # both groups to the same length puts each in a column of its own.
        bands = [(h, l) for h, l in zip(handles, leg_labels) if not l.startswith("Data /")]
        ratios = [(h, l) for h, l in zip(handles, leg_labels) if l.startswith("Data /")]
        if bands and ratios:
            rows = max(len(bands), len(ratios))
            spacer = plt.Line2D([], [], linestyle="none")
            bands += [(spacer, "")] * (rows - len(bands))
            ratios += [(spacer, "")] * (rows - len(ratios))
            handles = [h for h, _ in bands + ratios]
            leg_labels = [l for _, l in bands + ratios]
        leg = axes[0].legend(handles, leg_labels, loc="lower left",
                             bbox_to_anchor=(0, 1.005),
                             ncol=2, fontsize=PUB_LEGEND_FONTSIZE, frameon=False,
                             borderaxespad=0, columnspacing=1.6,
                             handletextpad=0.5, handlelength=1.8)
        axes[len(axes) // 2].set_ylabel("Data/Theory", fontsize=PUB_LABEL_FONTSIZE)
        axes[-1].set_xlabel(self._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
        axes[-1].set_xlim(*self._observable_xlim())
        fig.canvas.draw()
        lbox = leg.get_window_extent().transformed(axes[0].transAxes.inverted())
        label_arts = hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(),
                                   com=self._com_label(), ax=axes[0])
        # mplhep 1.2 has no pad kwarg: lift the label artists above the legend.
        dy = max(0.0, lbox.y1 - 1.0) + 0.015
        for art in label_arts:
            if art is not None:
                x, y = art.get_position()
                art.set_position((x, y + dy))
        save_path = (f"./{self.spec.output_dir}groomed_summary_ratio.pdf"
                     if self.groomed else
                     f"./{self.spec.output_dir}ungroomed_summary_ratio.pdf")
        self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_unfolded_summary_linear(self, show=True):
        markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
        npt = len(self.pt_edges) - 1
        stat_label = "Stat. Unc." if self.response_matrix_stat_available else "Input Stat. Unc."
        total_label = (
            r"Syst. $\oplus$ Stat. Unc."
            if self.response_matrix_stat_available and self.has_herwig
            else r"Partial Syst. $\oplus$ Stat. Unc."
        )
        title_list = []
        for i in range(npt):
            lo = int(self.pt_edges[i])
            if i + 1 < npt:
                hi = int(self.pt_edges[i + 1])
                title_list.append(rf"{lo}$<$$p_T$$<${hi} GeV")
            else:
                title_list.append(rf"{lo}$<$$p_T$$< \, \infty$  GeV")

        fig = plt.figure(figsize=(12, 10))
        for i in range(max(1, self.first_reported_pt_bin), npt):
            exponent = 2 * i - 1
            scale = 10 ** exponent
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)

            y_syst_up = scale * (unfolded + syst_up)
            y_syst_down = scale * (unfolded - syst_down)
            y_stat_up = scale * (unfolded + stat_unc)
            y_stat_down = scale * (unfolded - stat_unc)

            plt.stairs(scale * np.array(self.normalized_results[i]['true'], dtype=float), self.gen_edges_by_pt[i], label='PYTHIA8', color='b', ls='dotted', lw=3, baseline=None)
            plt.stairs(y_syst_up, self.gen_edges_by_pt[i], baseline=y_syst_down, fill=True, color="yellowgreen", label=total_label, alpha=0.8)
            plt.stairs(y_stat_up, self.gen_edges_by_pt[i], baseline=y_stat_down, fill=True, color="darkgreen", label=stat_label)
            rho_edges = np.array(self.gen_edges_by_pt[i], dtype=float)
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            plt.plot(centers, scale * unfolded, label=rf'$10^{{{exponent}}}$ x {title_list[i]}', color='k', lw=0, marker=markers[i])

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        for h, l in zip(handles, labels):
            if l == "" or l in seen:
                try:
                    h.set_label('_nolegend_')
                except Exception:
                    pass
            else:
                seen.add(l)
        plt.legend(fontsize=15)
        plt.xlabel(self._observable_label())
        plt.ylabel(self._normalized_ylabel())
        hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
        plt.xlim(*self._observable_xlim())

        save_path = f"./{self.spec.output_dir}groomed_summary_linear.pdf" if self.groomed else f"./{self.spec.output_dir}ungroomed_summary_linear.pdf"
        self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_herwig_pythia_comparison(self, show=True):
        """
        One plot per pT bin with three panels:
          - Main panel  : normalized gen-level jet mass for PYTHIA8 and HERWIG7.
          - Middle panel: gen-level model uncertainty (HERWIG7 - PYTHIA8) / PYTHIA8.
          - Bottom panel: post-unfolding model uncertainty, i.e. envelope of
                          |unfolded_herwigUp - nominal| / nominal and
                          |unfolded_herwigDown - nominal| / nominal.
        """
        true_herwig_pt_binned = unflatten_gen_by_pt(self.y_true_herwig, self.gen_edges_by_pt)
        npt = len(self.pt_edges) - 1
        title_list = [
            "",
            r"$p_T$ 200 – 290 GeV",
            r"$p_T$ 290 – 400 GeV",
            r"$p_T$ 400 – $\infty$ GeV",
        ]

        ratio_ymax = 0.5

        def _annotate_overflow(ax, values, edges, ymax, color, fontsize=11):
            """Write the numeric value above the clip line for bins that exceed ymax."""
            centers = 0.5 * (edges[:-1] + edges[1:])
            for x, y in zip(centers, values):
                if np.isfinite(y) and y > ymax:
                    ax.text(x, ymax - 0.02, f"{y:.2f}",
                            ha='center', va='top', fontsize=fontsize,
                            color=color, clip_on=True)

        for i in range(npt):
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            edges = np.array(self.gen_edges_by_pt[i], dtype=float)

            pythia_norm = np.array(self.normalized_results[i]['true'], dtype=float)
            herwig_raw = true_herwig_pt_binned[i]
            herwig_norm = herwig_raw / bin_widths / self._shown_norm_total(herwig_raw, i)

            # Unsigned gen-level difference
            gen_model_diff = np.abs(np.divide(
                herwig_norm - pythia_norm, pythia_norm,
                out=np.full_like(herwig_norm, np.nan), where=pythia_norm != 0,
            ))

            # Post-unfolding model uncertainty: envelope of herwigUp / herwigDown
            syst_frac = self._build_syst_fraction_dict(i)
            herwig_up_frac   = syst_frac.get('herwigUp',   np.zeros_like(pythia_norm))
            herwig_down_frac = syst_frac.get('herwigDown', np.zeros_like(pythia_norm))
            model_unc_unfolded = np.maximum(herwig_up_frac, herwig_down_frac)

            # PYTHIA8 / HERWIG7 ratio (gen-level, normalized)
            pythia_over_herwig = np.divide(
                pythia_norm, herwig_norm,
                out=np.full_like(pythia_norm, np.nan), where=herwig_norm != 0,
            )

            fig, (ax_main, ax_ph, ax_ratio, ax_model) = plt.subplots(
                4, 1, sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1, 1]},
                figsize=(10, 14),
            )

            # Main panel
            hep.histplot(pythia_norm, edges, label='PYTHIA8', color='#5790fc', ls='dotted', lw=3, ax=ax_main)
            hep.histplot(herwig_norm, edges, label='HERWIG7', color='#e42536', ls='dashdot', lw=2, ax=ax_main)
            ax_main.set_ylabel(self._normalized_ylabel())
            pt_title = title_list[i] if i < len(title_list) else ""
            ax_main.legend(title=pt_title, fontsize=14, title_fontsize=15)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20, ax=ax_main)

            # PYTHIA8 / HERWIG7 ratio panel
            ax_ph.axhline(1.0, color='gray', ls='--', lw=1)
            hep.histplot(pythia_over_herwig, edges, color='#5790fc', ls='dotted', lw=2, ax=ax_ph)
            ax_ph.set_ylabel('P8 / H7', fontsize=13)
            ax_ph.set_ylim(0.5, 1.5)
            ax_ph.set_yticks([0.5, 1.0, 1.5])

            # Gen-level model uncertainty panel: |HERWIG7 - PYTHIA8| / PYTHIA8
            ax_ratio.axhline(0.0, color='gray', ls='--', lw=1)
            hep.histplot(gen_model_diff, edges, color='#e42536', ls='dashdot', lw=2, ax=ax_ratio)
            ax_ratio.set_ylabel('Rel. Unc. GEN', fontsize=13)
            ax_ratio.set_ylim(0, ratio_ymax)
            ax_ratio.set_yticks([0.0, 0.25, 0.5])
            _annotate_overflow(ax_ratio, gen_model_diff, edges, ratio_ymax, color='#e42536')

            # Post-unfolding model uncertainty panel
            ax_model.axhline(0.0, color='gray', ls='--', lw=1)
            hep.histplot(model_unc_unfolded, edges, color='#7a21dd', ls='solid', lw=2, ax=ax_model)
            ax_model.set_ylabel('Rel. Unc. Unfolded', fontsize=13)
            ax_model.set_ylim(0, ratio_ymax)
            ax_model.set_yticks([0.0, 0.25, 0.5])
            ax_model.set_xlabel(self._observable_label())
            _annotate_overflow(ax_model, model_unc_unfolded, edges, ratio_ymax, color='#7a21dd')

            for ax in (ax_main, ax_ph, ax_ratio, ax_model):
                ax.set_xlim(*self._observable_xlim(i))

            ax_model.tick_params(axis='x', pad=8)
            ax_model.tick_params(axis='y', pad=6)
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.1)
            suffix = "groomed" if self.groomed else "ungroomed"
            save_path = f"./{self.spec.output_dir}herwig_pythia_comparison_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_unfolded(self, log=False, show=True):

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.gen_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.reco_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.reco_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.gen_edges_by_pt)
        true_herwig_pt_binned = (
            unflatten_gen_by_pt(self.y_true_herwig, self.gen_edges_by_pt)
            if getattr(self, "has_herwig", True)
            else None
        )
        #error_pt_binned = unflatten_gen_by_pt(self.ye_unf, self.gen_edges_by_pt)
        self.normalized_herwig = []
        #print("Herwig pt Binned", true_herwig_pt_binned)
        self.herwig_closure_unc = []
        for i in self._reported_pt_indices():
            yerr = self.normalized_results[i]['syst_unc']['up']
            bin_widths = np.diff(self.gen_edges_by_pt[i])
            bin_widths_reco = np.diff(self.reco_edges_by_pt[i])
            #self.normalized_herwig.append(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum())
            if self.herwig_closure:
                hep.histplot(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum(), self.gen_edges_by_pt[i], color='#964a8b', label = 'Herwig', alpha = 0.7, ls = 'dotted')
            else:
                hep.histplot(true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(), self.gen_edges_by_pt[i], color='#5790fc', label = 'PYTHIA', alpha = 0.8, ls = 'dotted', lw = 3)
            hep.histplot(unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), self.gen_edges_by_pt[i], label = 'Unfolded Herwig' if self.herwig_closure else 'Unfolded', color = 'k', ls = '--' )

            

            #hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.reco_edges_by_pt[i], color = 'k', ls= '--', alpha= 0.5, label = 'Meas' )
            #dhep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.reco_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f" {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title, fontsize = 18) 
            
            if self.groomed:
                #plt.xlim(0,250)
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
            save_path = f"./{self.spec.output_dir}unfold/unfolded_basic_groomed_{i-1}.pdf" if self.groomed else f"./{self.spec.output_dir}unfold/unfolded_basic_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show)
            # Plot relative difference: (true - unfolded) / true after normalization
            # true_norm = true_pt_binned[i] / np.diff(self.gen_edges_by_pt[i]) / true_pt_binned[i].sum()
            # true_norm = true_herwig_pt_binned[i] / np.diff(self.gen_edges_by_pt[i]) / true_herwig_pt_binned[i].sum()
            # unfolded_norm = unfolded_pt_binned[i] / np.diff(self.gen_edges_by_pt[i]) / unfolded_pt_binned[i].sum()
            
            # rel_diff = np.abs(true_norm - unfolded_norm) / true_norm
            # hep.histplot(rel_diff, self.gen_edges_by_pt[i], label="(Herwig - Unfolded) / Herwig", color="r")
            # title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            # plt.legend(title = title) 
            if self.herwig_closure:
                plt.figure(figsize=(10, 3))
                true_herwig = true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum()
                unfolded = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
                herwig_closure_unc = np.abs(true_herwig - unfolded) / true_herwig
                self.herwig_closure_unc.append(herwig_closure_unc)
                plt.stairs(herwig_closure_unc, self.gen_edges_by_pt[i], label = 'Closure Unc (|Herwig - Unfolded| / Herwig)', color='#7a21dd', ls = 'dotted')
                if not self.groomed:
                    plt.xlim(*self._observable_xlim(i))
                else:
                    plt.xlim(*self._observable_xlim(i))
                plt.ylim(0, 1)
                plt.xlabel(self._observable_label())
                #plt.legend()
                groomed_tag = "groomed" if self.groomed else "ungroomed"
                save_path = f"./{self.spec.output_dir}unfold/herwig_closure_unc_{groomed_tag}_{i-1}.pdf"
                self._finalize_plot(save_path=save_path, show=show)
        # Save uncertainty in a file for later use
        if self.herwig_closure:
            groomed_tag = "groomed" if self.groomed else "ungroomed"
            np.save(f"{self.spec.input_dir}herwig_closure_unc_{self.spec.name}_{groomed_tag}.npy", self.herwig_closure_unc)
            # if self.groomed:
            #     plt.xlim(0,250)
            #     plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            # #plt.ylim(0,0.02)
            # if not self.groomed:
            #     plt.xlim(20,250)
            #     plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            # plt.show()

    def _ensure_herwig_bias_inputs(self):
        """Build the HERWIG reco mosaic needed for the bias test, on demand.

        ``mosaic_herwig_2d`` (the HERWIG matched-reco spectrum, used as the
        measured input of the bias test) is normally only assembled when a
        herwig systematic is requested. The bias test needs it regardless, so
        build it here from the HERWIG response stored in ``_load_data``.
        """
        if getattr(self, "mosaic_herwig_2d", None) is not None:
            return
        self._prepare_herwig_inputs(
            self.herwig_4d,
            self.herwig_4d_gen,
            self.fakes_herwig,
            self.misses_herwig,
            self.edges,
            self.edges_gen,
            self.pt_edges,
            self.reco_edges_by_pt,
            self.gen_edges_by_pt,
        )
        self.mosaic_herwig_2d = merge_mass_flat(
            self.h2d_herwig, self.edges, self.reco_edges_by_pt
        )

    def _unfold_herwig_through_pythia(self):
        """Unfold the HERWIG reco spectrum through the PYTHIA response matrix.

        Returns ``(y_unf, ye_unf)``: the flattened unfolded result and TUnfold's
        propagated per-bin uncertainty on it. Runs a one-off nominal unfold in
        ``herwig_closure`` mode (measured = HERWIG reco, response = nominal
        PYTHIA, prior = PYTHIA truth) and restores the nominal data result so
        the rest of ``run_all_plots`` is unaffected. tau is already frozen from
        the nominal data unfold, so no re-scan happens here.
        """
        snapshot_attrs = (
            "y_meas", "ye_meas", "y_unf", "ye_unf", "y_true", "x_folded", "L",
            "h_resp", "cov", "cov_uncorr", "cov_uncorr_data", "cov_total",
            "cov_np", "cov_uncorr_np", "cov_data_np",
        )
        sentinel = object()
        saved = {name: getattr(self, name, sentinel) for name in snapshot_attrs}
        try:
            self._perform_unfold(systematic="nominal", herwig_closure=True)
            return np.array(self.y_unf, copy=True), np.array(self.ye_unf, copy=True)
        finally:
            for name, value in saved.items():
                if value is sentinel:
                    if hasattr(self, name):
                        delattr(self, name)
                else:
                    setattr(self, name, value)

    def plot_herwig_bias_test(self, show=True):
        """HERWIG bias (non-closure) test: PYTHIA matrix unfolds HERWIG reco.

        Unfolds the HERWIG reco spectrum with the nominal PYTHIA response and
        compares the result to HERWIG gen. Each panel shows:

          * Unfolded HERWIG as a dashed line.
          * HERWIG gen with its MC-stat uncertainty drawn as a band.
          * a ratio panel with the per-bin |HERWIG - Unfolded| / HERWIG and the
            HERWIG gen MC-stat uncertainty as a shaded envelope.

        The per-bin |HERWIG - Unfolded| / HERWIG is also saved as the
        model-dependence uncertainty input (``herwig_closure_unc_*.npy``).
        """
        if not getattr(self, "has_herwig", True) or self.y_true_herwig is None:
            return

        hep.style.use("CMS")
        self._ensure_herwig_bias_inputs()
        y_unf_herwig, _ = self._unfold_herwig_through_pythia()

        unfolded_pt_binned = unflatten_gen_by_pt(y_unf_herwig, self.gen_edges_by_pt)
        true_herwig_pt_binned = unflatten_gen_by_pt(
            self.y_true_herwig, self.gen_edges_by_pt
        )
        # HERWIG gen MC-stat error per flattened gen bin (sqrt of sumw2), or
        # None when the input pkls carry no variances.
        herwig_gen_err = (
            np.sqrt(np.clip(self.herwig_gen_var_flat, 0.0, None))
            if getattr(self, "herwig_gen_var_flat", None) is not None
            else None
        )
        herwig_gen_err_pt_binned = (
            unflatten_gen_by_pt(herwig_gen_err, self.gen_edges_by_pt)
            if herwig_gen_err is not None
            else None
        )

        def _rel(num, den):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.abs(
                    np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den != 0)
                )

        groomed_tag = "groomed" if self.groomed else "ungroomed"
        self.herwig_closure_unc = []
        for i in self._reported_pt_indices():
            edges = np.asarray(self.gen_edges_by_pt[i], dtype=float)
            bin_widths = np.diff(edges)

            herwig_sum = true_herwig_pt_binned[i].sum()
            unfolded_sum = unfolded_pt_binned[i].sum()
            herwig = true_herwig_pt_binned[i] / bin_widths / herwig_sum
            unfolded = unfolded_pt_binned[i] / bin_widths / unfolded_sum

            # HERWIG gen MC-stat (bin-width and per-slice normalization cancel
            # in the ratio, so apply the relative error to the density directly).
            # NB: TUnfold's propagated "unfolding uncertainty" on the HERWIG
            # pseudo-data is intentionally NOT drawn here -- closure mode feeds
            # no measured variances, so it is the sqrt(weighted-content) Poisson
            # error of weighted MC propagated through the unregularized inverse,
            # which is meaningless (and ~100%+) for a bias test.
            if herwig_gen_err_pt_binned is not None:
                herwig_rel = _rel(herwig_gen_err_pt_binned[i], true_herwig_pt_binned[i])
            else:
                herwig_rel = np.zeros_like(herwig)
            herwig_band = herwig * herwig_rel

            closure_unc = _rel(herwig - unfolded, herwig)
            self.herwig_closure_unc.append(closure_unc)

            fig, (ax, ax_ratio) = plt.subplots(
                2, 1, figsize=(10, 10), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            # HERWIG gen + MC-stat band
            hep.histplot(
                herwig, edges, ax=ax, color="#964a8b",
                label="Herwig (gen)", alpha=0.9, ls="dotted", lw=2,
            )
            if np.any(herwig_band > 0):
                ax.fill_between(
                    edges, np.r_[herwig - herwig_band, (herwig - herwig_band)[-1]],
                    np.r_[herwig + herwig_band, (herwig + herwig_band)[-1]],
                    step="post", color="#964a8b", alpha=0.2, lw=0,
                    label="Herwig stat. unc.",
                )

            # Unfolded HERWIG. The TUnfold uncertainty is intentionally omitted;
            # see the note above where the HERWIG MC-stat band is constructed.
            hep.histplot(
                unfolded, edges, ax=ax, color="k", ls="--", lw=2,
                label="Unfolded Herwig",
            )

            title = (
                f" {int(self.pt_edges[i])}-"
                f"{int(self.pt_edges[i + 1]) if i + 1 < len(self.pt_edges) - 1 else '∞'} GeV"
            )
            ax.legend(title=title, fontsize=15)
            ax.set_ylabel(self._normalized_ylabel())
            hep.cms.label(
                self.cms_label, data=False, lumi=self._lumi_label(),
                com=self._com_label(), fontsize=20, ax=ax,
            )

            # Ratio panel: compare the observed non-closure with HERWIG gen
            # MC statistics only.
            if np.any(herwig_rel > 0):
                ax_ratio.fill_between(
                    edges, 0.0, np.r_[herwig_rel, herwig_rel[-1]],
                    step="post", color="#964a8b", alpha=0.2, lw=0,
                    label="Herwig stat. unc.",
                )
            ax_ratio.stairs(
                closure_unc, edges, color="#7a21dd", lw=2,
                label="|Herwig - Unfolded| / Herwig",
            )
            ax_ratio.set_ylim(0, 1)
            ax_ratio.set_ylabel("Rel. diff.")
            ax_ratio.set_xlim(*self._observable_xlim(i))
            ax_ratio.set_xlabel(self._observable_label())
            ax_ratio.legend(fontsize=11, loc="upper left")

            save_path = (
                f"./{self.spec.output_dir}unfold/"
                f"herwig_bias_test_{groomed_tag}_{i - 1}.pdf"
            )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

        # Persist the non-closure as the model-dependence systematic input.
        np.save(
            f"{self.spec.input_dir}herwig_closure_unc_{self.spec.name}_{groomed_tag}.npy",
            np.array(self.herwig_closure_unc, dtype=object),
        )

    def _normalize_result(self):
        print("Normalizing results...")
        self.normalized_results = []
        gen_mass_bin_edges_by_pt = self.gen_edges_by_pt
        reco_mass_bin_edges_by_pt = self.reco_edges_by_pt

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, gen_mass_bin_edges_by_pt)
        #print("Unfolded pt binned:", unfolded_pt_binned)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, reco_mass_bin_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), reco_mass_bin_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, gen_mass_bin_edges_by_pt)
        error_pt_binned = unflatten_gen_by_pt(self.ye_unf, gen_mass_bin_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths = np.diff(gen_mass_bin_edges_by_pt[i])
            bin_widths_reco = np.diff(reco_mass_bin_edges_by_pt[i])
            # Denominators run over the SHOWN bins of the slice (full slice
            # unless spec.normalize_over_shown): the published spectra
            # integrate to 1 over the reported fiducial window, so hidden
            # buffer / catch-all bins do not enter the normalization.
            result = {
                "true": true_pt_binned[i]/bin_widths/self._shown_norm_total(true_pt_binned[i], i),
                "unfolded": unfolded_pt_binned[i]/bin_widths/self._shown_norm_total(unfolded_pt_binned[i], i),
                "unfolded_err": error_pt_binned[i]/bin_widths/self._shown_norm_total(unfolded_pt_binned[i], i),
                "measured": measured_pt_binned[i]/bin_widths_reco/self._shown_norm_total(measured_pt_binned[i], i, reco=True),
                "reco_mc": reco_mc_pt_binned[i]/bin_widths_reco/self._shown_norm_total(reco_mc_pt_binned[i], i, reco=True),
                "pt_bin": (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf')),
                "mgen_edges": self.gen_edges_by_pt[i]
            }
            self.normalized_results.append(result)
        
        # Storing normalized results for systematics
        self.normalized_systematics = []
        # Prepare normalized_systematics as a list of dicts, one per pt bin

        for i in range(len(self.pt_edges)-1):
            pt_bin = (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf'))
            unfolded = {}
            bin_widths = np.diff(gen_mass_bin_edges_by_pt[i])
            bin_widths_reco = np.diff(reco_mass_bin_edges_by_pt[i])
            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf_dict[syst], gen_mass_bin_edges_by_pt)
                unfolded[syst] = unfolded_pt_binned[i]/bin_widths/self._shown_norm_total(unfolded_pt_binned[i], i)
            self.normalized_systematics.append({
            "pt_bin": pt_bin,
            "unfolded": unfolded # Taking absolute values to avoid negative bins
            })

        self._compute_2d_normalized_result()

    def _compute_2d_normalized_result(self):
        """2D-normalized unfolded result: normalize over the full (m, pT) plane.

        Unlike ``normalized_results`` (each pT slice divided by its own sum so
        every slice integrates to 1), this divides the whole double-differential
        spectrum by a single global integral while keeping the mass-bin density
        (/ bin width). The result is 1/sigma * d^2sigma/(dm dpT), integrating to
        1 over the full (m, pT) plane, so the relative pT-to-pT normalization is
        preserved rather than discarded.

        Stores both the flat (unrolled) arrays and a per-pT-slice list mirroring
        ``normalized_results``, plus a copy of the absolute unfolded output.
        """
        widths_flat = np.concatenate(
            [np.diff(np.asarray(edges, dtype=float)) for edges in self.gen_edges_by_pt]
        )
        unf = np.asarray(self.y_unf, dtype=float)
        true = np.asarray(self.y_true, dtype=float)
        unf_err = np.asarray(self.ye_unf, dtype=float)

        total_unf = unf.sum()
        total_true = true.sum()

        # Preserved copy of the absolute (un-normalized) 2D unfolded output.
        self.unfolded_abs_flat = np.array(unf, copy=True)
        self.unfolded_abs_err_flat = np.array(unf_err, copy=True)

        self.unfolded_2dnorm_flat = unf / widths_flat / total_unf
        self.unfolded_2dnorm_err_flat = unf_err / widths_flat / total_unf
        self.true_2dnorm_flat = true / widths_flat / total_true

        unf_pt = unflatten_gen_by_pt(self.unfolded_2dnorm_flat, self.gen_edges_by_pt)
        err_pt = unflatten_gen_by_pt(self.unfolded_2dnorm_err_flat, self.gen_edges_by_pt)
        true_pt = unflatten_gen_by_pt(self.true_2dnorm_flat, self.gen_edges_by_pt)

        self.normalized_2d = []
        for i in range(len(self.pt_edges) - 1):
            self.normalized_2d.append({
                "unfolded": unf_pt[i],
                "unfolded_err": err_pt[i],
                "true": true_pt[i],
                "pt_bin": (
                    self.pt_edges[i],
                    self.pt_edges[i + 1] if i + 1 < len(self.pt_edges) - 1 else float("inf"),
                ),
                "mgen_edges": self.gen_edges_by_pt[i],
            })

    def save_2d_unfolded(self):
        """Pickle the 2D unfolded output (absolute + 2D-normalized) as hists.

        Companion to ``save_normalized_covariance`` (which stores the per-pT
        normalized result). Each entry is a ``hist.Hist`` over (pt, obs) (or a
        per-pT list when the binning is ragged, e.g. mass):
          - "unfolded_abs": absolute unfolded output (Weight, value + TUnfold
            stat variance), in counts.
          - "unfolded_2dnorm": globally (2D) normalized unfolded spectrum
            (Weight, value + stat variance), summing to 1 over the whole plane.
          - "true_2dnorm": PYTHIA gen truth in the same 2D-normalized units.
          - "layout": "2d" or "per_pt"; "pt_edges".

        The 2D-normalized entries are sum-normalized (not divided by bin width);
        recover the density with hist/mplhep ``binwnorm`` at plot time.
        """
        suffix = "groomed" if self.groomed else "ungroomed"
        # Undo the per-bin-width division so the stored 2D-normalized spectrum is
        # sum-normalized (sums to 1 over the plane); apply binwnorm for density.
        widths_flat = np.concatenate(
            [np.diff(np.asarray(e, dtype=float)) for e in self.gen_edges_by_pt]
        )
        summary = {
            "unfolded_abs": self._hist_from_flat(
                self.unfolded_abs_flat, self.unfolded_abs_err_flat
            ),
            "unfolded_2dnorm": self._hist_from_flat(
                self.unfolded_2dnorm_flat * widths_flat,
                self.unfolded_2dnorm_err_flat * widths_flat,
            ),
            "true_2dnorm": self._hist_from_flat(self.true_2dnorm_flat * widths_flat),
            "layout": "2d" if self._slices_share_binning() else "per_pt",
            "pt_edges": np.asarray(self.pt_edges, dtype=float),
        }
        save_path = self._relocate_output(f"unfolded_2d_{suffix}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as handle:
            pkl.dump(summary, handle)
        print(f"Saved 2D unfolded output (hist) to {save_path}")

    def _ptnorm_flat(self, val_flat, err_flat=None):
        """Per-pT-slice normalize a flat gen vector: v / bin_width / slice_sum.

        Matches the convention used for ``normalized_results`` and the MC-stat
        bands (relative error preserved: err / width / slice_sum). Returns the
        normalized values, or ``(values, errors)`` when ``err_flat`` is given.
        """
        val_flat = np.asarray(val_flat, dtype=float)
        out_val = np.zeros_like(val_flat)
        out_err = None if err_flat is None else np.zeros_like(val_flat)
        offset = 0
        for edges in self.gen_edges_by_pt:
            nbins = len(edges) - 1
            sl = slice(offset, offset + nbins)
            widths = np.diff(np.asarray(edges, dtype=float))
            slice_sum = val_flat[sl].sum()
            if slice_sum != 0:
                out_val[sl] = val_flat[sl] / widths / slice_sum
                if err_flat is not None:
                    out_err[sl] = np.asarray(err_flat[sl], dtype=float) / widths / slice_sum
            offset += nbins
        return (out_val, out_err) if err_flat is not None else out_val

    def _slices_share_binning(self):
        """True when every pT slice has identical gen edges (e.g. rho)."""
        e0 = np.asarray(self.gen_edges_by_pt[0], dtype=float)
        return all(
            np.array_equal(e0, np.asarray(e, dtype=float)) for e in self.gen_edges_by_pt
        )

    def _obs_axis(self, edges):
        return hist.axis.Variable(
            np.asarray(edges, dtype=float), name="obs",
            label=self._observable_short_label(),
            underflow=False, overflow=False,
        )

    def _pt_axis(self):
        return hist.axis.Variable(
            np.asarray(self.pt_edges, dtype=float), name="pt",
            label=r"jet $p_T$ [GeV]", underflow=False, overflow=False,
        )

    def _hist_from_flat(self, val_flat, err_flat=None, extra_axes=()):
        """Build hist(s) from a flat (unrolled) gen vector.

        Returns a single 2D ``Hist(pt, obs[, *extra_axes])`` when all pT slices
        share binning (rho), else a list of per-pT 1D ``Hist(obs[, *extra_axes])``
        (mass, whose binning differs per slice). With ``err_flat`` the storage is
        Weight (value + variance = err**2); otherwise Double. ``extra_axes`` are
        appended after the obs axis and ``val_flat``/``err_flat`` must then be
        shaped (n_bins, *extra_shape) per slice.
        """
        val_flat = np.asarray(val_flat, dtype=float)
        err_flat = None if err_flat is None else np.asarray(err_flat, dtype=float)

        def _fill(axes, val, err):
            if err is None:
                h = hist.Hist(*axes, storage=hist.storage.Double())
                h.view()[...] = val
            else:
                h = hist.Hist(*axes, storage=hist.storage.Weight())
                view = h.view()
                view["value"] = val
                view["variance"] = err ** 2
            return h

        if self._slices_share_binning():
            n_pt = len(self.pt_edges) - 1
            n_obs = len(self.gen_edges_by_pt[0]) - 1
            shape = (n_pt, n_obs) + val_flat.shape[1:]
            axes = (self._pt_axis(), self._obs_axis(self.gen_edges_by_pt[0]), *extra_axes)
            return _fill(axes, val_flat.reshape(shape),
                         None if err_flat is None else err_flat.reshape(shape))

        hists = []
        offset = 0
        for edges in self.gen_edges_by_pt:
            nbins = len(edges) - 1
            sl = slice(offset, offset + nbins)
            axes = (self._obs_axis(edges), *extra_axes)
            hists.append(_fill(axes, val_flat[sl],
                               None if err_flat is None else err_flat[sl]))
            offset += nbins
        return hists

    def save_2d_uncertainty_summary(self):
        """Pickle a per-bin (pT x obs) uncertainty + prediction summary as hists.

        Each entry is a ``hist.Hist`` over (pt, obs) when the gen binning is the
        same in every pT slice (rho), or a list of per-pT 1D ``hist.Hist`` when
        it differs per slice (mass). Requires a full ``do_syst=True`` run for the
        systematic content (otherwise the syst hists are zero).

        Values are sum-normalized per pT slice (each slice integrates to 1, i.e.
        .project("pt") == 1), NOT divided by bin width; recover the density
        (dN/dx) at plot time with hist/mplhep ``binwnorm``. Raw spectra are
        absolute counts.

        Dict keys:
          - "unfolded": Weight hist of the per-pT-normalized unfolded result
            (the usual analysis output); its variance is the per-bin stat.
          - "unfolded_syst", "unfolded_syst_up"/"_down": pure (stat-excluded)
            systematic, symmetric and asymmetric. "unfolded_total_up"/"_down":
            stat (+) syst in quadrature.
          - "syst_breakdown": hist with extra (source, direction) StrCategory
            axes giving the per-source systematic (JES, JER, JMS, ..., Model
            Uncertainty); quadrature over sources reproduces unfolded_syst_*.
            "syst_sources" lists the names.
          - "pythia_gen_raw", "pythia_gen_ptnorm": PYTHIA gen prediction (Weight,
            value + MC-stat variance), absolute and per-pT-normalized.
          - "herwig_gen_raw", "herwig_gen_ptnorm": same for HERWIG; the raw
            HERWIG is rescaled per pT slice to the unfolded-data yield (its
            stored normalization is unreliable), "herwig_pt_scale" holds it.
          - "layout": "2d" or "per_pt"; "pt_edges".
        """
        suffix = "groomed" if self.groomed else "ungroomed"

        # ---- unfolded (per-pT normalized) + stat ----
        unfolded = np.concatenate(
            [np.asarray(r["unfolded"], dtype=float) for r in self.normalized_results]
        )
        stat = np.concatenate(
            [np.asarray(r["stat_unc"], dtype=float) for r in self.normalized_results]
        )

        # ---- systematic breakdown by analysis summary source (stat excluded) ----
        # Each summary source's up/down is the quadrature sum of its individual
        # nuisances per side, matching _compute_total_systematic so that the
        # quadrature over sources reproduces the analysis up/down totals.
        nominal_by_pt = [np.asarray(r["unfolded"], dtype=float) for r in self.normalized_results]
        sources = []
        for syst_name in self.systematics:
            if syst_name == "nominal":
                continue
            src = self._get_systematic_summary_name(syst_name, grouped=False)
            if src not in sources:
                sources.append(src)

        up_sq = {src: [] for src in sources}    # source -> per-pT list of sq-sums
        down_sq = {src: [] for src in sources}
        for i, nominal in enumerate(nominal_by_pt):
            varied = self.normalized_systematics[i]["unfolded"]
            up_acc = {src: np.zeros_like(nominal) for src in sources}
            down_acc = {src: np.zeros_like(nominal) for src in sources}
            for syst_name, arr in varied.items():
                if syst_name == "nominal":
                    continue
                src = self._get_systematic_summary_name(syst_name, grouped=False)
                _, variation = self._split_systematic_variation(syst_name)
                diff_sq = (np.asarray(arr, dtype=float) - nominal) ** 2
                if variation == "Down":
                    down_acc[src] += diff_sq
                else:  # "Up" or unpaired
                    up_acc[src] += diff_sq
            for src in sources:
                up_sq[src].append(up_acc[src])
                down_sq[src].append(down_acc[src])

        syst_per_source_up = {src: np.sqrt(np.concatenate(up_sq[src])) for src in sources}
        syst_per_source_down = {src: np.sqrt(np.concatenate(down_sq[src])) for src in sources}

        n_flat = len(unfolded)
        syst_up = np.sqrt(np.sum([syst_per_source_up[s] ** 2 for s in sources], axis=0)
                          if sources else np.zeros(n_flat))
        syst_down = np.sqrt(np.sum([syst_per_source_down[s] ** 2 for s in sources], axis=0)
                            if sources else np.zeros(n_flat))
        syst_sym = 0.5 * (syst_up + syst_down)
        total_up = np.sqrt(stat ** 2 + syst_up ** 2)
        total_down = np.sqrt(stat ** 2 + syst_down ** 2)

        # ---- generator predictions: raw + per-pT-normalized, with MC stat ----
        def _gen(val_flat, var_flat):
            val = np.asarray(val_flat, dtype=float)
            err = (np.sqrt(np.clip(np.asarray(var_flat, float), 0.0, None))
                   if var_flat is not None else np.zeros_like(val))
            nval, nerr = self._ptnorm_flat(val, err)
            return val, err, nval, nerr

        # PYTHIA/HERWIG gen predictions exist only on the zjet spec path; the
        # dijet/trijet prepared inputs carry none, so emit None for those.
        has_gen_pred = hasattr(self, "pythia_gen_val_flat") and hasattr(
            self, "herwig_gen_val_flat"
        )
        if has_gen_pred:
            py_raw, py_raw_err, py_norm, py_norm_err = _gen(
                self.pythia_gen_val_flat, getattr(self, "pythia_gen_var_flat", None)
            )

            # HERWIG's stored gen normalization is unreliable, so rescale each pT
            # slice of the raw HERWIG prediction to the absolute unfolded-data
            # yield in that slice. The per-pT-normalized HERWIG is unaffected (a
            # constant per-slice factor cancels in the normalization).
            hw_scale_flat = np.ones_like(np.asarray(self.herwig_gen_val_flat, dtype=float))
            data_abs = np.asarray(self.y_unf, dtype=float)
            herwig_abs = np.asarray(self.herwig_gen_val_flat, dtype=float)
            offset = 0
            for edges in self.gen_edges_by_pt:
                nbins = len(edges) - 1
                sl = slice(offset, offset + nbins)
                hw_sum = herwig_abs[sl].sum()
                if hw_sum != 0:
                    hw_scale_flat[sl] = data_abs[sl].sum() / hw_sum
                offset += nbins
            hw_var_scaled = (
                np.asarray(self.herwig_gen_var_flat, dtype=float) * hw_scale_flat ** 2
                if getattr(self, "herwig_gen_var_flat", None) is not None
                else None
            )
            hw_raw, hw_raw_err, hw_norm, hw_norm_err = _gen(
                herwig_abs * hw_scale_flat, hw_var_scaled
            )
        else:
            py_raw = py_raw_err = py_norm = py_norm_err = None
            hw_raw = hw_raw_err = hw_norm = hw_norm_err = None
            hw_scale_flat = None

        # Store sum-normalized (per-bin) values, NOT densities: each pT slice
        # integrates to 1 over its bins, so .project("pt") == 1. Recover the
        # density (dN/dx) at plot time with hist/mplhep binwnorm. The normalized
        # quantities above carry a 1/bin_width from _normalize_result/_ptnorm_flat,
        # so multiply it back out here. Raw/absolute spectra are left untouched.
        widths_flat = np.concatenate(
            [np.diff(np.asarray(e, dtype=float)) for e in self.gen_edges_by_pt]
        )
        unfolded = unfolded * widths_flat
        stat = stat * widths_flat
        syst_sym = syst_sym * widths_flat
        syst_up = syst_up * widths_flat
        syst_down = syst_down * widths_flat
        total_up = total_up * widths_flat
        total_down = total_down * widths_flat
        for source in sources:
            syst_per_source_up[source] = syst_per_source_up[source] * widths_flat
            syst_per_source_down[source] = syst_per_source_down[source] * widths_flat
        if has_gen_pred:
            py_norm = py_norm * widths_flat
            py_norm_err = py_norm_err * widths_flat
            hw_norm = hw_norm * widths_flat
            hw_norm_err = hw_norm_err * widths_flat

        # ---- build hist objects (2D when uniform, per-pT list when ragged) ----
        summary = {
            "unfolded": self._hist_from_flat(unfolded, stat),
            "unfolded_syst": self._hist_from_flat(syst_sym),
            "unfolded_syst_up": self._hist_from_flat(syst_up),
            "unfolded_syst_down": self._hist_from_flat(syst_down),
            "unfolded_total_up": self._hist_from_flat(total_up),
            "unfolded_total_down": self._hist_from_flat(total_down),
            "pythia_gen_raw": self._hist_from_flat(py_raw, py_raw_err) if has_gen_pred else None,
            "pythia_gen_ptnorm": self._hist_from_flat(py_norm, py_norm_err) if has_gen_pred else None,
            "herwig_gen_raw": self._hist_from_flat(hw_raw, hw_raw_err) if has_gen_pred else None,
            "herwig_gen_ptnorm": self._hist_from_flat(hw_norm, hw_norm_err) if has_gen_pred else None,
            "herwig_pt_scale": self._hist_from_flat(hw_scale_flat) if has_gen_pred else None,
            "syst_sources": list(sources),
            "layout": "2d" if self._slices_share_binning() else "per_pt",
            "pt_edges": np.asarray(self.pt_edges, dtype=float),
        }

        # systematic breakdown with (source, direction) category axes
        if sources:
            syst_stack = np.zeros((n_flat, len(sources), 2))
            for j, source in enumerate(sources):
                syst_stack[:, j, 0] = syst_per_source_up[source]
                syst_stack[:, j, 1] = syst_per_source_down[source]
            extra = (
                hist.axis.StrCategory(sources, name="source"),
                hist.axis.StrCategory(["up", "down"], name="direction"),
            )
            summary["syst_breakdown"] = self._hist_from_flat(syst_stack, extra_axes=extra)
        else:
            summary["syst_breakdown"] = None

        save_path = self._relocate_output(f"uncertainty_summary_2d_{suffix}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as handle:
            pkl.dump(summary, handle)
        print(f"Saved 2D uncertainty summary (hist) to {save_path}")

    def plot_unfolded_unrolled_2d(self, show=True):
        """Full unrolled (mass x pT) unfolded data vs PYTHIA gen, absolute.

        Mirrors the reco-level input overlay, but at gen level: the unfolded
        result and the PYTHIA truth are shown at their original (un-normalized)
        absolute values on a linear y-axis, with a Data/MC ratio pad and dotted
        pT-slice dividers. The 2D-normalized arrays are still computed/saved by
        ``_compute_2d_normalized_result`` / ``save_2d_unfolded``.
        """
        hep.style.use("CMS")
        unf = np.asarray(self.unfolded_abs_flat, dtype=float)
        unf_err = np.asarray(self.unfolded_abs_err_flat, dtype=float)
        true = np.asarray(self.y_true, dtype=float)
        n = len(unf)
        x = np.arange(n)

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        ax.step(x, true, where="mid", color="blue", lw=1.4, label="PYTHIA8 (gen)")
        ax.errorbar(
            x, unf, yerr=unf_err, fmt="o", ms=3, color="black", lw=0.8,
            label="Unfolded data",
        )
        ax.set_ylabel("Unfolded events (absolute)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=13)
        hep.cms.label(
            self.cms_label, data=True, lumi=self._lumi_label(),
            com=self._com_label(), ax=ax, fontsize=18,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(unf, true, out=np.zeros_like(unf), where=true > 0)
            ratio_err = np.divide(unf_err, true, out=np.zeros_like(unf), where=true > 0)
        axr.axhline(1, color="gray", ls="--", lw=1)
        axr.errorbar(x, ratio, yerr=ratio_err, fmt="o", ms=3, color="black")
        axr.set_ylabel("Data / MC")
        axr.set_ylim(0.5, 1.5)
        axr.set_xlabel(
            "Unrolled bin index  (mass within each $p_T$ slice, slices concatenated)"
        )
        axr.set_xlim(-1, n)

        counts_per_slice = [len(edges) - 1 for edges in self.gen_edges_by_pt]
        boundaries = np.cumsum(counts_per_slice)
        pt_edges = np.asarray(self.pt_edges, dtype=float)
        start = 0
        for i, b in enumerate(boundaries):
            for a in (ax, axr):
                a.axvline(b - 0.5, color="steelblue", ls=":", lw=1, alpha=0.7)
            lo = pt_edges[i]
            hi = pt_edges[i + 1] if i + 1 < len(pt_edges) else np.inf
            lbl = f"{lo:g}-{'∞' if not np.isfinite(hi) or hi >= 13000 else f'{hi:g}'}"
            ax.text(
                (start + b - 1) / 2, ax.get_ylim()[1] * 0.5, lbl,
                ha="center", va="top", fontsize=9, color="steelblue", rotation=90,
            )
            start = b

        suffix = "groomed" if self.groomed else "ungroomed"
        save_path = f"./{self.spec.output_dir}unfold/unfolded_unrolled_2d_{suffix}.pdf"
        self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def _jackknife_convergence_fractions(self):
        """Per-bin fractional jackknife stat vs number of replicas.

        Recomputes the grouped-jackknife SE (the same estimator as
        _compute_stat_unc at the full count) of the unfolded result over the
        first n input-data and response-matrix jackknife replicas, for
        n = 2..N. The per-count scale on the sample std (ddof=1) is (n-1)/sqrt(n),
        equal to sqrt(n-1) applied to the population std used in _compute_stat_unc
        (9/sqrt(10) ~ 2.85 at n=10). Returns ``(ns, input_frac, matrix_frac,
        total_frac)`` with the fraction arrays shaped (len(ns), n_gen_bins); or
        None if no replicas.
        """
        input_reps = np.asarray(self.y_unf_jk_input_list, dtype=float)
        matrix_reps = np.asarray(self.y_unf_jk_matrix_list, dtype=float)
        if input_reps.ndim != 2 or matrix_reps.ndim != 2:
            return None
        n_rep = min(len(input_reps), len(matrix_reps))
        if n_rep < 2:
            return None
        ns = np.arange(2, n_rep + 1)
        nominal = np.abs(np.asarray(self.y_unf, dtype=float))

        def fracs(reps):
            rows = []
            for n in ns:
                jk_scale = (n - 1) / np.sqrt(n)
                std = jk_scale * np.std(reps[:n], axis=0, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rows.append(np.where(nominal > 0, std / nominal, 0.0))
            return np.asarray(rows)

        input_frac = fracs(input_reps)
        matrix_frac = fracs(matrix_reps)
        total_frac = np.sqrt(input_frac**2 + matrix_frac**2)
        return ns, input_frac, matrix_frac, total_frac

    def plot_jackknife_convergence(self, show=True):
        """Full sheet of jackknife stat convergence, one panel per (pT, rho) bin.

        Each panel shows the fractional statistical uncertainty (input/data,
        response-matrix/MC, and their quadrature sum) as a function of the
        number of jackknife replicas used (2..N). A flattening curve indicates
        the replica count is sufficient.
        """
        conv = self._jackknife_convergence_fractions()
        if conv is None:
            return
        ns, input_frac, matrix_frac, total_frac = conv

        counts = [len(edges) - 1 for edges in self.gen_edges_by_pt]
        n_pt = len(counts)
        n_col = max(counts)
        fig, axes = plt.subplots(
            n_pt, n_col, figsize=(2.4 * n_col, 2.3 * n_pt),
            squeeze=False, sharex=True,
        )
        offset = 0
        for pt_i, count in enumerate(counts):
            edges = np.asarray(self.gen_edges_by_pt[pt_i], dtype=float)
            lo = self.pt_edges[pt_i]
            hi = self.pt_edges[pt_i + 1] if pt_i + 1 < len(self.pt_edges) else np.inf
            pt_lbl = f"{lo:g}-{'∞' if not np.isfinite(hi) or hi >= 13000 else f'{hi:g}'} GeV"
            for r in range(n_col):
                ax = axes[pt_i][r]
                if r >= count:
                    ax.axis("off")
                    continue
                j = offset + r
                ax.plot(ns, input_frac[:, j], "o-", ms=3, lw=1, color="#1f77b4", label="input (data)")
                ax.plot(ns, matrix_frac[:, j], "s-", ms=3, lw=1, color="#ff7f0e", label="matrix (MC)")
                ax.plot(ns, total_frac[:, j], "^-", ms=3, lw=1.3, color="k", label="total")
                ax.axhline(total_frac[-1, j], color="gray", ls=":", lw=0.8)
                ax.set_title(rf"$\rho\in[{edges[r]:g},{edges[r+1]:g})$", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_ylabel(f"{pt_lbl}\nfrac. stat", fontsize=8)
                if pt_i == n_pt - 1:
                    ax.set_xlabel("# replicas", fontsize=8)
            offset += count

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=10, ncol=3)
        mode = "groomed" if self.groomed else "ungroomed"
        fig.suptitle(
            f"Jackknife stat-uncertainty convergence vs # replicas ({mode})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = f"./{self.spec.output_dir}unfold/jackknife_convergence_{mode}.pdf"
        self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_jackknife_convergence_pt_avg(self, show=True, pt_min=None):
        """Jackknife stat convergence averaged over rho within each pT bin.

        Same diagnostic as plot_jackknife_convergence, but instead of one panel
        per (pT, rho) gen bin it collapses the rho dimension: each panel is one
        pT slice and shows the mean over that slice's rho bins of the fractional
        statistical uncertainty (input/data, response-matrix/MC, and their
        quadrature sum) as a function of the number of jackknife replicas used
        (2..N). A flattening curve indicates the replica count is sufficient.

        Drawn in the mplhep CMS style for the analysis note. ``pt_min`` drops
        pT slices whose upper edge is <= pt_min from the panels (the rho average
        within each kept slice is unaffected).
        """
        conv = self._jackknife_convergence_fractions()
        if conv is None:
            return
        ns, input_frac, matrix_frac, total_frac = conv

        counts = [len(edges) - 1 for edges in self.gen_edges_by_pt]
        # Per-slice rho-averaged curves, keeping the flat-array offsets aligned
        # even for slices we end up dropping from the panels.
        panels = []
        offset = 0
        for pt_i, count in enumerate(counts):
            sl = slice(offset, offset + count)
            offset += count
            lo = self.pt_edges[pt_i]
            hi = self.pt_edges[pt_i + 1] if pt_i + 1 < len(self.pt_edges) else np.inf
            if pt_min is not None and np.isfinite(hi) and hi <= pt_min:
                continue
            panels.append((
                lo, hi, count,
                input_frac[:, sl].mean(axis=1),
                matrix_frac[:, sl].mean(axis=1),
                total_frac[:, sl].mean(axis=1),
            ))

        if not panels:
            return

        n_panel = len(panels)
        n_col = min(n_panel, 4)
        n_row = int(np.ceil(n_panel / n_col))
        mode = "groomed" if self.groomed else "ungroomed"

        with plt.style.context(hep.style.CMS):
            fig, axes = plt.subplots(
                n_row, n_col, figsize=(5.6 * n_col, 5.4 * n_row),
                squeeze=False, sharex=True, layout="constrained",
            )
            for idx, (lo, hi, count, inp, mat, tot) in enumerate(panels):
                ax = axes[idx // n_col][idx % n_col]
                ax.plot(ns, inp, "o-", ms=8, lw=2.0, color="#5790fc", label="Input (data)")
                ax.plot(ns, mat, "s-", ms=8, lw=2.0, color="#f89c20", label="Matrix (MC)")
                ax.plot(ns, tot, "^-", ms=9, lw=2.6, color="k", label="Total")
                ax.axhline(tot[-1], color="gray", ls=":", lw=1.4)

                if np.isfinite(hi) and hi < 13000:
                    pt_lbl = rf"${lo:g} < p_{{T}} < {hi:g}$ GeV"
                else:
                    pt_lbl = rf"$p_{{T}} > {lo:g}$ GeV"
                ax.text(
                    0.05, 0.95,
                    pt_lbl + "\n" + rf"$\langle\,{count}\ \rho$ bins$\,\rangle$",
                    transform=ax.transAxes, ha="left", va="top", fontsize=18,
                )

                vmax = max(float(inp.max()), float(mat.max()), float(tot.max()))
                ax.set_ylim(0.0, vmax * 1.45)  # headroom for the pT label
                ax.set_xlim(ns[0] - 0.4, ns[-1] + 0.4)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=17)
                ax.set_xlabel("Number of jackknife replicas", fontsize=19)

            for k in range(n_panel, n_row * n_col):
                axes[k // n_col][k % n_col].axis("off")

            # CMS multi-panel convention: "CMS Internal" over the first panel,
            # the luminosity over the last panel.
            first_ax = axes[0][0]
            last_ax = axes[(n_panel - 1) // n_col][(n_panel - 1) % n_col]
            first_ax.set_ylabel(
                "Mean fractional\nstatistical uncertainty", fontsize=19
            )
            hep.cms.label(self.cms_label, data=True, ax=first_ax, fontsize=20, rlabel="")
            last_ax.text(
                1.0, 1.01,
                rf"{self._lumi_label()} fb$^{{-1}}$ ({self._com_label()} TeV)",
                transform=last_ax.transAxes, ha="right", va="bottom", fontsize=18,
            )

            handles, labels = first_ax.get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="outside lower center", ncol=3, fontsize=17,
                frameon=False,
            )
            save_path = (
                f"./{self.spec.output_dir}unfold/"
                f"jackknife_convergence_pt_avg_{mode}.pdf"
            )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def _normalization_jacobian(self):
        """Jacobian of the per-pT-slice normalization y_i = x_i / (w_i * S_k).

        x is the absolute unfolded spectrum, w_i the gen bin width, and
        S_k the sum of x over the pT slice containing bin i, so
        dy_i/dx_j = (delta_ij - x_i / S_k) / (w_i * S_k) within a slice and
        zero across slices (each slice is normalized by its own sum).
        """
        n = len(self.y_unf)
        jacobian = np.zeros((n, n))
        offset = 0
        for k, edges in enumerate(self.gen_edges_by_pt):
            nbins = len(edges) - 1
            block = slice(offset, offset + nbins)
            x = np.asarray(self.y_unf[block], dtype=float)
            widths = np.diff(edges)
            # S_k sums shown bins only (all bins unless normalize_over_shown):
            # dy_i/dx_j picks up the -x_i/S term only for j in the denominator.
            mask = self._shown_gen_mask(k).astype(float)
            total = float((x * mask).sum())
            if total != 0:
                jacobian[block, block] = (
                    np.eye(nbins) - np.outer(x, mask) / total
                ) / (widths[:, None] * total)
            offset += nbins
        return jacobian

    def _absolute_stat_covariances(self):
        """Covariances of the absolute unfolded spectrum: (input, matrix).

        Uses TUnfold's exactly propagated covariances (input data stat,
        response MC stat) rather than the jackknife replicas: a covariance
        estimated from 10 replicas has rank <= 9 over the full gen vector,
        which fabricates near-unit correlations. The same matrices feed the
        legacy correlation plot, so only the normalization treatment differs
        between the legacy and jacobian tags' correlation figures.
        """
        return (
            np.array(self.cov_data_np, copy=True),
            np.array(self.cov_uncorr_np, copy=True),
        )

    def _compute_normalized_stat_covariance(self):
        """Propagate the absolute stat covariances through the normalization.

        Stores covariances of the normalized, per-bin-width result (the same
        units as ``normalized_results[i]['unfolded']``). The Jacobian is block
        diagonal per pT slice, but cross-slice correlations of the absolute
        covariance survive in the off-diagonal blocks. Within each slice the
        sum constraint makes the covariance singular (one zero eigenvalue) and
        introduces negative correlations; both are expected for a normalized
        measurement.
        """
        jacobian = self._normalization_jacobian()
        cov_input_abs, cov_matrix_abs = self._absolute_stat_covariances()
        self.norm_cov_input = jacobian @ cov_input_abs @ jacobian.T
        self.norm_cov_matrix = jacobian @ cov_matrix_abs @ jacobian.T
        self.norm_cov_stat = self.norm_cov_input + self.norm_cov_matrix

    def get_systematic_covariance(self):
        """Rank-1 systematic covariance of the normalized result.

        Each source contributes the outer product of its normalized shift,
        symmetrized as (up - down)/2 when both variations exist. The diagonal
        is consistent with the quadrature sum used for the plotted bands.
        """
        nominal_flat = np.concatenate(
            [np.asarray(result["unfolded"], dtype=float) for result in self.normalized_results]
        )
        varied_flat = {}
        for systematic in self.systematics:
            if systematic == "nominal":
                continue
            varied_flat[systematic] = np.concatenate(
                [
                    np.asarray(per_pt["unfolded"][systematic], dtype=float)
                    for per_pt in self.normalized_systematics
                ]
            )

        cov_syst = np.zeros((len(nominal_flat), len(nominal_flat)))
        seen = set()
        for systematic in varied_flat:
            if systematic in seen:
                continue
            if systematic.endswith("Up") or systematic.endswith("Down"):
                source = systematic[:-2] if systematic.endswith("Up") else systematic[:-4]
                up = varied_flat.get(source + "Up")
                down = varied_flat.get(source + "Down")
                seen.update({source + "Up", source + "Down"} & varied_flat.keys())
                if up is not None and down is not None:
                    shift = 0.5 * (up - down)
                else:
                    shift = (up if up is not None else down) - nominal_flat
            else:
                seen.add(systematic)
                shift = varied_flat[systematic] - nominal_flat
            cov_syst += np.outer(shift, shift)
        return cov_syst

    def get_total_covariance(self):
        """Total covariance of the normalized result (stat + systematics)."""
        return self.norm_cov_stat + self.get_systematic_covariance()

    # Pairs "<base>Up[_corr|_uncorr_YYYY]" with the matching Down key. The
    # Up/Down must sit immediately before the era suffix: a naive
    # replace("Up", "Down") would corrupt e.g. JES_PileUpPtBB.
    _UPDOWN_KEY = re.compile(r"^(.+)(Up|Down)((?:_corr|_uncorr_\d+)?)$")

    def _results_chi2_covariance(self):
        """Covariance of the normalized result for the quoted data-vs-MC chi2.

        Flat over all gen bins (normalized, per-bin-width units), cached.
        Three terms:

        * statistics: ``norm_cov_stat`` (TUnfold input + matrix-stat
          covariances through the normalization Jacobian);
        * detector systematics: rank-1 outer products of the normalized
          shifts, symmetrized (up-down)/2 when both legs exist. Sources
          superseded by the model envelope (herwig/fsr/isr, and the raw
          ``model_*`` variations) are excluded when the envelope is active,
          mirroring the plotted band;
        * modelling: the coherent PS/HAD shift vectors of the bottom-line
          construction, scaled to the normalized nominal and **projected onto
          the normalization-preserving subspace** of each pT slice
          (w . s = 0 over the shown bins, w = bin widths). The stored
          fractional shifts carry a rate-like component that a normalized
          measurement cannot fluctuate in; every detector source loses that
          component by construction (each variation is renormalized by its
          own sum), so the model term must as well.

        Within each slice the sum constraint leaves one exact null direction,
        so consumers must use a pseudo-inverse and ndof = n_bins - 1.
        """
        cached = getattr(self, "_results_chi2_cov", None)
        if cached is not None:
            return cached

        nominal_flat = np.concatenate(
            [np.asarray(r["unfolded"], dtype=float) for r in self.normalized_results]
        )
        covariance = np.array(self.norm_cov_stat, copy=True)

        use_model_envelope = getattr(self.spec, "model_envelope", False)
        superseded = ("herwig", "fsr", "isr", "model_") if use_model_envelope else ("model_",)
        shifts = {
            syst: np.concatenate(
                [
                    np.asarray(per_pt["unfolded"][syst], dtype=float)
                    for per_pt in self.normalized_systematics
                ]
            )
            for syst in self.systematics
            if syst != "nominal" and not syst.startswith(superseded)
        }
        done = set()
        for syst, varied in shifts.items():
            if syst in done:
                continue
            match = self._UPDOWN_KEY.match(syst)
            partner = None
            if match:
                base, direction, suffix = match.groups()
                other = "Down" if direction == "Up" else "Up"
                partner = f"{base}{other}{suffix}"
            if partner is not None and partner in shifts:
                done.update({syst, partner})
                up = varied if match.group(2) == "Up" else shifts[partner]
                down = shifts[partner] if match.group(2) == "Up" else varied
                shift = 0.5 * (up - down)
            else:
                done.add(syst)
                shift = varied - nominal_flat
            covariance += np.outer(shift, shift)

        ps_frac = getattr(self, "model_ps_shift_flat", None)
        had_frac = getattr(self, "model_had_shift_flat", None)
        if ps_frac is not None and had_frac is not None:
            offsets = np.concatenate(
                [[0], np.cumsum([len(e) - 1 for e in self.gen_edges_by_pt])]
            ).astype(int)
            for vector_frac in (ps_frac, had_frac):
                vector = np.asarray(vector_frac, dtype=float) * nominal_flat
                for i in range(len(self.gen_edges_by_pt)):
                    block = slice(offsets[i], offsets[i + 1])
                    mask = self._shown_gen_mask(i)
                    widths = np.diff(np.asarray(self.gen_edges_by_pt[i], float))
                    y_slice = nominal_flat[block]
                    # w . y = 1 over the shown bins by construction, so
                    # subtracting (w . s) y removes exactly the component
                    # that violates the sum constraint.
                    leak = float(np.sum(widths[mask] * vector[block][mask]))
                    vector[block] = np.where(
                        mask, vector[block] - leak * y_slice, vector[block]
                    )
                covariance += np.outer(vector, vector)

        self._results_chi2_cov = covariance
        return covariance

    def save_normalized_covariance(self):
        """Write the normalized-result covariances to an NPZ next to the plots."""
        suffix = "groomed" if self.groomed else "ungroomed"
        nominal_flat = np.concatenate(
            [np.asarray(result["unfolded"], dtype=float) for result in self.normalized_results]
        )
        cov_syst = self.get_systematic_covariance()
        save_path = self._relocate_output(f"normalized_covariance_{suffix}.npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            normalized=nominal_flat,
            cov_stat_input=self.norm_cov_input,
            cov_stat_matrix=self.norm_cov_matrix,
            cov_stat=self.norm_cov_stat,
            cov_syst=cov_syst,
            cov_total=self.norm_cov_stat + cov_syst,
            pt_edges=np.asarray(self.pt_edges, dtype=float),
            gen_bins_per_pt=np.asarray(
                [len(edges) - 1 for edges in self.gen_edges_by_pt], dtype=int
            ),
            tau=float(self.tau or 0.0),
        )
        print(f"Saved normalized covariances to {save_path}")

    def _compute_total_systematic(self):
        print("Computing total systematic uncertainty...")
        self._compute_normalized_stat_covariance()
        use_jacobian = self.stat_propagation == "jacobian"
        if use_jacobian:
            input_std_pt_binned = unflatten_gen_by_pt(
                np.sqrt(np.clip(np.diag(self.norm_cov_input), 0.0, None)),
                self.gen_edges_by_pt,
            )
            matrix_std_pt_binned = unflatten_gen_by_pt(
                np.sqrt(np.clip(np.diag(self.norm_cov_matrix), 0.0, None)),
                self.gen_edges_by_pt,
            )
        # Compute total systematic uncertainty for each pt bin
        # load from file
        #herwig_unc = np.load("./inputs/zjet/mass/herwig_closure_unc_mass_groomed.npy") if self.groomed else np.load("./inputs/zjet/mass/herwig_closure_unc_mass_ungroomed.npy")
        #self.herwig_unc = herwig_unc
        use_model_envelope = getattr(self.spec, "model_envelope", False)
        if use_model_envelope:
            # Offline column-scaled model variations (Vincia/CR/frag) unfolded
            # against the offline w=1 baseline; FSR joins the envelope below
            # from the stored PSWeight unfolds. Supersedes the legacy
            # herwig-diff model term (and the fsr/isr quadrature entries).
            from unfold.tools.model_envelope import (
                compute_model_shifts, group_model_shifts,
            )
            print("Computing model envelope (Vincia/CR/frag column-scaled responses)...")
            raw_shifts = compute_model_shifts(self)
            self.model_shift_components = group_model_shifts(
                raw_shifts, len(self.gen_edges_by_pt))
            self.model_signed_shifts = raw_shifts
            self.model_signed_shifts["fsrUp"] = {}
            self.model_signed_shifts["fsrDown"] = {}
        # Systematic-axis entries folded into the model envelope instead of the
        # quadrature sum when model_envelope is on.
        _model_superseded = ("herwig", "fsr", "isr")
        for i in range(len(self.normalized_results)):
            nominal = self.normalized_results[i]['unfolded']
            syst_up_total = np.zeros_like(nominal)
            syst_down_total = np.zeros_like(nominal)

            for syst in self.systematics:
                if use_model_envelope and syst.startswith(_model_superseded):
                    continue
                if 'Down' in syst:
                    if syst.startswith('herwig'):
                        ## This is for taking just the difference
                        syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_down = np.abs(syst_down - nominal)
                        #print("Systematic Down:", syst, "Diff Down:", diff_down)
                        syst_down_total += diff_down**2

                        ## This is for fethcing uncertainty from non-closure of herwig
                        # diff_down = herwig_unc[i] * nominal
                        # syst_down_total += diff_down**2
                    else:    
                        syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_down = np.abs(syst_down - nominal)
                        #print("Systematic Down:", syst, "Diff Down:", diff_down)
                        syst_down_total += diff_down**2
                else:
                    if syst=='nominal':
                        continue
                    if syst.startswith('herwig'):
                        #print("New Herwig adopted")
                        ## Difference way
                        syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_up = np.abs(syst_up - nominal)
                        #print("Systematic Up:", syst, "Diff Up:", diff_up)
                        syst_up_total += diff_up**2

                        ## Fetching uncertainty from non-closure of herwig
                        # diff_up = herwig_unc[i] * nominal
                        # syst_up_total += diff_up**2
                    else:
                        syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_up = np.abs(syst_up - nominal)
                        #print("Systematic up:", syst, "Diff up:", diff_up)
                        syst_up_total += diff_up**2
            if use_model_envelope:
                # Model term: per-bin max over {Vincia, CR, frag} re-unfold
                # shifts and the FSR PSWeight shift, added symmetrically.
                # Bins holding <0.5% of the normalized slice are below
                # measurement sensitivity (e.g. the ungroomed [-10,-2.5]
                # catch-all, ~0.3-1% and consistent with zero): a fractional
                # shift there is noise-over-noise, so the model fraction is
                # zeroed (its absolute band contribution is negligible either
                # way).
                widths = np.diff(np.asarray(self.gen_edges_by_pt[i], float))
                content = np.abs(nominal) * widths
                # positive density AND >0.5% of the slice: negative unfolded
                # bins are consistent with zero, so no relative uncertainty
                # is defined there.
                sensitive = (np.asarray(nominal) > 0) & (
                    content > 5e-3 * max(content.sum(), 1e-300))
                fsr_frac = np.zeros_like(nominal)
                for var in ("fsrUp", "fsrDown"):
                    varied = self.normalized_systematics[i]['unfolded'].get(
                        var, np.zeros_like(nominal))
                    if np.any(varied):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            signed_frac = np.divide(
                                varied - nominal, nominal,
                                out=np.zeros_like(nominal), where=nominal != 0)
                        self.model_signed_shifts[var][i] = np.where(
                            sensitive, signed_frac, 0.0)
                        fsr_frac = np.maximum(fsr_frac, np.abs(signed_frac))
                    else:
                        self.model_signed_shifts[var][i] = np.zeros_like(nominal)
                fsr_frac = np.where(sensitive, fsr_frac, 0.0)
                for name in ("Vincia", "CR", "frag"):
                    comp = np.where(
                        sensitive, self.model_shift_components[name][i], 0.0)
                    self.model_shift_components[name][i] = comp
                for source in ("vincia", "cr1", "cr2", "fraghard", "fragsoft"):
                    self.model_signed_shifts[source][i] = np.where(
                        sensitive, self.model_signed_shifts[source][i], 0.0)
                # ARC round-2 prescription (plan WS2): the shower and the
                # hadronization modelling are two INDEPENDENT uncertainties,
                # each an envelope over its own variations, combined in
                # quadrature -- never one max envelope over everything.
                #   PS leg  = Vincia shower swap (hadronization kept nominal),
                #             enveloped with the FSR PSWeight scale variation;
                #   HAD leg = hadronization-parameter variations on the default
                #             shower (CR modes 1/2, Lund frag hard/soft).
                ps_frac = np.maximum(
                    fsr_frac, self.model_shift_components["Vincia"][i])
                had_frac = np.maximum(
                    self.model_shift_components["CR"][i],
                    self.model_shift_components["frag"][i])
                model_frac = np.sqrt(ps_frac**2 + had_frac**2)
                if not hasattr(self, "model_fsr_frac"):
                    self.model_fsr_frac = {}
                if not hasattr(self, "model_ps_frac"):
                    self.model_ps_frac = {}
                    self.model_had_frac = {}
                self.model_fsr_frac[i] = fsr_frac
                self.model_ps_frac[i] = ps_frac
                self.model_had_frac[i] = had_frac
                model_unc = model_frac * np.abs(nominal)
                syst_up_total += model_unc**2
                syst_down_total += model_unc**2
                self.normalized_results[i]['model_unc_frac'] = model_frac
                self.normalized_results[i]['model_ps_frac'] = ps_frac
                self.normalized_results[i]['model_had_frac'] = had_frac
            if use_jacobian:
                # Errors of the normalized result itself: the Jacobian removes
                # the fluctuation common to all bins of a pT slice.
                input_stat_unc = input_std_pt_binned[i]
                matrix_stat_unc = matrix_std_pt_binned[i]
                with np.errstate(divide='ignore', invalid='ignore'):
                    input_stat_frac = np.abs(np.divide(
                        input_stat_unc, nominal,
                        out=np.zeros_like(input_stat_unc), where=nominal != 0,
                    ))
                    matrix_stat_frac = np.abs(np.divide(
                        matrix_stat_unc, nominal,
                        out=np.zeros_like(matrix_stat_unc), where=nominal != 0,
                    ))
                stat_frac = np.sqrt(input_stat_frac**2 + matrix_stat_frac**2)
            else:
                input_stat_unc = self.input_stat_unc_pt_binned[i] * nominal
                matrix_stat_unc = self.matrix_stat_unc_pt_binned[i] * nominal
                input_stat_frac = self.input_stat_unc_pt_binned[i]
                matrix_stat_frac = self.matrix_stat_unc_pt_binned[i]
                stat_frac = self.stat_unc_pt_binned[i]
            stat_unc = np.sqrt(input_stat_unc**2 + matrix_stat_unc**2)

            syst_up_total += stat_unc**2
            syst_down_total += stat_unc**2
            # Take sqrt of sum of squares for total uncertainty
            total_up_unc = np.sqrt(syst_up_total)
            total_down_unc = np.sqrt(syst_down_total)
            self.normalized_results[i]['syst_unc'] = {
            'up': total_up_unc,
            'down': total_down_unc
            }
            self.normalized_results[i]['input_stat_unc_frac'] = input_stat_frac
            self.normalized_results[i]['matrix_stat_unc_frac'] = matrix_stat_frac
            self.normalized_results[i]['stat_unc_frac'] = stat_frac
            self.normalized_results[i]['input_stat_unc'] = input_stat_unc
            self.normalized_results[i]['matrix_stat_unc'] = matrix_stat_unc
            self.normalized_results[i]['stat_unc'] = stat_unc

        # Flat per-gen-bin model-uncertainty fraction, concatenated over pT
        # slices in the same order as gen_edges_by_pt -> used by the bottom-line
        # test to fold the modelling uncertainty into the unfolded-data
        # covariance (ARC round-2 request). Zeros when no model envelope.
        if use_model_envelope:
            self.model_unc_frac_flat = np.concatenate([
                np.asarray(
                    self.normalized_results[i].get(
                        "model_unc_frac",
                        np.zeros(len(self.gen_edges_by_pt[i]) - 1)),
                    float)
                for i in range(len(self.normalized_results))
            ])
            # Preserve physical bin-to-bin correlations in the bottom-line
            # covariance. Select one coherent, signed variation for each model
            # leg using its global L2 norm over the full unfolded space; never
            # splice a different envelope source into each bin.
            def _flat_signed(source):
                return np.concatenate([
                    np.asarray(self.model_signed_shifts[source][i], float)
                    for i in range(len(self.normalized_results))
                ])

            ps_candidates = {
                source: _flat_signed(source)
                for source in ("vincia", "fsrUp", "fsrDown")
            }
            had_candidates = {
                source: _flat_signed(source)
                for source in ("cr1", "cr2", "fraghard", "fragsoft")
            }
            # Source selection mask: the production prescription uses exactly
            # the publication phase space. The 185-200 migration bin and the
            # hidden low-rho buffers remain in the response but cannot decide
            # which variation represents the reported covariance.
            shown_mask_parts = []
            shown_local_masks = []
            floors = self._bl_shown_floors()
            for i, edges in enumerate(self.gen_edges_by_pt):
                lower_edges = np.asarray(edges[:-1], float)
                local = ((self.pt_edges[i] >= 200)
                         & (lower_edges >= floors[i] - 1e-9))
                shown_local_masks.append(local)
                shown_mask_parts.append(local)
            shown_mask = np.concatenate(shown_mask_parts)
            scope = getattr(
                self.spec, "model_covariance_scope", "global_shown")
            selection_mask = (
                np.ones_like(shown_mask, dtype=bool)
                if scope == "global_all" else shown_mask
            )

            def _selected_norm(vector, mask):
                return np.linalg.norm(np.asarray(vector)[mask])

            self.model_ps_source = max(
                ps_candidates,
                key=lambda source: _selected_norm(
                    ps_candidates[source], selection_mask))
            self.model_had_source = max(
                had_candidates,
                key=lambda source: _selected_norm(
                    had_candidates[source], selection_mask))
            self.model_ps_shift_flat = ps_candidates[self.model_ps_source]
            self.model_had_shift_flat = had_candidates[self.model_had_source]

            # Diagnostic prescription with independent PS/HAD nuisances per
            # published pT slice. Each vector is embedded in the full flat
            # space, making _model_cov_unfolded block diagonal across pT.
            offsets = np.concatenate([
                [0], np.cumsum([
                    len(edges) - 1 for edges in self.gen_edges_by_pt
                ])
            ]).astype(int)
            self.model_ps_shifts_by_pt_flat = []
            self.model_had_shifts_by_pt_flat = []
            self.model_ps_sources_by_pt = {}
            self.model_had_sources_by_pt = {}
            for i, local_mask in enumerate(shown_local_masks):
                block = slice(offsets[i], offsets[i + 1])
                if self.pt_edges[i] < 200 or not np.any(local_mask):
                    continue
                ps_source = max(
                    ps_candidates,
                    key=lambda source: np.linalg.norm(
                        ps_candidates[source][block][local_mask]))
                had_source = max(
                    had_candidates,
                    key=lambda source: np.linalg.norm(
                        had_candidates[source][block][local_mask]))
                ps_vector = np.zeros_like(self.model_ps_shift_flat)
                had_vector = np.zeros_like(self.model_had_shift_flat)
                ps_vector[block] = ps_candidates[ps_source][block]
                had_vector[block] = had_candidates[had_source][block]
                self.model_ps_shifts_by_pt_flat.append(ps_vector)
                self.model_had_shifts_by_pt_flat.append(had_vector)
                self.model_ps_sources_by_pt[i] = ps_source
                self.model_had_sources_by_pt[i] = had_source
            print(
                "Bottom-line model covariance: correlated coherent shifts "
                f"scope={scope}, PS={self.model_ps_source}, "
                f"HAD={self.model_had_source}"
            )
        else:
            self.model_unc_frac_flat = None
            self.model_ps_shift_flat = None
            self.model_had_shift_flat = None

    def plot_statistical_fraction(self, show=True):
        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            plt.figure()
            pt_bin = result['pt_bin']
            input_stat_fraction = result['input_stat_unc_frac']
            matrix_stat_fraction = result['matrix_stat_unc_frac']

            hep.histplot(
                input_stat_fraction[1:],
                self.gen_edges_by_pt[i][1:],
                label="Input statistical Uncertainty",
                ls="--",
            )
            # The matrix (response-stat) term only exists when jackknife
            # replicas were run; the covariance fallback leaves it identically
            # zero, and drawing that flat line just advertises a missing input.
            draw_matrix = (
                self.response_matrix_stat_available
                and bool(np.any(matrix_stat_fraction[1:]))
            )
            if draw_matrix:
                hep.histplot(
                    matrix_stat_fraction[1:],
                    self.gen_edges_by_pt[i][1:],
                    label="Matrix uncertainty",
                    ls="-.",
                )

            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
            hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
            plt.xlim(*self._observable_xlim(i))
            # The y autoscale sees the buffer bins hidden outside the xlim
            # window (groomed panels blew up to 2-6 while the shown curve sits
            # at a few percent) -- scale to the shown bins only.
            edges = np.asarray(self.gen_edges_by_pt[i][1:], float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            xlo, xhi = self._observable_xlim(i)
            visible = (centers >= xlo) & (centers <= xhi)
            if np.any(visible):
                curves = [np.asarray(input_stat_fraction[1:], float)[visible]]
                if draw_matrix:
                    curves.append(
                        np.asarray(matrix_stat_fraction[1:], float)[visible])
                vis_max = max(float(c.max()) for c in curves)
                if vis_max > 0:
                    plt.ylim(0, 1.3 * vis_max)
            plt.xlabel(self._observable_label())
            plt.ylabel("Fractional Uncertainty")
            ax = plt.gca()
            ax.tick_params(axis='x', pad=8)
            ax.tick_params(axis='y', pad=8)
            plt.subplots_adjust(left=0.16, bottom=0.15)
            save_path = (
                f'./{self.spec.output_dir}uncertainties/stat_fraction_groomed_{i-1}.pdf'
                if self.groomed
                else f'./{self.spec.output_dir}uncertainties/stat_fraction_ungroomed_{i-1}.pdf'
            )
            self._finalize_plot(save_path=save_path, show=show)

    def plot_systematic_fraction(self, syst_name='all', show=True, log=True):
        self._plot_systematic_fraction_summary(grouped=False, show=show, log=log)

    def _get_systematic_group_name(self, syst_name):
        syst_lower = syst_name.lower()
        if syst_lower.startswith("jes") or syst_lower.startswith("jer"):
            return "Jet Energy"
        if syst_lower.startswith("jms") or syst_lower.startswith("jmr"):
            return "Jet Mass"
        if syst_lower.startswith("ele") or syst_lower.startswith("mu"):
            return "Lepton SFs"
        if syst_lower.startswith("showermodel"):
            return "Shower Model"
        if syst_lower.startswith("hadmodel"):
            return "Hadronization Model"
        if syst_lower.startswith("isr") or syst_lower.startswith("fsr"):
            return "Parton Shower"
        if syst_lower.startswith("modelenvelope") or syst_lower.startswith("herwig"):
            return "Model Uncertainty"
        if (
            syst_lower.startswith("pu")
            or syst_lower.startswith("pdf")
            or syst_lower.startswith("q2")
            or syst_lower.startswith("l1prefiring")
        ):
            return "Other Theory"
        return None

    def _get_systematic_label(self, syst_name):
        base_name, _ = self._split_systematic_variation(syst_name)
        label_map = {
            "pu": "Pileup",
            "l1prefiring": "L1 Prefiring",
            "q2": r"Q$^2$ Scale",
            "pdf": "PDF",
            "herwig": "Model Uncertainty",
            "modelenvelope": "Model Uncertainty",
            "showermodel": "Shower Model",
            "hadmodel": "Hadronization Model",
            "isr": "ISR",
            "fsr": "FSR",
            "jms": "JMS",
            "jmr": "JMR",
        }
        return label_map.get(base_name.lower(), base_name)

    def _split_systematic_variation(self, syst_name):
        match = re.match(r"^(.*?)(Up|Down)(?:_.*)?$", syst_name)
        if match:
            return match.group(1), match.group(2)
        return syst_name, None

    def _get_systematic_summary_name(self, syst_name, grouped=False):
        if grouped:
            group_name = self._get_systematic_group_name(syst_name)
            if group_name is not None:
                return group_name
            return self._get_systematic_label(syst_name)

        base_name, _ = self._split_systematic_variation(syst_name)
        base_lower = base_name.lower()
        summary_map = {
            "elereco": "Electron RECO",
            "eleid": "Electron ID",
            "eletrig": "Electron Trigger",
            "mureco": "Muon RECO",
            "muid": "Muon ID",
            "mutrig": "Muon Trigger",
            "muiso": "Muon ISO",
            "pu": "Pileup",
            "pdf": "PDF",
            "q2": "Q2 Scale",
            "l1prefiring": "L1 Prefiring",
            "herwig": "Model Uncertainty",
            "modelenvelope": "Model Uncertainty",
            "showermodel": "Shower Model",
            "hadmodel": "Hadronization Model",
        }
        if base_lower.startswith("jes"):
            return "JES"
        if base_lower.startswith("jer"):
            return "JER"
        if base_lower.startswith("isr"):
            return "ISR"
        if base_lower.startswith("fsr"):
            return "FSR"
        if base_lower.startswith("jmr"):
            return "JMR"
        if base_lower.startswith("jms"):
            return "JMS"
        return summary_map.get(base_lower, self._get_systematic_label(syst_name))

    def _build_syst_fraction_dict(self, pt_index):
        result = self.normalized_results[pt_index]
        nominal = result["unfolded"]
        total_syst_up = result["syst_unc"]["up"]
        total_syst_down = result["syst_unc"]["down"]
        syst_fraction_dict = {}
        use_model_envelope = getattr(self.spec, "model_envelope", False)

        for syst_name, syst_unfolded in self.normalized_systematics[pt_index]["unfolded"].items():
            # With the model envelope on, herwig/fsr/isr are superseded in the
            # total (see _compute_total_systematic); showing their legacy
            # curves would misrepresent the decomposition.
            if use_model_envelope and syst_name.startswith(("herwig", "fsr", "isr")):
                continue
            diff = syst_unfolded - nominal
            syst_fraction = np.abs(np.divide(diff, nominal, out=np.zeros_like(diff), where=nominal != 0))
            syst_fraction_dict[syst_name] = syst_fraction

        if use_model_envelope and "model_ps_frac" in result:
            # The two independent modelling legs entering the total in
            # quadrature (ARC round-2): shower (Vincia + FSR envelope) and
            # hadronization (CR/frag envelope). Shown separately.
            syst_fraction_dict["showermodelUp"] = np.asarray(
                result["model_ps_frac"], float)
            syst_fraction_dict["hadmodelUp"] = np.asarray(
                result["model_had_frac"], float)
        elif use_model_envelope and "model_unc_frac" in result:
            syst_fraction_dict["modelenvelopeUp"] = np.asarray(
                result["model_unc_frac"], float)

        stat_fraction = self.stat_unc_pt_binned[pt_index]
        total_syst_fraction_up = np.abs(np.divide(total_syst_up, np.abs(nominal), out=np.zeros_like(total_syst_up), where=np.abs(nominal) != 0))
        total_syst_fraction_down = np.abs(np.divide(total_syst_down, np.abs(nominal), out=np.zeros_like(total_syst_down), where=np.abs(nominal) != 0))

        syst_fraction_dict["Stat Unc"] = stat_fraction
        syst_fraction_dict["Total_Up"] = total_syst_fraction_up
        syst_fraction_dict["Total_Down"] = total_syst_fraction_down
        return syst_fraction_dict

    def _group_syst_fraction_dict(self, syst_fraction_dict, grouped=True):
        grouped_fraction_dict = {}
        accumulators = {}

        for syst_name, syst_fraction in syst_fraction_dict.items():
            if syst_name in {"Stat Unc", "Total_Up", "Total_Down"}:
                continue

            target_name = self._get_systematic_summary_name(syst_name, grouped=grouped)
            _, variation = self._split_systematic_variation(syst_name)
            target_key = f"{target_name}Down" if variation == "Down" else f"{target_name}Up"

            if target_key not in accumulators:
                accumulators[target_key] = np.zeros_like(syst_fraction)
            accumulators[target_key] += syst_fraction**2

        for target_key, target_sum in accumulators.items():
            grouped_fraction_dict[target_key] = np.sqrt(target_sum)

        grouped_fraction_dict["Stat Unc"] = syst_fraction_dict["Stat Unc"]
        grouped_fraction_dict["Total_Up"] = syst_fraction_dict["Total_Up"]
        grouped_fraction_dict["Total_Down"] = syst_fraction_dict["Total_Down"]
        return grouped_fraction_dict

    def _plot_systematic_fraction_summary(self, grouped=False, show=True, log=True):
        hep.style.use("CMS")
        self.syst_fraction_dicts = []
        # Linear y-range per grooming (Sal, approval dry-run): the groomed
        # totals peak at ~0.15, so a shared 0.5 ceiling left the groomed
        # panels mostly whitespace; the ungroomed first bin genuinely needs
        # 0.5 (its 0.77 total is clipped + annotated).
        # Approval comments (S. Consuegra, A. Meyer): the ungroomed pt0 first
        # bin (total 0.77) must be contained by the axis, not clipped at the
        # frame -- 0.8 keeps it inside while the over-range annotation logic
        # still guards anything larger.
        linear_ymax = 0.2 if self.groomed else 0.8
        grouped_legend_order = [
            "Jet Energy",
            "Jet Mass",
            "Parton Shower",
            "Lepton SFs",
            "Other Theory",
            "Model Uncertainty",
            "Shower Model",
            "Hadronization Model",
            "Stat Unc",
            "Total",
        ]
        # ARC round-2 casing for the legend; internal group keys unchanged.
        display_name = {
            "Jet Energy": "Jet energy",
            "Jet Mass": "Jet mass",
            "Parton Shower": "Parton shower",
            "Lepton SFs": "Lepton scale factors",
            "Other Theory": "Other theory",
            "Model Uncertainty": "Model uncertainty",
            "Shower Model": "Shower model",
            "Hadronization Model": "Hadronization model",
            "Stat Unc": "Stat unc.",
        }
        # Grouped entries on the CVD-friendly Petroff palette (ARC round-2
        # color check); ungrouped per-systematic colors on the 10-color set.
        summary_style_map = {
            "JES": {"color": "#3f90da", "ls": "-"},
            "JER": {"color": "#ffa90e", "ls": "-"},
            "Pileup": {"color": "#bd1f01", "ls": "-"},
            "Electron RECO": {"color": "#832db6", "ls": "-"},
            "Electron ID": {"color": "#a96b59", "ls": "-"},
            "Electron Trigger": {"color": "#e76300", "ls": "-"},
            "Muon RECO": {"color": "#94a4a2", "ls": "-"},
            "Muon ID": {"color": "#b9ac70", "ls": "-"},
            "Muon Trigger": {"color": "#92dadd", "ls": "-"},
            "Muon ISO": {"color": "#717581", "ls": "-"},
            "PDF": {"color": "#f89c20", "ls": "--"},
            "Q2 Scale": {"color": "#5790fc", "ls": "--"},
            "L1 Prefiring": {"color": "#964a8b", "ls": "--"},
            "ISR": {"color": "#e42536", "ls": "--"},
            "FSR": {"color": "#7a21dd", "ls": "--"},
            "JMR": {"color": "#9c9ca1", "ls": "--"},
            "JMS": {"color": "#bd1f01", "ls": ":"},
            "Jet Energy": {"color": "#5790fc", "ls": "-", "lw": 2.5},
            "Jet Mass": {"color": "#f89c20", "ls": "-", "lw": 2.5},
            "Parton Shower": {"color": "#e42536", "ls": "-.", "lw": 2.5},
            "Lepton SFs": {"color": "#964a8b", "ls": "-", "lw": 2.5},
            "Other Theory": {"color": "#9c9ca1", "ls": "-", "lw": 2.5},
            "Model Uncertainty": {"color": "#7a21dd", "ls": "-.", "lw": 2.5},
            "Shower Model": {"color": "#e42536", "ls": "-.", "lw": 2.5},
            "Hadronization Model": {"color": "#7a21dd", "ls": "--", "lw": 2.5},
            "Stat Unc": {"color": "k", "ls": ":", "lw": 2.5},
        }

        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            # CMS-default canvas (a forced figsize shrinks every font in the
            # scaled-down PDF; ARC round-2 "make all labels bigger").
            fig = plt.figure(layout="constrained")
            pt_bin = result["pt_bin"]
            syst_fraction_dict = self._build_syst_fraction_dict(i)
            result["syst_fraction_dict"] = syst_fraction_dict
            self.syst_fraction_dicts.append(syst_fraction_dict)

            plot_fraction_dict = self._group_syst_fraction_dict(syst_fraction_dict, grouped=grouped)
            rho_edges = np.asarray(self.gen_edges_by_pt[i], dtype=float)
            rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])

            # ARC round-2 s5ff: the total drawn as a filled silhouette BEHIND
            # the components, so the line hugging its top edge in each bin is
            # visibly the dominant contribution.
            hep.histplot(
                plot_fraction_dict["Total_Up"],
                self.gen_edges_by_pt[i],
                histtype="fill",
                facecolor="0.88",
                edgecolor="black",
                linewidth=2,
                label="Total",
            )

            plotted_labels = set()
            for syst_name, syst_fraction in plot_fraction_dict.items():
                label, variation = self._split_systematic_variation(syst_name)
                if syst_name in {"Stat Unc", "Total_Up", "Total_Down"} or variation == "Down":
                    continue

                if label in plotted_labels:
                    continue

                style = summary_style_map.get(label, {"ls": "-"})
                hep.histplot(
                    syst_fraction,
                    self.gen_edges_by_pt[i],
                    label=display_name.get(label, label),
                    **style,
                )
                plotted_labels.add(label)

            hep.histplot(
                plot_fraction_dict["Stat Unc"],
                self.gen_edges_by_pt[i],
                label=display_name["Stat Unc"],
                **summary_style_map["Stat Unc"],
            )

            if not log:
                ax = plt.gca()
                for x_pos, y_val in zip(rho_centers, np.asarray(plot_fraction_dict["Total_Up"], dtype=float)):
                    if y_val > linear_ymax:
                        ax.text(
                            x_pos,
                            linear_ymax - 0.015,
                            f"{y_val:.2f}",
                            ha="center",
                            va="top",
                            fontsize=PUB_LEGEND_FONTSIZE - 4,
                            clip_on=True,
                        )

            if log:
                plt.yscale("log")
            if pt_bin[1] == float("inf") or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            # ARC round-2: legend in-canvas
            if grouped:
                display_order = [display_name.get(l, l) for l in grouped_legend_order]
                label_to_handle = dict(zip(labels, handles))
                ordered_labels = [label for label in display_order if label in label_to_handle]
                ordered_handles = [label_to_handle[label] for label in ordered_labels]
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    title=rf"$p_{{\mathrm{{T}}}}$  {pt_bin_label} GeV",
                    loc="upper right",
                    fontsize=PUB_LEGEND_FONTSIZE + 2,
                    title_fontsize=PUB_LEGEND_FONTSIZE + 2,
                )
            else:
                ax.legend(
                    title=rf"$p_{{\mathrm{{T}}}}$  {pt_bin_label} GeV",
                    loc="upper right",
                    fontsize=13,
                    title_fontsize=15,
                    ncol=2,
                )
            if log:
                plt.ylim(10e-5, 1)
            else:
                plt.ylim(0, linear_ymax)
            plt.xlim(*self._observable_xlim(i))
            #plt.xlim(0,200)
            ax.set_xlabel(self._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
            ax.set_ylabel("Fractional uncertainty", fontsize=PUB_LABEL_FONTSIZE)
            ax.tick_params(axis="x", pad=8, labelsize=PUB_TICK_FONTSIZE)
            ax.tick_params(axis="y", pad=8, labelsize=PUB_TICK_FONTSIZE)
            # Stamp the CMS label LAST, after the axis labels: under
            # constrained layout a later set_ylabel shrinks the axes width and
            # the (already frozen) CMS<->suffix gap collapses. The draw lets
            # the layout settle before mplhep measures the text. Above-frame
            # tag here: the grey Total silhouette can fill the top-left
            # (ungroomed first bin), so in-frame loc=2 collides with it.
            fig.canvas.draw()
            hep.cms.label(self._cms_extra_label(), data=True, lumi=self._lumi_label(), com=self._com_label(), ax=ax)

            if grouped:
                if log:
                    save_path = (
                        f"./{self.spec.output_dir}uncertainties/summary_grouped_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./{self.spec.output_dir}uncertainties/summary_grouped_ungroomed_{i-1}.pdf"
                    )
                else:
                    save_path = (
                        f"./{self.spec.output_dir}uncertainties/summary_grouped_linear_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./{self.spec.output_dir}uncertainties/summary_grouped_linear_ungroomed_{i-1}.pdf"
                    )
            else:
                if log:
                    save_path = (
                        f"./{self.spec.output_dir}uncertainties/summary_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./{self.spec.output_dir}uncertainties/summary_ungroomed_{i-1}.pdf"
                    )
                else:
                    save_path = (
                        f"./{self.spec.output_dir}uncertainties/summary_linear_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./{self.spec.output_dir}uncertainties/summary_linear_ungroomed_{i-1}.pdf"
                    )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_systematic_fraction_grouped(self, show=True, log=True):
        self._plot_systematic_fraction_summary(grouped=True, show=show, log=log)

    def _resolve_raw_systematic_pair(self, syst_name):
        if syst_name in self.systematics:
            base_name, variation = self._split_systematic_variation(syst_name)
            if variation == "Up":
                return syst_name, f"{base_name}Down"
            if variation == "Down":
                return f"{base_name}Up", syst_name

        candidates = [name for name in self.systematics if name != "nominal"]
        matched_up = None
        matched_down = None

        for candidate in candidates:
            base_name, variation = self._split_systematic_variation(candidate)
            if base_name.lower() != syst_name.lower():
                continue
            if variation == "Up":
                matched_up = candidate
            elif variation == "Down":
                matched_down = candidate

        return matched_up, matched_down

    def plot_nominal_minus_variation(self, syst_names=["pu"], show=True):
        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            fig, ax = plt.subplots(figsize=(12, 8))
            nominal = result["unfolded"]
            pt_bin = result["pt_bin"]

            for syst in syst_names:
                up_key, down_key = self._resolve_raw_systematic_pair(syst)
                label = self._get_systematic_label(syst)

                if up_key in self.normalized_systematics[i]["unfolded"]:
                    up_shift = nominal - self.normalized_systematics[i]["unfolded"][up_key]
                    hep.histplot(
                        up_shift,
                        self.gen_edges_by_pt[i],
                        label=f"{label}: nominal - Up",
                        ls="-",
                    )

                if down_key in self.normalized_systematics[i]["unfolded"]:
                    down_shift = nominal - self.normalized_systematics[i]["unfolded"][down_key]
                    hep.histplot(
                        down_shift,
                        self.gen_edges_by_pt[i],
                        label=f"{label}: nominal - Down",
                        ls="--",
                    )

            if pt_bin[1] == float("inf") or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            ax.axhline(0.0, color="k", lw=1, alpha=0.5)
            ax.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize=15)
            hep.cms.label(self._cms_extra_label(), data=True, lumi=138, com=13, fontsize=20, ax=ax)
            #ax.set_xlim(*self._observable_xlim(i))
            ax.set_xlim(0,200)
            ax.set_xlabel(self._observable_label())
            ax.set_ylabel("Nominal - Variation")
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)

            save_stub = "_".join(str(name) for name in syst_names)
            save_path = (
                f"./{self.spec.output_dir}uncertainties/nominal_minus_{save_stub}_groomed_{i-1}.pdf"
                if self.groomed
                else f"./{self.spec.output_dir}uncertainties/nominal_minus_{save_stub}_ungroomed_{i-1}.pdf"
            )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    
    def plot_systematic_frac_indiv(self, syst_names=['JES', 'JER'], ylim=None, show=True):
        def build_plot_fraction_dict(raw_syst_fraction_dict):
            plot_fraction_dict = self._group_syst_fraction_dict(raw_syst_fraction_dict, grouped=False)

            electron_keys = [key for key in plot_fraction_dict if key.startswith("Electron ")]
            if electron_keys:
                electron_up = [plot_fraction_dict[key] for key in electron_keys if key.endswith("Up")]
                electron_down = [plot_fraction_dict[key] for key in electron_keys if key.endswith("Down")]
                if electron_up:
                    plot_fraction_dict["ElectronSFUp"] = np.sqrt(np.sum([value**2 for value in electron_up], axis=0))
                if electron_down:
                    plot_fraction_dict["ElectronSFDown"] = np.sqrt(np.sum([value**2 for value in electron_down], axis=0))

            muon_keys = [key for key in plot_fraction_dict if key.startswith("Muon ")]
            if muon_keys:
                muon_up = [plot_fraction_dict[key] for key in muon_keys if key.endswith("Up")]
                muon_down = [plot_fraction_dict[key] for key in muon_keys if key.endswith("Down")]
                if muon_up:
                    plot_fraction_dict["MuonSFUp"] = np.sqrt(np.sum([value**2 for value in muon_up], axis=0))
                if muon_down:
                    plot_fraction_dict["MuonSFDown"] = np.sqrt(np.sum([value**2 for value in muon_down], axis=0))

            return plot_fraction_dict

        def resolve_syst_keys(syst_fraction_dict, syst):
            alias_bases = [syst]
            summary_name = self._get_systematic_summary_name(syst, grouped=False)
            if summary_name not in alias_bases:
                alias_bases.append(summary_name)
            label_name = self._get_systematic_label(syst)
            if label_name not in alias_bases:
                alias_bases.append(label_name)

            for base_name in alias_bases:
                up_key = f"{base_name}Up"
                down_key = f"{base_name}Down"
                if up_key in syst_fraction_dict or down_key in syst_fraction_dict:
                    return up_key, down_key

            return f"{syst}Up", f"{syst}Down"

        # First, collect all values to determine global y-range
        all_values = []
        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            raw_syst_fraction_dict = result.get('syst_fraction_dict', {})
            syst_fraction_dict = build_plot_fraction_dict(raw_syst_fraction_dict)
            for syst in syst_names:
                up_key, down_key = resolve_syst_keys(syst_fraction_dict, syst)
                if up_key in syst_fraction_dict:
                    all_values.append(np.abs(syst_fraction_dict[up_key]))
                if down_key in syst_fraction_dict:
                    all_values.append(np.abs(syst_fraction_dict[down_key]))


        # Now plot with fixed y-range
        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            raw_syst_fraction_dict = result.get('syst_fraction_dict', {})
            syst_fraction_dict = build_plot_fraction_dict(raw_syst_fraction_dict)
            #plt.figure(figsize=(12, 8))
            pt_bin = result['pt_bin']
            panel_values = []
            for syst in syst_names:
                up_key, down_key = resolve_syst_keys(syst_fraction_dict, syst)
                color_map = ['#e42536', '#5790fc', '#964a8b']
                color = color_map[syst_names.index(syst)] if syst in syst_names and syst_names.index(syst) < len(color_map) else None
                if up_key not in syst_fraction_dict and down_key not in syst_fraction_dict:
                    print(
                        f"[DEBUG] Neither '{up_key}' nor '{down_key}' found in syst_fraction_dict "
                        f"for pt bin {pt_bin}. Available keys: {list(syst_fraction_dict.keys())}"
                    )

                # Plot Up uncertainty (solid)
                label_dic = {'pu':'Pileup', 'l1prefiring': 'L1 Prefiring', 'q2': r'Q$^2$ Scale', 'pdf': 'PDF', 'herwig': 'Model Unc.', 'isr': 'ISR', 'fsr': 'FSR', 'jms': 'JMS', 'jmr': 'JMR'}
                if up_key in syst_fraction_dict:
                    hep.histplot(syst_fraction_dict[up_key][1:], self.gen_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Up", color=color, ls='-')
                    panel_values.append(np.asarray(syst_fraction_dict[up_key][1:], float))
                # Plot Down uncertainty (dashed)
                if down_key in syst_fraction_dict:
                    hep.histplot(-syst_fraction_dict[down_key][1:], self.gen_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Down", color=color, ls='--')
                    panel_values.append(np.asarray(syst_fraction_dict[down_key][1:], float))


            # None of the requested systematics exist for this spec (e.g.
            # isr/fsr under model_envelope, where they live in the envelope
            # instead) -- don't save an empty frame.
            if not panel_values:
                plt.close()
                continue

            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            # if ylim is not None:
            #     plt.ylim(ylim)
            # Opaque legend frame (as in plot_bottom_line): with the tightened
            # symmetric y-range a flat curve can run through the legend text.
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize = 15,
                       frameon=True, framealpha=0.9, facecolor='white',
                       edgecolor='0.7')
            hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(), com=self._com_label(), fontsize=20)
            # Symmetric y-range from the bins inside the shown window only --
            # the hidden buffer bins otherwise inflate the autoscale and
            # flatten the visible curves (same trap as the stat/model plots).
            edges = np.asarray(self.gen_edges_by_pt[i][1:], float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            xlo, xhi = self._observable_xlim(i)
            visible = (centers >= xlo) & (centers <= xhi)
            if np.any(visible):
                vmax = max(float(np.abs(v[visible]).max()) for v in panel_values)
                if vmax > 0:
                    plt.ylim(-1.3 * vmax, 1.3 * vmax)

            if self.groomed:
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
                save_path = f'./{self.spec.output_dir}uncertainties/{syst_names[0]}_groomed_{i-1}.pdf'
            else:
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
                save_path = f'./{self.spec.output_dir}uncertainties/{syst_names[0]}_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)

    def plot_herwig_systematic(self, show=True):
        flat_uncertainty = np.sqrt(np.diag(self.cov_data_herwig_np))/np.abs(self.y_unf_dict['herwigUp'])
        uncertainty_pt_binned = unflatten_gen_by_pt(flat_uncertainty, self.gen_edges_by_pt)
        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.gen_edges_by_pt)
        
        for i, result in enumerate(self.normalized_results):
            syst_fraction_dict = result.get('syst_fraction_dict', {})
            # syst_fraction_dict is only populated for the reported pt bins
            # (first_reported_pt_bin..), so the unreported sink bin(s) carry no
            # 'herwigUp' -- skip them instead of raising.
            if 'herwigUp' not in syst_fraction_dict:
                continue
            error_in_syst = uncertainty_pt_binned[i]*syst_fraction_dict['herwigUp']  # Uncertainty on relative uncertainty
            pt_bin = result['pt_bin']
            if 'herwigUp' in syst_fraction_dict:
                hep.histplot(syst_fraction_dict['herwigUp'], self.gen_edges_by_pt[i], yerr = error_in_syst, label=f"Model Unc.", color='#964a8b', ls='-')


            # Fit a polynomial to the herwigUp systematic fraction
            if 'herwigUp' in syst_fraction_dict:
                edges = np.array(self.gen_edges_by_pt[i], dtype=float)
                centers = 0.5 * (edges[:-1] + edges[1:])
                #centers[0] = -100000000  # Set the first center to a very large negative value to exclude it from the fit
                y = syst_fraction_dict['herwigUp']
                mask = np.isfinite(y) & (y > 0)
                if mask.sum() > 3:
                    degree = 2
                    coeffs = np.polyfit(centers[mask], y[mask], degree, w=1.0/np.where(error_in_syst[mask] > 0, error_in_syst[mask], 1e-10))
                    poly = np.poly1d(coeffs)
                    x_fit = np.linspace(centers[mask][1], centers[mask][-1], 200)
                    plt.plot(x_fit, poly(x_fit), color='#5790fc', ls='--', lw=2, label=f"Poly fit (deg {degree})")

            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
            hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)
            plt.ylim(0,0.5)
            plt.xlim(*self._observable_xlim(i))
            plt.xlabel(self._observable_label())
            plt.ylabel("Relative Uncertainty")
            save_path = f'./{self.spec.output_dir}uncertainties/herwig_groomed_{i-1}.pdf' if self.groomed else f'./{self.spec.output_dir}uncertainties/herwig_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)


            
    def plot_purity_stability_herwig(self, show=True):
        """Overlay Pythia8 vs Herwig7 purity & stability to diagnose generator dependence."""
        hep.style.use("CMS")
        suffix = "groomed" if self.groomed else "ungroomed"
        len_underflow = len(self.gen_edges_by_pt[0]) - 1

        def _purity_stability(mosaic):
            diagonal = np.diag(mosaic)
            purity_denom = mosaic[len_underflow:, :].sum(axis=0)
            stability_denom = mosaic[:, len_underflow:].sum(axis=1)
            purity = np.divide(diagonal, purity_denom,
                               out=np.zeros_like(diagonal, dtype=float), where=purity_denom != 0)
            stability = np.divide(diagonal, stability_denom,
                                  out=np.zeros_like(diagonal, dtype=float), where=stability_denom != 0)
            return (
                unflatten_gen_by_pt(purity, self.gen_edges_by_pt),
                unflatten_gen_by_pt(stability, self.gen_edges_by_pt),
            )

        purity_py, stability_py = _purity_stability(self.mosaic_gen)
        purity_hw, stability_hw = _purity_stability(self.mosaic_gen_herwig)

        title_list = [
            "",
            r"200 $<$ $p_T$ $<$ 290 GeV",
            r"290 $<$ $p_T$ $<$ 400 GeV",
            r"400 $<$ $p_T$ $< \, \infty$ GeV",
        ]

        for i in range(len(self.pt_edges) - 1):
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            fig.subplots_adjust(wspace=0.05)

            for ax, (pur_py, stab_py, pur_hw, stab_hw, metric) in zip(
                axes,
                [
                    (purity_py[i], None, purity_hw[i], None, "Purity"),
                    (None, stability_py[i], None, stability_hw[i], "Stability"),
                ],
            ):
                vals_py = pur_py if metric == "Purity" else stab_py
                vals_hw = pur_hw if metric == "Purity" else stab_hw
                edges = self.gen_edges_by_pt[i]
                hep.histplot(vals_py, edges, label="Pythia8", ax=ax, color="steelblue")
                hep.histplot(vals_hw, edges, label="Herwig7", ax=ax, color="darkorange", linestyle="--")
                ax.axhline(0.5, color="k", linestyle=":", linewidth=1.2, label="0.5 threshold")
                ax.set_xlabel(self._observable_short_label(), fontsize=14)
                ax.set_xlim(*self._observable_xlim(i))
                ax.set_ylim(0.0, 1.05)
                ax.set_title(metric, fontsize=14)
                ax.legend(title=title_list[i], fontsize=12)

            axes[0].set_ylabel("Purity / Stability", fontsize=14)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=18, ax=axes[0])

            save_path = f"./{self.spec.output_dir}unfold/purity_stability_herwig_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)


    def plot_lcurve(self, show=True):
        """L-curve of the nominal-data tau scan (regularized runs only)."""
        scan = getattr(self, "lcurve_scan", None)
        if scan is None:
            return
        plt.figure(figsize=(8, 7))
        plt.plot(scan["x"], scan["y"], "-o", ms=3, lw=1.5, color="#5790fc",
                 label="L-curve scan")
        plt.plot(
            [scan["best_x"]], [scan["best_y"]], "*", ms=18, color="#e42536",
            label=rf"chosen: $\tau$ = {scan['tau']:.3g}",
        )
        plt.xlabel(r"$\log_{10}\,\chi^2_{\rm data}$")
        plt.ylabel(r"$\log_{10}\,(Lx)^2$")
        suffix = "groomed" if self.groomed else "ungroomed"
        plt.legend(title=f"{self.regularization}, {suffix}")
        save_path = f"./{self.spec.output_dir}unfold/lcurve_{suffix}.pdf"
        self._finalize_plot(save_path=save_path, show=show)

    def plot_correlation(self, show=True, shown_only=False):
        # ``shown_only`` crops each pT slice to the reported rho window (the
        # per-pT shown floors) before forming the correlation, matching the
        # approval-talk figure; the full-space version stays the default.
        if self.stat_propagation == "jacobian":
            # Correlation of the normalized result: stat covariance propagated
            # through the normalization Jacobian (negative correlations from
            # the per-pT-slice sum constraint are expected).
            cov_matrix = np.array(self.norm_cov_stat, copy=True)
        else:
            cov_matrix = self.cov_uncorr_np + self.cov_data_np
        # Reported slices only: the 185-200 GeV slice is the migration sink,
        # not a measurement (same max(1, ...) as plot_unfolded_fancy).
        first_pt_bin = max(1, getattr(self, "first_reported_pt_bin", 0))
        gen_offset = sum(
            len(edges) - 1 for edges in self.gen_edges_by_pt[:first_pt_bin]
        )
        cov_matrix = cov_matrix[gen_offset:, gen_offset:]
        reported_gen_edges = self.gen_edges_by_pt[first_pt_bin:]
        if shown_only:
            floors = self._bl_shown_floors()[first_pt_bin:]
            keep = []
            ncols_by_gp = []
            offset = 0
            for edges, floor in zip(reported_gen_edges, floors):
                edges = np.asarray(edges, float)
                local = np.flatnonzero(edges[:-1] >= floor - 1e-9)
                keep.extend((offset + local).tolist())
                ncols_by_gp.append(len(local))
                offset += len(edges) - 1
            cov_matrix = cov_matrix[np.ix_(keep, keep)]
        else:
            ncols_by_gp = [len(e) - 1 for e in reported_gen_edges]
        std_devs = np.sqrt(np.diag(cov_matrix))

        # Avoid division by zero by replacing zeros with a small number
        std_devs[std_devs == 0] = 1e-10

        # Compute the outer product of standard deviations
        std_matrix = np.outer(std_devs, std_devs)

        # Compute correlation matrix
        corr_matrix = cov_matrix / std_matrix

        self.corr_matrix = corr_matrix
        ## colormap
        num_bins = 20
        bounds = np.linspace(-1, 1, num_bins + 1)  # 21 boundaries for 20 bins

        # Get the 'seismic' colormap
        base_cmap = plt.get_cmap("seismic", num_bins)
        colors = base_cmap(np.linspace(0, 1, num_bins))  # Extract colors
        
        # Force the bins for -0.1 to 0.1 to be white
        for i in range(len(bounds) - 1):
            if -0.1 <= bounds[i] <= 0.1:
                colors[i] = [1, 1, 1, 1]  # Set to white (RGBA)


            
        #import matplotlib.colors as mcolors
        # Create colormap and normalizer
        cmap =  mcolors.ListedColormap(colors)  # Discrete 'seismic' colormap
        norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize for discrete bins
        

        fig, ax = plt.subplots(layout="constrained")
        img = ax.imshow(corr_matrix, cmap=cmap, norm=norm, origin='lower')

        # ---- Add grid lines and labels for pt bins ----
        # Block structure per reported pT bin (shown-cropped when shown_only)
        x_bounds = np.r_[0, np.cumsum(ncols_by_gp)]
        # Draw dashed lines at pt bin boundaries
        for x in x_bounds[1:-1]:
            ax.axvline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)
            ax.axhline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)
        # Optional: thin grid inside each block at every integer cell
        ax.set_xticks(np.arange(-0.5, corr_matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, corr_matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="w", alpha=0.15, lw=0.5)
        # Tick labels at block centers
        x_centers = (x_bounds[:-1] + x_bounds[1:] - 1) / 2.0
        pt_edges = self.pt_edges[first_pt_bin:]
        x_labels = [f"{int(pt_edges[i])}–{int(pt_edges[i+1]) if i+1 < len(pt_edges)-1 else '∞'}" for i in range(len(pt_edges)-1)]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(x_centers)
        ax.set_yticklabels(x_labels, rotation=90, va="center")

        # The aspect-equal matrix plus its colorbar leaves a much narrower axes
        # than a normal 10x10 panel, so the full PUB_* sizes do not fit here:
        # the pT-range tick strings run into each other and the 21 colorbar
        # boundaries overprint. Keep the axis labels at PUB and step the rest
        # down to what the axes can actually hold.
        ax.set_xlabel(r"GEN $p_T$ (GeV)", fontsize=PUB_LABEL_FONTSIZE)
        ax.set_ylabel(r"GEN $p_T$ (GeV)", fontsize=PUB_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE - 8)

        cbar = fig.colorbar(img, ax=ax, ticks=bounds[::2], boundaries=bounds,
                            fraction=0.046, pad=0.04)
        cbar.set_label("Correlation (Groomed)" if self.groomed else "Correlation (Ungroomed)",
                       fontsize=PUB_LABEL_FONTSIZE - 8)
        cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE - 12)
        # The colorbar renarrows the aspect-equal axes under constrained
        # layout, which would squeeze the frozen CMS-label geometry; settle
        # and freeze the layout, then stamp the label on the final axes
        # (same fix as plot_correlation_before_after).
        fig.canvas.draw()
        fig.set_layout_engine("none")
        # 20pt ran "CMS Internal" straight into the lumi text on this narrow axes.
        hep.cms.label(self.cms_label, data=True, lumi=self._lumi_label(),
                      com=self._com_label(), ax=ax, fontsize=15)
        mode = "groomed" if self.groomed else "ungroomed"
        tag = "_shown" if shown_only else ""
        save_path = f'{self.spec.output_dir}unfold/correlation_{mode}{tag}.pdf'
        self._finalize_plot(save_path=save_path, show=show)
    def _crop_matrix_view_to_floor(self, matrix, reco_by, gen_by, floor):
        """Drop per-pT rho bins below ``floor`` from both axes of a reported
        response-matrix view (display only). Returns the cropped matrix and the
        trimmed per-pT edge lists; the unfolding itself is unaffected."""
        def mask_and_edges(edges_by_pt):
            masks, new_edges = [], []
            for e in edges_by_pt:
                e = np.asarray(e, float)
                keep = e[:-1] >= floor - 1e-9      # bins whose lower edge >= floor
                masks.append(keep)
                idx = np.flatnonzero(keep)
                new_edges.append(list(e[idx[0]:idx[-1] + 2]) if idx.size else list(e[-1:]))
            return np.concatenate(masks), new_edges

        rmask, new_reco = mask_and_edges(reco_by)
        gmask, new_gen = mask_and_edges(gen_by)
        cropped = np.asarray(matrix)[np.ix_(rmask, gmask)]
        return cropped, new_reco, new_gen

    def plot_response_matrix(self, probability=True, log=False, show=True):
        (
            reported_mosaic,
            reported_reco_edges,
            reported_gen_edges,
            reported_pt_edges,
        ) = self._reported_matrix_view(self.mosaic)
        # Display "cheat" (ARC round-2): show the response matrix only over the
        # reported window (rho >= the display floor), hiding the low-rho buffer
        # bins that the unfold uses but that we never plot.
        floor = (self.spec.xlim_lower_groomed if self.groomed
                 else self.spec.xlim_lower_ungroomed)
        reported_mosaic, reported_reco_edges, reported_gen_edges = (
            self._crop_matrix_view_to_floor(
                reported_mosaic, reported_reco_edges, reported_gen_edges, floor)
        )
        fig, ax = self._plot_response_mosaic_cms(
            reported_mosaic,
            reco_mass_edges_by_pt=reported_reco_edges,
            gen_mass_edges_by_pt=reported_gen_edges,
            reco_pt_edges=reported_pt_edges,
            gen_pt_edges=reported_pt_edges,
            probability = probability,
            mask_zeros=True,
            log=log,                              # set False for linear
            rlabel="Groomed" if self.groomed else "Ungroomed",
        )
        self._finalize_plot(show=show, fig=fig)

    def plot_uncertainty_heatmap(self, show=True):
        """
        2D heatmap of fractional uncertainties: rows = systematic groups,
        columns = mass bins, color = uncertainty magnitude in %.
        Gives an at-a-glance budget matrix showing which source dominates
        in which part of the spectrum. One figure per pT bin.
        """
        group_order = [
            "Jet Energy",
            "Jet Mass",
            "Parton Shower",
            "Lepton SFs",
            "Other Theory",
            "Model Uncertainty",
            "Stat Unc",
            "Total",
        ]

        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            syst_fraction_dict = self._build_syst_fraction_dict(i)
            grouped_dict = self._group_syst_fraction_dict(syst_fraction_dict, grouped=True)

            # Compute per-group envelope: max(Up, Down) for each group
            envelopes = {}
            seen_bases = set()
            for key in grouped_dict:
                if key in {"Stat Unc", "Total_Up", "Total_Down"}:
                    continue
                base, variation = self._split_systematic_variation(key)
                if base not in seen_bases:
                    seen_bases.add(base)
                    up = grouped_dict.get(f"{base}Up", np.zeros_like(grouped_dict[key]))
                    down = grouped_dict.get(f"{base}Down", np.zeros_like(grouped_dict[key]))
                    envelopes[base] = np.maximum(up, down)

            envelopes["Stat Unc"] = grouped_dict["Stat Unc"]
            envelopes["Total"] = np.maximum(
                grouped_dict.get("Total_Up", np.zeros(1)),
                grouped_dict.get("Total_Down", np.zeros(1)),
            )

            # Build matrix in the predefined group order, skipping absent groups
            rows = [g for g in group_order if g in envelopes]
            matrix = np.array([envelopes[row] for row in rows]) * 100.0  # → percent

            edges = np.array(self.gen_edges_by_pt[i], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])

            # Trim columns below the observable lower limit:
            # ungroomed starts at 20 GeV (20-30 bin), groomed at 10 GeV (10-20 bin)
            x_lo, _ = self._observable_xlim(i)
            start_col = int(np.searchsorted(edges[:-1], x_lo, side='left'))
            centers = centers[start_col:]
            matrix = matrix[:, start_col:]

            n_rows, n_cols = matrix.shape

            fig_w = max(10, n_cols * 0.9 + 3)
            fig_h = n_rows * 0.8 + 2.5
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            vmax = 50.0

            im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                           vmin=0.0, vmax=vmax, origin='upper')

            # Annotate every cell with its numeric value
            for r in range(n_rows):
                for c in range(n_cols):
                    val = matrix[r, c]
                    text_color = 'white' if val > vmax * 0.65 else 'black'
                    ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                            fontsize=13, color=text_color, fontweight='bold')

            # This is a wide, densely annotated budget matrix rather than a
            # half-textwidth panel, so the cell/tick type is boosted well above
            # the old 8-9pt but kept below the full PUB_* sizes, which would
            # collide at this column density.
            # One decimal: log10(rho^2) bin centres are ~0.5 apart, so "%.0f"
            # collapsed distinct columns onto the same label ("-2", "-2").
            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels([f"{c:.1f}" for c in centers],
                               rotation=45, ha='right', fontsize=16)
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(rows, fontsize=18)

            # Visual separators: dashed blue before stat unc, solid black before total
            if "Stat Unc" in rows:
                ax.axhline(rows.index("Stat Unc") - 0.5, color='steelblue', lw=1.5, ls='--')
            if "Total" in rows:
                ax.axhline(rows.index("Total") - 0.5, color='black', lw=2.0)

            # Aligned colorbar: make_axes_locatable ensures it matches the heatmap height exactly
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.15)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Uncertainty (%)", fontsize=20)
            cbar.ax.tick_params(labelsize=16)

            pt_bin = result['pt_bin']
            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{int(pt_bin[0])}–∞ GeV"
            else:
                pt_bin_label = f"{int(pt_bin[0])}–{int(pt_bin[1])} GeV"

            ax.set_xlabel(self._observable_label(), fontsize=PUB_LABEL_FONTSIZE)

            # mplhep freezes the CMS<->lumi gap as an axes fraction when the
            # label is drawn, so with tight_layout the label must be stamped
            # AFTER the axes have been resized (skill rule 9). The pT bin moves
            # into rlabel now that the title is gone.
            plt.tight_layout()
            hep.cms.label(self._cms_extra_label(), data=True, ax=ax, fontsize=18,
                          rlabel=rf"$p_T$: {pt_bin_label}, "
                                 rf"{self._lumi_label()} fb$^{{-1}}$ "
                                 rf"({self._com_label()} TeV)")
            suffix = "groomed" if self.groomed else "ungroomed"
            save_path = f"./{self.spec.output_dir}uncertainties/heatmap_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_model_envelope(self, show=False):
        """Per-pT model-uncertainty composition: Vincia / CR / frag re-unfold
        shifts, the FSR PSWeight shift, and their per-bin envelope (the term
        added to the total band when ``spec.model_envelope`` is on)."""
        if not hasattr(self, "model_shift_components"):
            return
        groomed_tag = "groomed" if self.groomed else "ungroomed"
        colors = {"Vincia": "#e42536", "CR": "#5790fc",
                  "frag": "#f89c20", "FSR": "#7a21dd"}
        npz_payload = {}
        for i in self._reported_pt_indices():
            result = self.normalized_results[i]
            edges = np.asarray(self.gen_edges_by_pt[i], float)
            fig, ax = plt.subplots()
            for name in ("Vincia", "CR", "frag"):
                frac = np.asarray(self.model_shift_components[name][i], float)
                ax.stairs(frac, edges, color=colors[name], lw=2, label=name)
                npz_payload[f"{name}_{i}"] = frac
            fsr = np.asarray(self.model_fsr_frac[i], float)
            ax.stairs(fsr, edges, color=colors["FSR"], lw=2, label="FSR")
            npz_payload[f"FSR_{i}"] = fsr
            envelope = np.asarray(result["model_unc_frac"], float)
            ax.stairs(envelope, edges, color="black", lw=3, label="Envelope")
            npz_payload[f"envelope_{i}"] = envelope
            npz_payload[f"edges_{i}"] = edges
            ax.set_xlim(*self._observable_xlim(i))
            # Scale to the shown bins only: the buffer bins hidden outside the
            # xlim window carry large fractions that would otherwise squash
            # the visible curves (groomed panels blew up to 0.9-1.6).
            xlo, xhi = self._observable_xlim(i)
            centers = 0.5 * (edges[:-1] + edges[1:])
            visible = (centers >= xlo) & (centers <= xhi)
            curves = [
                np.asarray(self.model_shift_components[name][i], float)
                for name in ("Vincia", "CR", "frag")
            ] + [fsr, envelope]
            vis_max = (
                max(float(c[visible].max()) for c in curves)
                if np.any(visible) else 0.0
            )
            ax.set_ylim(0, max(0.1, 1.3 * vis_max))
            ax.set_xlabel(self._observable_label())
            ax.set_ylabel("Model uncertainty fraction")
            ax.legend(title=result["pt_bin"], fontsize=15)
            hep.cms.label(
                self.cms_label, data=True, lumi=self._lumi_label(),
                com=self._com_label(), fontsize=20, ax=ax,
            )
            save_path = (
                f"./{self.spec.output_dir}unfold/"
                f"model_envelope_{groomed_tag}_{i - 1}.pdf"
            )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        npz_path = Path(f"./{self.spec.output_dir}unfold")
        npz_path.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path / f"model_envelope_{groomed_tag}.npz", **npz_payload)

    def run_all_plots(self, show=False):
        # Initialize the shared style before the first figure. Otherwise the
        # first mode can use Matplotlib defaults until a later plot sets CMS.
        hep.style.use("CMS")
        self.plot_unfolded_fancy(show=show)
        self.plot_unfolded_summary_linear(show=show)
        self.plot_statistical_fraction(show=show)
        self.plot_systematic_fraction(show=show)
        self.plot_systematic_fraction_grouped(show=show)
        self.plot_systematic_fraction_grouped(show=show, log=False)
        if getattr(self.spec, "model_envelope", False):
            self.plot_model_envelope(show=show)
        if self.has_herwig:
            self.plot_herwig_systematic(show=show)
        q2_group = (
            ["q2"]
            if any(name in self.systematics for name in ("q2Up", "q2Down"))
            else ["q2muR", "q2muF"]
        )
        # The lepton entries are aggregates built inside
        # plot_systematic_frac_indiv from the per-leg systematics, whose raw
        # names (elereco/eleid/...) don't start with the aggregate name -- map
        # them to the prefixes that actually appear in self.systematics.
        group_prefix_aliases = {
            "ElectronSF": ("elereco", "eleid", "eletrig"),
            "MuonSF": ("mureco", "muid", "mutrig", "muiso"),
        }
        for systematic_group in (
            ["JES", "JER"],
            ["JMS", "JMR"],
            [*q2_group, "pdf"],
            ["pu", "l1prefiring", "lumi"],
            ["ElectronSF", "MuonSF"],
            ["isr", "fsr"],
        ):
            available_group = [
                name for name in systematic_group
                if self._has_systematic(*group_prefix_aliases.get(name, (name,)))
            ]
            if available_group:
                self.plot_systematic_frac_indiv(available_group, show=show)
        if self.has_herwig and self._has_systematic("herwig"):
            self.plot_systematic_frac_indiv(["herwig"], show=show)
        self.plot_correlation(show=show)
        if getattr(self.spec, "normalize_over_shown", False):
            # Reported-region companion (approval-talk convention): each pT
            # slice cropped to its shown rho window.
            self.plot_correlation(show=show, shown_only=True)
        self.plot_lcurve(show=show)
        self.save_normalized_covariance()
        self.save_2d_unfolded()
        self.save_2d_uncertainty_summary()
        self.plot_unfolded_unrolled_2d(show=show)
        self.plot_jackknife_convergence(show=show)
        self.plot_uncertainty_heatmap(show=show)
        self.plot_unfolded(show=show)
        if self.has_herwig:
            self.plot_herwig_bias_test(show=show)
        self.plot_response_matrix(probability=True, show=show)
        self.plot_folded(show=show)
        self.plot_bottom_line(show=show)
        # ARC round-2 (WS3.3): detector level rebinned to the gen binning, so
        # K is square and chi2_smeared is directly comparable to chi2_unfold.
        # These are the panels shown in the AN: no chi2 stamp, the numbers come
        # from the chi2 summary bar charts below.
        self.plot_bottom_line(show=show, rebin_reco_to_gen=True,
                              annotate_chi2=False)
        # PRIMARY (ARC round-2): chi2/ndof over the SHOWN space only -- each
        # slice cut at the rho value it is displayed to, with one global number.
        self.plot_bottom_line_chi2_summary(show=show, normalized=True, min_edge="shown")
        self.plot_bottom_line_chi2_summary(show=show, min_edge="shown")
        # Reference: the full-range and raw-chi2 versions.
        self.plot_bottom_line_chi2_summary(show=show)
        self.plot_bottom_line_chi2_summary(show=show, normalized=True)
        self.plot_fakes_misses(show=show)
        self.plot_purity_stability(show=show)
        if self.has_validation_inputs:
            self.plot_input_data_mc(show=show)
    def _plot_response_mosaic_cms(
        self,
        mosaic,
        reco_mass_edges_by_pt,   # list: per reco-pT slice, mass edges used (rows per block)
        gen_mass_edges_by_pt,    # list: per gen-pT slice,  mass edges used (cols per block)
        reco_pt_edges,           # e.g. [200, 290, 400, 13000]
        gen_pt_edges,            # e.g. [200, 290, 400, 13000]
        *,
        mask_zeros=True,
        probability=True,  # if True, normalize each block to sum to 1
        log=False,
        cmap="viridis",
        rlabel=None,             # e.g. "Ungroomed"
        show_cond=False,         # append the condition number to rlabel (ARC r2: leaning "don't show")
        vmin=None, vmax=None,
        ax=None
    ):
        """
        Draw a CMS-style gridded 'flattened' response plot.
        - `mosaic` is your unpadded 2D array (blocks concatenated).
        - Mass bin counts per (reco pT, gen pT) block come from the edge lists.
        - Dashed grid lines drawn at pT-block boundaries.
        """
        if probability:
            # Normalize each column to sum to 1
            mosaic = mosaic / np.sum(mosaic, axis=0, keepdims=True)
            # ensure no NaNs
            mosaic = np.nan_to_num(mosaic, nan=0.0)
        
        # Compute all the singular values of the mosaic matrix
        # and condition number (ratio of largest to smallest singular value)
        # to check for numerical stability.

        singular_values = np.linalg.svd(mosaic, compute_uv=False)
        #print("Singular values of the response mosaic:", singular_values)

        hep.style.use("CMS")
        if ax is None:
            # Keep the CMS-style ~10x10 per-axes footprint (a larger canvas
            # shrinks every font relative to it in the scaled-down PDF; ARC
            # round-2 "make all labels much bigger") -- the extra width pays
            # for the colorbar so the matrix axes stay square. Constrained
            # layout keeps the CMS-label spacing and the colorbar height.
            fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
        else:
            fig = ax.figure

        img = mosaic.copy()
        if mask_zeros:
            img = np.ma.masked_where(img == 0, img)

        # Choose norm
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else None

        # ---- Physical bin coordinates: stack the per-pT-block rho edges so
        # every cell is drawn with its true log10(rho^2) width. The matching
        # (i,i) block diagonal is then the real y=x line -- with index-space
        # imshow the unequal rho bin widths made it a fake diagonal.
        def _stacked_edges(edges_by_pt):
            coords = [0.0]
            bounds = [0.0]
            for edges in edges_by_pt:
                e = np.asarray(edges, dtype=float)
                coords.extend(bounds[-1] + (e[1:] - e[0]))
                bounds.append(bounds[-1] + (e[-1] - e[0]))
            return np.asarray(coords), np.asarray(bounds)

        x_coords, x_bounds = _stacked_edges(gen_mass_edges_by_pt)
        y_coords, y_bounds = _stacked_edges(reco_mass_edges_by_pt)

        # rasterized=True: the mesh is embedded as one high-res raster image
        # (see the dpi in savefig below) instead of per-cell vector rectangles,
        # whose shared edges every PDF rasterizer renders as hairline seams.
        im = ax.pcolormesh(x_coords, y_coords, img, cmap=cmap, norm=norm,
                           vmin=None if log else vmin, vmax=None if log else vmax,
                           antialiased=False, rasterized=True)
        ax.set_xlim(x_coords[0], x_coords[-1])
        ax.set_ylim(y_coords[0], y_coords[-1])

        # Dashed lines at pT-block boundaries
        for y in y_bounds[1:-1]:
            ax.axhline(y, color="r", ls="--", lw=2, alpha=0.6)
        for x in x_bounds[1:-1]:
            ax.axvline(x, color="r", ls="--", lw=2, alpha=0.6)

        # ---- Tick labels at pT bin centers ----
        x_centers = (x_bounds[:-1] + x_bounds[1:]) / 2.0
        y_centers = (y_bounds[:-1] + y_bounds[1:]) / 2.0
        # Label with pT edges (lower edges are fine; pick what you prefer)
        x_labels = [f"{int(gen_pt_edges[i])}–{int(gen_pt_edges[i+1]) if i+1 < len(gen_pt_edges)-1 else '∞'}" for i in range(len(gen_pt_edges)-1)]
        y_labels = [f"{int(reco_pt_edges[i])}–{int(reco_pt_edges[i+1]) if i+1 < len(reco_pt_edges)-1 else '∞'}" for i in range(len(reco_pt_edges)-1)]

        # This canvas is 12 in wide against the 10 in of the other published
        # figures, so it scales down further at the same \textwidth: give its
        # type the matching 1.2x so on-page sizes agree.
        mosaic_label_fs = round(PUB_LABEL_FONTSIZE * 1.2)
        mosaic_tick_fs = round(PUB_TICK_FONTSIZE * 1.2)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_centers)
        ax.set_yticklabels(y_labels, rotation=90, va="center")
        ax.tick_params(which="both", top=False, right=False,
                       labelsize=mosaic_tick_fs)

        ax.set_xlabel(r"Gen. $\log_{10}(\rho^{2}) \otimes p_{\mathrm{T}}$ (GeV)",
                      fontsize=mosaic_label_fs)
        ax.set_ylabel(r"Reco. $\log_{10}(\rho^{2}) \otimes p_{\mathrm{T}}$ (GeV)",
                      fontsize=mosaic_label_fs)

        # True y=x diagonal within the matching (i,i) blocks (gen and reco
        # cover the same rho range per block, so corner-to-corner is exact).
        for i in range(min(len(x_bounds)-1, len(y_bounds)-1)):
            ax.plot([x_bounds[i], x_bounds[i+1]], [y_bounds[i], y_bounds[i+1]],
                    color="r", lw=1, alpha=0.7)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Migration probability" if probability else "Counts",
                       fontsize=mosaic_label_fs)
        cbar.ax.tick_params(labelsize=mosaic_tick_fs)

        # CMS label
        rparts = [rlabel] if rlabel else []
        if show_cond:
            rparts.append(f"Cond. = {np.linalg.cond(mosaic):.2f}")
        hep.cms.label(self.cms_label, data=False, rlabel=", ".join(rparts))

        groomed_tag = "groomed" if self.groomed else "ungroomed"
        resp_path = self._relocate_output(f"response_{groomed_tag}.pdf")
        resp_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_label_flavors(fig, resp_path, dpi=300)
        return fig, ax
    
    def _make_inputs_numpy(self, filenames=None):
        if filenames is None:
            filenames = [self.spec.input_dir + f for f in self.spec.era_mc_files]
        # Load pickled inputs and save as numpy arrays for faster loading in the future
        # Define a mapping from era to key in the response dictionary
        from collections import defaultdict
        print("------------- Making numpy inputs from pickled files -----------------")

        keymap = {
            "2016": "pythia_UL16NanoAODv9",
            "2016APV": "pythia_UL16NanoAODAPVv9",
            "2017": "pythia_UL17NanoAODv9",
            "2018": "pythia_UL18NanoAODv9",
        }
        response_key_u = self.spec.hist_keys_ungroomed["response"]
        response_key_g = self.spec.hist_keys_groomed["response"]
        response_dict = {}
        ratio_data_mc = [1.0, 1.0, 1.0, 1.0]  # Placeholder for data/MC ratios per pt bin if needed for scaling
        for filename in filenames:
            era = Path(filename).stem.replace(self.spec.era_prefix, "", 1)
            print(f"Processing era: {era}")
            key = keymap[era]
            response_dict.setdefault('u', {})
            response_dict.setdefault('g', {})


            with open(filename, "rb") as f:
                data = pkl.load(f)
                # ensure top-level 'u' and 'g' keys exist
                # debug: check what keys are in the loaded data
                print(f"keys in data for {era}: {list(data.keys())}")
                if era == "2016APV" or era == "2016":
                    response_dict['u'][key] = data[response_key_u].project('ptreco','dataset', 'ptgen', self.gen_axis,  self.reco_axis, 'systematic')
                    response_dict['g'][key] = data[response_key_g].project('ptreco','dataset', 'ptgen', self.gen_axis,  self.reco_axis, 'systematic')
                    continue
                h_old = data[response_key_u].project('ptreco','dataset', 'ptgen', self.gen_axis,  self.reco_axis, 'systematic')
                #h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
                response_dict['u'][key] = h_old

                h_old = data[response_key_g].project('ptreco','dataset', 'ptgen', self.gen_axis,  self.reco_axis, 'systematic')
                #h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
                response_dict['g'][key] = h_old

                for i in range(4):
                    response_dict['u'][key].view()[i] *= ratio_data_mc[i]
                    response_dict['g'][key].view()[i] *= ratio_data_mc[i]

        correlation_dic = {
            'JES_AbsoluteMPFBias': 1,
            'JES_AbsoluteScale': 1,
            'JES_AbsoluteStat': 0,
            'JES_FlavorQCD': 1,
            'JES_Fragmentation': 1,
            'JES_PileUpDataMC': 0.5,
            'JES_PileUpPtBB': 0.5,
            'JES_PileUpPtEC1': 0.5,
            'JES_PileUpPtEC2': 0.5,
            'JES_PileUpPtHF': 0.5,
            'JES_PileUpPtRef': 0.5,
            'JES_RelativeFSR': 0.5,
            'JES_RelativeJEREC1': 0,
            'JES_RelativeJEREC2': 0,
            'JES_RelativeJERHF': 0.5,
            'JES_RelativePtBB': 0.5,
            'JES_RelativePtEC1': 0,
            'JES_RelativePtEC2': 0,
            'JES_RelativePtHF': 0.5,
            'JES_RelativeBal': 0.5,
            'JES_RelativeSample': 0,
            'JES_RelativeStatEC': 0,
            'JES_RelativeStatFSR': 0,
            'JES_RelativeStatHF': 0,
            'JES_SinglePionECAL': 1,
            'JES_SinglePionHCAL': 1,
            'JES_TimePtEta': 0,
            'JER': 0,
        }

        jes_sys_list = ['JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown',
                        'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp',
                        'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown',
                        'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
                        'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up',
                        'JES_RelativeJEREC1Down', 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown',
                        'JES_RelativePtBBUp', 'JES_RelativePtBBDown', 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 'JES_RelativePtEC2Down',
                        'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 
                        'JES_RelativeStatECUp', 'JES_RelativeStatECDown', 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown',
                        'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown', 'JERUp', 'JERDown']


        non_jes_sys_list = ['nominal', 'puUp', 'puDown', 'elerecoUp', 'elerecoDown',
                            'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp',
                            'murecoDown', 'muidUp', 'muidDown', 'mutrigUp', 'muisoUp', 'muisoDown','mutrigDown', 'pdfUp',
                            'pdfDown', 'q2Up', 'q2Down', 'l1prefiringUp', 'l1prefiringDown', 'isrUp', 'isrDown', 'fsrUp', 'fsrDown',
                            'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']

        # The lists above enumerate the full production; reduced skims (e.g.
        # the minimal_syst nominal+JER/JMS/JMR comparison pkls) carry fewer
        # categories, so keep only what the input actually provides.
        available_systematics = set()
        for era_response in response_dict['g'].values():
            available_systematics.update(str(c) for c in era_response.axes['systematic'])
        jes_sys_list = [sys for sys in jes_sys_list if sys in available_systematics]
        non_jes_sys_list = [sys for sys in non_jes_sys_list if sys in available_systematics]

        non_jes_sys_list_up = [sys for sys in non_jes_sys_list if sys[-2:] == 'Up' ]
        non_jes_sys_list_down = [sys for sys in non_jes_sys_list if sys[-4:] == 'Down' ]

        jes_sys_list_up = [sys for sys in jes_sys_list if sys[-2:] == 'Up' ]
        jes_sys_list_down = [sys for sys in jes_sys_list if sys[-4:] == 'Down' ]

        sys_matrix_dic_up = {}

        groomed = self.groomed
        if not groomed:
            response = response_dict['u']
        else:
            response = response_dict['g']





        for sys in jes_sys_list_up:
            m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() +\
                response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()

            m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() + \
                response['pythia_UL16NanoAODAPVv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            
            
            m_var_2016 = m_sys_2016 + m_nom_2017 + m_nom_2018
            m_var_2017 = m_nom_2016 + m_sys_2017 + m_nom_2018
            m_var_2018 = m_nom_2016 + m_nom_2017 + m_sys_2018
            
            
            rho = correlation_dic[sys[:-2]] ## correlation factor
            
            sigma_2016 = m_sys_2016 - m_nom_2016
            sigma_2017 = m_sys_2017 - m_nom_2017
            sigma_2018 = m_sys_2018 - m_nom_2018
            
            sigma_corr = rho*sigma_2016 + rho*sigma_2017 + rho*sigma_2018
            
            sigma_uncorr_2016 = (1-rho)*sigma_2016
            sigma_uncorr_2017 = (1-rho)*sigma_2017
            sigma_uncorr_2018 = (1-rho)*sigma_2018
            
            m_nom =  m_nom_2016 + m_nom_2017 + m_nom_2018
            m_corr = m_nom + sigma_corr
            
            m_uncorr_2016 = m_nom + sigma_uncorr_2016
            
            m_uncorr_2017 = m_nom + sigma_uncorr_2017

            m_uncorr_2018 = m_nom + sigma_uncorr_2018

            

            sys_matrix_dic_up[sys+'_corr'] = m_corr
            sys_matrix_dic_up[sys+'_uncorr_2016'] = m_uncorr_2016
            sys_matrix_dic_up[sys+'_uncorr_2017'] = m_uncorr_2017
            sys_matrix_dic_up[sys+'_uncorr_2018'] = m_uncorr_2018
            
        non_jes_sys_matrix_dic_up = {}
        for sys in non_jes_sys_list_up:
            sys_matrix_dic_up[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()

            non_jes_sys_matrix_dic_up[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()



        sys_matrix_dic_down = {}
        non_jes_sys_matrix_dic_down = {}
        for sys in jes_sys_list_down:
            m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() + \
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            
            m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() + \
                        response['pythia_UL16NanoAODAPVv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
            
            
            m_var_2016 = m_sys_2016 + m_nom_2017 + m_nom_2018
            m_var_2017 = m_nom_2016 + m_sys_2017 + m_nom_2018
            m_var_2018 = m_nom_2016 + m_nom_2017 + m_sys_2018
            
            
            rho = correlation_dic[sys[:-4]] ## correlation factor
            
            sigma_2016 = m_sys_2016 - m_nom_2016
            sigma_2017 = m_sys_2017 - m_nom_2017
            sigma_2018 = m_sys_2018 - m_nom_2018
            
            sigma_corr = rho*sigma_2016 + rho*sigma_2017 + rho*sigma_2018
            
            sigma_uncorr_2016 = (1-rho)*sigma_2016
            sigma_uncorr_2017 = (1-rho)*sigma_2017
            sigma_uncorr_2018 = (1-rho)*sigma_2018
            
            m_nom =  m_nom_2016 + m_nom_2017 + m_nom_2018
            m_corr = m_nom + sigma_corr
            
            m_uncorr_2016 = m_nom + sigma_uncorr_2016
            
            m_uncorr_2017 = m_nom + sigma_uncorr_2017

            m_uncorr_2018 = m_nom + sigma_uncorr_2018

            

            sys_matrix_dic_down[sys+'_corr'] = m_corr
            sys_matrix_dic_down[sys+'_uncorr_2016'] = m_uncorr_2016
            sys_matrix_dic_down[sys+'_uncorr_2017'] = m_uncorr_2017
            sys_matrix_dic_down[sys+'_uncorr_2018'] = m_uncorr_2018
            

        for sys in non_jes_sys_list_down:
            sys_matrix_dic_down[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()
        #                            + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values
            non_jes_sys_matrix_dic_down[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()

        if not groomed:
            response = response_dict['u']
        else:
            response = response_dict['g']

        sys_matrix_dic_up['nominal'] = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() +\
                response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()  +\
                    response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() +\
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()
        sys_matrix_dic_down['nominal'] = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() +\
                response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()  +\
                    response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values() +\
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis).values()

        prepend = self.spec.input_dir
        filename_herwig_1 = self._resolve_input_path(
            *(prepend + f for f in self.spec.reweighted_fallback_files)
        )
        with open(filename_herwig_1, "rb") as f:
            data_herwig = pkl.load(f)
            if not groomed:
                sys_matrix_dic_up['herwigUp'] = data_herwig[response_key_u].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()
                sys_matrix_dic_down['herwigDown'] = data_herwig[response_key_u].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()
            else:
                sys_matrix_dic_up['herwigUp'] = data_herwig[response_key_g].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()
                sys_matrix_dic_down['herwigDown'] = data_herwig[response_key_g].project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()

        ## Merging the up and down dictionaries
        merged = dict(sys_matrix_dic_up)  # shallow copy of up
        for k, v in sys_matrix_dic_down.items():
            if k == "nominal":
                continue  # skip duplicate nominal from the down dict
            if k in merged:
                # if a key collides, store the down version with a suffix
                merged_key = f"{k}_down"
                # ensure uniqueness
                i = 1
                unique_key = merged_key
                while unique_key in merged:
                    unique_key = f"{merged_key}_{i}"
                    i += 1
                merged[unique_key] = v
            else:
                merged[k] = v
        self.sys_matrix_dic = merged
        print(f"✅ saved {len(merged)} in sys_matrix_dic")

        # MC-stat variances (sumw2) of the nominal matched response, summed
        # over eras, in the same ('ptgen', gen, 'ptreco', reco) layout as
        # sys_matrix_dic['nominal']. These feed the TUnfold response-matrix
        # bin errors; without them ROOT defaults to sqrt(weighted content),
        # which badly overestimates the MC-stat error of weighted histograms.
        era_keys = [
            "pythia_UL16NanoAODv9",
            "pythia_UL16NanoAODAPVv9",
            "pythia_UL17NanoAODv9",
            "pythia_UL18NanoAODv9",
        ]
        variances = [
            response[era][..., "nominal"]
            .project("ptgen", self.gen_axis, "ptreco", self.reco_axis)
            .variances()
            for era in era_keys
        ]
        self.sys_matrix_variance = (
            np.sum(variances, axis=0) if all(v is not None for v in variances) else None
        )
    def _merge_eras(self, filenames=None):
        if filenames is None:
            filenames = [self.spec.input_dir + f for f in self.spec.era_mc_files]

        outputs = []
        for fname in filenames:
            with open(fname, "rb") as f:
                outputs.append(pkl.load(f))

        reco_keys = (self.spec.hist_keys_ungroomed["reco"], self.spec.hist_keys_groomed["reco"])
        response_keys = (self.spec.hist_keys_ungroomed["response"], self.spec.hist_keys_groomed["response"])
        gen_keys = (self.spec.hist_keys_ungroomed["gen"], self.spec.hist_keys_groomed["gen"])
        hist_keys = [*reco_keys, *response_keys, *gen_keys]
        out_dict = {}
        for i, output in enumerate(outputs):
            for key in hist_keys:
                if key in reco_keys:
                    if i == 0:
                        out_dict[key] = output[key].project('ptreco', self.reco_axis, 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptreco', self.reco_axis, 'systematic')
                elif key in response_keys:
                    if i == 0:
                        out_dict[key] = output[key].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis, 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptgen', self.gen_axis, 'ptreco', self.reco_axis, 'systematic')
                elif key in gen_keys:
                    if i == 0:
                        out_dict[key] = output[key].project('ptgen', self.gen_axis, 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptgen', self.gen_axis, 'systematic')
            self.pythia_hists = out_dict
        out_filename = self.spec.input_dir + self.spec.mc_file
        with open(out_filename, "wb") as f:
            pkl.dump(out_dict, f)

    def _merge_eras_jk(self, filenames=None):
        if filenames is None:
            filenames = [self.spec.input_dir + f for f in self.spec.era_jk_files]
        outputs = []
        for fname in filenames:
            with open(fname, "rb") as f:
                outputs.append(pkl.load(f))
        response_keys = (self.spec.hist_keys_ungroomed["response"], self.spec.hist_keys_groomed["response"])
        hist_keys = list(response_keys)
        out_dict = {}
        for i, output in enumerate(outputs):
            for key in hist_keys:
                if key in response_keys:
                    if i == 0:
                        out_dict[key] = output[key].project('jk','ptgen', self.gen_axis, 'ptreco', self.reco_axis, 'systematic')
                    else:
                        out_dict[key] += output[key].project('jk','ptgen', self.gen_axis, 'ptreco', self.reco_axis, 'systematic')
        self.pythia_hists_jk = out_dict
        return out_dict





# sys_matrix_dic['herwigUp'] = resp_matrix_4d_herwig.project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()
# sys_matrix_dic_down['herwigDown'] = resp_matrix_4d_herwig.project('ptgen',self.gen_axis,'ptreco',self.reco_axis).values()
