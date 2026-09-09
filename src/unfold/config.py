"""Analysis configuration: three channels, two observables, one tag registry.

``TAGS[(channel, observable)][tag]`` is what ``unfold run`` looks up.  A tag
is one of three kinds, which differ only in where the inputs come from:

* ``ObservableSpec``   Z+jet merged-era pickles (``zjet_inputs``);
* ``PairSplitTag``     dijet / trijet, full Run 2 pair-split pickles (``pairsplit``);
* ``ChannelTag``       dijet / trijet, single-year ``minimal_rho`` pickles (``channel_inputs``).

Every tag writes to ``outputs/<channel>/<observable>/<tag>/``.  The default
tag of every channel is ``original`` (the production configuration); older
or alternative configurations keep their own names.  ``describe(tag)`` prints
the resolved values, which is what the run manifest records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

CHANNELS = ("zjet", "dijet", "trijet")
OBSERVABLES = ("rho", "mass")
DEFAULT_TAG = "original"


def output_dir(channel, observable, tag):
    return f"outputs/{channel}/{observable}/{tag}/"


@dataclass(frozen=True)
class ObservableSpec:
    """Z+jet: inputs, binning names, plot labels and the unfolding knobs.

    The knobs (``method`` onward) are read by the engine for every channel;
    the pair-split and channel tags build a spec from ``RHO_BASE`` at run
    time and only override what they need.
    """
    # identity
    name: str                       # "rho" or "mass"
    reco_axis: str                  # histogram axis name on the reco side
    gen_axis: str                   # histogram axis name on the gen side

    # Z+jet input files (all relative to input_dir); unused by prepared inputs
    input_dir: str
    mc_file: str
    data_file: str
    herwig_file: str
    jk_data_file: str
    era_mc_files: tuple
    era_jk_files: tuple
    era_prefix: str                 # prefix stripped from a per-era stem to get the era
    reweighted_fallback_files: tuple  # response used for the "herwig" systematic
    bkg_file: str | None = None     # non-DY background subtracted from the input

    output_dir: str = "outputs/"

    # names in binning.ZJET_BINNINGS (ignored when a Binning is passed directly)
    binning_groomed: str = ""
    binning_ungroomed: str = ""
    gen_merge_below: float | None = None    # collapse gen bins below this value

    hist_keys_groomed: dict = None
    hist_keys_ungroomed: dict = None

    # plot labels and windows
    x_label_groomed: str = ""
    x_label_ungroomed: str = ""
    short_label_groomed: str = ""
    short_label_ungroomed: str = ""
    xlim_lower_groomed: float = 0.0
    xlim_lower_ungroomed: float = 0.0
    normalized_ylabel: str = ""
    band_color_total: str = "yellowgreen"
    band_color_stat: str = "darkgreen"
    bl_shown_floors_groomed: tuple | None = None     # per-pT rho floor of the shown space
    bl_shown_floors_ungroomed: tuple | None = None
    normalize_over_shown: bool = False               # normalize each slice over shown bins only
    normalization_window_groomed: tuple | None = None
    normalization_window_ungroomed: tuple | None = None
    display_window_groomed: tuple | None = None
    display_window_ungroomed: tuple | None = None

    # unfolding knobs
    method: str = "tunfold"         # "tunfold" or "roounfold_bayes"
    n_iter: int = 4                 # D'Agostini iterations
    regularization: str = "none"    # "none", "ratio_curvature", "curvature"
    tau: float | None = None        # None -> L-curve scan when regularized
    area_constraint: bool = True
    stat_propagation: str = "legacy"   # "legacy" or "jacobian"
    prediction_stat_method: str = "fixed_normalization"   # or "jacobian"

    # modelling uncertainty
    model_envelope: bool = False
    model_envelope_source: str = "zjet_offline"          # or "prepared_systematics"
    model_covariance_method: str = "selected_variation"  # or "enclosing_ellipsoid"
    model_covariance_scope: str = "global_shown"         # global_shown / global_all / per_pt / global_templates
    bottom_line_scale_mc_per_pt: bool = False

    def binning_name(self, groomed):
        return self.binning_groomed if groomed else self.binning_ungroomed

    def hist_keys(self, groomed):
        return self.hist_keys_groomed if groomed else self.hist_keys_ungroomed


@dataclass(frozen=True)
class PairSplitTag:
    """dijet / trijet rho from the Run-2 pair-split pickles (``unfold.pairsplit``)."""
    channel: str
    output_dir: str
    normalization_window: str = "full"          # "full" ([-3.5, 0]) or "peak" ([-2.0, -0.75]); ungroomed always [-2.5, 0]
    binning: str = "study_recommended"          # name in pairsplit.inputs.PAIR_SPLIT_BINNING_VARIANTS
    systematics: str = "nominal,all_safe_non_jes,JER,JES"
    model_envelope: bool = True
    model_covariance: str = "enclosing_ellipsoid"   # or "selected_variation"
    regularization: str = "none"                # "none" or "curvature"
    tau: float | None = None
    lumi: float = 138.0


@dataclass(frozen=True)
class ChannelTag:
    """dijet / trijet rho from one year's ``minimal_rho`` pickles under ``inputs/<channel>/rho/``."""
    channel: str
    output_dir: str
    year: str = "2018"
    lumi: float = 59.7


def describe(tag):
    """Plain dict of the resolved tag, for manifests and for reading."""
    return {"kind": type(tag).__name__, **asdict(tag)}


# ---------------------------------------------------------------------------
# Z+jet
# ---------------------------------------------------------------------------
_ZJET_ERAS = ("2016", "2016APV", "2017", "2018")

RHO_BASE = ObservableSpec(
    name="rho",
    reco_axis="mpt_reco",
    gen_axis="mpt_gen",
    input_dir="./inputs/zjet/rho/jmsjmr_unity/",
    mc_file="pythia_all.pkl",
    data_file="data_all.pkl",
    herwig_file="herwig_all.pkl",
    jk_data_file="jk_data_all.pkl",
    era_mc_files=tuple(f"pythia_{era}.pkl" for era in _ZJET_ERAS),
    era_jk_files=tuple(f"jk_pythia_{era}.pkl" for era in _ZJET_ERAS),
    era_prefix="pythia_",
    reweighted_fallback_files=("pythia_reweighted_all.pkl", "herwig_all.pkl"),
    output_dir=output_dir("zjet", "rho", "base"),
    binning_groomed="rho_groomed_v2",
    binning_ungroomed="rho_ungroomed_v2",
    hist_keys_groomed={"response": "response_matrix_rho_g", "reco": "ptjet_rhojet_g_reco", "gen": "ptjet_rhojet_g_gen"},
    hist_keys_ungroomed={"response": "response_matrix_rho_u", "reco": "ptjet_rhojet_u_reco", "gen": "ptjet_rhojet_u_gen"},
    x_label_groomed=r"$2\log_{10}(\rho)$, $\rho=m/(p_T R)$, groomed",
    x_label_ungroomed=r"$2\log_{10}(\rho)$, $\rho=m/(p_T R)$, ungroomed",
    short_label_groomed=r"$2\log_{10}(\rho)$, $\rho=m/(p_T R)$, Groomed",
    short_label_ungroomed=r"$2\log_{10}(\rho)$, $\rho=m/(p_T R)$, Ungroomed",
    xlim_lower_groomed=-4.5,
    xlim_lower_ungroomed=-2.5,
    normalized_ylabel=r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d[2\log_{10}(\rho)]\,dp_T}$",
)

# ARC round-2 settings (2026-07-10): groomed buffer binning, non-DY background
# subtracted, two-leg PS/HAD modelling envelope, report and normalize over the
# shown per-pT window.  HERWIG is overlay and closure only: the "herwig"
# systematic reads the nominal PYTHIA response (zero shift).
_ARC_R2_SETTINGS = dict(
    bkg_file="../jmsjmr_unity/bkg_all.pkl",
    reweighted_fallback_files=("pythia_all.pkl",),
    binning_groomed="rho_groomed_arcr2",
    binning_ungroomed="rho_ungroomed_arcr2",
    normalize_over_shown=True,
    model_envelope=True,
)
# Approved publication phase space: the response-limited groomed [-3.5, -3.0]
# bin of the 400-inf slice is dropped from the shown and normalized space.
_FLOOR3_SETTINGS = dict(xlim_lower_groomed=-3.0, bl_shown_floors_groomed=(-2.5, -3.0, -3.0, -3.0))

# The production: ARC round-2 settings and the floor-3 phase space on the
# reskim with the UL unity JMS/JMR tables (formerly jmsjmr_unity_groomed400_floor3).
ZJET_RHO_ORIGINAL = replace(RHO_BASE, output_dir=output_dir("zjet", "rho", "original"),
                            **_ARC_R2_SETTINGS, **_FLOOR3_SETTINGS)

# The ARC round-2 deliverable itself: arc_r2 reskim (old per-year JMS/JMR
# tables), groomed floors of the 5 GeV mass floor, no floor-3 cut.
ZJET_RHO_ARC_R2 = replace(RHO_BASE, input_dir="./inputs/zjet/rho/arc_r2/",
                          output_dir=output_dir("zjet", "rho", "arc_r2"),
                          xlim_lower_groomed=-3.5, bl_shown_floors_groomed=(-2.5, -3.0, -3.0, -3.5),
                          **_ARC_R2_SETTINGS)

MASS_BASE = replace(
    RHO_BASE,
    name="mass",
    reco_axis="mreco",
    gen_axis="mgen",
    input_dir="./inputs/zjet/mass/",
    mc_file="mass_pythia_all.pkl",
    data_file="mass_data_all.pkl",
    herwig_file="mass_herwig_all.pkl",
    jk_data_file="mass_jk_data_all.pkl",
    era_mc_files=tuple(f"mass_pythia_{era}.pkl" for era in _ZJET_ERAS),
    era_jk_files=tuple(f"mass_jk_pythia_{era}.pkl" for era in _ZJET_ERAS),
    era_prefix="mass_pythia_",
    reweighted_fallback_files=("mass_pythia_reweighted_all.pkl", "pythia_reweighted_all.pkl",
                               "mass_herwig_all.pkl", "herwig_all.pkl"),
    output_dir=output_dir("zjet", "mass", "original"),
    binning_groomed="mass_groomed",
    binning_ungroomed="mass_ungroomed",
    hist_keys_groomed={"response": "response_matrix_g", "reco": "ptjet_mjet_g_reco", "gen": "ptjet_mjet_g_gen"},
    hist_keys_ungroomed={"response": "response_matrix_u", "reco": "ptjet_mjet_u_reco", "gen": "ptjet_mjet_u_gen"},
    x_label_groomed="Groomed Jet Mass (GeV)",
    x_label_ungroomed="Ungroomed Jet Mass (GeV)",
    short_label_groomed="Jet Mass (GeV), Groomed",
    short_label_ungroomed="Jet Mass (GeV), Ungroomed",
    xlim_lower_groomed=10.0,
    xlim_lower_ungroomed=20.0,
    normalized_ylabel=r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{dm\,dp_T} (GeV^{-1})$",
)

# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
TAGS = {
    ("zjet", "rho"): {
        "original": ZJET_RHO_ORIGINAL,
        "arc_r2": ZJET_RHO_ARC_R2,
        "arc_r2_groomed400_floor3": replace(ZJET_RHO_ARC_R2, output_dir=output_dir("zjet", "rho", "arc_r2_groomed400_floor3"),
                                            **_FLOOR3_SETTINGS),
    },
    # mass inputs must be regenerated before this runs (inputs/zjet/mass/ is empty)
    ("zjet", "mass"): {"original": MASS_BASE},
    ("dijet", "rho"): {
        # the production: aligned binning, peak-window normalization, full safe systematics
        "original": PairSplitTag("dijet", output_dir("dijet", "rho", "original"), normalization_window="peak"),
        # single-year minimal_rho pickles under inputs/dijet/rho/ (pre pair-split)
        "2018": ChannelTag("dijet", output_dir("dijet", "rho", "2018")),
    },
    ("trijet", "rho"): {
        "original": PairSplitTag("trijet", output_dir("trijet", "rho", "original"), normalization_window="full"),
        "2018": ChannelTag("trijet", output_dir("trijet", "rho", "2018")),
    },
}


def get_tag(channel, observable="rho", tag=None):
    if (channel, observable) not in TAGS:
        available = ", ".join(f"{c}/{o}" for c, o in TAGS)
        raise KeyError(f"no tags for {channel}/{observable}; available: {available}")
    tags = TAGS[(channel, observable)]
    tag = tag or DEFAULT_TAG
    if tag not in tags:
        raise KeyError(f"unknown {channel}/{observable} tag {tag!r}; known: {', '.join(tags)}")
    return tags[tag]


def list_tags():
    lines = []
    for (channel, observable), tags in TAGS.items():
        for name, tag in tags.items():
            lines.append(f"{channel:7s} {observable:5s} {name:28s} {type(tag).__name__:15s} -> {tag.output_dir}")
    return "\n".join(lines)


def with_options(spec, *, jacobian=False, regularization=None, tau=None, method=None,
                 n_iter=None, model_covariance_scope=None):
    """Apply command-line overrides to an ``ObservableSpec``; option runs get a suffixed output dir."""
    suffix = option_suffix(jacobian=jacobian, regularization=regularization, method=method,
                           base_regularization=spec.regularization)
    if jacobian and spec.stat_propagation != "jacobian":
        spec = replace(spec, stat_propagation="jacobian")
    if regularization is not None and regularization != spec.regularization:
        spec = replace(spec, regularization=regularization)
    if tau is not None:
        if spec.regularization == "none":
            raise ValueError("--tau needs a regularization")
        spec = replace(spec, tau=tau)
    if method is not None and method != spec.method:
        spec = replace(spec, method=method)
    if n_iter is not None:
        spec = replace(spec, n_iter=n_iter)
    if model_covariance_scope is not None:
        spec = replace(spec, model_covariance_scope=model_covariance_scope)
    if suffix:
        spec = replace(spec, output_dir=spec.output_dir.rstrip("/") + suffix + "/")
    return spec


def option_suffix(*, jacobian=False, regularization=None, method=None, base_regularization="none"):
    """Output-directory suffix for option runs, so a tag's own outputs are never overwritten."""
    suffix = ""
    if jacobian:
        suffix += "_jacobian"
    if regularization is not None and regularization != base_regularization:
        suffix += "_reg" if regularization != "none" else "_noreg"
    if method == "roounfold_bayes":
        suffix += "_bayes"
    return suffix
