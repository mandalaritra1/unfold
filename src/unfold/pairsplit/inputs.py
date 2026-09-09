"""Read and validate Run-2 pair-split inputs without coupling to an unfolder.

The producer's histogram names are retained only at this boundary.  All arrays
and metadata exposed by this module use the physical observable definition
``rho = m / (pT * R)`` with ``R = 0.8`` and call the stored transformed
coordinate ``two_log10_rho = 2 * log10(rho)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle

import hist
import numpy as np

from unfold.paths import PAIRSPLIT_INPUTS


PAIR_SPLIT_ERAS = ("2016APV", "2016", "2017", "2018")
PAIR_SPLIT_CHANNELS = ("dijet", "trijet")
PAIR_SPLIT_INPUT_ROOT = PAIRSPLIT_INPUTS

JET_RADIUS = 0.8
PHYSICAL_RHO_DEFINITION = "rho = m / (pT * R), with R = 0.8"
TRANSFORMED_COORDINATE_NAME = "two_log10_rho"
TRANSFORMED_COORDINATE_DEFINITION = "two_log10_rho = 2 * log10(rho)"

# Run-2 JES/JER categories in the producer are combined-era variations.  The
# unfolding needs statistically meaningful alternatives, so we resolve them
# into virtual correlated and era-uncorrelated legs *before* any candidate
# rebinning.  Values use the square-root covariance factors; their sumw2 is
# intentionally the nominal Run-2 sumw2 because category-to-category MC
# statistical covariances are not stored in the inputs.
RUN2_ERA_GROUPS = {
    "2016": ("2016APV", "2016"),
    "2017": ("2017",),
    "2018": ("2018",),
}
JES_RUN2_CORRELATIONS = {
    "AbsoluteMPFBias": 1.0,
    "AbsoluteScale": 1.0,
    "AbsoluteStat": 0.0,
    "FlavorQCD": 1.0,
    "Fragmentation": 1.0,
    "PileUpDataMC": 0.5,
    "PileUpPtBB": 0.5,
    "PileUpPtEC1": 0.5,
    "PileUpPtEC2": 0.5,
    "PileUpPtHF": 0.5,
    "PileUpPtRef": 0.5,
    "RelativeFSR": 0.5,
    "RelativeJEREC1": 0.0,
    "RelativeJEREC2": 0.0,
    "RelativeJERHF": 0.5,
    "RelativePtBB": 0.5,
    "RelativePtEC1": 0.0,
    "RelativePtEC2": 0.0,
    "RelativePtHF": 0.5,
    "RelativeBal": 0.5,
    "RelativeSample": 0.0,
    "RelativeStatEC": 0.0,
    "RelativeStatFSR": 0.0,
    "RelativeStatHF": 0.0,
    "SinglePionECAL": 1.0,
    "SinglePionHCAL": 1.0,
    "TimePtEta": 0.0,
}
DEFECTIVE_JES_SOURCES = (
    "RelativeJEREC1",
    "RelativeJEREC2",
    "RelativeJERHF",
)
JER_RUN2_CORRELATION = 0.0

# This is a resolver expression, rather than a fixed category list, because
# the producer controls the safe non-JES nuisance inventory.
FULL_SAFE_SYSTEMATIC_REQUEST = "nominal,all_safe_non_jes,JER,JES"

# Luminosity is a coherent rate variation and is annihilated by this
# workflow's per-pT normalization window, so it is intentionally not a shape
# nuisance.  This is a narrow physics exclusion, not a generic filter.
PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS = {
    "LuminosityUp": (
        "normalization-only luminosity rate variation; per-pT normalization "
        "cancels a coherent scale exactly"
    ),
    "LuminosityDown": (
        "normalization-only luminosity rate variation; per-pT normalization "
        "cancels a coherent scale exactly"
    ),
}

# The pair-split producer intentionally keeps the GEN marginal nominal for
# detector-side variations, while response/reco use the varied event weights
# or detector kinematics.  Their central response, fakes, and misses remain
# meaningful, but no cross-category MC-statistical covariance is stored for
# ``Var(gen_nominal - matched_varied)``.  Use nominal Run-2 sumw2 for the
# *statistical errors* of these central-shift legs, matching the existing
# virtual JES/JER treatment.  TUnfold stores analytic covariance only for the
# nominal unfold; non-nominal legs contribute central shift vectors.
PAIR_SPLIT_NOMINAL_VARIANCE_SYSTEMATICS = (
    "puUp",
    "puDown",
    "l1prefiringUp",
    "l1prefiringDown",
    "JMSUp",
    "JMSDown",
    "JMRUp",
    "JMRDown",
)
PAIR_SPLIT_NOMINAL_VARIANCE_REASON = (
    "nominal Run-2 response/reco/gen sumw2 retained for detector-side central "
    "variation: the producer keeps the GEN marginal nominal and does not store "
    "cross-category MC-statistical covariance"
)


# These are the only intentionally legacy names in this module.  They map
# producer histogram keys to the terminology used by the returned arrays.
LEGACY_HISTOGRAM_KEYS = {
    "groomed": {
        "response": "response_matrix_rho_g",
        "reco": "ptjet_rhojet_g_reco",
        "gen": "ptjet_rhojet_g_gen",
        "reco_covariance": "reco_cov_rho_g",
    },
    "ungroomed": {
        "response": "response_matrix_rho_u",
        "reco": "ptjet_rhojet_u_reco",
        "gen": "ptjet_rhojet_u_gen",
        "reco_covariance": "reco_cov_rho_u",
    },
}

_ARRAY_AXES = {
    "response": ("ptreco", "mpt_reco", "ptgen", "mpt_gen"),
    "reco": ("ptreco", "mpt_reco"),
    "gen": ("ptgen", "mpt_gen"),
    "reco_covariance": (
        "ptreco_i",
        "mpt_reco_i",
        "ptreco_j",
        "mpt_reco_j",
    ),
}


@dataclass(frozen=True)
class PairSplitFineAxes:
    """Fine axes stored in the pair-split producer inputs for one mode."""

    pt_edges: tuple[float, ...]
    two_log10_rho_reco_edges: tuple[float, ...]
    two_log10_rho_gen_edges: tuple[float, ...]


PAIR_SPLIT_FINE_AXES = {
    # The Z+jet-ALIGNED producer lattice (2026-08-27 re-production): every
    # gen edge above the -3.5 shown floor on the quarter-integer grid, reco
    # the exact 2:1 halving (0.125 steps).  smp hist_utils.py `_had_rho_gen_g`.
    "groomed": PairSplitFineAxes(
        pt_edges=(185.0, 200.0, 290.0, 400.0, 480.0, 570.0, 680.0, 760.0, 820.0, 13000.0),
        two_log10_rho_reco_edges=(
            -10.0, -7.5, -5.0, -4.5, -4.0, -3.75, -3.5, -3.375,
            -3.25, -3.125, -3.0, -2.875, -2.75, -2.625, -2.5, -2.375,
            -2.25, -2.125, -2.0, -1.875, -1.75, -1.625, -1.5, -1.375,
            -1.25, -1.125, -1.0, -0.875, -0.75, -0.625, -0.5, -0.375,
            -0.25, -0.125, 0.0,
        ),
        two_log10_rho_gen_edges=(
            -10.0, -5.0, -4.0, -3.5, -3.25, -3.0, -2.75, -2.5,
            -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
            -0.25, 0.0,
        ),
    ),
    "ungroomed": PairSplitFineAxes(
        pt_edges=(185.0, 200.0, 290.0, 400.0, 480.0, 570.0, 680.0, 760.0, 820.0, 13000.0),
        two_log10_rho_reco_edges=(
            -10.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.75, -4.5,
            -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5,
            -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
            -0.25, 0.0,
        ),
        two_log10_rho_gen_edges=(
            -10.0, -8.0, -7.0, -6.0, -5.0, -4.5, -4.0, -3.5,
            -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
        ),
    ),
}


@dataclass(frozen=True)
class PairSplitBinning:
    """One explicit candidate binning for the pair-split inputs.

    ``pt_groups`` are source fine-bin indices.  The source 185--200 GeV bin
    is deliberately a sink bin, excluded from every group.  The first output
    pT bin is therefore also the first reported pT bin.  The two-log
    coordinate includes a hidden migration buffer; ``reported_two_log10_rho_minimum``
    is the lower edge of the published coordinate range.
    """

    channel: str
    variant: str
    grooming_mode: str
    pt_edges: tuple[float, ...]
    pt_groups: tuple[tuple[int, ...], ...]
    base_reco_two_log10_rho_edges: tuple[float, ...]
    base_to_gen_two_log10_rho_groups: tuple[tuple[int, ...], ...]
    sink_pt_source_bin_indices: tuple[int, ...]
    first_reported_pt_index: int
    reported_two_log10_rho_minimum: float

    @property
    def gen_two_log10_rho_edges(self) -> tuple[float, ...]:
        """Derived GEN edges after the candidate-only base-bin merges."""

        return _grouped_edges(
            self.base_reco_two_log10_rho_edges,
            self.base_to_gen_two_log10_rho_groups,
            context=f"{self.channel} {self.variant} base-to-gen edges",
        )


# Base RECO selections from the aligned producer lattice: 2:1 halves of the
# approved gen grids (study: outputs/studies/aligned_binning/, candidates
# "aligned_seven" dijet / "wide_tail_deep" trijet), with the LAST gen bin
# kept 1:1 at the m ~ pT*R kinematic edge (near-empty upper reco half, the
# recurring lesson from the ungroomed 2:1 rebuild) and the [-10, -3.5]
# migration buffer a single 1:1 bin.
_GROOMED_CANDIDATE_COORDINATE_EDGES_BY_CHANNEL = {
    "dijet": (
        -10.0, -3.5, -3.0, -2.5, -2.25, -2.0, -1.75, -1.5,
        -1.375, -1.25, -1.125, -1.0, -0.875, -0.75, 0.0,
    ),
    "trijet": (
        -10.0, -3.5, -3.0, -2.5, -2.25, -2.0, -1.75, -1.5,
        -1.25, -1.0, 0.0,
    ),
}

# The ungroomed producer truth axis is intrinsically coarser than the groomed
# one: 0.5-wide GEN bins above the hidden [-10, -2.5] migration catch-all.
# The producer RECO axis does carry 0.25-wide bins there, so the candidate
# keeps them — the standard 2:1 reco:gen refinement (as in Z+jet) that gives
# the least-squares unfold real residual degrees of freedom and makes the
# refold comparison a genuine check (a square response refolds exactly by
# construction).  The catch-all stays 1:1, and so does the LAST gen bin
# [-0.5, 0]: at the m ~ pT*R kinematic edge the data leave its upper 0.25
# half nearly empty (single weighted events), and those near-zero-variance
# cells dominated every reco-space chi2 diagnostic when split.
_UNGROOMED_RECO_COORDINATE_EDGES = (
    -10.0, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0,
    -0.75, -0.5, 0.0,
)
_UNGROOMED_BASE_TO_GEN_GROUPS = (
    (0,), (1, 2), (3, 4), (5, 6), (7, 8), (9,),
)


def _candidate_binning(
    channel: str,
    variant: str,
    pt_groups: tuple[tuple[int, ...], ...],
    two_log10_rho_groups: tuple[tuple[int, ...], ...],
) -> PairSplitBinning:
    fine_pt_edges = PAIR_SPLIT_FINE_AXES["groomed"].pt_edges
    output_pt_edges = tuple(
        fine_pt_edges[group[0]] for group in pt_groups
    ) + (fine_pt_edges[pt_groups[-1][-1] + 1],)
    return PairSplitBinning(
        channel=channel,
        variant=variant,
        grooming_mode="groomed",
        pt_edges=output_pt_edges,
        pt_groups=pt_groups,
        base_reco_two_log10_rho_edges=_GROOMED_CANDIDATE_COORDINATE_EDGES_BY_CHANNEL[channel],
        base_to_gen_two_log10_rho_groups=two_log10_rho_groups,
        sink_pt_source_bin_indices=(0,),
        first_reported_pt_index=0,
        reported_two_log10_rho_minimum=-3.5,
    )


def _ungroomed_candidate_binning(
    groomed_candidate: PairSplitBinning,
) -> PairSplitBinning:
    """Reuse only a candidate's pT aggregation for the ungroomed observable."""

    return PairSplitBinning(
        channel=groomed_candidate.channel,
        variant=groomed_candidate.variant,
        grooming_mode="ungroomed",
        pt_edges=groomed_candidate.pt_edges,
        pt_groups=groomed_candidate.pt_groups,
        base_reco_two_log10_rho_edges=_UNGROOMED_RECO_COORDINATE_EDGES,
        base_to_gen_two_log10_rho_groups=_UNGROOMED_BASE_TO_GEN_GROUPS,
        sink_pt_source_bin_indices=groomed_candidate.sink_pt_source_bin_indices,
        first_reported_pt_index=groomed_candidate.first_reported_pt_index,
        reported_two_log10_rho_minimum=-2.5,
    )


# These variants document candidate aggregation choices; no physics-status
# claim is encoded in their names or metadata.  Pre-2026-08-27 variants
# (coarse_tail / two_to_one / window_aligned*) lived on the old free-edge
# producer lattice and are gone with it — git history has them.
PAIR_SPLIT_BINNING_VARIANTS = {
    # The APPROVED aligned grids (2026-08-27): dijet = study "aligned_seven"
    # [-3.5,-2.5,-2,-1.5,-1.25,-1,-0.75,0], trijet = "wide_tail_deep"
    # [-3.5,-2.5,-2,-1.5,-1,0].  Every dijet edge merges onto the common
    # grid, so combined plots rebin exactly.
    "aligned": {
        "dijet": _candidate_binning(
            "dijet",
            "aligned",
            ((1,), (2,), (3,), (4,), (5, 6, 7, 8)),
            ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13,)),
        ),
        "trijet": _candidate_binning(
            "trijet",
            "aligned",
            ((1,), (2,), (3, 4, 5, 6, 7, 8)),
            ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9,)),
        ),
    },
    # Cross-check: dijet regrouped onto the common three-channel grid
    # [-3.5,-2.5,-2,-1.5,-1,0] (trijet identical to "aligned").
    "aligned_common": {
        "dijet": _candidate_binning(
            "dijet",
            "aligned_common",
            ((1,), (2,), (3,), (4,), (5, 6, 7, 8)),
            ((0,), (1, 2), (3, 4), (5, 6), (7, 8, 9, 10), (11, 12, 13)),
        ),
        "trijet": _candidate_binning(
            "trijet",
            "aligned_common",
            ((1,), (2,), (3, 4, 5, 6, 7, 8)),
            ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9,)),
        ),
    },
}


@dataclass(frozen=True)
class PairSplitSourceFiles:
    channel: str
    era: str
    mc: Path
    data: Path


@dataclass(frozen=True)
class PairSplitModeArrays:
    """Fine-bin, dataset- and era-summed arrays for one grooming mode.

    The response order is ``(ptreco, two_log10_rho_reco, ptgen,
    two_log10_rho_gen)``.  Reco and gen marginals have their matching two-axis
    order.  The covariance has paired reco axes in that same order.
    """

    grooming_mode: str
    fine_axes: PairSplitFineAxes
    systematics: tuple[str, ...]
    response_by_systematic: Mapping[str, np.ndarray]
    response_variance_by_systematic: Mapping[str, np.ndarray]
    reco_by_systematic: Mapping[str, np.ndarray]
    reco_variance_by_systematic: Mapping[str, np.ndarray]
    gen_by_systematic: Mapping[str, np.ndarray]
    gen_variance_by_systematic: Mapping[str, np.ndarray]
    nominal_data: np.ndarray
    nominal_data_variance: np.ndarray
    nominal_data_covariance: np.ndarray


@dataclass(frozen=True)
class PairSplitRun2Inputs:
    """One channel's validated Run-2 inputs, loaded without runner coupling."""

    channel: str
    eras: tuple[str, ...]
    modes: Mapping[str, PairSplitModeArrays]
    source_files: tuple[PairSplitSourceFiles, ...]
    observable_metadata: Mapping[str, object]


@dataclass(frozen=True)
class PairSplitPreparedBinning:
    """Binning attributes consumed by ``Unfolder.from_prepared_inputs``.

    The attribute names deliberately preserve the physical distinction between
    rho and its stored coordinate.  The caller's ``ObservableSpec`` selects
    these names through its prepared-binning accessor fields.
    """

    pt_edges: tuple[float, ...]
    two_log10_rho_reco_edges: tuple[float, ...]
    two_log10_rho_gen_edges: tuple[float, ...]
    reco_two_log10_rho_edges_by_pt: tuple[tuple[float, ...], ...]
    gen_two_log10_rho_edges_by_pt: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PairSplitPreparedInputs:
    """Rebinned histograms and covariance for the maintained core API."""

    channel: str
    candidate: PairSplitBinning
    analysis_binning: PairSplitPreparedBinning
    mc_inputs: Mapping[str, hist.Hist]
    data_inputs: Mapping[str, hist.Hist]
    systematics: tuple[str, ...]
    measured_covariance: np.ndarray
    first_reported_pt_bin: int
    metadata: Mapping[str, object]


def _linear_group_map(
    n_source_bins: int,
    groups: Sequence[Sequence[int]],
    *,
    context: str,
    require_full_coverage: bool,
) -> np.ndarray:
    """Return the exact sum map from source bins to grouped bins."""

    matrix = np.zeros((len(groups), n_source_bins), dtype=float)
    used: set[int] = set()
    for output_bin, raw_group in enumerate(groups):
        group = tuple(int(index) for index in raw_group)
        if not group:
            raise ValueError(f"{context}: empty source-bin group")
        if tuple(sorted(group)) != group or any(
            right != left + 1 for left, right in zip(group, group[1:])
        ):
            raise ValueError(f"{context}: source-bin groups must be ordered and contiguous")
        for source_bin in group:
            if not 0 <= source_bin < n_source_bins:
                raise ValueError(f"{context}: source-bin index {source_bin} is out of range")
            if source_bin in used:
                raise ValueError(f"{context}: source-bin index {source_bin} appears twice")
            used.add(source_bin)
            matrix[output_bin, source_bin] = 1.0
    if require_full_coverage and used != set(range(n_source_bins)):
        raise ValueError(f"{context}: grouped bins do not cover every source bin")
    return matrix


def _groups_for_target_edges(
    source_edges: Sequence[float],
    target_edges: Sequence[float],
    *,
    context: str,
) -> tuple[tuple[int, ...], ...]:
    """Map a nested numeric edge set to contiguous source-bin groups."""

    source_edges = np.asarray(source_edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)
    if target_edges.ndim != 1 or target_edges.size < 2:
        raise ValueError(f"{context}: at least two target edges are required")
    source_indices: list[int] = []
    for edge in target_edges:
        matches = np.flatnonzero(np.isclose(source_edges, edge, rtol=0.0, atol=1e-12))
        if matches.size != 1:
            raise ValueError(f"{context}: target edge {edge} is absent from the fine axis")
        source_indices.append(int(matches[0]))
    if source_indices[0] != 0 or source_indices[-1] != len(source_edges) - 1:
        raise ValueError(f"{context}: target edges must cover the complete fine axis")
    return tuple(
        tuple(range(first, last))
        for first, last in zip(source_indices, source_indices[1:])
    )


def _grouped_edges(
    source_edges: Sequence[float],
    groups: Sequence[Sequence[int]],
    *,
    context: str,
) -> tuple[float, ...]:
    """Return the numeric edges implied by a full, contiguous group partition."""

    source_edges = tuple(float(edge) for edge in source_edges)
    _linear_group_map(
        len(source_edges) - 1,
        groups,
        context=context,
        require_full_coverage=True,
    )
    group_tuples = tuple(tuple(int(index) for index in group) for group in groups)
    if group_tuples[0][0] != 0 or group_tuples[-1][-1] != len(source_edges) - 2:
        raise ValueError(f"{context}: groups must span the full source edge range")
    return tuple(source_edges[group[0]] for group in group_tuples) + (
        source_edges[group_tuples[-1][-1] + 1],
    )


def _rebin_2d_values(
    values: np.ndarray,
    pt_map: np.ndarray,
    coordinate_map: np.ndarray,
) -> np.ndarray:
    return np.einsum("ai,bj,ij->ab", pt_map, coordinate_map, values, optimize=True)


def _rebin_response_values(
    values: np.ndarray,
    reco_pt_map: np.ndarray,
    reco_coordinate_map: np.ndarray,
    gen_pt_map: np.ndarray,
    gen_coordinate_map: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "ai,bj,ck,dl,ijkl->abcd",
        reco_pt_map,
        reco_coordinate_map,
        gen_pt_map,
        gen_coordinate_map,
        values,
        optimize=True,
    )


def _validate_fake_miss_variances(
    reco_variances: np.ndarray,
    gen_variances: np.ndarray,
    response_variances: np.ndarray,
    *,
    systematic: str,
) -> None:
    """Fail before TUnfold when rebinned marginal sumw2 is inconsistent."""

    fake_variances = reco_variances - response_variances.sum(axis=(2, 3))
    miss_variances = gen_variances - response_variances.sum(axis=(0, 1))
    scale = max(
        1.0,
        float(np.max(np.abs(reco_variances))),
        float(np.max(np.abs(gen_variances))),
        float(np.max(np.abs(response_variances))),
    )
    tolerance = 1e-10 * scale
    if np.min(fake_variances) < -tolerance:
        raise ValueError(
            f"{systematic}: rebinned fake variance is negative beyond tolerance "
            f"({np.min(fake_variances):.6g})"
        )
    if np.min(miss_variances) < -tolerance:
        raise ValueError(
            f"{systematic}: rebinned miss variance is negative beyond tolerance "
            f"({np.min(miss_variances):.6g})"
        )


def _histogram_from_systematic_arrays(
    systematic_names: Sequence[str],
    axes: Sequence[object],
    values_by_systematic: Mapping[str, np.ndarray],
    variances_by_systematic: Mapping[str, np.ndarray],
) -> hist.Hist:
    output = hist.Hist(
        hist.axis.StrCategory(list(systematic_names), name="systematic"),
        *axes,
        storage=hist.storage.Weight(),
    )
    view = output.view(flow=False)
    for index, systematic in enumerate(systematic_names):
        view.value[index, ...] = values_by_systematic[systematic]
        view.variance[index, ...] = variances_by_systematic[systematic]
    return output


def _prepared_histogram_axes(
    binning: PairSplitPreparedBinning,
) -> dict[str, tuple[object, ...]]:
    return {
        "response": (
            hist.axis.Variable(binning.pt_edges, name="ptreco"),
            hist.axis.Variable(binning.two_log10_rho_reco_edges, name="mpt_reco"),
            hist.axis.Variable(binning.pt_edges, name="ptgen"),
            hist.axis.Variable(binning.two_log10_rho_gen_edges, name="mpt_gen"),
        ),
        "reco": (
            hist.axis.Variable(binning.pt_edges, name="ptreco"),
            hist.axis.Variable(binning.two_log10_rho_reco_edges, name="mpt_reco"),
        ),
        "gen": (
            hist.axis.Variable(binning.pt_edges, name="ptgen"),
            hist.axis.Variable(binning.two_log10_rho_gen_edges, name="mpt_gen"),
        ),
    }


def prepare_pairsplit_inputs(
    inputs: PairSplitRun2Inputs,
    variant: str,
    systematics: Sequence[str] | None = None,
    *,
    grooming_mode: str = "groomed",
    model_variations: Mapping[str, np.ndarray] | None = None,
    model_metadata: Mapping[str, object] | None = None,
) -> PairSplitPreparedInputs:
    """Bridge one pair-split grooming mode into ``Unfolder.from_prepared_inputs``.

    Reco quantities use the candidate's base coordinate bins.  Candidate
    ``coarse_tail``/``two_to_one`` groups are applied only to GEN bins.  The
    source 185--200 GeV pT bin is not mapped into the active pT space.
    """

    candidate = pair_split_binning(inputs.channel, variant, grooming_mode=grooming_mode)
    source = inputs.modes[grooming_mode]
    requested_systematics = tuple(source.systematics if systematics is None else systematics)
    if not requested_systematics or requested_systematics[0] != "nominal":
        raise ValueError("Prepared pair-split systematics must start with 'nominal'")
    if len(set(requested_systematics)) != len(requested_systematics):
        raise ValueError("Prepared pair-split systematics contain duplicates")
    unavailable = [name for name in requested_systematics if name not in source.systematics]
    if unavailable:
        raise ValueError(f"Requested pair-split systematics are unavailable: {unavailable}")
    excluded = [
        name for name in requested_systematics
        if name in PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS
    ]
    if excluded:
        details = "; ".join(
            f"{name}: {PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS[name]}"
            for name in excluded
        )
        raise ValueError(f"Requested pair-split systematic is excluded: {details}")

    model_variations = {
        str(name): np.asarray(weight, dtype=float)
        for name, weight in (model_variations or {}).items()
    }
    overlap = set(requested_systematics) & set(model_variations)
    if overlap:
        raise ValueError(
            f"Prepared model variations collide with producer systematics: {sorted(overlap)}"
        )
    expected_model_shape = (
        len(source.fine_axes.pt_edges) - 1,
        len(source.fine_axes.two_log10_rho_gen_edges) - 1,
    )
    for name, weight in model_variations.items():
        if weight.shape != expected_model_shape:
            raise ValueError(
                f"Prepared model variation {name!r} has shape {weight.shape}; "
                f"expected {expected_model_shape}"
            )
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError(
                f"Prepared model variation {name!r} contains non-finite or nonpositive weights"
            )
    prepared_systematics = requested_systematics + tuple(model_variations)

    fine_axes = source.fine_axes
    pt_map = _linear_group_map(
        len(fine_axes.pt_edges) - 1,
        candidate.pt_groups,
        context="pair-split pT groups",
        require_full_coverage=False,
    )
    expected_sink = set(range(len(fine_axes.pt_edges) - 1)) - {
        source_bin
        for group in candidate.pt_groups
        for source_bin in group
    }
    if expected_sink != set(candidate.sink_pt_source_bin_indices):
        raise ValueError(
            "Pair-split pT groups do not match the declared source sink-bin behavior"
        )
    if _grouped_edges(
        fine_axes.pt_edges,
        ((0,),) + candidate.pt_groups,
        context="pair-split pT group edge contract",
    )[1:] != candidate.pt_edges:
        raise ValueError("Pair-split candidate pT edges do not match its pT groups")

    reco_coordinate_groups = _groups_for_target_edges(
        fine_axes.two_log10_rho_reco_edges,
        candidate.base_reco_two_log10_rho_edges,
        context="reco two_log10_rho base bins",
    )
    reco_coordinate_map = _linear_group_map(
        len(fine_axes.two_log10_rho_reco_edges) - 1,
        reco_coordinate_groups,
        context="reco two_log10_rho base bins",
        require_full_coverage=True,
    )
    final_gen_edges = candidate.gen_two_log10_rho_edges
    # Map the fine GEN axis directly onto the candidate's final GEN edges.
    # The former composition through the base RECO edges required every base
    # reco edge to exist on the fine GEN axis — a requirement a
    # finer-than-gen reco binning (the 2:1 ungroomed candidate) deliberately
    # violates.
    gen_coordinate_groups = _groups_for_target_edges(
        fine_axes.two_log10_rho_gen_edges,
        final_gen_edges,
        context="gen two_log10_rho bins",
    )
    gen_coordinate_map = _linear_group_map(
        len(fine_axes.two_log10_rho_gen_edges) - 1,
        gen_coordinate_groups,
        context="gen two_log10_rho bins",
        require_full_coverage=True,
    )
    analysis_binning = PairSplitPreparedBinning(
        pt_edges=candidate.pt_edges,
        two_log10_rho_reco_edges=candidate.base_reco_two_log10_rho_edges,
        two_log10_rho_gen_edges=final_gen_edges,
        reco_two_log10_rho_edges_by_pt=(candidate.base_reco_two_log10_rho_edges,) * len(candidate.pt_groups),
        gen_two_log10_rho_edges_by_pt=(final_gen_edges,) * len(candidate.pt_groups),
    )

    response_values: dict[str, np.ndarray] = {}
    response_variances: dict[str, np.ndarray] = {}
    reco_values: dict[str, np.ndarray] = {}
    reco_variances: dict[str, np.ndarray] = {}
    gen_values: dict[str, np.ndarray] = {}
    gen_variances: dict[str, np.ndarray] = {}
    model_fine_arrays: dict[str, tuple[np.ndarray, ...]] = {}
    if model_variations:
        nominal_response = np.asarray(
            source.response_by_systematic["nominal"], dtype=float
        )
        nominal_response_variance = np.asarray(
            source.response_variance_by_systematic["nominal"], dtype=float
        )
        nominal_reco = np.asarray(source.reco_by_systematic["nominal"], dtype=float)
        nominal_reco_variance = np.asarray(
            source.reco_variance_by_systematic["nominal"], dtype=float
        )
        nominal_gen = np.asarray(source.gen_by_systematic["nominal"], dtype=float)
        nominal_gen_variance = np.asarray(
            source.gen_variance_by_systematic["nominal"], dtype=float
        )
        nominal_matched_reco = nominal_response.sum(axis=(2, 3))
        nominal_matched_reco_variance = nominal_response_variance.sum(axis=(2, 3))
        nominal_fakes = nominal_reco - nominal_matched_reco
        nominal_fake_variance = nominal_reco_variance - nominal_matched_reco_variance
        difference_scale = max(
            1.0,
            float(np.max(np.abs(nominal_reco))),
            float(np.max(np.abs(nominal_reco_variance))),
        )
        difference_tolerance = 1.0e-10 * difference_scale
        if np.min(nominal_fakes) < -difference_tolerance:
            raise ValueError(
                "Nominal fine-bin fake yield is negative beyond tolerance before model reweighting"
            )
        if np.min(nominal_fake_variance) < -difference_tolerance:
            raise ValueError(
                "Nominal fine-bin fake variance is negative beyond tolerance before model reweighting"
            )
        nominal_fakes = np.clip(nominal_fakes, 0.0, None)
        nominal_fake_variance = np.clip(nominal_fake_variance, 0.0, None)
        for systematic, weight in model_variations.items():
            response_weight = weight[None, None, :, :]
            varied_response = nominal_response * response_weight
            varied_response_variance = nominal_response_variance * response_weight**2
            varied_gen = nominal_gen * weight
            varied_gen_variance = nominal_gen_variance * weight**2
            # Follow the Z+jet column-scaled-response treatment: the absolute
            # unmatched-reco contribution stays nominal while the matched
            # projection and inclusive GEN marginal receive the gen weight.
            varied_reco = nominal_fakes + varied_response.sum(axis=(2, 3))
            varied_reco_variance = (
                nominal_fake_variance + varied_response_variance.sum(axis=(2, 3))
            )
            model_fine_arrays[systematic] = (
                varied_response,
                varied_response_variance,
                varied_reco,
                varied_reco_variance,
                varied_gen,
                varied_gen_variance,
            )

    for systematic in prepared_systematics:
        if systematic in model_fine_arrays:
            (
                fine_response,
                fine_response_variance,
                fine_reco,
                fine_reco_variance,
                fine_gen,
                fine_gen_variance,
            ) = model_fine_arrays[systematic]
        else:
            fine_response = source.response_by_systematic[systematic]
            fine_response_variance = source.response_variance_by_systematic[systematic]
            fine_reco = source.reco_by_systematic[systematic]
            fine_reco_variance = source.reco_variance_by_systematic[systematic]
            fine_gen = source.gen_by_systematic[systematic]
            fine_gen_variance = source.gen_variance_by_systematic[systematic]
        response_values[systematic] = _rebin_response_values(
            fine_response,
            pt_map,
            reco_coordinate_map,
            pt_map,
            gen_coordinate_map,
        )
        response_variances[systematic] = _rebin_response_values(
            fine_response_variance,
            pt_map,
            reco_coordinate_map,
            pt_map,
            gen_coordinate_map,
        )
        reco_values[systematic] = _rebin_2d_values(
            fine_reco, pt_map, reco_coordinate_map
        )
        reco_variances[systematic] = _rebin_2d_values(
            fine_reco_variance, pt_map, reco_coordinate_map
        )
        gen_values[systematic] = _rebin_2d_values(
            fine_gen,
            pt_map,
            gen_coordinate_map,
        )
        gen_variances[systematic] = _rebin_2d_values(
            fine_gen_variance,
            pt_map,
            gen_coordinate_map,
        )
        if systematic in PAIR_SPLIT_NOMINAL_VARIANCE_SYSTEMATICS:
            # The non-nominal arrays still define this nuisance's central
            # response/fake/miss shift.  Only its unobservable cross-category
            # MC-stat covariance is fixed to the nominal Run-2 estimate.
            response_variances[systematic] = response_variances["nominal"].copy()
            reco_variances[systematic] = reco_variances["nominal"].copy()
            gen_variances[systematic] = gen_variances["nominal"].copy()
        _validate_fake_miss_variances(
            reco_variances[systematic],
            gen_variances[systematic],
            response_variances[systematic],
            systematic=systematic,
        )

    data_values = _rebin_2d_values(
        source.nominal_data, pt_map, reco_coordinate_map
    )
    data_variances = _rebin_2d_values(
        source.nominal_data_variance, pt_map, reco_coordinate_map
    )
    data_covariance = rebin_reco_covariance(
        source.nominal_data_covariance,
        len(fine_axes.pt_edges) - 1,
        len(fine_axes.two_log10_rho_reco_edges) - 1,
        candidate.pt_groups,
        reco_coordinate_groups,
    )
    flat_data_covariance = data_covariance.reshape(data_values.size, data_values.size)
    axes = _prepared_histogram_axes(analysis_binning)
    keys = LEGACY_HISTOGRAM_KEYS[grooming_mode]
    mc_inputs = {
        keys["response"]: _histogram_from_systematic_arrays(
            prepared_systematics, axes["response"], response_values, response_variances
        ),
        keys["reco"]: _histogram_from_systematic_arrays(
            prepared_systematics, axes["reco"], reco_values, reco_variances
        ),
        keys["gen"]: _histogram_from_systematic_arrays(
            prepared_systematics, axes["gen"], gen_values, gen_variances
        ),
    }
    data_inputs = {
        keys["reco"]: _histogram_from_systematic_arrays(
            ("nominal",),
            axes["reco"],
            {"nominal": data_values},
            {"nominal": data_variances},
        )
    }
    return PairSplitPreparedInputs(
        channel=inputs.channel,
        candidate=candidate,
        analysis_binning=analysis_binning,
        mc_inputs=mc_inputs,
        data_inputs=data_inputs,
        systematics=prepared_systematics,
        measured_covariance=flat_data_covariance,
        first_reported_pt_bin=candidate.first_reported_pt_index,
        metadata={
            "rho_definition": PHYSICAL_RHO_DEFINITION,
            "transformed_coordinate_name": TRANSFORMED_COORDINATE_NAME,
            "transformed_coordinate_definition": TRANSFORMED_COORDINATE_DEFINITION,
            "grooming_mode": grooming_mode,
            "candidate": candidate.variant,
            "source_sink_pt_bin_indices": candidate.sink_pt_source_bin_indices,
            "first_reported_pt_bin": candidate.first_reported_pt_index,
            "reported_two_log10_rho_minimum": candidate.reported_two_log10_rho_minimum,
            "pt_groups": candidate.pt_groups,
            "reco_two_log10_rho_groups": reco_coordinate_groups,
            "fine_to_gen_two_log10_rho_groups": gen_coordinate_groups,
            "base_to_gen_two_log10_rho_groups": candidate.base_to_gen_two_log10_rho_groups,
            "data_covariance_source": inputs.observable_metadata[
                "data_covariance_source_by_mode"
            ][grooming_mode],
            "systematic_exclusions": inputs.observable_metadata.get(
                "systematic_exclusions", dict(PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS)
            ),
            "systematic_variance_policy": {
                "nominal_variance_systematics": list(
                    PAIR_SPLIT_NOMINAL_VARIANCE_SYSTEMATICS
                ),
                "reason": PAIR_SPLIT_NOMINAL_VARIANCE_REASON,
            },
            "model_envelope": dict(model_metadata or {}),
            "run2_era_correlation": inputs.observable_metadata.get(
                "run2_era_correlation",
                {
                    "applied": False,
                    "generated_legs_by_mode": {},
                    "excluded_defective_jes_sources": list(DEFECTIVE_JES_SOURCES),
                },
            ),
        },
    )


def prepare_pairsplit_groomed_inputs(
    inputs: PairSplitRun2Inputs,
    variant: str,
    systematics: Sequence[str] | None = None,
    *,
    model_variations: Mapping[str, np.ndarray] | None = None,
    model_metadata: Mapping[str, object] | None = None,
) -> PairSplitPreparedInputs:
    """Backward-compatible groomed wrapper around :func:`prepare_pairsplit_inputs`."""

    return prepare_pairsplit_inputs(
        inputs,
        variant,
        systematics,
        grooming_mode="groomed",
        model_variations=model_variations,
        model_metadata=model_metadata,
    )


def _is_raw_run2_jes_jer_category(name: str) -> bool:
    """Return whether *name* is a producer-era-combined JES/JER variation."""

    if name in {"JERUp", "JERDown"}:
        return True
    return (
        name.startswith("JES_")
        and name.endswith(("Up", "Down"))
        and "_corr" not in name
        and "_uncorr_" not in name
    )


def _run2_virtual_name(prefix: str, component: str, direction: str, era_group: str | None = None) -> str:
    if component == "corr":
        return f"{prefix}_corr{direction}"
    assert era_group is not None
    return f"{prefix}_uncorr_{era_group}{direction}"


def _add_run2_era_correlation_virtuals(
    accumulator: dict[str, object],
    era_accumulators: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    raw_systematics: Sequence[str],
    eras: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Replace raw combined JES/JER categories by Run-2 virtual variations.

    The caller supplies dataset-summed, but not era-summed, arrays.  For a
    source with correlation coefficient ``rho`` we build a correlated leg as
    ``nominal + sqrt(rho) * sum_group(varied - nominal)`` and one leg per era
    group as ``nominal + sqrt(1-rho) * (group varied - group nominal)``.  The
    virtual response/reco/gen sumw2 are copied from the nominal Run-2 arrays:
    producer inputs do not encode statistical covariances between systematic
    categories, so deriving a new variance would fabricate MC statistics.
    """

    required_eras = {era for members in RUN2_ERA_GROUPS.values() for era in members}
    if set(eras) != required_eras:
        return tuple(raw_systematics), {
            "applied": False,
            "reason": "requires the complete 2016APV, 2016, 2017, 2018 Run-2 era set",
            "generated_legs": [],
        }

    array_roles = ("response", "reco", "gen")
    variance_roles = {
        "response": "response_variance",
        "reco": "reco_variance",
        "gen": "gen_variance",
    }
    raw_set = set(raw_systematics)
    generated_legs: list[str] = []
    generated_leg_details: list[dict[str, object]] = []
    unavailable_sources: list[str] = []

    source_definitions = [("JER", JER_RUN2_CORRELATION)] + [
        (f"JES_{source}", rho)
        for source, rho in JES_RUN2_CORRELATIONS.items()
        if source not in DEFECTIVE_JES_SOURCES
    ]
    for prefix, rho in source_definitions:
        raw_variations = {f"{prefix}Up", f"{prefix}Down"}
        if not raw_variations <= raw_set:
            unavailable_sources.append(prefix)
            continue
        for direction in ("Up", "Down"):
            group_deltas: dict[str, dict[str, np.ndarray]] = {}
            for group_name, group_eras in RUN2_ERA_GROUPS.items():
                group_deltas[group_name] = {}
                for role in array_roles:
                    varied = sum(
                        era_accumulators[era][role][f"{prefix}{direction}"]
                        for era in group_eras
                    )
                    nominal = sum(
                        era_accumulators[era][role]["nominal"]
                        for era in group_eras
                    )
                    group_deltas[group_name][role] = varied - nominal

            virtual_specs: list[tuple[str, float, str | None]] = []
            if rho > 0.0:
                virtual_specs.append(
                    ("corr", float(np.sqrt(rho)), None)
                )
            if rho < 1.0:
                for group_name in RUN2_ERA_GROUPS:
                    virtual_specs.append(
                        ("uncorr", float(np.sqrt(1.0 - rho)), group_name)
                    )

            for component, factor, group_name in virtual_specs:
                name = _run2_virtual_name(prefix, component, direction, group_name)
                for role in array_roles:
                    if component == "corr":
                        delta = sum(
                            group_deltas[existing_group][role]
                            for existing_group in RUN2_ERA_GROUPS
                        )
                    else:
                        assert group_name is not None
                        delta = group_deltas[group_name][role]
                    accumulator[role][name] = accumulator[role]["nominal"] + factor * delta
                    accumulator[variance_roles[role]][name] = accumulator[
                        variance_roles[role]
                    ]["nominal"].copy()
                generated_legs.append(name)
                generated_leg_details.append(
                    {
                        "name": name,
                        "source": prefix,
                        "direction": direction,
                        "component": component,
                        "era_group": group_name,
                        "rho": float(rho),
                        "coefficient": factor,
                    }
                )

    retained = [
        systematic
        for systematic in raw_systematics
        if not _is_raw_run2_jes_jer_category(systematic)
    ]
    return tuple(retained + generated_legs), {
        "applied": True,
        "era_groups": {name: list(group) for name, group in RUN2_ERA_GROUPS.items()},
        "correlation_coefficients": {
            "JER": JER_RUN2_CORRELATION,
            **{f"JES_{source}": rho for source, rho in JES_RUN2_CORRELATIONS.items()},
        },
        "prescription": (
            "corr=sqrt(rho) and uncorr=sqrt(1-rho), applied to "
            "Run-2 nominal plus grouped varied-minus-nominal shifts"
        ),
        "generated_legs": generated_legs,
        "generated_leg_details": generated_leg_details,
        "unavailable_safe_sources": unavailable_sources,
        "excluded_defective_jes_sources": [f"JES_{source}" for source in DEFECTIVE_JES_SOURCES],
        "virtual_variance": (
            "nominal Run-2 sumw2 copied for every virtual response/reco/gen leg; "
            "systematic-category statistical covariance is unavailable"
        ),
        "raw_combined_categories_removed": True,
    }


def resolve_pairsplit_systematics(
    available_systematics: Sequence[str],
    requested: str | Sequence[str],
) -> tuple[str, ...]:
    """Resolve safe bases to exact nominal or derived Run-2 variation legs."""

    raw_requested = (
        requested.split(",") if isinstance(requested, str) else list(requested)
    )
    requested_names = [name.strip() for name in raw_requested if name.strip()]
    if not requested_names:
        raise ValueError("At least one systematic request is required")
    available = tuple(str(name) for name in available_systematics)
    resolved: list[str] = []
    for request in requested_names:
        lower_request = request.lower()
        if _is_raw_run2_jes_jer_category(request):
            raise ValueError(
                f"Raw combined Run-2 category {request!r} is not selectable; "
                "request JER, JES, or an exact derived JES source instead"
            )
        if lower_request == "nominal":
            matches = [name for name in available if name == "nominal"]
        elif lower_request == "all_safe_non_jes":
            matches = [
                name
                for name in available
                if name != "nominal"
                and not name.startswith("JES_")
                and not name.startswith("JER_")
                and not _is_raw_run2_jes_jer_category(name)
                and name not in PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS
            ]
        elif lower_request == "jes":
            matches = [
                name for name in available
                if name.startswith("JES_")
                and ("_corr" in name or "_uncorr_" in name)
            ]
        elif lower_request == "jer":
            matches = [
                name for name in available
                if name.startswith("JER_")
                and ("_corr" in name or "_uncorr_" in name)
            ]
        elif request in available:
            matches = [request]
        elif request.startswith("JES_"):
            matches = [
                name
                for name in available
                if name.startswith(f"{request}_corr")
                or name.startswith(f"{request}_uncorr_")
            ]
        else:
            matches = [
                name
                for name in available
                if name.lower() in {f"{lower_request}up", f"{lower_request}down"}
            ]
        if not matches:
            raise ValueError(
                f"No available pair-split systematic matches {request!r}; available={available}"
            )
        excluded_matches = [
            name for name in matches if name in PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS
        ]
        if excluded_matches:
            details = "; ".join(
                f"{name}: {PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS[name]}"
                for name in excluded_matches
            )
            raise ValueError(f"Requested pair-split systematic is excluded: {details}")
        for name in matches:
            if name not in resolved:
                resolved.append(name)
    if "nominal" not in resolved:
        resolved.insert(0, "nominal")
    elif resolved[0] != "nominal":
        resolved.remove("nominal")
        resolved.insert(0, "nominal")
    return tuple(resolved)


def pair_split_binning(
    channel: str,
    variant: str,
    *,
    grooming_mode: str = "groomed",
) -> PairSplitBinning:
    """Return an explicitly named candidate binning for one grooming mode."""

    if channel not in PAIR_SPLIT_CHANNELS:
        raise ValueError(f"Unsupported channel {channel!r}; choose from {PAIR_SPLIT_CHANNELS}")
    try:
        candidate = PAIR_SPLIT_BINNING_VARIANTS[variant][channel]
    except KeyError as error:
        available = tuple(PAIR_SPLIT_BINNING_VARIANTS)
        raise ValueError(f"Unsupported binning variant {variant!r}; choose from {available}") from error
    if grooming_mode == "groomed":
        return candidate
    if grooming_mode == "ungroomed":
        return _ungroomed_candidate_binning(candidate)
    raise ValueError("grooming_mode must be 'groomed' or 'ungroomed'")


def discover_pairsplit_run2_files(
    channel: str,
    era: str | int,
    input_root: str | Path = PAIR_SPLIT_INPUT_ROOT,
) -> PairSplitSourceFiles:
    """Discover exactly one nested data and MC all-systematics pickle.

    The producer layout has a channel directory below each era, but discovery
    remains nested to tolerate a non-physics directory level.  The archived
    ``2018_lhe_basis`` tree is explicitly excluded rather than searched.
    """

    if channel not in PAIR_SPLIT_CHANNELS:
        raise ValueError(f"Unsupported channel {channel!r}; choose from {PAIR_SPLIT_CHANNELS}")
    era = str(era)
    era_dir = Path(input_root) / era
    if not era_dir.is_dir():
        raise FileNotFoundError(f"Missing pair-split era directory: {era_dir}")

    def discover_one(kind: str, filename: str) -> Path:
        pattern = f"**/{channel}_{kind}/**/{filename}"
        matches = [
            path for path in era_dir.glob(pattern)
            if "2018_lhe_basis" not in path.relative_to(era_dir).parts
        ]
        if len(matches) != 1:
            listed = "\n".join(f"  {path}" for path in sorted(matches)) or "  (none)"
            raise ValueError(
                f"Expected exactly one {kind} pair-split pickle for {channel} {era} "
                f"using {pattern!r}; found {len(matches)}:\n{listed}"
            )
        return matches[0]

    return PairSplitSourceFiles(
        channel=channel,
        era=era,
        mc=discover_one("mc", f"minimal_rho_{channel}_mg_pythia8_{era}.pkl"),
        data=discover_one("data", f"minimal_rho_{channel}_data_{era}.pkl"),
    )


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _axis_categories(histogram, axis_name: str, context: str) -> tuple[str, ...]:
    if axis_name not in histogram.axes.name:
        raise ValueError(f"{context}: missing required {axis_name!r} category axis")
    return tuple(str(category) for category in histogram.axes[axis_name])


def _axis_edges(histogram, axis_name: str, context: str) -> tuple[float, ...]:
    if axis_name not in histogram.axes.name:
        raise ValueError(f"{context}: missing required {axis_name!r} physics axis")
    axis = histogram.axes[axis_name]
    if not hasattr(axis, "edges"):
        raise ValueError(f"{context}: axis {axis_name!r} is not a numeric physics axis")
    return tuple(float(edge) for edge in axis.edges)


def _validate_edges(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    context: str,
    axis_name: str,
) -> None:
    if len(actual) != len(expected) or not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{context}: physics-axis mismatch for {axis_name!r}; "
            f"expected {expected}, found {actual}"
        )


def _validate_payload_keys(
    payload: Mapping[str, object],
    mode: str,
    source: str,
    roles: Sequence[str],
) -> None:
    missing = [LEGACY_HISTOGRAM_KEYS[mode][role] for role in roles if LEGACY_HISTOGRAM_KEYS[mode][role] not in payload]
    if missing:
        raise KeyError(f"{source}: missing histograms for {mode}: {missing}")


def _validate_histogram_axes(
    histogram,
    mode: str,
    role: str,
    context: str,
) -> None:
    fine_axes = PAIR_SPLIT_FINE_AXES[mode]
    expected = {
        "ptreco": fine_axes.pt_edges,
        "ptgen": fine_axes.pt_edges,
        "mpt_reco": fine_axes.two_log10_rho_reco_edges,
        "mpt_gen": fine_axes.two_log10_rho_gen_edges,
        "ptreco_i": fine_axes.pt_edges,
        "ptreco_j": fine_axes.pt_edges,
        "mpt_reco_i": fine_axes.two_log10_rho_reco_edges,
        "mpt_reco_j": fine_axes.two_log10_rho_reco_edges,
    }
    for axis_name in _ARRAY_AXES[role]:
        _validate_edges(
            _axis_edges(histogram, axis_name, context),
            expected[axis_name],
            context,
            axis_name,
        )


def _dataset_summed(histogram, context: str):
    _axis_categories(histogram, "dataset", context)
    return histogram[{"dataset": sum}]


def _values_and_variances(histogram, context: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(histogram.values(flow=False), dtype=float)
    variances = histogram.variances(flow=False)
    if variances is None:
        raise ValueError(f"{context}: histogram has no sumw2/variance storage")
    return values, np.asarray(variances, dtype=float)


def _selected_arrays(histogram, role: str, systematic: str, context: str) -> tuple[np.ndarray, np.ndarray]:
    selected = histogram[{"systematic": systematic}]
    projected = selected.project(*_ARRAY_AXES[role])
    return _values_and_variances(projected, context)


def _selected_covariance(histogram, context: str) -> np.ndarray:
    selected = histogram[{"systematic": "nominal"}]
    projected = selected.project(*_ARRAY_AXES["reco_covariance"])
    return np.asarray(projected.values(flow=False), dtype=float)


def _add_arrays(
    accumulator: dict[str, np.ndarray],
    systematic: str,
    values: np.ndarray,
) -> None:
    if systematic not in accumulator:
        accumulator[systematic] = values.copy()
    else:
        accumulator[systematic] += values


def _validate_source_payloads(
    source: PairSplitSourceFiles,
    mc_payload: Mapping[str, object],
    data_payload: Mapping[str, object],
    mc_systematics_by_mode: dict[str, tuple[str, ...]],
    data_systematics_by_mode: dict[str, tuple[str, ...]],
    data_covariance_source_by_mode: dict[str, str],
) -> None:
    """Validate one era and compare its axes/categories with prior eras."""

    for mode, keys in LEGACY_HISTOGRAM_KEYS.items():
        mc_context = f"{source.mc} ({mode})"
        data_context = f"{source.data} ({mode})"
        _validate_payload_keys(mc_payload, mode, mc_context, ("response", "reco", "gen"))
        data_roles = ("reco", "reco_covariance") if source.channel == "dijet" else ("reco",)
        _validate_payload_keys(data_payload, mode, data_context, data_roles)

        mc_categories: tuple[str, ...] | None = None
        for role in ("response", "reco", "gen"):
            histogram = mc_payload[keys[role]]
            _validate_histogram_axes(histogram, mode, role, mc_context)
            categories = _axis_categories(histogram, "systematic", mc_context)
            if mc_categories is None:
                mc_categories = categories
            elif set(categories) != set(mc_categories):
                raise ValueError(
                    f"{mc_context}: systematic category mismatch between "
                    f"response/reco/gen: expected {mc_categories}, found {categories}"
                )
        assert mc_categories is not None
        if "nominal" not in mc_categories:
            raise ValueError(f"{mc_context}: MC systematic categories lack 'nominal'")

        data_categories: tuple[str, ...] | None = None
        for role in data_roles:
            histogram = data_payload[keys[role]]
            _validate_histogram_axes(histogram, mode, role, data_context)
            categories = _axis_categories(histogram, "systematic", data_context)
            if data_categories is None:
                data_categories = categories
            elif set(categories) != set(data_categories):
                raise ValueError(
                    f"{data_context}: data systematic category mismatch between "
                    f"reco and covariance: expected {data_categories}, found {categories}"
                )
        assert data_categories is not None
        if "nominal" not in data_categories:
            raise ValueError(f"{data_context}: data systematic categories lack 'nominal'")

        covariance_source = (
            "full_reco_covariance"
            if keys["reco_covariance"] in data_payload
            else "diagonal_reco_sumw2"
        )
        if source.channel == "dijet" and covariance_source != "full_reco_covariance":
            raise AssertionError("Dijet preflight must require an event-clustered reco covariance")
        if covariance_source == "full_reco_covariance" and "reco_covariance" not in data_roles:
            covariance_histogram = data_payload[keys["reco_covariance"]]
            _validate_histogram_axes(covariance_histogram, mode, "reco_covariance", data_context)
            covariance_categories = _axis_categories(covariance_histogram, "systematic", data_context)
            if set(covariance_categories) != set(data_categories):
                raise ValueError(
                    f"{data_context}: data systematic category mismatch between "
                    f"reco and covariance: expected {data_categories}, found {covariance_categories}"
                )

        if mode not in mc_systematics_by_mode:
            mc_systematics_by_mode[mode] = mc_categories
        elif set(mc_categories) != set(mc_systematics_by_mode[mode]):
            raise ValueError(
                f"{mc_context}: systematic category mismatch across eras: "
                f"expected {mc_systematics_by_mode[mode]}, found {mc_categories}"
            )
        if mode not in data_systematics_by_mode:
            data_systematics_by_mode[mode] = data_categories
        elif set(data_categories) != set(data_systematics_by_mode[mode]):
            raise ValueError(
                f"{data_context}: data systematic category mismatch across eras: "
                f"expected {data_systematics_by_mode[mode]}, found {data_categories}"
            )
        if mode not in data_covariance_source_by_mode:
            data_covariance_source_by_mode[mode] = covariance_source
        elif covariance_source != data_covariance_source_by_mode[mode]:
            raise ValueError(
                f"{data_context}: mixed data covariance sources across eras: "
                f"expected {data_covariance_source_by_mode[mode]!r}, found {covariance_source!r}"
            )


def load_pairsplit_run2_inputs(
    channel: str,
    eras: Sequence[str | int] = PAIR_SPLIT_ERAS,
    input_root: str | Path = PAIR_SPLIT_INPUT_ROOT,
) -> PairSplitRun2Inputs:
    """Load one channel, validating all eras before retaining summed arrays.

    Each pickle is opened and reduced one era at a time.  Dataset components
    are summed before cross-era values and sumw2 are added, keeping the memory
    footprint bounded to one channel plus the returned fine arrays.
    """

    if channel not in PAIR_SPLIT_CHANNELS:
        raise ValueError(f"Unsupported channel {channel!r}; choose from {PAIR_SPLIT_CHANNELS}")
    normalized_eras = tuple(str(era) for era in eras)
    if not normalized_eras:
        raise ValueError("At least one era is required")
    if len(set(normalized_eras)) != len(normalized_eras):
        raise ValueError(
            "Each requested era must occur exactly once so source files and eras "
            "remain one-to-one"
        )

    sources = tuple(
        discover_pairsplit_run2_files(channel, era, input_root)
        for era in normalized_eras
    )
    source_eras = tuple(source.era for source in sources)
    assert len(sources) == len(normalized_eras)
    assert source_eras == normalized_eras
    # Do not start an era sum until the complete selected-era schema has passed
    # the axis and category preflight.  This prevents a partial Run-2 total
    # from being returned or reused after a later-era incompatibility.
    mc_systematics_by_mode: dict[str, tuple[str, ...]] = {}
    data_systematics_by_mode: dict[str, tuple[str, ...]] = {}
    data_covariance_source_by_mode: dict[str, str] = {}
    for source in sources:
        mc_payload = _load_pickle(source.mc)
        data_payload = _load_pickle(source.data)
        _validate_source_payloads(
            source,
            mc_payload,
            data_payload,
            mc_systematics_by_mode,
            data_systematics_by_mode,
            data_covariance_source_by_mode,
        )
        del mc_payload
        del data_payload

    accumulators = {
        mode: {
            "response": {}, "response_variance": {},
            "reco": {}, "reco_variance": {},
            "gen": {}, "gen_variance": {},
            "data": None, "data_variance": None, "data_covariance": None,
        }
        for mode in LEGACY_HISTOGRAM_KEYS
    }
    # Keep the selected-era arrays only until the virtual Run-2 JES/JER legs
    # have been built.  These per-era shifts must be combined before candidate
    # rebinning; the ordinary accumulators remain the returned Run-2 totals.
    era_accumulators: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
        mode: {} for mode in LEGACY_HISTOGRAM_KEYS
    }

    for source in sources:
        mc_payload = _load_pickle(source.mc)
        data_payload = _load_pickle(source.data)
        for mode, keys in LEGACY_HISTOGRAM_KEYS.items():
            mc_context = f"{source.mc} ({mode})"
            data_context = f"{source.data} ({mode})"
            accumulator = accumulators[mode]
            era_accumulator = era_accumulators[mode].setdefault(
                source.era,
                {
                    "response": {}, "response_variance": {},
                    "reco": {}, "reco_variance": {},
                    "gen": {}, "gen_variance": {},
                },
            )
            mc_systematics = mc_systematics_by_mode[mode]

            for role, accumulator_name, variance_name in (
                ("response", "response", "response_variance"),
                ("reco", "reco", "reco_variance"),
                ("gen", "gen", "gen_variance"),
            ):
                dataset_summed = _dataset_summed(mc_payload[keys[role]], mc_context)
                for systematic in mc_systematics:
                    values, variances = _selected_arrays(
                        dataset_summed, role, systematic, mc_context
                    )
                    _add_arrays(accumulator[accumulator_name], systematic, values)
                    _add_arrays(accumulator[variance_name], systematic, variances)
                    era_accumulator[accumulator_name][systematic] = values.copy()
                    era_accumulator[variance_name][systematic] = variances.copy()

            data_reco = _dataset_summed(data_payload[keys["reco"]], data_context)
            data_values, data_variances = _selected_arrays(
                data_reco, "reco", "nominal", data_context
            )
            if data_covariance_source_by_mode[mode] == "full_reco_covariance":
                data_covariance = _dataset_summed(
                    data_payload[keys["reco_covariance"]], data_context
                )
                covariance_values = _selected_covariance(data_covariance, data_context)
            else:
                covariance_values = np.diag(data_variances.reshape(-1)).reshape(
                    data_values.shape + data_values.shape
                )
            for name, values in (
                ("data", data_values),
                ("data_variance", data_variances),
                ("data_covariance", covariance_values),
            ):
                if accumulator[name] is None:
                    accumulator[name] = values.copy()
                else:
                    accumulator[name] += values

        del mc_payload
        del data_payload

    systematics_by_mode: dict[str, tuple[str, ...]] = {}
    correlation_by_mode: dict[str, dict[str, object]] = {}
    for mode, accumulator in accumulators.items():
        systematics, correlation_metadata = _add_run2_era_correlation_virtuals(
            accumulator,
            era_accumulators[mode],
            mc_systematics_by_mode[mode],
            normalized_eras,
        )
        systematics_by_mode[mode] = systematics
        correlation_by_mode[mode] = correlation_metadata

    modes = {
        mode: PairSplitModeArrays(
            grooming_mode=mode,
            fine_axes=PAIR_SPLIT_FINE_AXES[mode],
            systematics=systematics_by_mode[mode],
            response_by_systematic=accumulator["response"],
            response_variance_by_systematic=accumulator["response_variance"],
            reco_by_systematic=accumulator["reco"],
            reco_variance_by_systematic=accumulator["reco_variance"],
            gen_by_systematic=accumulator["gen"],
            gen_variance_by_systematic=accumulator["gen_variance"],
            nominal_data=accumulator["data"],
            nominal_data_variance=accumulator["data_variance"],
            nominal_data_covariance=accumulator["data_covariance"],
        )
        for mode, accumulator in accumulators.items()
    }
    return PairSplitRun2Inputs(
        channel=channel,
        eras=normalized_eras,
        modes=modes,
        source_files=sources,
        observable_metadata={
            "jet_radius": JET_RADIUS,
            "rho_definition": PHYSICAL_RHO_DEFINITION,
            "transformed_coordinate_name": TRANSFORMED_COORDINATE_NAME,
            "transformed_coordinate_definition": TRANSFORMED_COORDINATE_DEFINITION,
            "data_covariance_source_by_mode": dict(data_covariance_source_by_mode),
            "systematic_exclusions": dict(PAIR_SPLIT_SYSTEMATIC_EXCLUSIONS),
            "run2_era_correlation": {
                "applied": all(
                    details["applied"] for details in correlation_by_mode.values()
                ),
                "by_mode": correlation_by_mode,
                "era_groups": {
                    name: list(group) for name, group in RUN2_ERA_GROUPS.items()
                },
                "excluded_defective_jes_sources": [
                    f"JES_{source}" for source in DEFECTIVE_JES_SOURCES
                ],
                "virtual_variance": (
                    "nominal Run-2 sumw2 copied for every virtual response/reco/gen leg; "
                    "systematic-category statistical covariance is unavailable"
                ),
                "generated_legs_by_mode": {
                    mode: details["generated_legs"]
                    for mode, details in correlation_by_mode.items()
                },
            },
        },
    )


def covariance_rebin_matrix(
    n_pt_bins: int,
    n_two_log10_rho_bins: int,
    pt_groups: Sequence[Sequence[int]],
    two_log10_rho_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Build the linear map for summed reco bins in row-major (pT, coordinate) order."""

    source_size = n_pt_bins * n_two_log10_rho_bins
    output_size = len(pt_groups) * len(two_log10_rho_groups)
    matrix = np.zeros((output_size, source_size), dtype=float)
    for output_pt, source_pt_bins in enumerate(pt_groups):
        for output_coordinate, source_coordinate_bins in enumerate(two_log10_rho_groups):
            output_index = output_pt * len(two_log10_rho_groups) + output_coordinate
            for source_pt in source_pt_bins:
                for source_coordinate in source_coordinate_bins:
                    if not 0 <= source_pt < n_pt_bins or not 0 <= source_coordinate < n_two_log10_rho_bins:
                        raise ValueError("Rebin group index is outside the fine reco array")
                    source_index = source_pt * n_two_log10_rho_bins + source_coordinate
                    matrix[output_index, source_index] = 1.0
    return matrix


def rebin_reco_covariance(
    covariance: np.ndarray,
    n_pt_bins: int,
    n_two_log10_rho_bins: int,
    pt_groups: Sequence[Sequence[int]],
    two_log10_rho_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Rebin a full reco covariance as ``G @ covariance @ G.T``.

    The output shape is ``(output_pt, output_coordinate, output_pt,
    output_coordinate)`` so later unfolding integration keeps explicit axes.
    """

    source_size = n_pt_bins * n_two_log10_rho_bins
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (n_pt_bins, n_two_log10_rho_bins) * 2:
        flat_covariance = covariance.reshape(source_size, source_size)
    elif covariance.shape == (source_size, source_size):
        flat_covariance = covariance
    else:
        raise ValueError(
            "Covariance shape does not match the supplied fine pT and "
            "two_log10_rho dimensions"
        )
    matrix = covariance_rebin_matrix(
        n_pt_bins, n_two_log10_rho_bins, pt_groups, two_log10_rho_groups
    )
    rebinned = matrix @ flat_covariance @ matrix.T
    output_shape = (len(pt_groups), len(two_log10_rho_groups)) * 2
    return rebinned.reshape(output_shape)
