"""Input discovery and schema adaptation for dijet/trijet rho unfolding."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl
from typing import Mapping

import hist
import numpy as np

from unfold.utils.integrate_and_rebin import rebin_hist


CHANNELS = ("dijet", "trijet")
RHO_HIST_KEYS = {
    "ungroomed": {
        "response": "response_matrix_rho_u",
        "reco": "ptjet_rhojet_u_reco",
        "gen": "ptjet_rhojet_u_gen",
    },
    "groomed": {
        "response": "response_matrix_rho_g",
        "reco": "ptjet_rhojet_g_reco",
        "gen": "ptjet_rhojet_g_gen",
    },
}
RHO_PHYSICS_AXES = {
    "response": ("ptgen", "mpt_gen", "ptreco", "mpt_reco"),
    "reco": ("ptreco", "mpt_reco"),
    "gen": ("ptgen", "mpt_gen"),
}
SYSTEMATIC_RENAMES = {
    "PUSFUp": "puUp",
    "PUSFDown": "puDown",
    "PDFUp": "pdfUp",
    "PDFDown": "pdfDown",
    "L1prefiringUp": "l1prefiringUp",
    "L1prefiringDown": "l1prefiringDown",
    "Q2muRUp": "q2muRUp",
    "Q2muRDown": "q2muRDown",
    "Q2muFUp": "q2muFUp",
    "Q2muFDown": "q2muFDown",
    "LuminosityUp": "lumiUp",
    "LuminosityDown": "lumiDown",
    "isrUp": "ISRUp",
    "isrDown": "ISRDown",
    "fsrUp": "FSRUp",
    "fsrDown": "FSRDown",
}


@dataclass(frozen=True)
class RhoChannelFiles:
    channel: str
    year: str
    data: Path
    mc: Path
    herwig: Path | None = None


@dataclass(frozen=True)
class RhoAnalysisBinning:
    pt_edges: list[float]
    rho_edges: list[float]
    rho_edges_gen: list[float]
    reco_rho_edges_by_pt: list[list[float]]
    gen_rho_edges_by_pt: list[list[float]]


@dataclass
class PreparedRhoInputs:
    mc: dict[str, object]
    data: dict[str, object]
    systematics: list[str]
    binning: dict[str, RhoAnalysisBinning]
    source_files: RhoChannelFiles
    herwig: dict[str, object] | None = None


def channel_rho_binning(channel: str, groomed: bool) -> RhoAnalysisBinning:
    """Return producer-compatible rho edges for one dijet/trijet mode."""

    if channel not in CHANNELS:
        raise ValueError(f"Unsupported channel {channel!r}; choose from {CHANNELS}")

    pt_edges = [0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0]
    rho_edges = [
        -10.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.75, -4.5,
        -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5,
        -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
        -0.25, 0.0,
    ]
    rho_edges_gen = [
        -10.0, -8.0, -7.0, -6.0, -5.0, -4.5, -4.0, -3.5,
        -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
    ]

    reco_lower_edge = -4.75 if groomed else -2.75
    gen_lower_edge = -4.5 if groomed else -2.5
    reco_edges_by_pt = [-10.0] + [
        edge for edge in rho_edges if edge >= reco_lower_edge
    ]
    gen_edges_by_pt = [-10.0] + [
        edge for edge in rho_edges_gen if edge >= gen_lower_edge
    ]
    n_pt = len(pt_edges) - 1
    reco_edges_by_pt = [reco_edges_by_pt[:] for _ in range(n_pt)]
    gen_edges_by_pt = [gen_edges_by_pt[:] for _ in range(n_pt)]

    if channel == "dijet" and groomed:
        # Merge the low-rho gen tail into coarse bins to suppress oscillatory
        # unfolding. Low-pT intervals stop the merge at -4.5; the two central
        # pT intervals (where statistics support it) carry one extra, deeper
        # truth bin reaching -6.0. Reco splits every gen bin into two.
        fine_gen_edges = [edge for edge in rho_edges_gen if edge >= -2.5]
        fine_reco_edges = [edge for edge in rho_edges if edge >= -2.5]

        low_pt_gen_edges = [-10.0, -4.5, -3.5] + fine_gen_edges
        low_pt_reco_edges = [-10.0, -7.0, -4.5, -4.0, -3.5, -3.0] + fine_reco_edges

        central_pt_gen_edges = [-10.0, -6.0, -4.5, -3.5] + fine_gen_edges
        central_pt_reco_edges = (
            [-10.0, -8.0, -6.0, -5.5, -4.5, -4.0, -3.5, -3.0] + fine_reco_edges
        )

        for pt_index in range(n_pt):
            if pt_index in (3, 4):
                gen_edges_by_pt[pt_index] = central_pt_gen_edges[:]
                reco_edges_by_pt[pt_index] = central_pt_reco_edges[:]
            else:
                gen_edges_by_pt[pt_index] = low_pt_gen_edges[:]
                reco_edges_by_pt[pt_index] = low_pt_reco_edges[:]

    return RhoAnalysisBinning(
        pt_edges=pt_edges,
        rho_edges=rho_edges,
        rho_edges_gen=rho_edges_gen,
        reco_rho_edges_by_pt=reco_edges_by_pt,
        gen_rho_edges_by_pt=gen_edges_by_pt,
    )


def discover_rho_channel_files(
    input_root: str | Path,
    channel: str,
    year: str | int,
) -> RhoChannelFiles:
    if channel not in CHANNELS:
        raise ValueError(f"Unsupported channel {channel!r}; choose from {CHANNELS}")

    year = str(year)
    # Channel rho pickles live under inputs/<channel>/rho/. Fall back to the
    # legacy flat inputs/<channel>/ layout if the rho/ subdir is absent.
    channel_dir = Path(input_root) / channel / "rho"
    if not channel_dir.is_dir():
        channel_dir = Path(input_root) / channel
    data = channel_dir / f"minimal_rho_{channel}_data_{year}.pkl"
    mc = channel_dir / f"minimal_rho_{channel}_mg_pythia8_{year}.pkl"
    missing = [path for path in (data, mc) if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required rho inputs:\n{formatted}")
    # HERWIG is optional (alternate-generator model uncertainty); use it if the
    # producer pickle is present, otherwise the channel runs without it.
    herwig = channel_dir / f"minimal_rho_{channel}_herwig_{year}.pkl"
    return RhoChannelFiles(
        channel=channel, year=year, data=data, mc=mc,
        herwig=herwig if herwig.is_file() else None,
    )


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def _storage(h_obj):
    storage_type = getattr(h_obj, "storage_type", None)
    return storage_type() if callable(storage_type) else h_obj._storage_type()


def _values_and_variances(h_obj):
    values = np.asarray(h_obj.values(flow=False), dtype=float)
    variances = h_obj.variances(flow=False)
    if variances is not None:
        variances = np.asarray(variances, dtype=float)
    return values, variances


def _build_systematic_histogram(
    sample_hist,
    physics_axes: tuple[str, ...],
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray | None]],
):
    systematic_axis = hist.axis.StrCategory(
        list(arrays),
        name="systematic",
        growth=False,
    )
    axes = [sample_hist.axes[name] for name in physics_axes] + [systematic_axis]
    output = hist.Hist(*axes, storage=_storage(sample_hist))
    output_values = output.values(flow=False)
    output_variances = output.variances(flow=False)

    for index, (systematic, (values, variances)) in enumerate(arrays.items()):
        output_values[(..., index)] = values
        if output_variances is not None and variances is not None:
            output_variances[(..., index)] = variances
    return output


def _rebin_rho_histogram(h_obj, binning: RhoAnalysisBinning):
    output = h_obj
    for axis_name, edges in (
        ("ptreco", binning.pt_edges),
        ("ptgen", binning.pt_edges),
        ("mpt_reco", binning.rho_edges),
        ("mpt_gen", binning.rho_edges_gen),
    ):
        if axis_name in output.axes.name:
            output = rebin_hist(output, axis_name, edges)
    return output


def _adapt_histogram(
    h_obj,
    role: str,
    binning: RhoAnalysisBinning,
    systematic_renames: Mapping[str, str],
):
    physics_axes = RHO_PHYSICS_AXES[role]
    rebinned = _rebin_rho_histogram(h_obj, binning)
    available = (
        list(rebinned.axes["systematic"])
        if "systematic" in rebinned.axes.name
        else ["nominal"]
    )
    arrays: OrderedDict[str, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
    sample_hist = None

    for systematic in available:
        selected = rebinned
        if "systematic" in selected.axes.name:
            selected = selected[{"systematic": systematic}]
        if "jk" in selected.axes.name:
            jk_categories = list(selected.axes["jk"])
            jk_value = -1 if -1 in jk_categories else jk_categories[0]
            selected = selected[{"jk": hist.loc(jk_value)}]
        projected = selected.project(*physics_axes)
        sample_hist = projected
        renamed = systematic_renames.get(systematic, systematic)
        if renamed in arrays:
            raise ValueError(
                f"Systematic rename collision for {systematic!r} -> {renamed!r}"
            )
        arrays[renamed] = _values_and_variances(projected)

    if sample_hist is None:
        raise ValueError(f"No histogram content found for role {role!r}")
    return _build_systematic_histogram(sample_hist, physics_axes, arrays)


def _required_keys_for_mc() -> set[str]:
    return {
        key
        for mode_keys in RHO_HIST_KEYS.values()
        for key in mode_keys.values()
    }


def _required_keys_for_data() -> set[str]:
    return {
        mode_keys["reco"]
        for mode_keys in RHO_HIST_KEYS.values()
    }


def _validate_payload(payload, required_keys: set[str], role: str, path: Path):
    if not isinstance(payload, dict):
        raise TypeError(f"{role} input {path} must contain a dictionary")
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise KeyError(f"{role} input {path} is missing histograms: {missing}")


def build_prepared_rho_inputs(
    files: RhoChannelFiles,
    *,
    systematic_renames: Mapping[str, str] | None = None,
) -> PreparedRhoInputs:
    """Adapt producer files without inventing missing systematics or samples."""

    mc_payload = _load_pickle(files.mc)
    data_payload = _load_pickle(files.data)
    _validate_payload(mc_payload, _required_keys_for_mc(), "MC", files.mc)
    _validate_payload(data_payload, _required_keys_for_data(), "data", files.data)

    renames = dict(SYSTEMATIC_RENAMES)
    if systematic_renames:
        renames.update(systematic_renames)
    binning_by_mode = {
        "ungroomed": channel_rho_binning(files.channel, False),
        "groomed": channel_rho_binning(files.channel, True),
    }
    adapted_mc = {}
    adapted_data = {}

    for mode, keys in RHO_HIST_KEYS.items():
        mode_binning = binning_by_mode[mode]
        for role, key in keys.items():
            adapted_mc[key] = _adapt_histogram(
                mc_payload[key],
                role,
                mode_binning,
                renames,
            )
        adapted_data[keys["reco"]] = _adapt_histogram(
            data_payload[keys["reco"]],
            "reco",
            mode_binning,
            renames,
        )

    ungroomed_systematics = list(
        adapted_mc[RHO_HIST_KEYS["ungroomed"]["response"]].axes["systematic"]
    )
    groomed_systematics = list(
        adapted_mc[RHO_HIST_KEYS["groomed"]["response"]].axes["systematic"]
    )
    if ungroomed_systematics != groomed_systematics:
        raise ValueError(
            "Groomed and ungroomed response matrices have different systematics"
        )
    if "nominal" not in ungroomed_systematics:
        raise ValueError("MC response matrices do not contain a nominal category")

    for mode, keys in RHO_HIST_KEYS.items():
        data_systematics = list(adapted_data[keys["reco"]].axes["systematic"])
        if "nominal" not in data_systematics:
            raise ValueError(f"{mode} data histogram has no nominal category")

    # Optional HERWIG sample for the alternate-generator (model) uncertainty.
    # Adapted like the MC (reco/gen/response, both modes); only the nominal
    # category is used downstream.
    adapted_herwig = None
    if files.herwig is not None:
        herwig_payload = _load_pickle(files.herwig)
        _validate_payload(herwig_payload, _required_keys_for_mc(), "HERWIG", files.herwig)
        adapted_herwig = {}
        for mode, keys in RHO_HIST_KEYS.items():
            mode_binning = binning_by_mode[mode]
            for role, key in keys.items():
                adapted_herwig[key] = _adapt_histogram(
                    herwig_payload[key],
                    role,
                    mode_binning,
                    renames,
                )

    return PreparedRhoInputs(
        mc=adapted_mc,
        data=adapted_data,
        systematics=ungroomed_systematics,
        binning=binning_by_mode,
        source_files=files,
        herwig=adapted_herwig,
    )
