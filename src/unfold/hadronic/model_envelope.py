"""Hadronic inputs for the shared Z+jet two-leg model envelope.

This module derives generator-space weights on the *stored fine* hadronic
coordinate before the candidate GEN-bin merge.  The response adapter applies
those weights to the nominal response columns and inclusive GEN marginal, then
the maintained :class:`~unfold.engine.Unfolder` re-unfolds the
same data through each varied response.

The reported prescription is the Z+jet ARC round-2 one::

    PS  = max(MESS+Vincia, FSR)
    HAD = max(CR1, CR2, fragmentation-hard, fragmentation-soft)
    model = sqrt(PS**2 + HAD**2)

The 2026-08-13 MESS+Vincia numerator is on the exact hadronic fiducial.  The
available CR/fragmentation transfer ratios predate that fiducial, so their
provenance is retained explicitly and the resulting HAD leg is provisional.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from unfold.paths import INTERNAL_RESULTS
from unfold.hadronic.inputs import HadronicRun2Inputs, hadronic_binning
from unfold.hadronic.internal_variations import (
    load_hadronic_internal_transfers,
)
from unfold.hadronic.vincia import (
    HadronicVinciaSource,
    derive_hadronic_vincia_prediction,
)


MODEL_SOURCES = ("vincia", "cr1", "cr2", "fraghard", "fragsoft")
PREPARED_SYSTEMATIC_BY_SOURCE = {
    source: f"model_{source}" for source in MODEL_SOURCES
}
DEFAULT_INTERNAL_TRANSFER_DIRECTORY = INTERNAL_RESULTS / "final_1000of1000"
DEFAULT_ANCHOR_FRACTION = 0.02
DEFAULT_MIN_MESS_EFFECTIVE_ENTRIES = 25.0
DEFAULT_WEIGHT_CLIP = (0.2, 5.0)
DEFAULT_CLOSURE_TOLERANCE = 0.01
DEFAULT_MAX_ITERATIONS = 8


class HadronicModelEnvelopeError(RuntimeError):
    """A model source cannot satisfy the hadronic reweight contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_mask(
    edges: Sequence[float], window: tuple[float, float]
) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    low, high = (float(value) for value in window)
    mask = (edges[:-1] >= low - 1.0e-12) & (edges[1:] <= high + 1.0e-12)
    if not np.any(mask):
        raise HadronicModelEnvelopeError(
            f"normalization window {window} contains no complete fine bins"
        )
    return mask


def _nearest_anchor_fill(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Flat-extrapolate/interpolate from the nearest populated anchor bin."""

    values = np.asarray(values, dtype=float)
    anchors = np.asarray(anchors, dtype=bool)
    anchor_indices = np.flatnonzero(anchors)
    if anchor_indices.size == 0:
        raise HadronicModelEnvelopeError("model reweight has no populated anchor bins")
    output = values.copy()
    for index in np.flatnonzero(~anchors):
        nearest = anchor_indices[np.argmin(np.abs(anchor_indices - index))]
        output[index] = values[nearest]
    return output


def _renormalize_weight(
    nominal: np.ndarray,
    weight: np.ndarray,
    normalization_mask: np.ndarray,
) -> np.ndarray:
    nominal_total = float(np.asarray(nominal)[normalization_mask].sum())
    varied_total = float(
        (np.asarray(nominal)[normalization_mask] * np.asarray(weight)[normalization_mask]).sum()
    )
    if nominal_total <= 0.0 or varied_total <= 0.0:
        raise HadronicModelEnvelopeError(
            "model reweight has a nonpositive normalization-window integral"
        )
    return np.asarray(weight, dtype=float) * (nominal_total / varied_total)


def _derive_iterated_weight(
    nominal: np.ndarray,
    target: np.ndarray,
    *,
    normalization_mask: np.ndarray,
    anchors: np.ndarray,
    clip: tuple[float, float],
    closure_tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Iteratively match a target shape on populated fine-coordinate bins.

    Target and nominal are binned on the same fine axis.  The iteration is
    deliberately retained even though an exact binwise ratio usually closes in
    one pass: it is the fail-closed validation required by the Z+jet reweight
    workflow and remains correct if a conditioned/extrapolated bin is added.
    """

    nominal = np.asarray(nominal, dtype=float)
    target = np.asarray(target, dtype=float)
    anchors = np.asarray(anchors, dtype=bool)
    if nominal.shape != target.shape or anchors.shape != nominal.shape:
        raise ValueError("nominal, target, and anchor arrays must have identical shapes")
    if np.any(nominal < 0.0) or np.any(target < 0.0):
        raise HadronicModelEnvelopeError("model reweight inputs contain negative bins")
    nominal_norm = float(nominal[normalization_mask].sum())
    target_norm = float(target[normalization_mask].sum())
    if nominal_norm <= 0.0 or target_norm <= 0.0:
        raise HadronicModelEnvelopeError(
            "model reweight nominal or target has no normalization-window yield"
        )

    nominal_shape = nominal / nominal_norm
    target_shape = target / target_norm
    weight = np.ones_like(nominal_shape)
    residual_max = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        current = nominal * weight
        current_shape = current / float(current[normalization_mask].sum())
        residual = np.divide(
            target_shape,
            current_shape,
            out=np.ones_like(target_shape),
            where=current_shape > 0.0,
        )
        anchor_residual = residual[anchors]
        residual_max = float(np.max(np.abs(anchor_residual - 1.0)))
        if residual_max <= closure_tolerance:
            break
        updated = weight.copy()
        updated[anchors] *= anchor_residual
        updated = np.clip(updated, *clip)
        updated = _nearest_anchor_fill(updated, anchors)
        weight = _renormalize_weight(nominal, updated, normalization_mask)

    current = nominal * weight
    current_shape = current / float(current[normalization_mask].sum())
    closure = np.divide(
        current_shape,
        target_shape,
        out=np.ones_like(current_shape),
        where=target_shape > 0.0,
    )
    residual_max = float(np.max(np.abs(closure[anchors] - 1.0)))
    if residual_max > closure_tolerance:
        raise HadronicModelEnvelopeError(
            "iterated model reweight failed its populated-bin closure gate: "
            f"{residual_max:.6g} > {closure_tolerance:.6g}"
        )
    return weight, {
        "iterations": iteration,
        "anchor_bin_indices": np.flatnonzero(anchors).tolist(),
        "anchor_count": int(np.count_nonzero(anchors)),
        "max_relative_closure_residual": residual_max,
    }


def _condition_transfer_weight(
    nominal: np.ndarray,
    transfer_ratio: np.ndarray,
    *,
    normalization_mask: np.ndarray,
    anchor_fraction: float,
    clip: tuple[float, float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Condition a harvested variation/CP5 ratio on populated nominal bins."""

    nominal = np.asarray(nominal, dtype=float)
    transfer_ratio = np.asarray(transfer_ratio, dtype=float)
    if nominal.shape != transfer_ratio.shape:
        raise ValueError("nominal and transfer-ratio arrays must have identical shapes")
    scale = float(np.max(nominal[normalization_mask]))
    anchors = (
        np.isfinite(transfer_ratio)
        & (transfer_ratio > 0.0)
        & (nominal >= anchor_fraction * max(scale, 1.0e-300))
    )
    conditioned = np.where(anchors, transfer_ratio, 1.0)
    conditioned = np.clip(conditioned, *clip)
    conditioned = _nearest_anchor_fill(conditioned, anchors)
    conditioned = _renormalize_weight(nominal, conditioned, normalization_mask)

    raw_target = nominal * np.where(
        np.isfinite(transfer_ratio) & (transfer_ratio > 0.0), transfer_ratio, 1.0
    )
    varied = nominal * conditioned
    raw_shape = raw_target / float(raw_target[normalization_mask].sum())
    varied_shape = varied / float(varied[normalization_mask].sum())
    closure = np.divide(
        varied_shape,
        raw_shape,
        out=np.ones_like(varied_shape),
        where=raw_shape > 0.0,
    )
    return conditioned, {
        "anchor_bin_indices": np.flatnonzero(anchors).tolist(),
        "anchor_count": int(np.count_nonzero(anchors)),
        "max_relative_conditioning_residual": float(
            np.max(np.abs(closure[anchors] - 1.0))
        ),
    }


@dataclass(frozen=True)
class HadronicModelEnvelopeInputs:
    """Fine-axis weights and complete provenance for one hadronic channel."""

    channel: str
    grooming_mode: str
    weights_by_source: Mapping[str, np.ndarray]
    systematic_by_source: Mapping[str, str]
    closure: Mapping[str, object]
    provenance: Mapping[str, object]

    def prepared_variations(self) -> dict[str, np.ndarray]:
        return {
            self.systematic_by_source[source]: np.asarray(weight, dtype=float)
            for source, weight in self.weights_by_source.items()
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "grooming_mode": self.grooming_mode,
            "systematic_by_source": dict(self.systematic_by_source),
            "source_identity": self.provenance["source_identity"],
            "derivation": self.provenance["derivation"],
        }

    def provenance_payload(self) -> dict[str, object]:
        return {
            **self.provenance,
            "systematic_by_source": dict(self.systematic_by_source),
            "closure": self.closure,
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        source_names = tuple(self.systematic_by_source)
        return {
            "model_source_names": np.asarray(source_names),
            "model_fine_weights": np.stack(
                [np.asarray(self.weights_by_source[source], dtype=float) for source in source_names]
            ),
        }


def derive_hadronic_model_envelope_inputs(
    inputs: HadronicRun2Inputs,
    vincia_source: HadronicVinciaSource,
    *,
    variant: str,
    normalization_window: tuple[float, float],
    grooming_mode: str = "groomed",
    internal_transfer_directory: Path = DEFAULT_INTERNAL_TRANSFER_DIRECTORY,
    anchor_fraction: float = DEFAULT_ANCHOR_FRACTION,
    min_mess_effective_entries: float = DEFAULT_MIN_MESS_EFFECTIVE_ENTRIES,
    clip: tuple[float, float] = DEFAULT_WEIGHT_CLIP,
    closure_tolerance: float = DEFAULT_CLOSURE_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> HadronicModelEnvelopeInputs:
    """Derive the five fine-axis response weights used by the model envelope."""

    if inputs.channel != vincia_source.channel:
        raise ValueError(
            f"hadronic input channel {inputs.channel!r} does not match "
            f"MESS source channel {vincia_source.channel!r}"
        )
    if grooming_mode not in {"groomed", "ungroomed"}:
        raise ValueError("grooming_mode must be 'groomed' or 'ungroomed'")
    source = inputs.modes[grooming_mode]
    fine_axes = source.fine_axes
    fine_pt_edges = np.asarray(fine_axes.pt_edges, dtype=float)
    fine_coordinate_edges = np.asarray(
        fine_axes.two_log10_rho_gen_edges, dtype=float
    )
    nominal_gen = np.asarray(
        source.gen_by_systematic["nominal"], dtype=float
    )
    expected_shape = (len(fine_pt_edges) - 1, len(fine_coordinate_edges) - 1)
    if nominal_gen.shape != expected_shape:
        raise HadronicModelEnvelopeError(
            f"unexpected fine nominal GEN shape {nominal_gen.shape}; expected {expected_shape}"
        )
    normalization_mask = _normalization_mask(
        fine_coordinate_edges, normalization_window
    )

    # Derive the MESS transfer on the analysis pT slices.  This is important
    # for trijet and the high-pT dijet slice: the final-all sample has adequate
    # statistics after the candidate pT merge, but not in every internal
    # 680--760/760--820/820--infinity source slice separately.
    candidate = hadronic_binning(
        inputs.channel, variant, grooming_mode=grooming_mode
    )
    mess_prediction = derive_hadronic_vincia_prediction(
        vincia_source,
        pt_edges=candidate.pt_edges,
        gen_edges_by_pt=(tuple(fine_coordinate_edges),) * len(candidate.pt_groups),
        normalization_window=normalization_window,
        grooming_mode=grooming_mode,
    )
    mess_sumw = np.stack(mess_prediction.sumw_by_pt)
    mess_sumw2 = np.stack(mess_prediction.sumw2_by_pt)

    weights_by_source: dict[str, np.ndarray] = {
        source_name: np.ones(expected_shape, dtype=float) for source_name in MODEL_SOURCES
    }
    closure: dict[str, object] = {source_name: {} for source_name in MODEL_SOURCES}
    for candidate_index, fine_pt_group in enumerate(candidate.pt_groups):
        nominal = nominal_gen[np.asarray(fine_pt_group, dtype=int)].sum(axis=0)
        target = mess_sumw[candidate_index]
        neff = np.divide(
            target**2,
            mess_sumw2[candidate_index],
            out=np.zeros_like(target),
            where=mess_sumw2[candidate_index] > 0.0,
        )
        nominal_scale = float(np.max(nominal[normalization_mask]))
        anchors = (
            (nominal >= anchor_fraction * max(nominal_scale, 1.0e-300))
            & (target > 0.0)
            & (neff >= min_mess_effective_entries)
        )
        weight, metrics = _derive_iterated_weight(
            nominal,
            target,
            normalization_mask=normalization_mask,
            anchors=anchors,
            clip=clip,
            closure_tolerance=closure_tolerance,
            max_iterations=max_iterations,
        )
        for fine_pt_index in fine_pt_group:
            weights_by_source["vincia"][fine_pt_index] = weight
        closure["vincia"][str(candidate_index)] = {
            "pt_range_GeV": [
                float(candidate.pt_edges[candidate_index]),
                float(candidate.pt_edges[candidate_index + 1]),
            ],
            "source_fine_pt_bin_indices": list(fine_pt_group),
            "minimum_effective_entries": float(min_mess_effective_entries),
            **metrics,
        }

    # Both grooming modes harvest the RAW standalone internal-variation rows
    # and histogram them directly onto the current fine GEN axis.  Until the
    # 2026-08-27 aligned-axes re-production, groomed instead read the
    # pre-binned ``internal_fullsimref_<channel>.npz`` (FullSim-anchored
    # presentation of the SAME campaign's variation/CP5 ratios) — but that
    # file pins ``rho_edges`` to the retired free-edge gen axis and cannot
    # follow a binning change, while the raw harvest feeds the identical
    # conditioned transfer-ratio application below.  The npz path is gone
    # with the old axes; the ratios remain "normalized variation /
    # normalized standalone CP5" applied to the FullSim nominal.
    internal_context = None
    raw_internal = load_hadronic_internal_transfers(
        inputs.channel,
        grooming_mode=grooming_mode,
        coordinate_edges=fine_coordinate_edges,
    )
    internal_source_identity = {
        **raw_internal.provenance,
        "prescription": "normalized variation / normalized standalone CP5",
        "fiducial_status": "same internal standalone campaign selection",
        "status": "raw_manifest_harvest",
    }

    try:
        for fine_pt_index in range(1, len(fine_pt_edges) - 1):
            pt_low = fine_pt_edges[fine_pt_index]
            transfer_pt_low = 200 if pt_low < 290 else 290 if pt_low < 400 else 400
            nominal = nominal_gen[fine_pt_index]
            for model_source in ("cr1", "cr2", "fraghard", "fragsoft"):
                ratio = np.asarray(
                    raw_internal.ratio_by_source_and_pt_low[model_source][transfer_pt_low],
                    dtype=float,
                )
                weight, metrics = _condition_transfer_weight(
                    nominal,
                    ratio,
                    normalization_mask=normalization_mask,
                    anchor_fraction=anchor_fraction,
                    clip=clip,
                )
                weights_by_source[model_source][fine_pt_index] = weight
                closure[model_source][str(fine_pt_index)] = {
                    "pt_range_GeV": [
                        float(fine_pt_edges[fine_pt_index]),
                        float(fine_pt_edges[fine_pt_index + 1]),
                    ],
                    "transfer_pt_low_GeV": transfer_pt_low,
                    **metrics,
                }
    finally:
        if internal_context is not None:
            internal_context.__exit__(None, None, None)

    source_identity = {
        "mess_vincia": vincia_source.identity_payload(),
        "vincia_denominator": {
            "source": "combined Run-2 FullSim nominal GEN spectrum from the hadronic response inputs",
            "status": "provisional_no_matched_pair_split_standalone_cp5_denominator",
            "interpretation": (
                "MESS+Vincia / FullSim CP5 is an alternate-model response leg; "
                "it is not a pure Vincia-only shower ratio"
            ),
        },
        "hadronization_transfer": internal_source_identity,
    }
    derivation = {
        "coordinate": (
            "2*log10(rho), rho=m_g/(pt*0.8)"
            if grooming_mode == "groomed"
            else "2*log10(rho), rho=m_u/(pt*0.8)"
        ),
        "application_point": "stored fine GEN coordinate before candidate GEN merge",
        "candidate": variant,
        "normalization_window": list(normalization_window),
        "anchor_fraction_of_window_peak": float(anchor_fraction),
        "minimum_mess_effective_entries": float(min_mess_effective_entries),
        "weight_clip": list(clip),
        "closure_tolerance": float(closure_tolerance),
        "max_iterations": int(max_iterations),
        "fake_treatment": "nominal absolute fakes retained; matched response is reweighted",
        "miss_treatment": "inclusive nominal GEN and matched response reweighted coherently",
        "combination": "PS=max(MESS+Vincia,FSR); HAD=max(CR1,CR2,frag-hard,frag-soft); model=sqrt(PS^2+HAD^2)",
        "isr_treatment": "excluded from model band",
    }
    return HadronicModelEnvelopeInputs(
        channel=inputs.channel,
        grooming_mode=grooming_mode,
        weights_by_source=weights_by_source,
        systematic_by_source=PREPARED_SYSTEMATIC_BY_SOURCE,
        closure=closure,
        provenance={
            "source_identity": source_identity,
            "derivation": derivation,
            "fine_pt_edges_GeV": fine_pt_edges.tolist(),
            "fine_gen_two_log10_rho_edges": fine_coordinate_edges.tolist(),
        },
    )
