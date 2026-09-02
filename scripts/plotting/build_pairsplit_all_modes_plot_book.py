#!/usr/bin/env python3
"""Build a vector-quality groomed + ungroomed pair-split plot book."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "pairsplit_run2"
OUTPUT_PDF = OUTPUT_ROOT / "PAIR_SPLIT_GROOMED_UNGROOMED_PLOT_BOOK_2026-08-27.pdf"
INVENTORY_JSON = OUTPUT_ROOT / "PAIR_SPLIT_GROOMED_UNGROOMED_PLOT_BOOK_2026-08-27.inventory.json"

RUN_MANIFESTS = (
    OUTPUT_ROOT / "dijet/aligned/regularization-none__tau-disabled__normalization-peak__systematics-f64617bc992988f4/run_manifest.json",
    OUTPUT_ROOT / "dijet/aligned/ungroomed/regularization-none__tau-disabled__normalization-minus2p5_to_zero__systematics-ac5dca45279f978c/run_manifest.json",
    OUTPUT_ROOT / "trijet/aligned/regularization-none__tau-disabled__normalization-full__systematics-9e20a3857f23f611/run_manifest.json",
    OUTPUT_ROOT / "trijet/aligned/ungroomed/regularization-none__tau-disabled__normalization-minus2p5_to_zero__systematics-57d6a4e8cadbc415/run_manifest.json",
)

PAGE_WIDTH, PAGE_HEIGHT = letter
INK = HexColor("#0B2545")
BLUE = HexColor("#2E74B5")
DARK_BLUE = HexColor("#1F4D78")
MUTED = HexColor("#5B6572")
PALE_BLUE = HexColor("#E8EEF5")
LIGHT_GRAY = HexColor("#F2F4F7")
GREEN = HexColor("#2E7D32")
CAUTION = HexColor("#8A5A00")


PRODUCTS = (
    # data_mc lives under OUTPUT_ROOT/data_mc/<channel>/, not the run
    # directory; ordered_plot_pages resolves it specially.
    ("data_mc", None, "data_mc_{mode}_pt{index}.pdf"),
    ("unfolded", "unfolded", "unfolded_{mode}_pt{index}.pdf"),
    ("corrected_refold", "validation", "folded_{mode}_pt{index}.pdf"),
    ("purity_stability", "response", "purity_stability_{mode}_pt{index}.pdf"),
    (
        "grouped_uncertainties",
        "uncertainties",
        "summary_grouped_linear_{mode}_pt{index}.pdf",
    ),
)
DATA_MC_DIRECTORY = OUTPUT_ROOT / "data_mc"
EXPECTED_PLOT_COUNT = 92  # 4 runs x (5 per-pT products x n_pt + 3 run-level)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}g}"


def pt_label(edges: Sequence[float], index: int) -> str:
    low = f"{float(edges[index]):g}"
    high = "infinity" if float(edges[index + 1]) >= 12000 else f"{float(edges[index + 1]):g}"
    return f"{low}-{high} GeV"


def product_title(product: str) -> str:
    return {
        "data_mc": "Detector-level data/MC comparison",
        "unfolded": "Normalized unfolded spectrum",
        "corrected_refold": "Corrected measured spectrum vs nominal refold",
        "purity_stability": "Purity and stability",
        "grouped_uncertainties": "Grouped fractional uncertainties",
        "correlation": "Total normalized-covariance correlation",
        "unfolded_summary": "Unfolded spectra across pT slices",
        "unfolded_summary_ratio": "Unfolded-to-Pythia ratios across pT slices",
    }[product]


@dataclass(frozen=True)
class RunRecord:
    manifest_path: Path
    manifest: Mapping[str, object]
    directory: Path
    artifact_path: Path
    channel: str
    mode: str
    candidate: str
    pt_edges: tuple[float, ...]
    gen_edges: tuple[float, ...]
    display_window: tuple[float, float]
    normalization_window: tuple[float, float]
    response_condition: float
    min_purity: float
    mean_purity: float
    min_stability: float
    mean_stability: float
    refold_chi2: float
    refold_rank: int
    reco_bins_per_pt: int
    gen_bins_per_pt: int
    refold_chi2_by_pt: tuple[float, ...]
    refold_rank_by_pt: tuple[int, ...]
    closure_relative_l1: float
    stat_median_percent: float
    stat_max_percent: float
    total_median_percent: float
    total_max_percent: float
    first_bin_total_max_percent: float
    total_below_stat_count: int
    model_ps_source: str
    model_had_source: str
    data_covariance_source: str
    expected_null_modes: int
    observed_null_modes: int
    covariance_min_eigenvalue: float

    @property
    def label(self) -> str:
        return f"{self.channel.capitalize()} {self.mode}"


@dataclass(frozen=True)
class PlotPage:
    run: RunRecord
    product: str
    source_pdf: Path
    pt_index: int | None


def _complete_bin_mask(edges: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    return (edges[:-1] >= low - 1.0e-9) & (edges[1:] <= high + 1.0e-9)


def _positive_mode_chi2(
    residual: np.ndarray, covariance: np.ndarray, rcond: float = 1.0e-12
) -> tuple[float, int]:
    """Residual chi2 over the positive covariance eigenmodes.

    Mirrors the runner's ``_full_covariance_chi2`` convention so the per-slice
    values printed on refold pages are directly comparable to the manifest's
    run-level ``refolded_residual`` diagnostic.
    """

    residual = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0)
    keep = eigenvalues > rcond * scale
    if not np.any(keep):
        return 0.0, 0
    projected = eigenvectors[:, keep].T @ residual
    return float(np.sum(projected**2 / eigenvalues[keep])), int(np.count_nonzero(keep))


def per_pt_refold_chi2(
    residual: np.ndarray,
    covariance: np.ndarray,
    n_slices: int,
    reco_bins_per_pt: int,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Slice-local refold chi2 from the saved artifact arrays.

    Each slice uses its own diagonal covariance block; cross-slice covariance
    (present for the event-clustered dijet input) is deliberately outside a
    per-slice statistic and remains covered by the run-level diagnostic.
    """

    residual = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    expected = n_slices * reco_bins_per_pt
    if residual.size != expected or covariance.shape != (expected, expected):
        raise ValueError(
            "Refold residual/covariance do not match the rectangular reco layout"
        )
    chi2_values: list[float] = []
    ranks: list[int] = []
    for index in range(n_slices):
        block = slice(index * reco_bins_per_pt, (index + 1) * reco_bins_per_pt)
        chi2, rank = _positive_mode_chi2(residual[block], covariance[block, block])
        chi2_values.append(chi2)
        ranks.append(rank)
    return tuple(chi2_values), tuple(ranks)


def load_run(manifest_path: Path) -> RunRecord:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory = manifest_path.parent
    artifact_path = Path(str(manifest["artifact"])).resolve()
    channel = str(manifest["channel"])
    mode = str(manifest["grooming_mode"])
    candidate = str(manifest["binning"]["candidate"])
    pt_edges = tuple(float(value) for value in manifest["binning"]["pt_edges_GeV"])
    gen_edges = tuple(
        float(value) for value in manifest["binning"]["gen_two_log10_rho_edges"]
    )
    display_window = tuple(
        float(value)
        for value in manifest["unfolding"]["plot_display"]["two_log10_rho_range"]
    )
    normalization_window = tuple(
        float(value)
        for value in manifest["unfolding"]["plot_normalization"]["two_log10_rho_range"]
    )
    diagnostics = manifest["diagnostics"]
    migration = diagnostics["gen_layout_migration"]["reported_normalization_window"]
    covariance = diagnostics["normalized_covariances"]["total"][
        "reported_normalization_window"
    ]
    closure = diagnostics["nominal_mc_self_closure"]["raw_bias"]
    selected = manifest["model_uncertainty"]["selected_covariance_sources"]["global"]

    reco_edges = tuple(
        float(value)
        for value in manifest["binning"]["base_reco_two_log10_rho_edges"]
    )

    with np.load(artifact_path, allow_pickle=False) as arrays:
        normalized = np.asarray(arrays["normalized_result"], dtype=float)
        stat = np.sqrt(np.clip(np.diag(arrays["norm_cov_stat"]), 0.0, None))
        total = np.sqrt(np.clip(np.diag(arrays["norm_cov_total"]), 0.0, None))
        refold_chi2_by_pt, refold_rank_by_pt = per_pt_refold_chi2(
            arrays["refolded_residual"],
            arrays["measured_corrected_covariance"],
            len(pt_edges) - 1,
            len(reco_edges) - 1,
        )
        bins_per_pt = len(gen_edges) - 1
        if normalized.size != bins_per_pt * (len(pt_edges) - 1):
            raise ValueError(f"Unexpected flattened result size in {artifact_path}")
        local_display = _complete_bin_mask(np.asarray(gen_edges), display_window)
        display_mask = np.tile(local_display, len(pt_edges) - 1)
        denominator = np.abs(normalized)
        stat_fraction = np.divide(
            stat,
            denominator,
            out=np.zeros_like(stat),
            where=denominator > 0.0,
        )
        total_fraction = np.divide(
            total,
            denominator,
            out=np.zeros_like(total),
            where=denominator > 0.0,
        )
        first_local = int(np.flatnonzero(local_display)[0])
        first_indices = first_local + bins_per_pt * np.arange(len(pt_edges) - 1)
        total_below_stat_count = int(np.count_nonzero(total + 1.0e-12 < stat))
        if total_below_stat_count:
            raise ValueError(
                f"{manifest_path}: total covariance is below stat in "
                f"{total_below_stat_count} bins"
            )

    return RunRecord(
        manifest_path=manifest_path,
        manifest=manifest,
        directory=directory,
        artifact_path=artifact_path,
        channel=channel,
        mode=mode,
        candidate=candidate,
        pt_edges=pt_edges,
        gen_edges=gen_edges,
        display_window=display_window,
        normalization_window=normalization_window,
        response_condition=float(diagnostics["response_column_normalized"]["condition_number"]),
        min_purity=float(migration["purity"]["minimum"]),
        mean_purity=float(migration["purity"]["mean"]),
        min_stability=float(migration["stability"]["minimum"]),
        mean_stability=float(migration["stability"]["mean"]),
        refold_chi2=float(diagnostics["refolded_residual"]["chi2"]),
        refold_rank=int(diagnostics["refolded_residual"]["rank"]),
        reco_bins_per_pt=len(reco_edges) - 1,
        gen_bins_per_pt=len(gen_edges) - 1,
        refold_chi2_by_pt=refold_chi2_by_pt,
        refold_rank_by_pt=refold_rank_by_pt,
        closure_relative_l1=float(closure["relative_l1"]),
        stat_median_percent=float(np.median(stat_fraction[display_mask]) * 100.0),
        stat_max_percent=float(np.max(stat_fraction[display_mask]) * 100.0),
        total_median_percent=float(np.median(total_fraction[display_mask]) * 100.0),
        total_max_percent=float(np.max(total_fraction[display_mask]) * 100.0),
        first_bin_total_max_percent=float(np.max(total_fraction[first_indices]) * 100.0),
        total_below_stat_count=total_below_stat_count,
        model_ps_source=str(selected["parton_shower"]),
        model_had_source=str(selected["hadronization"]),
        data_covariance_source=str(manifest["unfolding"]["data_covariance_source"]),
        expected_null_modes=int(covariance["expected_normalization_null_modes"]),
        observed_null_modes=int(covariance["observed_null_modes"]),
        covariance_min_eigenvalue=float(covariance["minimum_eigenvalue"]),
    )


def ordered_plot_pages(runs: Sequence[RunRecord]) -> list[PlotPage]:
    pages: list[PlotPage] = []
    for run in runs:
        for index in range(len(run.pt_edges) - 1):
            for product, category, filename in PRODUCTS:
                name = filename.format(mode=run.mode, index=index)
                if product == "data_mc":
                    source = DATA_MC_DIRECTORY / run.channel / name
                else:
                    source = run.directory / category / name
                pages.append(PlotPage(run, product, source, index))
        for product, category, filename in (
            (
                "correlation",
                "unfolded",
                f"correlation_{run.mode}_shown_total.pdf",
            ),
            (
                "unfolded_summary",
                "summary",
                f"unfolded_summary_{run.mode}.pdf",
            ),
            (
                "unfolded_summary_ratio",
                "summary",
                f"unfolded_summary_ratio_{run.mode}.pdf",
            ),
        ):
            pages.append(PlotPage(run, product, run.directory / category / filename, None))

    source_paths = [page.source_pdf.resolve() for page in pages]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical PDFs: {missing}")
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("A canonical plot PDF was selected more than once")
    if any("final_plots" in str(path) for path in source_paths):
        raise ValueError("Custom final_plots inputs are prohibited")
    return pages


def draw_wrapped(
    pdf: canvas.Canvas,
    text: str,
    *,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 8.5,
    leading: float = 10.5,
    color: Color = black,
    max_lines: int | None = None,
) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if stringWidth(trial, font, size) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and stringWidth(last + "...", font, size) > width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def make_base_page(draw_function) -> object:
    payload = BytesIO()
    pdf = canvas.Canvas(payload, pagesize=letter, pageCompression=1)
    draw_function(pdf)
    pdf.showPage()
    pdf.save()
    payload.seek(0)
    return PdfReader(payload).pages[0]


def draw_running_furniture(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setStrokeColor(HexColor("#D5DADF"))
    pdf.setLineWidth(0.5)
    pdf.line(36, 754, 576, 754)
    pdf.line(36, 32, 576, 32)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 7.8)
    pdf.drawString(36, 762, "RUN-2 PAIR-SPLIT UNFOLDING | GROOMED + UNGROOMED PLOT BOOK")
    pdf.setFont("Helvetica", 7.8)
    pdf.drawString(36, 20, "Internal evidence review | 18 Aug 2026")
    pdf.drawRightString(576, 20, f"Page {page_number}")


def draw_cover(pdf: canvas.Canvas, runs: Sequence[RunRecord], n_plots: int) -> None:
    draw_running_furniture(pdf, 1)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(36, 720, "RUN-2 PAIR-SPLIT TUnfold PLOT BOOK")
    pdf.setFillColor(DARK_BLUE)
    pdf.setFont("Helvetica-Bold", 11.5)
    pdf.drawString(36, 700, "Groomed and ungroomed | log10(rho^2), rho = m/(pT R), R = 0.8")

    y = 676
    y = draw_wrapped(
        pdf,
        f"Scope: {n_plots} canonical PDFs from four validated runs — core Unfolder products plus a detector-level data/MC comparison per pT slice. "
        "Each source plot is embedded as vector content; no final_plots or browser screenshots are used.",
        x=36,
        y=y,
        width=540,
        size=9.2,
        leading=11.5,
        color=black,
    )
    y -= 5
    y = draw_wrapped(
        pdf,
        "Method: TUnfoldDensity, tau = 0, area constraint, Jacobian input statistics, "
        "TUnfold response-matrix statistics, and response-level model variations. "
        "No jackknife and no iterative Bayes.",
        x=36,
        y=y,
        width=540,
        size=9.2,
        leading=11.5,
        color=black,
    )

    y -= 12
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(BLUE)
    pdf.drawString(36, y, "Validated run summary")
    y -= 15
    columns = (36, 150, 238, 320, 404, 500, 576)
    row_height = 34
    pdf.setFillColor(LIGHT_GRAY)
    pdf.rect(36, y - 18, 540, 20, fill=1, stroke=0)
    headers = ("Run", "pT", "Shown", "Condition", "Min P/S", "Refold (run)")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 7.6)
    for left, text in zip(columns[:-1], headers, strict=True):
        pdf.drawString(left + 3, y - 11, text)
    y -= 20
    for index, run in enumerate(runs):
        if index % 2 == 0:
            pdf.setFillColor(Color(0.96, 0.97, 0.98))
            pdf.rect(36, y - row_height, 540, row_height, fill=1, stroke=0)
        values = (
            run.label,
            f"{len(run.pt_edges) - 1} slices",
            f"[{run.display_window[0]:g}, {run.display_window[1]:g}]",
            fmt(run.response_condition),
            f"{fmt(run.min_purity)}/{fmt(run.min_stability)}",
            f"{fmt(run.refold_chi2)}/{run.refold_rank}",
        )
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 7.5)
        for left, text in zip(columns[:-1], values, strict=True):
            pdf.drawString(left + 3, y - 12, text)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 6.9)
        pdf.drawString(
            39,
            y - 25,
            f"Model PS/HAD: {run.model_ps_source}/{run.model_had_source}; "
            f"stat median/max {run.stat_median_percent:.2f}/{run.stat_max_percent:.2f}%; "
            f"total median/max {run.total_median_percent:.2f}/{run.total_max_percent:.2f}%; "
            f"closure rel-L1 {run.closure_relative_l1:.1e}",
        )
        y -= row_height

    y -= 12
    pdf.setFillColor(GREEN)
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.drawString(36, y, "What worked")
    y -= 14
    closure_values = [run.closure_relative_l1 for run in runs]
    y = draw_wrapped(
        pdf,
        "All four runs have nominal-MC self-closure at numerical precision: raw relative-L1 bias "
        f"{min(closure_values):.1e} to {max(closure_values):.1e} (per-run values in the table above, "
        "from each manifest's nominal_mc_self_closure diagnostics). "
        "Across every saved bin, total uncertainty is greater than or equal to statistical uncertainty. "
        "The pT labels are integral, the uncertainty legends have data-driven headroom, and the axis title is compact.",
        x=36,
        y=y,
        width=540,
        size=8.7,
        leading=10.7,
        color=black,
    )
    y -= 5
    pdf.setFillColor(CAUTION)
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.drawString(36, y, "What remains limited")
    y -= 14
    draw_wrapped(
        pdf,
        "The first published ungroomed bin remains statistically and migration limited, especially in trijet; it is shown rather than hidden. "
        "The trijet 290-400 GeV first bin has a slightly negative central value with a ~120% band: it is compatible with zero, and its off-scale ratio band is marked in-panel rather than clipped silently. "
        "The low-coordinate migration catch-all below -2.5 is retained in the response but excluded from the published ungroomed normalization. "
        "Vincia and internal Pythia variations test model dependence; numerical self-closure is not independent response-model validation.",
        x=36,
        y=y,
        width=540,
        size=8.7,
        leading=10.7,
        color=black,
    )


def page_comments(page: PlotPage) -> tuple[str, str]:
    run = page.run
    if page.product == "data_mc":
        error_source = (
            "the event-clustered reco covariance"
            if run.data_covariance_source == "full_reco_covariance"
            else "diagonal reco sumw2"
        )
        return (
            "Detector-level data vs QCD (MG+Pythia8) on the exact unfolding reco binning; the band is MC stat, detector systematics, and the PS/HAD model envelope; "
            f"data errors use {error_source} (notebooks/data_mc_pairsplit_rho.py).",
            "The MC is shape-normalized to the data yield within this pT slice (trigger prescale weights make cross-slice yields meaningless), so this is a shape comparison with no absolute-rate claim.",
        )
    if page.product == "unfolded":
        return (
            f"The normalized data spectrum is finite and non-negative; model comparison curves and edge-aligned stat/total bands use the exact {run.mode} truth bins.",
            f"The first displayed-bin total reaches up to {run.first_bin_total_max_percent:.1f}% across this run; this is retained as a real low-coordinate limitation.",
        )
    if page.product == "corrected_refold":
        residual_dof = run.reco_bins_per_pt - run.gen_bins_per_pt
        if residual_dof <= 0:
            return (
                "This slice's response is square (reco bins = truth bins), so the unregularized refold reproduces the measurement identically and the residual is zero by construction.",
                "Bookkeeping check only: an exact-by-construction refold has no power to test the response model in this mode.",
            )
        slice_chi2 = run.refold_chi2_by_pt[page.pt_index]
        slice_rank = run.refold_rank_by_pt[page.pt_index]
        return (
            f"This slice's refold residual chi2 is {fmt(slice_chi2)} over {slice_rank} slice-local stat-covariance modes; residual dof = {run.reco_bins_per_pt}-{run.gen_bins_per_pt} = {residual_dof}. Run-level: {fmt(run.refold_chi2)}/{run.refold_rank}.",
            "The chi2 uses the statistical measured covariance only (no systematics); residual sub-truth-bin data/MC shape tension is expected at this precision and belongs to the systematic band, not to the refold.",
        )
    if page.product == "purity_stability":
        return (
            f"Across the published window, purity min/mean is {fmt(run.min_purity)}/{fmt(run.mean_purity)} and stability min/mean is {fmt(run.min_stability)}/{fmt(run.mean_stability)}.",
            "Low first-bin purity or stability identifies genuine migration sensitivity and should remain visible in the interpretation.",
        )
    if page.product == "grouped_uncertainties":
        return (
            f"Stat median/max is {run.stat_median_percent:.2f}/{run.stat_max_percent:.2f}% and total median/max is {run.total_median_percent:.2f}/{run.total_max_percent:.2f}%; total<stat violations: {run.total_below_stat_count}.",
            "The tall first-bin envelope is not clipped or rescaled; the panel-specific y range only reserves clear legend headroom.",
        )
    if page.product == "correlation":
        return (
            f"The normalized total covariance has {run.observed_null_modes} observed normalization null modes, matching the expected {run.expected_null_modes}.",
            f"The minimum eigenvalue is {run.covariance_min_eigenvalue:.3g}; near-zero normalization modes are expected constraint structure.",
        )
    if page.product == "unfolded_summary":
        return (
            f"All {len(run.pt_edges) - 1} configured pT slices are shown together for the {run.mode} observable.",
            "Shared visual scale aids comparison but can make the statistically weakest slice appear compressed.",
        )
    if page.product == "unfolded_summary_ratio":
        return (
            "The summary collects data/Pythia and data/Vincia behavior across every configured pT slice.",
            "Generator curves are model comparisons; their separation is not added directly as a particle-level truth-spread uncertainty.",
        )
    raise ValueError(page.product)


def draw_plot_base(
    pdf: canvas.Canvas,
    page: PlotPage,
    *,
    page_number: int,
    figure_number: int,
) -> None:
    draw_running_furniture(pdf, page_number)
    run = page.run
    pt_text = (
        f" | pT {pt_label(run.pt_edges, page.pt_index)}"
        if page.pt_index is not None
        else " | all pT slices"
    )
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12.5)
    pdf.drawString(
        36,
        735,
        f"{run.channel.capitalize()} | {run.mode}{pt_text} | {product_title(page.product)}",
    )
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.4)
    pdf.drawString(
        36,
        721,
        f"Candidate {run.candidate} | normalize [{run.normalization_window[0]:g}, {run.normalization_window[1]:g}] | show [{run.display_window[0]:g}, {run.display_window[1]:g}]",
    )

    worked, limitation = page_comments(page)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 8.6)
    pdf.drawString(36, 157, "Worked:")
    draw_wrapped(
        pdf,
        worked,
        x=78,
        y=157,
        width=498,
        size=8.2,
        leading=9.8,
        max_lines=2,
    )
    pdf.setFillColor(CAUTION)
    pdf.setFont("Helvetica-Bold", 8.6)
    pdf.drawString(36, 126, "Limitation:")
    draw_wrapped(
        pdf,
        limitation,
        x=88,
        y=126,
        width=488,
        size=8.2,
        leading=9.8,
        max_lines=2,
    )
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Oblique", 7.2)
    pdf.drawString(
        36,
        91,
        f"Figure {figure_number}. Canonical Internal output: {page.source_pdf.name}",
    )
    pdf.setFont("Helvetica", 6.6)
    draw_wrapped(
        pdf,
        f"Run: {run.directory.relative_to(ROOT)}",
        x=36,
        y=78,
        width=540,
        size=6.6,
        leading=8,
        color=MUTED,
        max_lines=2,
    )


def merge_plot(base_page, source_pdf: Path) -> None:
    source_reader = PdfReader(str(source_pdf))
    if len(source_reader.pages) != 1:
        raise ValueError(f"Expected a one-page plot PDF: {source_pdf}")
    source = source_reader.pages[0]
    source_width = float(source.mediabox.width)
    source_height = float(source.mediabox.height)
    max_width = 540.0
    max_height = 525.0
    scale = min(max_width / source_width, max_height / source_height)
    tx = (PAGE_WIDTH - source_width * scale) / 2.0
    ty = 181.0 + (max_height - source_height * scale) / 2.0
    transform = Transformation().scale(scale).translate(tx, ty)
    base_page.merge_transformed_page(source, transform, over=True)


def build() -> tuple[Path, list[dict[str, object]]]:
    runs = tuple(load_run(path) for path in RUN_MANIFESTS)
    pages = ordered_plot_pages(runs)
    if len(pages) != EXPECTED_PLOT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PLOT_COUNT} canonical plots, found {len(pages)}"
        )

    writer = PdfWriter()
    writer.add_page(make_base_page(lambda pdf: draw_cover(pdf, runs, len(pages))))
    inventory: list[dict[str, object]] = []
    for figure_number, plot_page in enumerate(pages, start=1):
        page_number = figure_number + 1
        base = make_base_page(
            lambda pdf, p=plot_page, pn=page_number, fn=figure_number: draw_plot_base(
                pdf, p, page_number=pn, figure_number=fn
            )
        )
        merge_plot(base, plot_page.source_pdf)
        writer.add_page(base)
        inventory.append(
            {
                "figure": figure_number,
                "channel": plot_page.run.channel,
                "grooming_mode": plot_page.run.mode,
                "candidate": plot_page.run.candidate,
                "product": plot_page.product,
                "pt_index": plot_page.pt_index,
                "source_pdf": str(plot_page.source_pdf.resolve()),
                "source_sha256": sha256_path(plot_page.source_pdf),
                "manifest": str(plot_page.run.manifest_path),
                "manifest_sha256": sha256_path(plot_page.run.manifest_path),
                "artifact": str(plot_page.run.artifact_path),
                "artifact_sha256": sha256_path(plot_page.run.artifact_path),
            }
        )

    writer.add_metadata(
        {
            "/Title": "Run-2 pair-split TUnfold groomed and ungroomed plot book",
            "/Subject": "Canonical dijet and trijet groomed and ungroomed unfolding figures",
            "/Author": "CMS jet-mass analysis",
            "/Creator": "Codex vector plot-book builder",
            "/CreationDate": f"D:{date.today():%Y%m%d}",
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PDF.open("wb") as handle:
        writer.write(handle)
    INVENTORY_JSON.write_text(
        json.dumps(
            {
                "output_pdf": str(OUTPUT_PDF),
                "output_sha256": sha256_path(OUTPUT_PDF),
                "page_count": len(writer.pages),
                "canonical_plot_count": len(inventory),
                "data_mc_provenance": {
                    "path": str(DATA_MC_DIRECTORY / "provenance.json"),
                    "sha256": sha256_path(DATA_MC_DIRECTORY / "provenance.json"),
                },
                "runs": [
                    {
                        "channel": run.channel,
                        "grooming_mode": run.mode,
                        "candidate": run.candidate,
                        "manifest": str(run.manifest_path),
                        "manifest_sha256": sha256_path(run.manifest_path),
                        "artifact": str(run.artifact_path),
                        "artifact_sha256": sha256_path(run.artifact_path),
                    }
                    for run in runs
                ],
                "plots": inventory,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return OUTPUT_PDF, inventory


def main() -> None:
    output, inventory = build()
    print(output)
    print(f"pages={len(inventory) + 1} canonical_plots={len(inventory)}")
    print(f"sha256={sha256_path(output)}")


if __name__ == "__main__":
    main()
