#!/usr/bin/env python3
"""Build a 16:9 Typst slide deck of the pair-split results, product-centric.

One slide per product per run, showing ALL pT slices of that product together
(dijet 3+2 grid, trijet one row of 3): detector-level data/MC, unfolded
results, forward-folding checks, purity/stability, uncertainties, and the
Z+jet-parity validation suite, then the run-level summaries.  Reuses the
plot-book builder's run records and metrics as the single source of truth and
renders each canonical PDF to PNG.  The hash-inventoried plot book remains
the delivery; this deck is its meeting companion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "pairsplit_plot_book", ROOT / "scripts" / "plotting" / "build_pairsplit_all_modes_plot_book.py"
)
plot_book = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = plot_book
_spec.loader.exec_module(plot_book)

DECK_DATE = "2026-08-27"
OUTPUT_ROOT = ROOT / "outputs" / "pairsplit_run2"
BUILD_DIR = OUTPUT_ROOT / "slides_build"
FIG_DIR = BUILD_DIR / "figs"
TYP_PATH = BUILD_DIR / "pairsplit_slides.typ"
OUTPUT_PDF = OUTPUT_ROOT / f"PAIR_SPLIT_GROOMED_UNGROOMED_SLIDES_{DECK_DATE}.pdf"
PNG_DPI = 200

# Per-pT products, one slide each per run, in presentation order.  Book
# products resolve through the book builder's page map; check products live
# in the run directory per the core's categorized layout.
BOOK_PRODUCTS = {
    "data_mc", "unfolded", "corrected_refold", "purity_stability",
    "grouped_uncertainties",
}
PER_PT_PRODUCT_ORDER = (
    ("data_mc", "Detector-level data/MC"),
    ("unfolded", "Unfolded results"),
    ("corrected_refold", "Forward-folding checks"),
    ("closure", "Nominal-MC self-closure"),
    ("purity_stability", "Purity and stability"),
    ("fakes_misses", "Fake and miss rates"),
    ("grouped_uncertainties", "Grouped uncertainties"),
    ("stat_fraction", "Statistical components"),
    ("model_envelope", "Model envelope"),
    ("bottom_line", "Bottom-line test"),
    ("bottom_line_rebinned", "Bottom-line test (square K)"),
    ("heatmap", "Uncertainty heatmaps"),
)
CHECK_PATHS = {
    "closure": ("closure/validation", "closure_{mode}_pt{i}.pdf"),
    "model_envelope": ("uncertainties", "model_envelope_{mode}_pt{i}.pdf"),
    "fakes_misses": ("response", "fakes_misses_{mode}_pt{i}.pdf"),
    "stat_fraction": ("uncertainties", "stat_fraction_{mode}_pt{i}.pdf"),
    "bottom_line": ("bottom_line", "bottom_line_{mode}_pt{i}.pdf"),
    "bottom_line_rebinned": ("bottom_line", "bottom_line_rebinned_{mode}_pt{i}.pdf"),
    "heatmap": ("uncertainties", "heatmap_{mode}_pt{i}.pdf"),
    "response_matrix": ("response", "response_matrix_{mode}.pdf"),
    "correlation_shown": ("unfolded", "correlation_{mode}_shown.pdf"),
    "blt_chi2_perndf_shown": ("summary", "bottom_line_chi2_perndf_summary_shown_{mode}.pdf"),
    "blt_chi2_shown": ("summary", "bottom_line_chi2_summary_shown_{mode}.pdf"),
}
RUN_LEVEL_GROUPS = (
    (("unfolded_summary", "unfolded_summary_ratio"), "Unfolded summaries"),
    (("correlation", "correlation_shown"), "Correlation matrices (total | stat-only)"),
    (("response_matrix",), "Response matrix"),
    (("blt_chi2_perndf_shown", "blt_chi2_shown"), "Bottom-line chi2 summaries"),
)


def _escape(text: str) -> str:
    for char in ("\\", "#", "$", "*", "_", "@", "<", ">"):
        text = text.replace(char, "\\" + char)
    return text


def source_pdf(run, product, pt_index, book_pages) -> Path:
    if product in BOOK_PRODUCTS or product in {
        "correlation", "unfolded_summary", "unfolded_summary_ratio"
    }:
        return book_pages[(run.channel, run.mode, product, pt_index)].source_pdf
    category, template = CHECK_PATHS[product]
    return run.directory / category / template.format(mode=run.mode, i=pt_index)


def render_png(source: Path, name: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Missing canonical figure: {source}")
    target = FIG_DIR / name
    png = target.with_suffix(".png")
    if not png.exists() or png.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(PNG_DPI), "-singlefile",
             str(source), str(target)],
            check=True,
        )
    return f"figs/{name}.png"


def slide_caption(run, product) -> str:
    fmt = plot_book.fmt
    square = run.reco_bins_per_pt == run.gen_bins_per_pt
    return {
        "data_mc": (
            "Data vs QCD (MG+Pythia8) on the unfolding reco binning; MC shape-"
            "normalized to the data yield within each pT slice (trigger prescale); "
            "band = MC stat + detector systematics + PS/HAD model envelope."
        ),
        "unfolded": (
            "Normalized unfolded data vs Pythia8 and Vincia; the first "
            f"displayed-bin total reaches {run.first_bin_total_max_percent:.1f}% "
            "across this run — a real low-coordinate limitation, shown not hidden."
        ),
        "corrected_refold": (
            "Fake-corrected measured data vs nominal refold (stat-only covariance). "
            + ("The per-slice response is square, so the refold is exact by "
               "construction: a bookkeeping check with no model-testing power."
               if square else
               f"Run-level residual chi2/rank {fmt(run.refold_chi2)}/{run.refold_rank}; "
               "residual sub-truth-bin shape tension belongs to the systematic band.")
        ),
        "closure": (
            "Nominal MC unfolded through the identical prepared TUnfold path, "
            f"stat-only band; manifest raw relative-L1 bias {run.closure_relative_l1:.1e}."
        ),
        "purity_stability": (
            f"Published-window purity min/mean {fmt(run.min_purity)}/{fmt(run.mean_purity)}, "
            f"stability {fmt(run.min_stability)}/{fmt(run.mean_stability)}; low first-bin "
            "values flag genuine migration sensitivity."
        ),
        "fakes_misses": (
            "Fake and miss rates from the prepared response; per-systematic fakes "
            "and misses feed the fake-survival correction."
        ),
        "grouped_uncertainties": (
            f"Stat median/max {run.stat_median_percent:.2f}/{run.stat_max_percent:.2f}%, "
            f"total {run.total_median_percent:.2f}/{run.total_max_percent:.2f}%; "
            "total < stat violations: 0."
        ),
        "stat_fraction": (
            "Input-statistics vs response-matrix-statistics components "
            "(GetEmatrixInput / GetEmatrixSysUncorr through the normalization Jacobian)."
        ),
        "model_envelope": (
            (
                "Diagnostic binwise envelopes: PS = max(Vincia, FSR); HAD = "
                "max(CR1, CR2, frag-hard, frag-soft). The uncertainty band uses "
                "the diagonals of the two enclosing-template covariances."
                if run.model_ps_source == "enclosing templates"
                else "Two-leg composition: PS = max(Vincia, FSR); HAD = max(CR1, CR2, "
                "frag-hard, frag-soft); the black total enters the uncertainty band. "
                f"Selected covariance sources: PS={run.model_ps_source}, HAD={run.model_had_source}."
            )
        ),
        "bottom_line": (
            "Bottom-line diagnostic on the native reco binning; the MC side is "
            "scaled to the data yield per pT slice (prescaled data), so both chi2 "
            "sides are shape comparisons; the raw-chi2 inequality is the criterion."
        ),
        "bottom_line_rebinned": (
            "Square-K variant: detector level rebinned to the truth binning so "
            "chi2 smeared is directly comparable to chi2 unfolded."
        ),
        "heatmap": "Per-bin uncertainty composition per pT slice.",
        "unfolded_summary": (
            "All pT slices together (offset by powers of ten) and the per-slice "
            "data/theory ratio strips."
        ),
        "correlation": (
            f"Normalized-result correlations over the shown bins; {run.observed_null_modes} "
            f"observed normalization null modes match the expected {run.expected_null_modes}."
        ),
        "response_matrix": (
            "Column-normalized (probability) response; condition number "
            f"{fmt(run.response_condition)}."
        ),
        "blt_chi2_perndf_shown": (
            "Shown-space bottom-line chi2 per pT slice: chi2/ndf (left) and raw "
            "chi2 (right); the unfolded bar must not exceed the smeared bar."
        ),
    }[product]


def slide(body: str) -> str:
    return "#page[\n" + body + "]\n"


def grid_slide(title: str, caption: str, images: list[str], height_cm: float,
               columns: int) -> str:
    # Cap BOTH dimensions with fit:"contain": height-only scaling let wide
    # panels (uncertainty heatmaps, whose reco axes grew with the aligned
    # binning) overflow the page width and clip their own margins.
    usable_width_cm = 29.70 - 2.0  # presentation-16-9 minus x margins
    width_cm = (usable_width_cm - 0.3 * (columns - 1)) / columns
    cells = "".join(
        "  align(center + horizon)[#box("
        f"width: {width_cm:.2f}cm, height: {height_cm}cm)"
        f"[#image(\"{png}\", width: 100%, height: 100%, fit: \"contain\")]],\n"
        for png in images
    )
    column_spec = ", ".join(["1fr"] * columns)
    return slide(
        f"  #text(size: 15pt, weight: \"bold\", fill: rgb(\"#0B2545\"))[{_escape(title)}]\n"
        "  #v(0.15cm)\n"
        f"  #block(breakable: false)[#grid(columns: ({column_spec}), "
        "column-gutter: 0.3cm, row-gutter: 0.15cm,\n"
        + cells
        + "  )]\n"
        "  #v(0.1cm)\n"
        f"  #text(size: 8pt, fill: rgb(\"#5B6572\"))[{_escape(caption)}]\n"
    )


def run_title(run) -> str:
    return f"{run.channel.capitalize()} {run.mode}"


def divider_slide(run) -> str:
    facts = [
        f"Candidate: {run.candidate} | {len(run.pt_edges) - 1} pT slices",
        (
            f"Normalize [{run.normalization_window[0]:g}, {run.normalization_window[1]:g}] | "
            f"show [{run.display_window[0]:g}, {run.display_window[1]:g}]"
        ),
        (
            f"Min purity/stability {plot_book.fmt(run.min_purity)}/"
            f"{plot_book.fmt(run.min_stability)} | response condition "
            f"{plot_book.fmt(run.response_condition)}"
        ),
        (
            f"Refold chi2/rank (run, stat-only) {plot_book.fmt(run.refold_chi2)}/"
            f"{run.refold_rank} | closure rel-L1 {run.closure_relative_l1:.1e}"
        ),
        (
            f"Model PS/HAD: {run.model_ps_source}/{run.model_had_source} | "
            f"data errors: {run.data_covariance_source.replace('_', ' ')}"
        ),
        (
            f"Stat median/max {run.stat_median_percent:.2f}/{run.stat_max_percent:.2f}% | "
            f"total {run.total_median_percent:.2f}/{run.total_max_percent:.2f}%"
        ),
    ]
    return slide(
        "  #v(4cm)\n"
        f"  #align(center)[#text(size: 26pt, weight: \"bold\", "
        f"fill: rgb(\"#1F4D78\"))[{run_title(run)}]]\n"
        "  #v(0.6cm)\n"
        "  #align(center)[#text(size: 11pt)[\n"
        + "".join(f"    {_escape(line)} \\\n" for line in facts)
        + "  ]]\n"
    )


def cover_slide(runs) -> str:
    lines = [
        "Run-2 pair-split TUnfold: dijet + trijet, groomed + ungroomed, "
        "log10(rho^2) with rho = m/(pT R), R = 0.8.",
    ]
    return slide(
        "  #v(3cm)\n"
        "  #align(center)[#text(size: 28pt, weight: \"bold\", "
        "fill: rgb(\"#0B2545\"))[Run-2 pair-split unfolding]]\n"
        "  #align(center)[#text(size: 16pt, fill: rgb(\"#1F4D78\"))"
        f"[Groomed and ungroomed plot review | {DECK_DATE}]]\n"
        "  #v(0.8cm)\n"
        + "".join(
            f"  #align(center)[#text(size: 11pt)[{_escape(line)}]]\n  #v(0.15cm)\n"
            for line in lines
        )
    )


def build() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = tuple(plot_book.load_run(path) for path in plot_book.RUN_MANIFESTS)
    book_pages = {
        (page.run.channel, page.run.mode, page.product, page.pt_index): page
        for page in plot_book.ordered_plot_pages(runs)
    }

    slides = [cover_slide(runs)]
    n_figures = 0
    for run in runs:
        slides.append(divider_slide(run))
        n_pt = len(run.pt_edges) - 1
        for product, label in PER_PT_PRODUCT_ORDER:
            images = []
            for index in range(n_pt):
                source = source_pdf(run, product, index, book_pages)
                images.append(render_png(
                    source, f"{run.channel}_{run.mode}_{product}_pt{index}"
                ))
            n_figures += len(images)
            # 3-column grid: <=3 panels sit in one tall row, 4-5 wrap to 3+2.
            height = 9.0 if n_pt <= 3 else 6.4
            slides.append(grid_slide(
                f"{run_title(run)} | {label}",
                slide_caption(run, product),
                images,
                height_cm=height,
                columns=min(3, n_pt),
            ))
        for group, label in RUN_LEVEL_GROUPS:
            images = [
                render_png(
                    source_pdf(run, product, None, book_pages),
                    f"{run.channel}_{run.mode}_{product}",
                )
                for product in group
            ]
            n_figures += len(images)
            slides.append(grid_slide(
                f"{run_title(run)} | {label}",
                slide_caption(run, group[0]),
                images,
                height_cm=10.6 if len(images) == 2 else 11.4,
                columns=len(images),
            ))

    # Combined three-channel comparison on the aligned common grid — the
    # payoff of the 2026-08-27 binning alignment.
    combined_dir = plot_book.OUTPUT_ROOT / "combined"
    combined_caption = (
        "All three channels rebinned EXACTLY onto the common gen grid and "
        "re-normalized to unit area over the shown window per pT slice, so "
        "per-run normalization-window choices drop out. Bands: total "
        "(stat + detector quadrature + PS/HAD model legs; zjet uses its "
        "published total covariance). Lower panels: channel/dijet with the "
        "dijet total band hatched around 1; channels treated as uncorrelated. "
        "Common pT slices 200-290, 290-400, >400 GeV (dijet's three high-pT "
        "slices merged at the count level with full stat covariance)."
    )
    for mode in ("groomed", "ungroomed"):
        images = []
        for index in range(3):
            source = combined_dir / f"combined_{mode}_pt{index}.pdf"
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing combined figure: {source} — run "
                    "scripts/plotting/build_combined_channels_rho.py first"
                )
            images.append(render_png(source, f"combined_{mode}_pt{index}"))
        n_figures += len(images)
        slides.append(grid_slide(
            f"Combined channels | Z+jet vs dijet vs trijet, {mode}",
            combined_caption,
            images,
            height_cm=9.0,
            columns=3,
        ))

    header = (
        "#set page(paper: \"presentation-16-9\", margin: (x: 1.0cm, y: 0.7cm),\n"
        "  footer: context [#text(size: 8pt, fill: rgb(\"#5B6572\"))"
        "[Run-2 pair-split unfolding | " + DECK_DATE + " "
        "#h(1fr) #counter(page).display()]])\n"
        "#set text(font: \"Helvetica\", size: 11pt)\n"
    )
    TYP_PATH.write_text(header + "".join(slides), encoding="utf-8")
    subprocess.run(
        ["typst", "compile", str(TYP_PATH), str(OUTPUT_PDF)],
        check=True, cwd=BUILD_DIR,
    )
    print(f"{OUTPUT_PDF}\nslides={len(slides)} figures={n_figures}")


if __name__ == "__main__":
    build()
