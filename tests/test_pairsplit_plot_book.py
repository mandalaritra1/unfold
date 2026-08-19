"""Caption regressions for the pair-split plot-book builder.

The 2026-08-15 review found the refold pages stamped the run-level residual
chi2/rank on every per-pT page.  These tests pin the corrected behavior:
per-slice values on per-slice pages, and an explicit exact-by-construction
statement when the per-slice response is square.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plotting"
    / "build_pairsplit_all_modes_plot_book.py"
)
spec = importlib.util.spec_from_file_location("pairsplit_plot_book", MODULE_PATH)
plot_book = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot_book
spec.loader.exec_module(plot_book)


def _run_record(*, reco_bins_per_pt, gen_bins_per_pt, refold_chi2_by_pt, refold_rank_by_pt):
    return plot_book.RunRecord(
        manifest_path=Path("run_manifest.json"),
        manifest={},
        directory=Path("."),
        artifact_path=Path("artifacts/groomed_results.npz"),
        channel="dijet",
        mode="groomed",
        candidate="coarse_tail",
        pt_edges=(200.0, 290.0, 400.0),
        gen_edges=tuple(float(v) for v in range(gen_bins_per_pt + 1)),
        display_window=(-4.0, 0.0),
        normalization_window=(-1.8, -0.55),
        response_condition=8.0,
        min_purity=0.4,
        mean_purity=0.5,
        min_stability=0.4,
        mean_stability=0.5,
        refold_chi2=4825.0,
        refold_rank=65,
        reco_bins_per_pt=reco_bins_per_pt,
        gen_bins_per_pt=gen_bins_per_pt,
        refold_chi2_by_pt=refold_chi2_by_pt,
        refold_rank_by_pt=refold_rank_by_pt,
        closure_relative_l1=5.0e-15,
        stat_median_percent=1.0,
        stat_max_percent=2.0,
        total_median_percent=3.0,
        total_max_percent=4.0,
        first_bin_total_max_percent=5.0,
        total_below_stat_count=0,
        model_ps_source="vincia",
        model_had_source="cr1",
        data_covariance_source="full_reco_covariance",
        expected_null_modes=2,
        observed_null_modes=2,
        covariance_min_eigenvalue=1.0e-12,
    )


def test_refold_caption_is_per_slice_and_labels_the_run_level_value():
    run = _run_record(
        reco_bins_per_pt=13,
        gen_bins_per_pt=11,
        refold_chi2_by_pt=(101.0, 202.0),
        refold_rank_by_pt=(13, 13),
    )
    captions = [
        plot_book.page_comments(
            plot_book.PlotPage(run, "corrected_refold", Path("a.pdf"), index)
        )
        for index in (0, 1)
    ]
    assert captions[0] != captions[1]
    assert "101" in captions[0][0] and "202" not in captions[0][0]
    assert "202" in captions[1][0] and "101" not in captions[1][0]
    for worked, limitation in captions:
        assert "Run-level" in worked
        assert "residual dof = 13-11 = 2" in worked
        assert "statistical measured covariance only" in limitation


def test_square_response_refold_caption_states_exact_by_construction():
    run = _run_record(
        reco_bins_per_pt=6,
        gen_bins_per_pt=6,
        refold_chi2_by_pt=(0.0, 0.0),
        refold_rank_by_pt=(6, 6),
    )
    worked, limitation = plot_book.page_comments(
        plot_book.PlotPage(run, "corrected_refold", Path("a.pdf"), 0)
    )
    assert "zero by construction" in worked
    assert "no power" in limitation
    assert "chi2" not in worked.lower()  # no numeric chi2 quoted for an exact refold
    assert "4825" not in worked


def test_per_pt_refold_chi2_matches_full_chi2_for_block_diagonal_covariance():
    rng = np.random.default_rng(7)
    n_slices, reco_bins = 3, 4
    blocks = []
    for _ in range(n_slices):
        basis = rng.normal(size=(reco_bins, reco_bins))
        blocks.append(basis @ basis.T + reco_bins * np.eye(reco_bins))
    covariance = np.zeros((n_slices * reco_bins, n_slices * reco_bins))
    for index, block in enumerate(blocks):
        sel = slice(index * reco_bins, (index + 1) * reco_bins)
        covariance[sel, sel] = block
    residual = rng.normal(size=n_slices * reco_bins)
    chi2_by_pt, rank_by_pt = plot_book.per_pt_refold_chi2(
        residual, covariance, n_slices, reco_bins
    )
    expected_total = float(residual @ np.linalg.inv(covariance) @ residual)
    assert rank_by_pt == (reco_bins,) * n_slices
    assert np.isclose(sum(chi2_by_pt), expected_total, rtol=1.0e-10)
    with pytest.raises(ValueError):
        plot_book.per_pt_refold_chi2(residual[:-1], covariance, n_slices, reco_bins)


def test_data_mc_caption_states_shape_normalization_and_error_source():
    run = _run_record(
        reco_bins_per_pt=13,
        gen_bins_per_pt=11,
        refold_chi2_by_pt=(1.0, 2.0),
        refold_rank_by_pt=(13, 13),
    )
    worked, limitation = plot_book.page_comments(
        plot_book.PlotPage(run, "data_mc", Path("a.pdf"), 0)
    )
    assert "event-clustered reco covariance" in worked
    assert "shape-normalized to the data yield" in limitation
    assert "no absolute-rate claim" in limitation
