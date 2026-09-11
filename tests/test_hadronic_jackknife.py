"""Selection, label alignment, normalization, and consumer covariance checks."""

from pathlib import Path

import hist
import numpy as np
import pytest

from unfold.cli import build_parser, resolve_output_dir
from unfold.config import HadronicTag
from unfold.hadronic.jackknife import (
    covariance, full_sample, extract_histogram, load_jackknife_inputs, check_data_sample,
)
from unfold.hadronic.inputs import HADRONIC_FINE_AXES
from unfold.hadronic.run import HadronicOptions


def test_cli_default_and_explicit_analytic():
    parser = build_parser()
    args = parser.parse_args(["run", "--channel", "dijet"])
    assert args.stat_method is None
    assert HadronicOptions().stat_method == "jackknife"
    assert HadronicTag("dijet", "outputs/test").stat_method == "jackknife"
    assert parser.parse_args(["run", "--channel", "trijet", "--stat-method", "analytic"]).stat_method == "analytic"


def test_explicit_method_preserves_default_output_directory():
    parser = build_parser()
    tag = HadronicTag("dijet", "outputs/dijet/rho/original/")
    default = parser.parse_args(["run", "--channel", "dijet"])
    analytic = parser.parse_args(["run", "--channel", "dijet", "--stat-method", "analytic"])
    assert resolve_output_dir(tag, default).name == "original"
    assert resolve_output_dir(tag, analytic).name == "original_stat_analytic"
    analytic.output_dir = "/tmp/explicit-unfold-output"
    assert resolve_output_dir(tag, analytic) == Path(analytic.output_dir).resolve()


def test_missing_and_partial_campaign_fall_back(tmp_path):
    result, metadata = load_jackknife_inputs(tmp_path, "dijet", "groomed")
    assert result is None and metadata["resolved"] == "analytic"
    assert len(metadata["missing_files"]) == 8
    path = tmp_path / "data/rho_jk_dijet_data_2016APV.pkl"
    path.parent.mkdir()
    path.touch()
    result, metadata = load_jackknife_inputs(tmp_path, "dijet", "groomed")
    assert result is None and len(metadata["missing_files"]) == 7


def test_analytic_does_not_read_replica_files(tmp_path):
    (tmp_path / "manifest.json").write_text("invalid json")
    result, metadata = load_jackknife_inputs(tmp_path, "dijet", "groomed", requested="analytic")
    assert result is None and "fallback_reason" not in metadata


def test_complete_but_corrupt_campaign_errors(tmp_path):
    for kind, sample in (("data", "data"), ("mc", "mg_pythia8")):
        (tmp_path / kind).mkdir()
        for era in ("2016APV", "2016", "2017", "2018"):
            (tmp_path / kind / f"rho_jk_dijet_{sample}_{era}.pkl").write_bytes(b"broken")
    with pytest.raises(Exception):
        load_jackknife_inputs(tmp_path, "dijet", "groomed")


def test_label_order_and_sumw2_reconstruction():
    fine = HADRONIC_FINE_AXES["groomed"]
    order = [9, 3, 7, 0, 2, 8, 4, 1, 6, 5]
    h = hist.Hist(hist.axis.IntCategory(order, name="jk"),
                  hist.axis.StrCategory(["nominal"], name="systematic"),
                  hist.axis.Variable(fine.pt_edges, name="ptreco"),
                  hist.axis.Variable(fine.two_log10_rho_reco_edges, name="mpt_reco"),
                  storage=hist.storage.Weight())
    for label in order:
        index = h.axes["jk"].index(label)
        h.values()[index] = 55 - (label + 1)
        h.variances()[index] = 385 - (label + 1)**2
    values, variances = extract_histogram(h, "reco", "groomed")
    np.testing.assert_array_equal(values[:, 0, 0], 55 - np.arange(1, 11))
    nominal = full_sample({"reco": values, "reco_variance": variances})
    np.testing.assert_allclose(nominal["reco"], 55)
    np.testing.assert_allclose(nominal["reco_variance"], 385)


def test_covariance_equals_independent_group_sum_estimator():
    groups = np.array([[3., 1.], [2., 4.], [7., 2.], [5., 3.]])
    g = len(groups)
    estimates = (groups.sum(axis=0) - groups) * g / (g - 1)
    # Unbiased covariance of a sum of independent, equally distributed groups.
    np.testing.assert_allclose(covariance(estimates), g * np.cov(groups, rowvar=False, ddof=1))
    with pytest.raises(ValueError):
        covariance([[1, 2]])


def test_normalization_null_and_independent_components():
    rng = np.random.default_rng(20260911)
    widths = np.array([.5, 1., 2.])
    data = rng.uniform(1, 5, size=(10, 3))
    mc = rng.uniform(1, 5, size=(10, 3))
    normalize = lambda x: x / x.sum(axis=1)[:, None] / widths
    total = covariance(normalize(data)) + covariance(normalize(mc))
    np.testing.assert_allclose(total @ widths, 0, atol=1e-15)
    assert np.linalg.eigvalsh(total).min() > -1e-15
    assert not np.allclose(total, covariance(normalize(data) + normalize(mc)))


def test_mismatched_data_is_not_silently_accepted():
    check_data_sample(np.array([10., 20.]), np.array([10., 20.]))
    with pytest.raises(ValueError, match="samples differ"):
        check_data_sample(np.array([10., 20.]), np.array([11., 22.]))


def test_engine_uses_exact_normalized_replica_covariance():
    from unfold.engine import Unfolder
    fake = Unfolder.__new__(Unfolder)
    a = np.array([[.1, -.1], [-.1, .1]])
    b = a * 2
    fake.jackknife_normalized_covariances = (a, b)
    fake.gen_edges_by_pt = [[0, 1, 2]]
    fake.normalized_results = [{}]
    # No y_unf/analytic covariance: invoking the analytic Jacobian would fail.
    fake._compute_normalized_stat_covariance()
    np.testing.assert_array_equal(fake.norm_cov_stat, a + b)
    np.testing.assert_allclose(fake.normalized_results[0]["unfolded_err"], np.sqrt(np.diag(a + b)))
    fake.stat_propagation = "jacobian"
    np.testing.assert_array_equal(fake._correlation_covariance("stat"), a + b)
