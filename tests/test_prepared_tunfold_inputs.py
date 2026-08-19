import inspect
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import hist
import numpy as np
import pytest

from unfold.tools.unfolder_core import RHO_FIXED_JEC_SPEC, Unfolder
from unfold.tools import unfolder_core


PT_EDGES = [0.0, 200.0, 400.0]
RHO_EDGES = [-3.0, -2.0, -1.0, 0.0]
SYSTEMATICS = ["nominal", "shiftUp"]
KEYS = {
    "response": "response_matrix_rho_g",
    "reco": "ptjet_rhojet_g_reco",
    "gen": "ptjet_rhojet_g_gen",
}


def _weighted_histogram(*axes):
    return hist.Hist(*axes, storage=hist.storage.Weight())


def _response_histogram():
    output = _weighted_histogram(
        hist.axis.StrCategory(["sample"], name="dataset"),
        hist.axis.StrCategory(SYSTEMATICS, name="systematic"),
        hist.axis.Variable(PT_EDGES, name="ptreco"),
        hist.axis.Variable(PT_EDGES, name="ptgen"),
        hist.axis.Variable(RHO_EDGES, name="mpt_reco"),
        hist.axis.Variable(RHO_EDGES, name="mpt_gen"),
    )
    view = output.view(flow=False)
    for systematic_index, (content, variance) in enumerate(((10.0, 4.0), (20.0, 9.0))):
        for pt_index in range(2):
            for rho_index in range(3):
                view.value[0, systematic_index, pt_index, pt_index, rho_index, rho_index] = content
                view.variance[0, systematic_index, pt_index, pt_index, rho_index, rho_index] = variance
    return output


def _marginal_histogram(axis_pt, axis_rho, fake_or_miss):
    output = _weighted_histogram(
        hist.axis.StrCategory(["sample"], name="dataset"),
        hist.axis.StrCategory(SYSTEMATICS, name="systematic"),
        hist.axis.Variable(PT_EDGES, name=axis_pt),
        hist.axis.Variable(RHO_EDGES, name=axis_rho),
    )
    view = output.view(flow=False)
    # The response has 10 (nominal) / 20 (shiftUp) diagonal entries. The
    # marginals add a deliberately different fake/miss component to each one.
    for systematic_index, (matched, extra, variance) in enumerate(
        ((10.0, fake_or_miss[0], 4.0 + fake_or_miss[1]),
         (20.0, fake_or_miss[2], 9.0 + fake_or_miss[3]))
    ):
        view.value[0, systematic_index, :, :] = matched + extra
        view.variance[0, systematic_index, :, :] = variance
    return output


def _data_histogram():
    output = _weighted_histogram(
        hist.axis.StrCategory(["data"], name="dataset"),
        hist.axis.StrCategory(["nominal"], name="systematic"),
        hist.axis.Variable(PT_EDGES, name="ptreco"),
        hist.axis.Variable(RHO_EDGES, name="mpt_reco"),
    )
    output.view(flow=False).value[...] = 100.0
    output.view(flow=False).variance[...] = 25.0
    return output


def _prepared_unfolder():
    unfolder = Unfolder.__new__(Unfolder)
    unfolder.spec = SimpleNamespace(
        hist_keys_groomed=KEYS,
        hist_keys_ungroomed=KEYS,
        normalization_window_groomed=None,
        normalization_window_ungroomed=None,
        normalize_over_shown=False,
        x_label_groomed=r"$2\log_{10}(\rho)$",
        x_label_ungroomed=r"$2\log_{10}(\rho)$",
        short_label_groomed=r"$2\log_{10}(\rho)$",
        short_label_ungroomed=r"$2\log_{10}(\rho)$",
        xlim_lower_groomed=-3.0,
        xlim_lower_ungroomed=-3.0,
        name="rho",
        output_dir="unused/",
    )
    unfolder.groomed = True
    unfolder.cms_label = "Internal"
    unfolder.lumi = 138.0
    unfolder.com = 13.0
    unfolder.reco_axis = "mpt_reco"
    unfolder.gen_axis = "mpt_gen"
    unfolder._uses_prepared_inputs = True
    unfolder._setup_prepared_binning(
        SimpleNamespace(
            pt_edges=PT_EDGES,
            rho_edges=RHO_EDGES,
            rho_edges_gen=RHO_EDGES,
            reco_rho_edges_by_pt=[RHO_EDGES, RHO_EDGES],
            gen_rho_edges_by_pt=[RHO_EDGES, RHO_EDGES],
        )
    )
    mc_inputs = {
        KEYS["response"]: _response_histogram(),
        KEYS["reco"]: _marginal_histogram("ptreco", "mpt_reco", (2.0, 1.0, 5.0, 4.0)),
        KEYS["gen"]: _marginal_histogram("ptgen", "mpt_gen", (3.0, 2.0, 7.0, 5.0)),
    }
    data_inputs = {KEYS["reco"]: _data_histogram()}
    covariance = np.eye(6) * 25.0
    covariance[0, 1] = covariance[1, 0] = 6.0
    unfolder._load_prepared_inputs(
        mc_inputs,
        data_inputs,
        SYSTEMATICS,
        measured_covariance={"groomed": covariance},
    )
    return unfolder, covariance


def test_prepared_systematics_keep_own_fakes_misses_variances_and_covariance():
    unfolder, covariance = _prepared_unfolder()

    assert np.allclose(unfolder.fakes_2d_dict["nominal"], 2.0)
    assert np.allclose(unfolder.fakes_2d_dict["shiftUp"], 5.0)
    assert np.allclose(unfolder.misses_2d_dict["nominal"], 3.0)
    assert np.allclose(unfolder.misses_2d_dict["shiftUp"], 7.0)
    assert np.allclose(unfolder.mosaic_var_dict["nominal"].diagonal(), 4.0)
    assert np.allclose(unfolder.mosaic_var_dict["shiftUp"].diagonal(), 9.0)
    assert np.allclose(unfolder.misses_var_dict["shiftUp"], 5.0)

    # shiftUp survival is 20 / (20 + 5) = 0.8 in every flattened reco bin.
    shifted_covariance = unfolder.corrected_measured_covariance_dict["shiftUp"]
    assert shifted_covariance[0, 1] == pytest.approx(covariance[0, 1] * 0.8**2)
    assert shifted_covariance[0, 0] == pytest.approx(covariance[0, 0] * 0.8**2)


def test_prepared_inputs_require_every_requested_mc_category():
    unfolder, _ = _prepared_unfolder()
    response = _response_histogram()
    reco = _marginal_histogram("ptreco", "mpt_reco", (2.0, 1.0, 5.0, 4.0))
    gen = _weighted_histogram(
        hist.axis.StrCategory(["sample"], name="dataset"),
        hist.axis.StrCategory(["nominal"], name="systematic"),
        hist.axis.Variable(PT_EDGES, name="ptgen"),
        hist.axis.Variable(RHO_EDGES, name="mpt_gen"),
    )
    data = _data_histogram()

    with pytest.raises(ValueError, match="gen marginal=.*shiftUp"):
        unfolder._load_prepared_inputs(
            {KEYS["response"]: response, KEYS["reco"]: reco, KEYS["gen"]: gen},
            {KEYS["reco"]: data},
            SYSTEMATICS,
        )


def test_prepared_first_reported_bin_and_curvature_ranges_follow_pt_slices():
    unfolder, _ = _prepared_unfolder()
    unfolder.first_reported_pt_bin = unfolder._validated_first_reported_pt_bin(0)

    class Recorder:
        def __init__(self):
            self.calls = []

        def RegularizeBins(self, *args):
            self.calls.append(args)

    assert list(unfolder._reported_pt_indices()) == [0, 1]
    assert list(unfolder._summary_pt_indices()) == [0, 1]
    assert unfolder._curvature_regularization_ranges() == [(1, 3), (4, 6)]
    recorder = Recorder()
    unfolder._add_curvature_regularization(recorder)
    assert recorder.calls == [
        (1, 1, 3, unfolder_core.ROOT.TUnfold.kRegModeCurvature),
        (4, 1, 3, unfolder_core.ROOT.TUnfold.kRegModeCurvature),
    ]
    with pytest.raises(ValueError, match="outside"):
        unfolder._validated_first_reported_pt_bin(2)


def test_prepared_panel_names_start_at_pt0_and_fancy_uses_bin_edges():
    unfolder, _ = _prepared_unfolder()
    unfolder.first_reported_pt_bin = 0

    assert unfolder._output_panel_index(0) == 0
    assert unfolder._categorize_output("groomed_0") == (
        "unfolded", "unfolded_groomed_pt0"
    )
    # A legacy migration-sink layout continues to label its first reported
    # slice as pt0 as well.
    unfolder.first_reported_pt_bin = 1
    assert unfolder._output_panel_index(1) == 0

    fancy_source = inspect.getsource(Unfolder.plot_unfolded_fancy)
    assert fancy_source.find('hep.style.use("CMS")') < fancy_source.find("plt.subplots(")
    assert 'layout="constrained"' in fancy_source
    assert "plt.stairs(" in fancy_source
    assert "rho_edges" in fancy_source
    assert "fill_between(centers" not in fancy_source


def test_prepared_correlation_can_use_total_covariance():
    unfolder, _ = _prepared_unfolder()
    unfolder.first_reported_pt_bin = 0
    unfolder.stat_propagation = "jacobian"
    unfolder.norm_cov_stat = np.eye(6)
    total_covariance = np.diag([4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
    total_covariance[0, 1] = total_covariance[1, 0] = 3.0
    unfolder.get_total_covariance = lambda: total_covariance
    assert np.array_equal(
        unfolder._correlation_covariance("total"), total_covariance
    )
    assert np.array_equal(
        unfolder._correlation_covariance("stat"), unfolder.norm_cov_stat
    )
    with pytest.raises(ValueError, match="stat.*total"):
        unfolder._correlation_covariance("input")


def test_systematic_summary_uses_normalized_statistical_component():
    unfolder = Unfolder.__new__(Unfolder)
    unfolder.spec = SimpleNamespace(model_envelope=False)
    unfolder.normalized_results = [
        {
            "unfolded": np.array([0.2, 0.3]),
            "syst_unc": {
                "up": np.array([0.03, 0.045]),
                "down": np.array([0.025, 0.036]),
            },
            "stat_unc_frac": np.array([0.10, 0.08]),
        }
    ]
    unfolder.normalized_systematics = [{"unfolded": {}}]
    # Deliberately incompatible pre-normalization fractions reproduce the old
    # plotting bug: these values must not feed a normalized uncertainty panel.
    unfolder.stat_unc_pt_binned = [np.array([0.50, 0.40])]

    fractions = unfolder._build_syst_fraction_dict(0)

    assert np.array_equal(fractions["Stat Unc"], [0.10, 0.08])
    assert np.all(fractions["Total_Up"] >= fractions["Stat Unc"])
    assert np.all(fractions["Total_Down"] >= fractions["Stat Unc"])


def test_prepared_folded_counts_uses_corrected_covariance_and_window():
    unfolder, _ = _prepared_unfolder()
    unfolder.first_reported_pt_bin = 0
    unfolder.spec.normalization_window_groomed = (-2.0, -1.0)
    unfolder.x_folded = np.array([5.0, 10.0, 20.0, 4.0, 8.0, 16.0])
    unfolder.y_meas = np.array([6.0, 12.0, 21.0, 5.0, 7.0, 15.0])
    unfolder.corrected_measured_covariance = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])

    payload = unfolder._folded_counts_payload(0)

    # One fully contained reco bin remains. The count error is sqrt(C_ii)=2
    # from the corrected (not raw) covariance, and both bins/crop use native
    # reco edges rather than the GEN display layout.
    assert np.array_equal(payload["edges"], np.array([-2.0, -1.0]))
    assert np.array_equal(payload["folded"], np.array([10.0]))
    assert np.array_equal(payload["measured"], np.array([12.0]))
    assert np.array_equal(payload["measured_error"], np.array([2.0]))


def test_normalization_window_excludes_low_buffer_and_high_catchall():
    unfolder, _ = _prepared_unfolder()
    unfolder.spec.normalization_window_groomed = (-2.0, -1.0)

    assert np.array_equal(unfolder._shown_gen_mask(0), [False, True, False])
    assert np.array_equal(unfolder._shown_reco_mask(1), [False, True, False])
    display_slice, display_edges = unfolder._gen_display_slice(0)
    assert display_slice == slice(1, 2)
    assert np.array_equal(display_edges, [-2.0, -1.0])


def test_narrow_normalization_can_retain_a_wider_display_window():
    unfolder, _ = _prepared_unfolder()
    unfolder.spec.normalization_window_groomed = (-2.0, -1.0)
    unfolder.spec.display_window_groomed = (-3.0, 0.0)
    values = np.array([10.0, 20.0, 30.0])

    normalized = unfolder._normalized_slice(values, RHO_EDGES, 0)
    display_slice, display_edges = unfolder._gen_display_slice(0)

    assert display_slice == slice(0, 3)
    assert np.array_equal(display_edges, RHO_EDGES)
    assert unfolder._display_xlim(0) == (-3.0, 0.0)
    # Only the middle bin defines unit area; the two diagnostic side bins are
    # still drawn using that same denominator.
    assert normalized[1] == pytest.approx(1.0)
    assert np.array_equal(normalized, [0.5, 1.0, 1.5])


def test_bottom_line_window_normalization_and_covariance_use_the_display_mask():
    unfolder, _ = _prepared_unfolder()
    unfolder.spec.normalization_window_groomed = (-2.0, -1.0)
    values = np.array([10.0, 20.0, 30.0])

    normalized = unfolder._normalized_slice(values, RHO_EDGES, 0)
    jacobian = unfolder._slice_normalization_jacobian(values, RHO_EDGES, 0)
    display_slice, _ = unfolder._gen_display_slice(0)

    assert np.sum(normalized[display_slice] * np.diff(RHO_EDGES)[display_slice]) == pytest.approx(1.0)
    assert np.allclose((jacobian @ np.eye(3) @ jacobian.T)[display_slice, display_slice], 0.0)


def test_prepared_inputs_reject_material_negative_fake_or_miss_variance():
    unfolder, _ = _prepared_unfolder()
    response = _response_histogram()
    reco = _marginal_histogram("ptreco", "mpt_reco", (2.0, 1.0, 5.0, 4.0))
    reco.view(flow=False).variance[...] = 1.0
    gen = _marginal_histogram("ptgen", "mpt_gen", (3.0, 2.0, 7.0, 5.0))

    with pytest.raises(ValueError, match="reco fake variance.*'nominal'"):
        unfolder._load_prepared_inputs(
            {KEYS["response"]: response, KEYS["reco"]: reco, KEYS["gen"]: gen},
            {KEYS["reco"]: _data_histogram()},
            SYSTEMATICS,
        )


def _small_prepared_inputs_for_response_variance(shift_response_variance):
    """Return a full-rank detector-shift toy with fixed central arrays."""

    response = hist.Hist(
        hist.axis.StrCategory(SYSTEMATICS, name="systematic"),
        hist.axis.Variable([200.0, 400.0], name="ptreco"),
        hist.axis.Variable([-3.0, -2.0, -1.0], name="mpt_reco"),
        hist.axis.Variable([200.0, 400.0], name="ptgen"),
        hist.axis.Variable([-3.0, -2.0, -1.0], name="mpt_gen"),
        storage=hist.storage.Weight(),
    )
    response_view = response.view(flow=False)
    nominal = np.array([[10.0, 2.0], [1.0, 10.0]])
    shifted = np.array([[12.0, 1.0], [1.0, 8.0]])
    response_view.value[0, 0, :, 0, :] = nominal
    response_view.value[1, 0, :, 0, :] = shifted
    response_view.variance[0, 0, :, 0, :] = 1.0
    response_view.variance[1, 0, :, 0, :] = shift_response_variance

    reco = hist.Hist(
        hist.axis.StrCategory(SYSTEMATICS, name="systematic"),
        hist.axis.Variable([200.0, 400.0], name="ptreco"),
        hist.axis.Variable([-3.0, -2.0, -1.0], name="mpt_reco"),
        storage=hist.storage.Weight(),
    )
    reco_view = reco.view(flow=False)
    reco_view.value[0, 0, :] = nominal.sum(axis=1) + 1.0
    reco_view.value[1, 0, :] = shifted.sum(axis=1) + 1.0
    reco_view.variance[0, 0, :] = nominal.sum(axis=1) + 1.0
    reco_view.variance[1, 0, :] = shift_response_variance.sum(axis=1) + 1.0

    gen = hist.Hist(
        hist.axis.StrCategory(SYSTEMATICS, name="systematic"),
        hist.axis.Variable([200.0, 400.0], name="ptgen"),
        hist.axis.Variable([-3.0, -2.0, -1.0], name="mpt_gen"),
        storage=hist.storage.Weight(),
    )
    gen_view = gen.view(flow=False)
    # Detector-side shifts intentionally keep the particle-level marginal fixed.
    gen_view.value[:, 0, :] = nominal.sum(axis=0) + 1.0
    gen_view.variance[:, 0, :] = 3.0

    data = hist.Hist(
        hist.axis.StrCategory(["nominal"], name="systematic"),
        hist.axis.Variable([200.0, 400.0], name="ptreco"),
        hist.axis.Variable([-3.0, -2.0, -1.0], name="mpt_reco"),
        storage=hist.storage.Weight(),
    )
    data_view = data.view(flow=False)
    data_view.value[0, 0, :] = (15.0, 16.0)
    data_view.variance[0, 0, :] = (15.0, 16.0)
    binning = SimpleNamespace(
        pt_edges=[200.0, 400.0],
        rho_edges=[-3.0, -2.0, -1.0],
        rho_edges_gen=[-3.0, -2.0, -1.0],
        reco_rho_edges_by_pt=[[-3.0, -2.0, -1.0]],
        gen_rho_edges_by_pt=[[-3.0, -2.0, -1.0]],
    )
    return (
        {KEYS["response"]: response, KEYS["reco"]: reco, KEYS["gen"]: gen},
        {KEYS["reco"]: data},
        binning,
    )


def test_non_nominal_response_sumw2_does_not_change_tunfold_central_shift(tmp_path):
    """A detector variation's central shift is independent of response errors."""

    import ROOT

    spec = replace(
        RHO_FIXED_JEC_SPEC,
        output_dir=f"{tmp_path}/",
        regularization="none",
        tau=None,
        area_constraint=True,
        model_envelope=False,
        normalization_window_groomed=None,
        normalize_over_shown=False,
    )
    common = {
        "analysis_binning": None,
        "systematics": tuple(SYSTEMATICS),
        "measured_covariance": np.diag([15.0, 16.0]),
        "first_reported_pt_bin": 0,
    }
    previous_error_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kError
    try:
        nominal_variance_inputs, data_inputs, binning = _small_prepared_inputs_for_response_variance(
            np.ones((2, 2))
        )
        nominal_variance = Unfolder.from_prepared_inputs(
            spec,
            True,
            mc_inputs=nominal_variance_inputs,
            data_inputs=data_inputs,
            analysis_binning=binning,
            systematics=common["systematics"],
            measured_covariance=common["measured_covariance"],
            first_reported_pt_bin=common["first_reported_pt_bin"],
        )
        shifted_variance_inputs, data_inputs, binning = _small_prepared_inputs_for_response_variance(
            np.full((2, 2), 1.4)
        )
        shifted_variance = Unfolder.from_prepared_inputs(
            spec,
            True,
            mc_inputs=shifted_variance_inputs,
            data_inputs=data_inputs,
            analysis_binning=binning,
            systematics=common["systematics"],
            measured_covariance=common["measured_covariance"],
            first_reported_pt_bin=common["first_reported_pt_bin"],
        )
    finally:
        ROOT.gErrorIgnoreLevel = previous_error_level

    assert not np.allclose(
        nominal_variance.mosaic_var_dict["shiftUp"],
        shifted_variance.mosaic_var_dict["shiftUp"],
    )
    assert np.allclose(
        nominal_variance.y_unf_dict["shiftUp"],
        shifted_variance.y_unf_dict["shiftUp"],
        rtol=0.0,
        atol=1e-12,
    )


def test_model_envelope_plot_names_all_prepared_pt_slices(tmp_path):
    """Prepared inputs start at pT index zero; no slice may become ``ptall``."""

    import matplotlib.pyplot as plt

    unfolder = Unfolder.__new__(Unfolder)
    unfolder.spec = SimpleNamespace(output_dir=f"{tmp_path}/")
    unfolder.groomed = True
    unfolder.first_reported_pt_bin = 0
    unfolder.pt_edges = np.asarray([200.0, 290.0, 400.0])
    unfolder.gen_edges_by_pt = [
        np.asarray([-4.0, -2.0, 0.0]),
        np.asarray([-4.0, -2.0, 0.0]),
    ]
    unfolder.normalized_results = [
        {"pt_bin": (200.0, 290.0), "model_unc_frac": np.asarray([0.04, 0.05])},
        {"pt_bin": (290.0, 400.0), "model_unc_frac": np.asarray([0.06, 0.07])},
    ]
    unfolder.model_shift_components = {
        name: [np.asarray([0.01, 0.02]), np.asarray([0.02, 0.03])]
        for name in ("Vincia", "CR", "frag")
    }
    unfolder.model_fsr_frac = [
        np.asarray([0.02, 0.01]),
        np.asarray([0.03, 0.01]),
    ]
    unfolder.cms_label = "Internal"
    unfolder.lumi = 138.0
    unfolder.com = 13.0
    unfolder._observable_xlim = lambda _i: (-4.0, 0.0)
    unfolder._observable_label = lambda: r"$2\log_{10}(\rho)$"
    saved = []

    def _record_plot(*, save_path, show, fig):
        saved.append(Path(save_path).name)
        plt.close(fig)

    unfolder._finalize_plot = _record_plot
    unfolder.plot_model_envelope(show=False)

    assert saved == [
        "model_envelope_groomed_0.pdf",
        "model_envelope_groomed_1.pdf",
    ]


def test_total_covariance_uses_two_leg_model_once():
    """Raw FSR/model inputs are superseded by coherent PS/HAD nuisances."""

    unfolder = Unfolder.__new__(Unfolder)
    nominal = np.asarray([0.4, 0.6])
    detector_up = np.asarray([0.42, 0.58])
    unfolder.spec = SimpleNamespace(
        model_envelope=True,
        model_covariance_scope="global_shown",
    )
    unfolder.systematics = (
        "nominal",
        "detectorUp",
        "fsrUp",
        "model_vincia",
    )
    unfolder.normalized_results = [{"unfolded": nominal}]
    unfolder.normalized_systematics = [
        {
            "unfolded": {
                "nominal": nominal,
                "detectorUp": detector_up,
                # Deliberately huge raw shifts: neither may enter directly.
                "fsrUp": np.asarray([0.8, 0.2]),
                "model_vincia": np.asarray([0.1, 0.9]),
            }
        }
    ]
    unfolder.gen_edges_by_pt = [np.asarray([0.0, 1.0, 2.0])]
    unfolder._shown_gen_mask = lambda _i: np.asarray([True, True])
    unfolder.model_ps_shift_flat = np.asarray([0.10, -1.0 / 15.0])
    unfolder.model_had_shift_flat = np.asarray([-0.05, 1.0 / 30.0])

    detector_shift = detector_up - nominal
    ps_shift = unfolder.model_ps_shift_flat * nominal
    had_shift = unfolder.model_had_shift_flat * nominal
    expected = (
        np.outer(detector_shift, detector_shift)
        + np.outer(ps_shift, ps_shift)
        + np.outer(had_shift, had_shift)
    )

    assert np.allclose(unfolder.get_systematic_covariance(), expected)


def test_offscale_ratio_band_marks_only_clipped_bins():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    centers = np.array([0.5, 1.5, 2.5])
    frac = np.array([1.84, 0.4, 0.2])  # only the first bin exceeds (0, 2)
    Unfolder._mark_offscale_ratio_band(ax, centers, frac, frac, top=2.0, bottom=0.0)
    assert [text.get_text() for text in ax.texts] == ["band ±184%"]
    assert sorted(line.get_marker() for line in ax.lines) == ["^", "v"]
    assert all(line.get_xdata()[0] == 0.5 for line in ax.lines)
    plt.close(fig)


def test_offscale_ratio_band_labels_a_one_sided_clip():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    Unfolder._mark_offscale_ratio_band(
        ax, np.array([0.5]), np.array([1.2]), np.array([0.3]), top=2.0, bottom=0.0
    )
    assert [text.get_text() for text in ax.texts] == ["band +120%"]
    assert [line.get_marker() for line in ax.lines] == ["^"]
    plt.close(fig)


def test_offscale_ratio_band_handles_a_negative_central_value():
    """A negative unfolded bin flips the plotted fractions' signs; the cue must
    key on the drawn band edges (trijet ungroomed 290-400 GeV first-bin case)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    Unfolder._mark_offscale_ratio_band(
        ax,
        np.array([0.5]),
        np.array([-1.84]),
        np.array([-1.84]),
        top=2.0,
        bottom=0.0,
    )
    assert [text.get_text() for text in ax.texts] == ["band ±184%"]
    assert sorted(line.get_marker() for line in ax.lines) == ["^", "v"]
    plt.close(fig)
