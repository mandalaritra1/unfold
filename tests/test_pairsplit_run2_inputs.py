from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest

import hist
import numpy as np

from unfold.tools.pairsplit_run2_inputs import (
    LEGACY_HISTOGRAM_KEYS,
    PAIR_SPLIT_FINE_AXES,
    PairSplitRun2Inputs,
    PHYSICAL_RHO_DEFINITION,
    TRANSFORMED_COORDINATE_DEFINITION,
    TRANSFORMED_COORDINATE_NAME,
    covariance_rebin_matrix,
    discover_pairsplit_run2_files,
    load_pairsplit_run2_inputs,
    pair_split_binning,
    prepare_pairsplit_groomed_inputs,
    rebin_reco_covariance,
)


def weighted_histogram(*axes, value: float, variance: float):
    output = hist.Hist(*axes, storage=hist.storage.Weight())
    output.values(flow=False)[...] = value
    output.variances(flow=False)[...] = variance
    return output


def _mc_histogram(mode: str, role: str, systematics, value: float, variance: float, reco_edges=None):
    fine = PAIR_SPLIT_FINE_AXES[mode]
    reco_edges = fine.two_log10_rho_reco_edges if reco_edges is None else reco_edges
    dataset = hist.axis.StrCategory(["sample_a", "sample_b"], name="dataset")
    systematic = hist.axis.StrCategory(systematics, name="systematic")
    ptreco = hist.axis.Variable(fine.pt_edges, name="ptreco")
    ptgen = hist.axis.Variable(fine.pt_edges, name="ptgen")
    two_log10_reco = hist.axis.Variable(reco_edges, name="mpt_reco")
    two_log10_gen = hist.axis.Variable(fine.two_log10_rho_gen_edges, name="mpt_gen")
    axes_by_role = {
        "response": (dataset, systematic, ptreco, ptgen, two_log10_reco, two_log10_gen),
        "reco": (dataset, systematic, ptreco, two_log10_reco),
        "gen": (dataset, systematic, ptgen, two_log10_gen),
    }
    return weighted_histogram(*axes_by_role[role], value=value, variance=variance)


def _data_histogram(mode: str, role: str, value: float, variance: float, reco_edges=None):
    fine = PAIR_SPLIT_FINE_AXES[mode]
    reco_edges = fine.two_log10_rho_reco_edges if reco_edges is None else reco_edges
    dataset = hist.axis.StrCategory(["data"], name="dataset")
    systematic = hist.axis.StrCategory(["nominal"], name="systematic")
    if role == "reco":
        return weighted_histogram(
            dataset,
            systematic,
            hist.axis.Variable(fine.pt_edges, name="ptreco"),
            hist.axis.Variable(reco_edges, name="mpt_reco"),
            value=value,
            variance=variance,
        )
    return weighted_histogram(
        dataset,
        systematic,
        hist.axis.Variable(fine.pt_edges, name="ptreco_i"),
        hist.axis.Variable(reco_edges, name="mpt_reco_i"),
        hist.axis.Variable(fine.pt_edges, name="ptreco_j"),
        hist.axis.Variable(reco_edges, name="mpt_reco_j"),
        value=value,
        variance=variance,
    )


def make_payloads(
    *,
    systematics=("nominal", "JERUp"),
    gen_systematics=None,
    groomed_reco_edges=None,
    mc_value=1.0,
    mc_variance=3.0,
    data_value=5.0,
):
    mc = {}
    data = {}
    gen_systematics = systematics if gen_systematics is None else gen_systematics
    for mode, keys in LEGACY_HISTOGRAM_KEYS.items():
        reco_edges = groomed_reco_edges if mode == "groomed" else None
        mc[keys["response"]] = _mc_histogram(
            mode, "response", systematics, mc_value, mc_variance, reco_edges
        )
        mc[keys["reco"]] = _mc_histogram(
            mode, "reco", systematics, mc_value, mc_variance, reco_edges
        )
        mc[keys["gen"]] = _mc_histogram(
            mode, "gen", gen_systematics, mc_value, mc_variance, reco_edges
        )
        data[keys["reco"]] = _data_histogram(
            mode, "reco", data_value, data_value + 1.0, reco_edges
        )
        data[keys["reco_covariance"]] = _data_histogram(
            mode, "reco_covariance", data_value + 2.0, data_value + 3.0, reco_edges
        )
    return mc, data


def write_pair_split_era(root: Path, era: str, mc, data, *, channel="dijet", nested="producer"):
    base = root / era / nested
    mc_directory = base / f"{channel}_mc"
    data_directory = base / f"{channel}_data"
    mc_directory.mkdir(parents=True)
    data_directory.mkdir(parents=True)
    mc_path = mc_directory / f"minimal_rho_{channel}_mg_pythia8_{era}.pkl"
    data_path = data_directory / f"minimal_rho_{channel}_data_{era}.pkl"
    with mc_path.open("wb") as handle:
        pickle.dump(mc, handle)
    with data_path.open("wb") as handle:
        pickle.dump(data, handle)
    return mc_path, data_path


class PairSplitRun2InputTests(unittest.TestCase):
    def test_nested_discovery_excludes_lhe_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            expected_mc, expected_data = write_pair_split_era(root, "2018", mc, data)
            archive_mc, _ = write_pair_split_era(
                root, "2018", mc, data, nested="2018_lhe_basis"
            )

            files = discover_pairsplit_run2_files("dijet", "2018", root)

            self.assertEqual(files.mc, expected_mc)
            self.assertEqual(files.data, expected_data)
            self.assertNotEqual(files.mc, archive_mc)

    def test_discovery_rejects_a_second_nonarchive_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            write_pair_split_era(root, "2018", mc, data, nested="first")
            write_pair_split_era(root, "2018", mc, data, nested="second")

            with self.assertRaisesRegex(ValueError, "Expected exactly one mc"):
                discover_pairsplit_run2_files("dijet", "2018", root)

    def test_axis_mismatch_is_rejected_before_era_sum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            write_pair_split_era(root, "2016", mc, data)
            incompatible = tuple(
                edge for edge in PAIR_SPLIT_FINE_AXES["groomed"].two_log10_rho_reco_edges
                if not np.isclose(edge, -7.5)
            )
            mc_bad, data_bad = make_payloads(groomed_reco_edges=incompatible)
            write_pair_split_era(root, "2017", mc_bad, data_bad)

            with self.assertRaisesRegex(ValueError, "physics-axis mismatch"):
                load_pairsplit_run2_inputs("dijet", ("2016", "2017"), root)

    def test_systematic_categories_must_match_each_mc_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads(gen_systematics=("nominal",))
            write_pair_split_era(root, "2018", mc, data)

            with self.assertRaisesRegex(ValueError, "systematic category mismatch"):
                load_pairsplit_run2_inputs("dijet", ("2018",), root)

    def test_dataset_and_era_variances_are_added(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for era in ("2016", "2017"):
                mc, data = make_payloads()
                write_pair_split_era(root, era, mc, data)

            inputs = load_pairsplit_run2_inputs("dijet", ("2016", "2017"), root)
            groomed = inputs.modes["groomed"]

            self.assertEqual(inputs.eras, ("2016", "2017"))
            self.assertEqual(tuple(source.era for source in inputs.source_files), inputs.eras)
            self.assertEqual(len(inputs.source_files), len(inputs.eras))
            # Two MC datasets per era, each with values=1 and sumw2=3.
            self.assertTrue(np.all(groomed.response_by_systematic["nominal"] == 4.0))
            self.assertTrue(np.all(groomed.response_variance_by_systematic["nominal"] == 12.0))
            self.assertTrue(np.all(groomed.nominal_data == 10.0))
            self.assertTrue(np.all(groomed.nominal_data_covariance == 14.0))
            self.assertTrue(
                np.all(inputs.modes["ungroomed"].nominal_data_covariance == 14.0)
            )

    def test_duplicate_era_is_rejected_to_preserve_source_file_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            write_pair_split_era(root, "2018", mc, data)

            with self.assertRaisesRegex(ValueError, "occur exactly once"):
                load_pairsplit_run2_inputs("dijet", ("2018", "2018"), root)

    def test_trijet_without_reco_covariance_uses_diagonal_reco_sumw2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            for mode, keys in LEGACY_HISTOGRAM_KEYS.items():
                del data[keys["reco_covariance"]]
            write_pair_split_era(root, "2016APV", mc, data, channel="trijet")

            inputs = load_pairsplit_run2_inputs("trijet", ("2016APV",), root)
            covariance = inputs.modes["groomed"].nominal_data_covariance.reshape(
                inputs.modes["groomed"].nominal_data.size,
                inputs.modes["groomed"].nominal_data.size,
            )

            self.assertTrue(np.allclose(covariance, np.diag(np.diag(covariance))))
            self.assertTrue(np.all(np.diag(covariance) == 6.0))
            self.assertEqual(
                inputs.observable_metadata["data_covariance_source_by_mode"]["groomed"],
                "diagonal_reco_sumw2",
            )
            groomed = inputs.modes["groomed"]
            zero_response = {
                systematic: np.zeros_like(values)
                for systematic, values in groomed.response_by_systematic.items()
            }
            prepared_inputs = PairSplitRun2Inputs(
                channel=inputs.channel,
                eras=inputs.eras,
                modes={
                    "groomed": replace(
                        groomed,
                        response_by_systematic=zero_response,
                        response_variance_by_systematic=zero_response,
                    )
                },
                source_files=inputs.source_files,
                observable_metadata=inputs.observable_metadata,
            )
            prepared = prepare_pairsplit_groomed_inputs(
                prepared_inputs, "two_to_one", ("nominal",)
            )
            self.assertEqual(prepared.metadata["data_covariance_source"], "diagonal_reco_sumw2")

    def test_trijet_rejects_mixed_full_and_diagonal_covariance_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc_full, data_full = make_payloads()
            write_pair_split_era(root, "2016", mc_full, data_full, channel="trijet")
            mc_diagonal, data_diagonal = make_payloads()
            for keys in LEGACY_HISTOGRAM_KEYS.values():
                del data_diagonal[keys["reco_covariance"]]
            write_pair_split_era(root, "2017", mc_diagonal, data_diagonal, channel="trijet")

            with self.assertRaisesRegex(ValueError, "mixed data covariance sources"):
                load_pairsplit_run2_inputs("trijet", ("2016", "2017"), root)

    def test_dijet_still_requires_event_clustered_reco_covariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            for keys in LEGACY_HISTOGRAM_KEYS.values():
                del data[keys["reco_covariance"]]
            write_pair_split_era(root, "2018", mc, data)

            with self.assertRaisesRegex(KeyError, "reco_cov"):
                load_pairsplit_run2_inputs("dijet", ("2018",), root)

    def test_full_covariance_rebinning_uses_g_covariance_g_transpose(self):
        covariance = np.diag([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 2, 2)
        grouping = covariance_rebin_matrix(2, 2, ((0, 1),), ((0, 1),))
        rebinned = rebin_reco_covariance(covariance, 2, 2, ((0, 1),), ((0, 1),))

        self.assertEqual(grouping.shape, (1, 4))
        self.assertTrue(np.all(grouping == 1.0))
        self.assertEqual(rebinned.shape, (1, 1, 1, 1))
        self.assertEqual(rebinned[0, 0, 0, 0], 10.0)

    def test_exact_candidate_edges_and_explicit_sink_behavior(self):
        fine = PAIR_SPLIT_FINE_AXES["groomed"]
        coarse_tail = pair_split_binning("dijet", "coarse_tail")
        two_to_one = pair_split_binning("trijet", "two_to_one")
        dijet_window_aligned = pair_split_binning("dijet", "window_aligned")
        trijet_window_aligned = pair_split_binning("trijet", "window_aligned")
        dijet_window_aligned_coarse = pair_split_binning("dijet", "window_aligned_coarse")
        trijet_window_aligned_coarse = pair_split_binning("trijet", "window_aligned_coarse")

        self.assertEqual(
            fine.pt_edges,
            (185.0, 200.0, 290.0, 400.0, 480.0, 570.0, 680.0, 760.0, 820.0, 13000.0),
        )
        self.assertEqual(coarse_tail.pt_edges, (200.0, 290.0, 400.0, 480.0, 570.0, 13000.0))
        self.assertEqual(
            coarse_tail.base_reco_two_log10_rho_edges,
            (-10.0, -4.0, -3.4, -2.85, -2.25, -1.8, -1.5, -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.0),
        )
        self.assertEqual(coarse_tail.sink_pt_source_bin_indices, (0,))
        self.assertEqual(coarse_tail.first_reported_pt_index, 0)
        self.assertEqual(coarse_tail.reported_two_log10_rho_minimum, -4.0)
        self.assertEqual(
            coarse_tail.gen_two_log10_rho_edges,
            (-10.0, -4.0, -2.85, -1.8, -1.5, -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.0),
        )
        self.assertEqual(
            sum(edge >= -4.0 for edge in coarse_tail.gen_two_log10_rho_edges[:-1]),
            10,
        )
        self.assertEqual(
            two_to_one.base_to_gen_two_log10_rho_groups,
            ((0,), (1, 2), (3, 4), (5, 6), (7,)),
        )
        self.assertEqual(
            two_to_one.base_reco_two_log10_rho_edges,
            (-10.0, -4.0, -2.85, -2.25, -1.8, -1.5, -1.1, -0.75, 0.0),
        )
        self.assertEqual(
            two_to_one.gen_two_log10_rho_edges,
            (-10.0, -4.0, -2.25, -1.5, -0.75, 0.0),
        )
        self.assertEqual(
            sum(edge >= -4.0 for edge in two_to_one.gen_two_log10_rho_edges[:-1]),
            4,
        )
        self.assertEqual(
            dijet_window_aligned.base_to_gen_two_log10_rho_groups,
            ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11,), (12,)),
        )
        self.assertEqual(
            dijet_window_aligned.gen_two_log10_rho_edges,
            (-10.0, -4.0, -2.85, -1.8, -1.3, -0.9, -0.65, -0.55, 0.0),
        )
        self.assertEqual(
            trijet_window_aligned.base_to_gen_two_log10_rho_groups,
            ((0,), (1,), (2,), (3, 4), (5, 6), (7,)),
        )
        self.assertEqual(
            trijet_window_aligned.gen_two_log10_rho_edges,
            (-10.0, -4.0, -2.85, -2.25, -1.5, -0.75, 0.0),
        )
        self.assertEqual(
            dijet_window_aligned_coarse.base_to_gen_two_log10_rho_groups,
            ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10, 11), (12,)),
        )
        self.assertEqual(
            dijet_window_aligned_coarse.gen_two_log10_rho_edges,
            (-10.0, -4.0, -2.85, -1.8, -1.3, -0.9, -0.55, 0.0),
        )
        self.assertEqual(
            trijet_window_aligned_coarse.base_to_gen_two_log10_rho_groups,
            ((0,), (1,), (2, 3, 4), (5, 6), (7,)),
        )
        self.assertEqual(
            trijet_window_aligned_coarse.gen_two_log10_rho_edges,
            (-10.0, -4.0, -2.85, -1.5, -0.75, 0.0),
        )
        for candidate, window, expected_reported_bins in (
            (dijet_window_aligned, (-2.85, -0.55), 5),
            (trijet_window_aligned, (-3.0, -0.7), 3),
            (dijet_window_aligned_coarse, (-2.85, -0.55), 4),
            (trijet_window_aligned_coarse, (-3.0, -0.7), 2),
        ):
            groups = candidate.base_to_gen_two_log10_rho_groups
            self.assertEqual(
                tuple(index for group in groups for index in group),
                tuple(range(len(candidate.base_reco_two_log10_rho_edges) - 1)),
            )
            edges = candidate.gen_two_log10_rho_edges
            contained = [
                index
                for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
                if low >= window[0] and high <= window[1]
            ]
            self.assertEqual(len(contained), expected_reported_bins)
            for boundary in window:
                crossing_groups = [
                    group
                    for group in groups
                    if candidate.base_reco_two_log10_rho_edges[group[0]] < boundary
                    < candidate.base_reco_two_log10_rho_edges[group[-1] + 1]
                ]
                self.assertTrue(all(len(group) == 1 for group in crossing_groups))

    def test_observable_metadata_keeps_physical_and_transformed_names_distinct(self):
        self.assertEqual(PHYSICAL_RHO_DEFINITION, "rho = m / (pT * R), with R = 0.8")
        self.assertEqual(TRANSFORMED_COORDINATE_NAME, "two_log10_rho")
        self.assertEqual(
            TRANSFORMED_COORDINATE_DEFINITION,
            "two_log10_rho = 2 * log10(rho)",
        )


if __name__ == "__main__":
    unittest.main()
