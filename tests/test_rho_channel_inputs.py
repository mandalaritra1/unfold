from __future__ import annotations

from pathlib import Path
import ast
import pickle
import tempfile
import unittest

import hist
import numpy as np

from unfold.tools.rho_channel_inputs import (
    RHO_HIST_KEYS,
    build_prepared_rho_inputs,
    channel_rho_binning,
    discover_rho_channel_files,
)


SOURCE_PT_EDGES = [0.0, 200.0, 290.0, 400.0, 480.0, 570.0, 680.0, 760.0, 820.0, 13000.0]
SOURCE_RECO_EDGES = [
    -10.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.75, -4.5,
    -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5,
    -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
    -0.25, 0.0,
]
SOURCE_GEN_EDGES = [
    -10.0, -8.0, -7.0, -6.0, -5.0, -4.5, -4.0, -3.5,
    -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
]


def weighted_histogram(*axes):
    output = hist.Hist(*axes, storage=hist.storage.Weight())
    output.values(flow=False)[...] = 1.0
    output.variances(flow=False)[...] = 1.0
    return output


def make_reco(systematics, reco_edges=SOURCE_RECO_EDGES):
    return weighted_histogram(
        hist.axis.StrCategory(["sample"], name="dataset"),
        hist.axis.StrCategory(systematics, name="systematic"),
        hist.axis.Variable(SOURCE_PT_EDGES, name="ptreco"),
        hist.axis.Variable(reco_edges, name="mpt_reco"),
    )


def make_gen(systematics):
    return weighted_histogram(
        hist.axis.StrCategory(["sample"], name="dataset"),
        hist.axis.StrCategory(systematics, name="systematic"),
        hist.axis.Variable(SOURCE_PT_EDGES, name="ptgen"),
        hist.axis.Variable(SOURCE_GEN_EDGES, name="mpt_gen"),
    )


def make_response(systematics):
    return weighted_histogram(
        hist.axis.StrCategory(["sample"], name="dataset"),
        hist.axis.StrCategory(systematics, name="systematic"),
        hist.axis.Variable(SOURCE_PT_EDGES, name="ptreco"),
        hist.axis.Variable(SOURCE_PT_EDGES, name="ptgen"),
        hist.axis.Variable(SOURCE_RECO_EDGES, name="mpt_reco"),
        hist.axis.Variable(SOURCE_GEN_EDGES, name="mpt_gen"),
    )


def make_payloads(reco_edges=SOURCE_RECO_EDGES):
    response_systematics = ["nominal", "Q2muRUp", "Q2muRDown"]
    mc = {}
    data = {}
    for keys in RHO_HIST_KEYS.values():
        mc[keys["response"]] = make_response(response_systematics)
        mc[keys["reco"]] = make_reco(response_systematics, reco_edges)
        mc[keys["gen"]] = make_gen(["nominal"])
        data[keys["reco"]] = make_reco(["nominal"], reco_edges)
    return mc, data


class RhoChannelInputTests(unittest.TestCase):
    def write_inputs(self, root: Path, mc, data):
        channel_dir = root / "dijet"
        channel_dir.mkdir(parents=True)
        data_path = channel_dir / "minimal_rho_dijet_data_2018.pkl"
        mc_path = channel_dir / "minimal_rho_dijet_mg_pythia8_2018.pkl"
        with data_path.open("wb") as handle:
            pickle.dump(data, handle)
        with mc_path.open("wb") as handle:
            pickle.dump(mc, handle)
        return discover_rho_channel_files(root, "dijet", 2018)

    def test_exact_filename_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            files = self.write_inputs(root, mc, data)
            self.assertEqual(files.data.name, "minimal_rho_dijet_data_2018.pkl")
            self.assertEqual(files.mc.name, "minimal_rho_dijet_mg_pythia8_2018.pkl")

    def test_role_aware_adaptation_and_no_systematic_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            prepared = build_prepared_rho_inputs(self.write_inputs(root, mc, data))

            self.assertEqual(
                prepared.systematics,
                ["nominal", "q2muRUp", "q2muRDown"],
            )
            gen_key = RHO_HIST_KEYS["ungroomed"]["gen"]
            data_key = RHO_HIST_KEYS["ungroomed"]["reco"]
            self.assertEqual(list(prepared.mc[gen_key].axes["systematic"]), ["nominal"])
            self.assertEqual(list(prepared.data[data_key].axes["systematic"]), ["nominal"])
            self.assertEqual(
                prepared.binning["groomed"].pt_edges,
                [0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0],
            )

    def test_missing_mc_histogram_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc, data = make_payloads()
            del mc[RHO_HIST_KEYS["groomed"]["gen"]]
            files = self.write_inputs(root, mc, data)
            with self.assertRaisesRegex(KeyError, "missing histograms"):
                build_prepared_rho_inputs(files)

    def test_incompatible_rho_edges_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incompatible_edges = [
                edge for edge in SOURCE_RECO_EDGES if not np.isclose(edge, -8.0)
            ]
            mc, data = make_payloads(reco_edges=incompatible_edges)
            files = self.write_inputs(root, mc, data)
            with self.assertRaisesRegex(ValueError, "incompatible edges"):
                build_prepared_rho_inputs(files)

    def test_binning_uses_established_rho_edges(self):
        ungroomed = channel_rho_binning("trijet", False)
        groomed = channel_rho_binning("trijet", True)
        self.assertEqual(ungroomed.reco_rho_edges_by_pt[0][1], -2.75)
        self.assertEqual(ungroomed.gen_rho_edges_by_pt[0][1], -2.5)
        self.assertEqual(groomed.reco_rho_edges_by_pt[0][1], -4.75)
        self.assertEqual(groomed.gen_rho_edges_by_pt[0][1], -4.5)

    def test_dijet_groomed_uses_studied_variable_low_rho_binning(self):
        dijet = channel_rho_binning("dijet", True)
        trijet = channel_rho_binning("trijet", True)

        expected_dijet_gen = [
            [-10.0, -4.5],
            [-10.0, -4.5],
            [-10.0, -4.5],
            [-10.0, -5.0, -4.5],
            [-10.0, -5.0, -4.5],
            [-10.0, -4.5],
        ]
        for pt_index, expected_prefix in enumerate(expected_dijet_gen):
            self.assertEqual(
                dijet.gen_rho_edges_by_pt[pt_index][:len(expected_prefix)],
                expected_prefix,
            )

        self.assertEqual(
            dijet.reco_rho_edges_by_pt[3][:6],
            [-10.0, -6.0, -5.5, -5.0, -4.75, -4.5],
        )
        self.assertEqual(
            dijet.reco_rho_edges_by_pt[4][:6],
            [-10.0, -6.0, -5.5, -5.0, -4.75, -4.5],
        )
        self.assertTrue(
            all(edges[1] == -4.5 for edges in trijet.gen_rho_edges_by_pt)
        )

    def test_runner_delegates_plotting_to_shared_unfolder(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_rho_unfolding.py"
        )
        tree = ast.parse(runner.read_text(encoding="utf-8"))
        function_names = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertFalse(
            any(name.startswith(("make_", "plot_")) for name in function_names)
        )
        self.assertIn(
            "unfolder.run_all_plots(show=False)",
            runner.read_text(encoding="utf-8"),
        )

    def test_shared_plot_suite_initializes_cms_style_first(self):
        core = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "unfold"
            / "tools"
            / "unfolder_core.py"
        )
        tree = ast.parse(core.read_text(encoding="utf-8"))
        unfolder_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Unfolder"
        )
        run_all_plots = next(
            node
            for node in unfolder_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_all_plots"
        )
        first_statement = run_all_plots.body[0]
        self.assertIsInstance(first_statement, ast.Expr)
        self.assertEqual(
            ast.unparse(first_statement),
            "hep.style.use('CMS')",
        )


if __name__ == "__main__":
    unittest.main()
