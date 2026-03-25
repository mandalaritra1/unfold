from array import array
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pickle as pkl
import ROOT
from cycler import cycler
from matplotlib.colors import LogNorm

from unfold.tools import *
from unfold.tools import binning
from unfold.utils.integrate_and_rebin import *
from unfold.utils.merge_helpers import *

plt.rcParams["axes.prop_cycle"] = cycler(color=plt.cm.tab20.colors)

DEFAULT_MC_FILE = "./inputs/rhoInputs/jms_pythiaV2_all_syst.pkl"
DEFAULT_DATA_FILE = "./inputs/rhoInputs/data_all.pkl"
DEFAULT_HERWIG_FILE = "./inputs/rhoInputs/herwig_all.pkl"
DEFAULT_JK_DATA_FILE = "./inputs/rhoInputs/jk_data_all.pkl"

JES_SYSTEMATICS = [
    "JES_AbsoluteMPFBiasUp", "JES_AbsoluteMPFBiasDown", "JES_AbsoluteScaleUp", "JES_AbsoluteScaleDown",
    "JES_AbsoluteStatUp", "JES_AbsoluteStatDown", "JES_FlavorQCDUp", "JES_FlavorQCDDown", "JES_FragmentationUp",
    "JES_FragmentationDown", "JES_PileUpDataMCUp", "JES_PileUpDataMCDown", "JES_PileUpPtBBUp", "JES_PileUpPtBBDown",
    "JES_PileUpPtEC1Up", "JES_PileUpPtEC1Down", "JES_PileUpPtEC2Up", "JES_PileUpPtEC2Down", "JES_PileUpPtHFUp",
    "JES_PileUpPtHFDown", "JES_PileUpPtRefUp", "JES_PileUpPtRefDown", "JES_RelativeFSRUp", "JES_RelativeFSRDown",
    "JES_RelativeJEREC1Up", "JES_RelativeJEREC1Down", "JES_RelativeJEREC2Up", "JES_RelativeJEREC2Down",
    "JES_RelativeJERHFUp", "JES_RelativeJERHFDown", "JES_RelativePtBBUp", "JES_RelativePtBBDown",
    "JES_RelativePtEC1Up", "JES_RelativePtEC1Down", "JES_RelativePtEC2Up", "JES_RelativePtEC2Down",
    "JES_RelativePtHFUp", "JES_RelativePtHFDown", "JES_RelativeBalUp", "JES_RelativeBalDown",
    "JES_RelativeSampleUp", "JES_RelativeSampleDown", "JES_RelativeStatECUp", "JES_RelativeStatECDown",
    "JES_RelativeStatFSRUp", "JES_RelativeStatFSRDown", "JES_RelativeStatHFUp", "JES_RelativeStatHFDown",
    "JES_SinglePionECALUp", "JES_SinglePionECALDown", "JES_SinglePionHCALUp", "JES_SinglePionHCALDown",
    "JES_TimePtEtaUp", "JES_TimePtEtaDown",
]

NON_JES_SYSTEMATICS = [
    "nominal", "puUp", "puDown", "elerecoUp", "elerecoDown", "eleidUp", "eleidDown", "eletrigUp", "eletrigDown",
    "murecoUp", "murecoDown", "muidUp", "muidDown", "mutrigUp", "muisoUp", "muisoDown", "mutrigDown", "pdfUp",
    "pdfDown", "q2Up", "q2Down", "l1prefiringUp", "l1prefiringDown", "ISRUp", "ISRDown", "FSRUp", "FSRDown",
    "JERUp", "JERDown", "JMSUp", "JMSDown", "herwigUp", "herwigDown",
]



class Unfolder:
    def __init__(self, groomed, closure=False, herwig_closure=False, do_syst=False, cms_label="Internal"):
        self.groomed = groomed
        self.cms_label = cms_label
        self.closure = closure
        self.herwig_closure = herwig_closure
        self.y_unf_dict = {}
        self._setup_binning()
        self._make_inputs_numpy()
        self._configure_systematics(do_syst)
        self._load_data(
            filename_mc=DEFAULT_MC_FILE,
            filename_data=DEFAULT_DATA_FILE,
            filename_herwig=DEFAULT_HERWIG_FILE,
        )
        self._perform_unfold(closure=self.closure, herwig_closure=self.herwig_closure)
        for syst in self.systematics:
            self._perform_unfold(systematic=syst, closure=self.closure, herwig_closure=self.herwig_closure)
        self._compute_stat_unc()
        self._normalize_result()
        self._compute_total_systematic()

    def _setup_binning(self):
        self.bins = binning.bin_edges(self.groomed)
        self.edges = self.bins.rho_edges
        self.edges_gen = self.bins.rho_edges_gen
        self.pt_edges = self.bins.pt_edges

    def _configure_systematics(self, do_syst):
        available_systematics = list(self.sys_matrix_dic.keys())
        self.systematics = available_systematics if do_syst else ["nominal"]

    def _load_pickle(self, filename):
        with open(filename, "rb") as handle:
            return pkl.load(handle)

    def _finalize_plot(self, save_path=None, show=True, fig=None):
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            target_fig = fig if fig is not None else plt.gcf()
            target_fig.savefig(path)
        if show:
            plt.show()
        else:
            plt.close(fig if fig is not None else plt.gcf())

    def _histogram_keys(self):
        if self.groomed:
            return {
                "response": "response_matrix_rho_g",
                "reco": "ptjet_rhojet_g_reco",
                "gen": "ptjet_rhojet_g_gen",
            }
        return {
            "response": "response_matrix_rho_u",
            "reco": "ptjet_rhojet_u_reco",
            "gen": "ptjet_rhojet_u_gen",
        }

    def _prepare_jackknife_inputs(self, data2d_jk, mass_edges_reco, pt_edges, reco_mass_edges_by_pt):
        mosaic_2d_jk_list = []
        for i in range(10):
            reco_proj_jk = data2d_jk.project("jk", "ptreco", "mpt_reco")[i, ...]
            h2d_jk = reco_proj_jk.values()
            h2d_jk_reordered, _ = reorder_to_expected_2d(h2d_jk, mass_edges_reco, pt_edges)
            mosaic_2d_jk_list.append(
                merge_mass_flat(h2d_jk_reordered, mass_edges_reco, reco_mass_edges_by_pt)
            )
        return mosaic_2d_jk_list

    def _prepare_nominal_inputs(
        self,
        input_data,
        fakes,
        misses,
        resp_matrix_4d_gen,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        reco_proj = input_data.project("ptreco", "mpt_reco")
        self.h2d, _ = reorder_to_expected_2d(reco_proj.values(), mass_edges_reco, pt_edges)

        reco_proj_fakes = fakes.project("ptreco", "mpt_reco")
        self.h2d_fakes, _ = reorder_to_expected_2d(reco_proj_fakes.values(), mass_edges_reco, pt_edges)

        reco_proj_misses = misses.project("ptgen", "mpt_gen")
        self.h2d_misses, _ = reorder_to_expected_2d(reco_proj_misses.values(), mass_edges_gen, pt_edges)

        nominal_gen = resp_matrix_4d_gen[{"systematic": "nominal"}]
        proj_gen = nominal_gen.project("ptreco", "mpt_reco", "ptgen", "mpt_gen")
        self.M_np_2d_gen, _ = reorder_to_expected(proj_gen.values(flow=False), mass_edges_gen, pt_edges, mass_edges_gen)
        self.mosaic_gen, _ = mosaic_no_padding(
            self.M_np_2d_gen,
            mass_edges_gen,
            mass_edges_gen,
            gen_mass_edges_by_pt,
            gen_mass_edges_by_pt,
        )

        self.fakes_2d = merge_mass_flat(self.h2d_fakes, mass_edges_reco, reco_mass_edges_by_pt)
        self.misses_2d = merge_mass_flat(self.h2d_misses, mass_edges_gen, gen_mass_edges_by_pt)

    def _prepare_herwig_inputs(
        self,
        resp_matrix_4d_herwig,
        fakes_herwig,
        misses_herwig,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        resp_matrix_4d_syst = resp_matrix_4d_herwig[{"systematic": "nominal"}]
        h2d_herwig = resp_matrix_4d_syst.project("ptreco", "mpt_reco").values(flow=False)
        self.h2d_herwig, _ = reorder_to_expected_2d(h2d_herwig, mass_edges_reco, pt_edges)

        reco_proj_fakes = fakes_herwig.project("ptreco", "mpt_reco")
        self.h2d_fakes_herwig, _ = reorder_to_expected_2d(reco_proj_fakes.values(), mass_edges_reco, pt_edges)

        reco_proj_misses = misses_herwig.project("ptgen", "mpt_gen")
        self.h2d_misses_herwig, _ = reorder_to_expected_2d(reco_proj_misses.values(), mass_edges_gen, pt_edges)

        self.fakes_2d_herwig = merge_mass_flat(self.h2d_fakes_herwig, mass_edges_reco, reco_mass_edges_by_pt)
        self.misses_2d_herwig = merge_mass_flat(self.h2d_misses_herwig, mass_edges_gen, gen_mass_edges_by_pt)

    def _response_matrix_for_systematic(self, syst, resp_matrix_4d_herwig, sys_matrix_dic):
        if syst in {"herwigUp", "herwigDown"}:
            resp_matrix_4d_syst = resp_matrix_4d_herwig[{"systematic": "nominal"}]
            proj = resp_matrix_4d_syst.project("ptreco", "mpt_reco", "ptgen", "mpt_gen")
            return sys_matrix_dic[syst], proj.variances(flow=False)
        return sys_matrix_dic[syst], None

    def _finalize_reco_views(self, mass_edges_reco, reco_mass_edges_by_pt):
        self.M_np_2d = self.M_np_2d_dict["nominal"]
        self.mosaic = self.mosaic_dict["nominal"]
        self.mosaic_2d = merge_mass_flat(self.h2d, mass_edges_reco, reco_mass_edges_by_pt)

        if "herwigUp" in self.systematics or "herwigDown" in self.systematics:
            self.mosaic_herwig_2d = merge_mass_flat(self.h2d_herwig, mass_edges_reco, reco_mass_edges_by_pt)

    def _load_data(self, filename_mc='latest_pkl/0508/mc_0508_full.pkl', filename_data="latest_pkl/0508/data_0508_full.pkl", filename_herwig='latest_pkl/0508/herwig_0508_full.pkl', filename_jk_data=DEFAULT_JK_DATA_FILE):
        print("------------- Adding inputs to unfolder -----------------")
        self._merge_eras()
        output_pythia = self.pythia_hists
        print("Keys in pythia file:", output_pythia.keys())
        output_herwig = self._load_pickle(filename_herwig)
        output_data = self._load_pickle(filename_data)
        output_jk_data = self._load_pickle(filename_jk_data)
        print("Keys in jk data file:", output_jk_data.keys())

        sys_matrix_dic = self.sys_matrix_dic
        keys = self._histogram_keys()

        pythia4d = output_pythia[keys["response"]]
        herwig4d = output_herwig[keys["response"]]
        pythia2d = output_pythia[keys["reco"]]
        herwig2d = output_herwig[keys["reco"]]
        data2d = output_data[keys["reco"]]
        data2d_jk = output_jk_data[keys["reco"]]
        pythia_gen2d = output_pythia[keys["gen"]]
        herwig_gen2d = output_herwig[keys["gen"]]

        pythia4d_gen = rebin_hist(pythia4d.copy(), 'mpt_reco',self.edges_gen )
        herwig4d_gen = rebin_hist(herwig4d.copy(), 'mpt_reco',self.edges_gen )

        resp_matrix_4d_gen = pythia4d_gen

        reco_mass_edges_by_pt = self.bins.reco_rho_edges_by_pt
        gen_mass_edges_by_pt = self.bins.gen_rho_edges_by_pt

        fakes = pythia2d.project('ptreco', 'mpt_reco', 'systematic')[:, :, 'nominal'] + (-1)*pythia4d.project('ptreco', 'mpt_reco', 'systematic')[:, :, 'nominal']
        fakes_herwig = herwig2d.project('ptreco', 'mpt_reco', 'systematic') + (-1)*herwig4d.project('ptreco', 'mpt_reco', 'systematic')
        self.fakes = fakes
        self.fakes_herwig = fakes_herwig

        misses = pythia_gen2d.project('ptgen', 'mpt_gen', 'systematic')[:, :, 'nominal'] + (-1)*pythia4d.project('ptgen', 'mpt_gen', 'systematic')[:, :, 'nominal']
        misses_herwig = herwig_gen2d.project('ptgen', 'mpt_gen', 'systematic') + (-1)*herwig4d.project('ptgen', 'mpt_gen', 'systematic')
        self.misses = misses
        self.misses_herwig = misses_herwig

        self.mosaic_dict = {}
        self.M_np_2d_dict = {}
        resp_matrix_4d = pythia4d
        resp_matrix_4d_herwig = herwig4d
        input_data = data2d


        self.input_data = input_data
        self.pythia_2d = pythia2d
        self.pythia_4d = pythia4d
        pt_edges        = self.bins.pt_edges
        mass_edges_reco = self.bins.rho_edges
        mass_edges_gen  = self.bins.rho_edges_gen

        print("Loaded pkl files and rebinned histograms.")

        print("Processing jk inputs...")
        self.mosaic_2d_jk_list = self._prepare_jackknife_inputs(
            data2d_jk,
            mass_edges_reco,
            pt_edges,
            reco_mass_edges_by_pt,
        )
        
        self.mosaic_jk_list = []

        for syst in self.systematics:
            if syst == 'nominal':
                print("Processing nominal systematic:", syst)
                self._prepare_nominal_inputs(
                    input_data,
                    fakes,
                    misses,
                    resp_matrix_4d_gen,
                    mass_edges_reco,
                    mass_edges_gen,
                    pt_edges,
                    reco_mass_edges_by_pt,
                    gen_mass_edges_by_pt,
                )
            if syst in {"herwigUp", "herwigDown"}:
                print("Processing Herwig systematic:", syst)
                self._prepare_herwig_inputs(
                    resp_matrix_4d_herwig,
                    fakes_herwig,
                    misses_herwig,
                    mass_edges_reco,
                    mass_edges_gen,
                    pt_edges,
                    reco_mass_edges_by_pt,
                    gen_mass_edges_by_pt,
                )

            M_np, _ = self._response_matrix_for_systematic(syst, resp_matrix_4d_herwig, sys_matrix_dic)
            self.M_np_orig = M_np
            self.M_np_2d_dict[syst], _ = reorder_to_expected(M_np, mass_edges_reco, pt_edges, mass_edges_gen)
            self.mosaic_dict[syst], _ = mosaic_no_padding(
                self.M_np_2d_dict[syst], mass_edges_reco, mass_edges_gen,
                reco_mass_edges_by_pt, gen_mass_edges_by_pt
            )
                
        print("Loaded data and prepared response matrices.")
        self.y_unf_jk_list = []
        self._finalize_reco_views(mass_edges_reco, reco_mass_edges_by_pt)
        print("h2d shape:", self.h2d.shape)
        print("reco_mass_edges_by_pt:", reco_mass_edges_by_pt)
        print("len reco_mass_edges_by_pt:", len(reco_mass_edges_by_pt))
    def plot_fakes_misses(self, show=True):
        title_list = ["",r"200 $<$ $p_T$ $<$ 290 GeV", r"290 $<$ $p_T$ $<$ 400 GeV", r"400 $<$ $p_T$ $< \, \infty$  GeV"]

        fakerate = self.fakes_2d/self.mosaic_2d

        fakerate = np.nan_to_num(fakerate, nan=0.0)# posinf=0.0, neginf=0.0)
        efficiency = 1 - (self.misses_2d/(self.misses_2d + self.mosaic.sum(axis=0)))
        efficiency_pt_binned = unflatten_gen_by_pt(efficiency, self.bins.gen_rho_edges_by_pt)
        fakerate_pt_binned = unflatten_gen_by_pt(fakerate, self.bins.reco_rho_edges_by_pt)

        for i in range(4):
            plt.stairs(1-fakerate_pt_binned[i], self.bins.reco_rho_edges_by_pt[i], label = f"Fake rate", lw = 1.5)
            plt.stairs(efficiency_pt_binned[i],self.bins.gen_rho_edges_by_pt[i], label = f"Efficiency", lw = 1.5)
            plt.legend(title = title_list[i])
            plt.xlabel(r"log($\rho^2$)")
            if self.groomed:
                plt.xlim(-4.5 , 0)
            else:
                plt.xlim(-2.5 , 0)
            plt.ylim(0,1.05)
            hep.cms.label("Internal", data = False, lumi = 138, fontsize = 20)
            if self.groomed:
                save_path = f"./outputs/rho/fakerates_groomed_{i-1}.png"
            else:
                save_path = f"./outputs/rho/fakerates_ungroomed_{i-1}.png"
            self._finalize_plot(save_path=save_path, show=show)

    def _compute_stat_unc(self):
        for i in range(10):
            meas_flat = self.mosaic_2d_jk_list[i]
            self._perform_unfold(systematic = 'nominal', closure = self.closure, herwig_closure = self.herwig_closure, meas_flat = meas_flat, do_jk = True)
        
        std = np.std(self.y_unf_jk_list, axis = 0)
        self.stat_unc_frac = np.abs(std/self.y_unf)

        self.stat_unc_pt_binned = unflatten_gen_by_pt(self.stat_unc_frac, self.bins.gen_rho_edges_by_pt)

        #plt.stairs(self.stat_unc_frac)
        plt.show()

    def _select_measured_spectrum(self, closure, herwig_closure, meas_flat):
        if meas_flat is None:
            if closure:
                meas_flat = self.mosaic.sum(axis=1)
            else:
                meas_flat = self.mosaic_2d
            if herwig_closure:
                meas_flat = self.mosaic_herwig_2d
        return meas_flat

    def _build_root_binning(self):
        truth_root = ROOT.TUnfoldBinning("truth")
        reco_root = ROOT.TUnfoldBinning("reco")

        truth_signal = truth_root.AddBinning("signal")
        reco_primary = reco_root.AddBinning("primary")

        for i, edges in enumerate(self.bins.gen_rho_edges_by_pt):
            truth_node = truth_signal.AddBinning(f"pt{i}")
            truth_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        for i, edges in enumerate(self.bins.reco_rho_edges_by_pt):
            reco_node = reco_primary.AddBinning(f"pt{i}")
            reco_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        return truth_root, reco_root

    def _fill_root_histogram(self, hist, values):
        for index, value in enumerate(values, 1):
            hist.SetBinContent(index, float(value))

    def _fill_response_histogram(self, h_resp, resp_np, misses):
        n_reco, n_true = resp_np.shape
        for i_reco in range(n_reco):
            for j_true in range(n_true):
                h_resp.SetBinContent(j_true + 1, i_reco + 1, resp_np[i_reco, j_true])
        for j_true in range(n_true):
            h_resp.SetBinContent(j_true + 1, 0, misses[j_true])

    def _store_covariances(self, unfold, systematic):
        if systematic == "nominal":
            self.cov = unfold.GetEmatrixTotal("cov", "Covariance Matrix")
            self.cov_uncorr = unfold.GetEmatrixSysUncorr(
                "cov_uncorr",
                "Covariance Matrix from Uncorrelated Uncertainties",
            )
            self.cov_uncorr_data = unfold.GetEmatrixInput(
                "cov_uncorr_data",
                "Covariance Matrix from Stat Uncertainties of Input Data",
            )
            self.cov_total = unfold.GetEmatrixTotal("total", "Cov")

            _, n_true = self.mosaic.shape
            self.cov_np = np.zeros((n_true, n_true))
            self.cov_uncorr_np = np.zeros((n_true, n_true))
            self.cov_data_np = np.zeros((n_true, n_true))
            for i in range(1, n_true + 1):
                for j in range(1, n_true + 1):
                    self.cov_np[i - 1, j - 1] = self.cov.GetBinContent(i, j)
                    self.cov_uncorr_np[i - 1, j - 1] = self.cov_uncorr.GetBinContent(i, j)
                    self.cov_data_np[i - 1, j - 1] = self.cov_uncorr_data.GetBinContent(i, j)

        if systematic == "herwigUp":
            self.cov_uncorr_data = unfold.GetEmatrixInput(
                "cov_uncorr_data",
                "Covariance Matrix from Stat Uncertainties of Input Data",
            )
            _, n_true = self.mosaic.shape
            self.cov_data_herwig_np = np.zeros((n_true, n_true))
            for i in range(1, n_true + 1):
                for j in range(1, n_true + 1):
                    self.cov_data_herwig_np[i - 1, j - 1] = self.cov_uncorr_data.GetBinContent(i, j)

    def _store_unfold_result(self, systematic, do_jk, unfold, h_meas, h_true):
        h_unfold = unfold.GetOutput("unfold")
        h_folded = unfold.GetFoldedOutput("folded")

        y_meas, ye_meas = self._th1_to_arrays(h_meas)
        y_true, ye_true = self._th1_to_arrays(h_true)
        x_folded, _ = self._th1_to_arrays(h_folded)
        y_unf, ye_unf = self._th1_to_arrays(h_unfold)

        if do_jk and systematic == "nominal":
            self.y_unf_jk_list.append(y_unf)
            return

        if systematic == "herwigUp":
            self.y_true_herwig = self.mosaic_dict["herwigUp"].sum(axis=0)

        if systematic == "nominal":
            self.y_meas = y_meas
            self.ye_meas = ye_meas
            self.y_unf = y_unf
            self.ye_unf = ye_unf
            self.y_true = y_true
            self.x_folded = x_folded
            self.L = unfold.GetL("Lmatrix", "Lmatrix")
        else:
            self.y_unf_dict[systematic] = y_unf

    def _perform_unfold(self, systematic = 'nominal', closure = False, herwig_closure = False, meas_flat = None, do_jk = False):
        resp_np = self.mosaic_dict[systematic]
        meas_flat = self._select_measured_spectrum(closure, herwig_closure, meas_flat)

        true_flat = self.mosaic.sum(axis = 0) + self.misses_2d
        n_reco, n_true = resp_np.shape
        assert len(meas_flat) == n_reco, "measured spectrum must have n_reco bins"
        truth_root, reco_root = self._build_root_binning()
        h_meas = reco_root.CreateHistogram("hRecoData")
        h_true = truth_root.CreateHistogram("hTruthPrior")
        h_resp = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(truth_root, reco_root, "hResponse")

        misses = self.misses_2d_herwig if systematic in {"herwigUp", "herwigDown"} else self.misses_2d
        self._fill_response_histogram(h_resp, resp_np, misses)
        self._fill_root_histogram(h_meas, meas_flat)
        self._fill_root_histogram(h_true, true_flat)
        self.h_resp = h_resp

        unfold = ROOT.TUnfoldDensity(
            h_resp,
            ROOT.TUnfold.kHistMapOutputHoriz,          # mapping of TH2 axes
            ROOT.TUnfold.kRegModeDerivative,            # curvature regularisation
            ROOT.TUnfold.kEConstraintArea,             # one global area constraint
            ROOT.TUnfoldDensity.kDensityModeBinWidth,  # bin-width aware scaling
            truth_root,                              # output (truth) binning tree
            reco_root,                               # input  (reco)  binning tree
            "signal",                                # regularisationDistributionName
            "*[UOB]"                                 # regularisationAxisSteering
        )

        # feed measured spectrum
        status = unfold.SetInput(h_meas)
        if status >= 10000:
            raise RuntimeError("TUnfold input had overflow/underflow – check your hist.")
        unfold.DoUnfold(0.0)
        self._store_covariances(unfold, systematic)
        self._store_unfold_result(systematic, do_jk, unfold, h_meas, h_true)
        
        
    
    def plot_L(self, show=True):
        lMatrix = self.L
        #try plotting the L matrix root way
        c = ROOT.TCanvas("c", "L-curve Matrix", 800, 600)
        lMatrix.Draw("colz")
        c.SaveAs("outputs/rho/unfold/L_matrix_root.png")
        nx, ny = lMatrix.GetNbinsX(), lMatrix.GetNbinsY() 
        l_np = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                l_np[i, j] = lMatrix.GetBinContent(i + 1, j + 1)
        #mask zeros for better visualization
        l_np_masked = np.ma.masked_where(l_np == 0, l_np)
        plt.imshow(l_np_masked, origin='lower', aspect='auto')
        plt.colorbar(label='L-curve Matrix Value')
        hep.cms.label(self.cms_label, data = False, lumi = 138 , com = 13, fontsize = 20)
        self._finalize_plot(save_path="outputs/rho/unfold/L_matrix_matplotlib.png", show=show)
        
        


    
    def _th1_to_arrays(self,h):
        nb = h.GetNbinsX()                       # bin numbers
        x  = np.arange(1, nb + 1)
        y  = np.array([h.GetBinContent(int(i)) for i in x])
        ye = np.array([h.GetBinError(int(i))   for i in x])
        return  y, ye
    def plot_folded(self, show=True):
        folded_pt_binned = unflatten_gen_by_pt(self.x_folded, self.bins.reco_rho_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.bins.reco_rho_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.bins.reco_rho_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths_reco = np.diff(self.bins.reco_rho_edges_by_pt[i])
            # two-panel plot: main + ratio
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            plt.sca(ax_top)
            hep.histplot(folded_pt_binned[i]/bin_widths_reco/folded_pt_binned[i].sum(), self.bins.reco_rho_edges_by_pt[i], label='Folded', color='r', alpha=0.8, ls='dotted', lw=3, ax=ax_top)
            hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.bins.reco_rho_edges_by_pt[i], color='k', ls='--', alpha=1, label='Measured Data', ax=ax_top)

            # ratio (Measured / Folded)
            edges = np.array(self.bins.reco_rho_edges_by_pt[i], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            folded = folded_pt_binned[i]/bin_widths_reco/folded_pt_binned[i].sum()
            meas = measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum()
            ratio = np.divide(meas, folded, out=np.full_like(meas, np.nan), where=folded != 0)

            plt.sca(ax_bot)
            ax_bot.axhline(1.0, color='gray', ls='--')
            ax_bot.errorbar(centers, ratio, yerr=None, fmt='o', color='k')
            ax_bot.set_ylabel('Data / Folded')
            ax_bot.set_xlim(edges[0], edges[-1])
            ax_bot.set_ylim(0.5, 1.5)
            plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")

            # switch back to top axes so subsequent plotting (reco_mc, legend, labels) goes to the main panel
            plt.sca(ax_top)
            #hep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.bins.reco_rho_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title) 
            
            if self.groomed:
                #plt.xlim(0,250)
                plt.xlim(-4.5, 0)
                hep.cms.label(self.cms_label, data = True,lumi = 138, com = 13, fontsize = 20)
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(-2.5, 0)
                #plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")
                hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
            save_path = f"./outputs/rho/unfold/folded_groomed_{i-1}.pdf" if self.groomed else f"./outputs/rho/unfold/folded_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
    
    def plot_jk(self, show= True):
        # Outputs
        n_pt_bins = len(self.pt_edges) - 1
        jk_pt_binned = [
            unflatten_gen_by_pt(sample, self.bins.gen_rho_edges_by_pt)
            for sample in self.y_unf_jk_list
        ]

        for pt_index in range(n_pt_bins):
            fig, ax = plt.subplots()
            for jk_index, unfolded_pt_binned in enumerate(jk_pt_binned):
                ax.stairs(
                    unfolded_pt_binned[pt_index],
                    self.bins.gen_rho_edges_by_pt[pt_index],
                    label=f"JK sample {jk_index}",
                    alpha=0.6,
                )

            pt_low = int(self.pt_edges[pt_index])
            pt_high = self.pt_edges[pt_index + 1]
            pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

            ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
            
            ax.set_ylabel("Unfolded entries")
            if self.groomed:
                ax.set_xlim(-4.5, 0)
                ax.set_xlabel(r"log($\rho^2$), Groomed")
                save_path = f"./outputs/rho/unfold/jk_outputs_groomed_pt{pt_index-1}.pdf"
            else:
                ax.set_xlim(-2.5, 0)
                ax.set_xlabel(r"log($\rho^2$), Ungroomed")
                save_path = f"./outputs/rho/unfold/jk_outputs_ungroomed_pt{pt_index-1}.pdf"

            plt.sca(ax)
            hep.cms.label(self.cms_label, data=True, lumi=138, com=13, fontsize=20)
            plt.tight_layout()
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        # Inputs
        for pt_index in range(n_pt_bins):
            fig, ax = plt.subplots()
            for jk_index, mosaic_2d_jk in enumerate(self.mosaic_2d_jk_list):
                reco_pt_binned_jk = unflatten_gen_by_pt(self.mosaic_2d_jk_list[jk_index], self.bins.reco_rho_edges_by_pt)
                ax.stairs(
                    reco_pt_binned_jk[pt_index],
                    self.bins.reco_rho_edges_by_pt[pt_index],
                    label=f"JK sample {jk_index}",
                    alpha=0.6,
                )

            pt_low = int(self.pt_edges[pt_index])
            pt_high = self.pt_edges[pt_index + 1]
            pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

            ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
            
            ax.set_ylabel("Entries")
            if self.groomed:
                ax.set_xlim(-4.5, 0)
                ax.set_xlabel(r"log($\rho^2$), Groomed")
                save_path = f"./outputs/rho/unfold/jk_inputs_groomed_pt{pt_index-1}.pdf"
            else:
                ax.set_xlim(-2.5, 0)
                ax.set_xlabel(r"log($\rho^2$), Ungroomed")
                save_path = f"./outputs/rho/unfold/jk_inputs_ungroomed_pt{pt_index-1}.pdf"

            plt.sca(ax)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20)
            self._finalize_plot(save_path=save_path, show=show, fig=fig)



    def plot_bottom_line(self, show=True):
        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.bins.gen_rho_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.bins.gen_rho_edges_by_pt)

        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.bins.reco_rho_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.bins.reco_rho_edges_by_pt)
        
        #now plot the ratio of unfolded to true and measured to reco mc in the same axis, just the ratio plot (no main panel)
        for i in range(len(self.pt_edges)-1):
            # two-panel plot: main + ratio
            error = self.normalized_results[i]['stat_unc']/self.normalized_results[i]['unfolded']
            fig, ax = plt.subplots(figsize=(12, 9))
            bin_widths = np.diff(self.bins.gen_rho_edges_by_pt[i])
            unfolded = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
            true = true_pt_binned[i]/bin_widths/true_pt_binned[i].sum()
            
            ratio_unf_true = np.divide(unfolded, true, out=np.full_like(unfolded, np.nan), where=true != 0)
            
            bin_widths_reco = np.diff(self.bins.reco_rho_edges_by_pt[i])
            measured = measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum()
            reco_mc = reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum()
            ratio_meas_reco = np.divide(measured, reco_mc, out=np.full_like(measured, np.nan), where=reco_mc != 0)
            
            ax.axhline(1.0, color='gray', ls='--')
            hep.histplot(ratio_unf_true, self.bins.gen_rho_edges_by_pt[i], yerr = np.abs(error),label='Unfolded / True', color='k', ls='--')
            hep.histplot(ratio_meas_reco, self.bins.reco_rho_edges_by_pt[i],  label='Measured / Reco_MC', color='r', ls=':')
            ax.set_ylabel('Ratio')
            ax.set_xlim(self.bins.gen_rho_edges_by_pt[i][0], self.bins.gen_rho_edges_by_pt[i][-1])
            ax.set_ylim(0.5, 1.5)
            if self.groomed:
                plt.xlim(-4.5, 0)
            else:
                plt.xlim(-2.5, 0)
            plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title) 
            hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
            save_path = f"./outputs/rho/bottom_line_groomed_{i-1}.pdf" if self.groomed else f"./outputs/rho/bottom_line_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_unfolded_fancy(self, log=False, show=True):
        markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
        npt = len(self.pt_edges)-1
        title_list = ["",r"200$<$$p_T$$<$290 GeV", r"290 $<$$p_T$$<$400 GeV", r"400 $<$$p_T$$< \, \infty$  GeV"]
        for i in range(npt):
            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 10)
            )
            plt.sca(ax_main)
            plt.stairs( self.normalized_results[i]['unfolded'] + self.normalized_results[i]['syst_unc']['up'],
                self.bins.gen_rho_edges_by_pt[i],
                baseline = self.normalized_results[i]['unfolded'] - self.normalized_results[i]['syst_unc']['down'],
                fill = True, color = "yellowgreen" , label = r"Syst. $\oplus$ Stat. Unc.")
            plt.stairs( self.normalized_results[i]['unfolded'] + self.normalized_results[i]['stat_unc'],
                self.bins.gen_rho_edges_by_pt[i],
                baseline = self.normalized_results[i]['unfolded'] - self.normalized_results[i]['stat_unc'],
                fill = True, color = "darkgreen" , label = "Stat. Unc.")
            plt.stairs(self.normalized_results[i]['true'], self.bins.gen_rho_edges_by_pt[i], label = 'PYTHIA7', color = 'b', ls = 'dotted', lw = 3)

            plt.legend(title = title_list[i])
            hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
            plt.ylabel(r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d(\log{\rho^2})dp_T} (GeV^{-1})$")

            # Ratio Plot
            plt.sca(ax_ratio)
            plt.axhline(1.0, color='gray', ls='--')
            ratio = np.divide(self.normalized_results[i]['unfolded'], self.normalized_results[i]['true'])
            ratio_syst_up = np.divide(self.normalized_results[i]['unfolded'] + self.normalized_results[i]['syst_unc']['up'], self.normalized_results[i]['true'])
            ratio_syst_down = np.divide(self.normalized_results[i]['unfolded'] - self.normalized_results[i]['syst_unc']['down'], self.normalized_results[i]['true'])

            ratio_stat_up = np.divide(self.normalized_results[i]['unfolded'] + self.normalized_results[i]['stat_unc'], self.normalized_results[i]['true'])
            ratio_stat_down = np.divide(self.normalized_results[i]['unfolded'] - self.normalized_results[i]['stat_unc'], self.normalized_results[i]['true'])

            plt.stairs(ratio_syst_up, self.bins.gen_rho_edges_by_pt[i], baseline=ratio_syst_down, fill=True, color="yellowgreen", label=r"Syst. $\oplus$ Stat. Unc.")
            plt.stairs(ratio_stat_up, self.bins.gen_rho_edges_by_pt[i], baseline=ratio_stat_down, fill=True, color="darkgreen", label="Stat. Unc.")
            plt.ylim(0, 2)
            plt.xlabel(r"Groomed Jet  $\log_{10}{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log_{10}{\rho^2}$")
            plt.ylabel(r"$\frac{Data}{Simulation}$")
            if self.groomed:
                plt.xlim(-4.5 , 0)
            else:
                plt.xlim(-2.5 , 0)
            if self.closure:
                save_path = f"./outputs/rho/closure_groomed_{i-1}.pdf" if self.groomed else f"./outputs/rho/closure_ungroomed_{i-1}.pdf"
            else:
                save_path = f"./outputs/rho/unfold/groomed_{i-1}.pdf" if self.groomed else f"./outputs/rho/unfold/ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        
        # Now also plot a summary plot, with all of them together, but shifted on y axis for visibility

        min_floor = None
        for i in range(1, npt):
            j = i 
            scale = 10 ** j
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)

            y_syst_up = scale * (unfolded + syst_up)
            y_syst_down = scale * (unfolded - syst_down)
            y_stat_up = scale * (unfolded + stat_unc)
            y_stat_down = scale * (unfolded - stat_unc)

            # User requested floor = 10**(i-1) for plotting the i-th overlay
            floor = 10 ** (j - 2)
            if min_floor is None or floor < min_floor:
                min_floor = floor
            plt.stairs(scale * np.array(self.normalized_results[i]['true'], dtype=float), self.bins.gen_rho_edges_by_pt[i], label='PYTHIA7', color='b', ls='dotted', lw=3)
            y_syst_down_clipped = np.where(y_syst_down <= floor, floor, y_syst_down)
            y_stat_down_clipped = np.where(y_stat_down <= floor, floor, y_stat_down)

            plt.stairs(y_syst_up, self.bins.gen_rho_edges_by_pt[i], baseline=y_syst_down_clipped, fill=True, color="yellowgreen", label=r"Syst. $\oplus$ Stat. Unc.", alpha = 0.8)
            plt.stairs(y_stat_up, self.bins.gen_rho_edges_by_pt[i], baseline=y_stat_down, fill=True, color="darkgreen", label="Stat. Unc.")
            rho_edges = np.array(self.bins.gen_rho_edges_by_pt[i], dtype=float)
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            plt.plot(centers, scale * unfolded, label=rf'$10^{j}$ x {title_list[i]}', color='k', lw=0, marker=markers[i])
            

        plt.yscale('log')
        plt.yscale('log')
        # Group duplicate legend entries: keep first occurrence, hide subsequent ones
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        for h, l in zip(handles, labels):
            if l == "" or l in seen:
                try:
                    h.set_label('_nolegend_')  # matplotlib ignores labels starting with '_'
                except Exception:
                    pass
            else:
                seen.add(l)
        plt.legend(fontsize  = 15)
        plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")
        plt.ylabel(r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d(\log{\rho^2})dp_T} (GeV^{-1})$")
        hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
        # set a reasonable lower y-limit based on the smallest floor used
        # try:
        #     if min_floor is not None:
        #         plt.ylim(min_floor * 1e-3, None)
        # except Exception:
        #     pass

        if self.groomed:
            plt.xlim(-4.5 , 0)
        else:
            plt.xlim(-2.5 , 0)

        save_path = f"./outputs/rho/groomed_summary.pdf" if self.groomed else f"./outputs/rho/ungroomed_summary.pdf"
        self._finalize_plot(save_path=save_path, show=show)




    def plot_unfolded(self, log=False, show=True):

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.bins.gen_rho_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.bins.reco_rho_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.bins.reco_rho_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.bins.gen_rho_edges_by_pt)
        true_herwig_pt_binned = unflatten_gen_by_pt(self.y_true_herwig, self.bins.gen_rho_edges_by_pt) 
        #error_pt_binned = unflatten_gen_by_pt(self.ye_unf, self.bins.gen_rho_edges_by_pt)
        self.normalized_herwig = []
        #print("Herwig pt Binned", true_herwig_pt_binned)
        self.herwig_closure_unc = []
        for i in range(len(self.pt_edges)-1):
            yerr = self.normalized_results[i]['syst_unc']['up']
            bin_widths = np.diff(self.bins.gen_rho_edges_by_pt[i])
            bin_widths_reco = np.diff(self.bins.reco_rho_edges_by_pt[i])
            #self.normalized_herwig.append(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum())
            if self.herwig_closure:
                hep.histplot(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum(), self.bins.gen_rho_edges_by_pt[i], color = 'green', label = 'Herwig', alpha = 0.7, ls = 'dotted')
            else:
                hep.histplot(true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(), self.bins.gen_rho_edges_by_pt[i], color = 'b', label = 'PYTHIA', alpha = 0.8, ls = 'dotted', lw = 3)
            hep.histplot(unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), self.bins.gen_rho_edges_by_pt[i], label = 'Unfolded Herwig' if self.herwig_closure else 'Unfolded', color = 'k', ls = '--' )

            

            #hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], color = 'k', ls= '--', alpha= 0.5, label = 'Meas' )
            #dhep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f" {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title, fontsize = 18) 
            
            if self.groomed:
                #plt.xlim(0,250)
                plt.xlim(-4.5, 0)
                plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(-2.5, 0)
                #plt.xlim(20,250)
                plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")
            save_path = f"./outputs/rho/unfold/unfolded_basic_groomed_{i-1}.pdf" if self.groomed else f"./outputs/rho/unfold/unfolded_basic_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show)
            # Plot relative difference: (true - unfolded) / true after normalization
            # true_norm = true_pt_binned[i] / np.diff(self.bins.gen_mass_edges_by_pt[i]) / true_pt_binned[i].sum()
            # true_norm = true_herwig_pt_binned[i] / np.diff(self.bins.gen_mass_edges_by_pt[i]) / true_herwig_pt_binned[i].sum()
            # unfolded_norm = unfolded_pt_binned[i] / np.diff(self.bins.gen_mass_edges_by_pt[i]) / unfolded_pt_binned[i].sum()
            
            # rel_diff = np.abs(true_norm - unfolded_norm) / true_norm
            # hep.histplot(rel_diff, self.bins.gen_mass_edges_by_pt[i], label="(Herwig - Unfolded) / Herwig", color="r")
            # title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            # plt.legend(title = title) 
            if self.herwig_closure:
                plt.figure(figsize=(10, 3))
                true_herwig = true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum()
                unfolded = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
                herwig_closure_unc = np.abs(true_herwig - unfolded) / true_herwig
                self.herwig_closure_unc.append(herwig_closure_unc)
                plt.stairs(herwig_closure_unc, self.bins.gen_rho_edges_by_pt[i], label = 'Closure Unc (|Herwig - Unfolded| / Herwig)', color = 'g', ls = 'dotted')
                if not self.groomed:
                    plt.xlim(-2.5, 0)
                else:
                    plt.xlim(-4.5, 0)
                plt.ylim(0, 1)
                plt.xlabel(r"Groomed Jet  $\log{\rho^2}$" if self.groomed else r"Ungroomed Jet $\log{\rho^2}$")
                #plt.legend()
                save_path = f"./outputs/rho/unfold/herwig_closure_unc_groomed_{i-1}.pdf" if self.groomed else f"./outputs/rho/unfold/herwig_closure_unc_ungroomed_{i-1}.pdf"
                self._finalize_plot(save_path=save_path, show=show)
        # Save uncertainty in a file for later use
        if self.herwig_closure:
            if self.groomed:
                np.save("./inputs/rho/herwig_closure_unc_groomed.npy", self.herwig_closure_unc)
            else:
                np.save("./inputs/rho/herwig_closure_unc_ungroomed.npy", self.herwig_closure_unc)
            # if self.groomed:
            #     plt.xlim(0,250)
            #     plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            # #plt.ylim(0,0.02)
            # if not self.groomed:
            #     plt.xlim(20,250)
            #     plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            # plt.show()
    def _normalize_result(self):
        print("Normalizing results...")
        self.normalized_results = []
        gen_mass_bin_edges_by_pt = self.bins.gen_rho_edges_by_pt
        reco_mass_bin_edges_by_pt = self.bins.reco_rho_edges_by_pt

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, gen_mass_bin_edges_by_pt)
        print("Unfolded pt binned:", unfolded_pt_binned)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, reco_mass_bin_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), reco_mass_bin_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, gen_mass_bin_edges_by_pt)
        error_pt_binned = unflatten_gen_by_pt(self.ye_unf, gen_mass_bin_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths = np.diff(gen_mass_bin_edges_by_pt[i])
            bin_widths_reco = np.diff(reco_mass_bin_edges_by_pt[i])
            result = {
                "true": true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(),
                "unfolded": unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), # Taking absolute values to avoid negative bins
                "unfolded_err": error_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(),
                "measured": measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(),
                "reco_mc": reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(),
                "pt_bin": (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf')),
                "mgen_edges": self.bins.gen_mass_edges_by_pt[i]
            }
            self.normalized_results.append(result)
        
        # Storing normalized results for systematics
        self.normalized_systematics = []
        # Prepare normalized_systematics as a list of dicts, one per pt bin

        for i in range(len(self.pt_edges)-1):
            pt_bin = (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf'))
            unfolded = {}
            bin_widths = np.diff(gen_mass_bin_edges_by_pt[i])
            bin_widths_reco = np.diff(reco_mass_bin_edges_by_pt[i])
            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf_dict[syst], gen_mass_bin_edges_by_pt)
                unfolded[syst] = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum() # Taking absolute values to avoid negative bins
            self.normalized_systematics.append({
            "pt_bin": pt_bin,
            "unfolded": unfolded # Taking absolute values to avoid negative bins
            })
    def _compute_total_systematic(self):
        print("Computing total systematic uncertainty...")
        # Compute total systematic uncertainty for each pt bin
        # load from file
        #herwig_unc = np.load("./inputs/rho/herwig_closure_unc_groomed.npy") if self.groomed else np.load("./inputs/rho/herwig_closure_unc_ungroomed.npy")
        #self.herwig_unc = herwig_unc
        for i in range(len(self.normalized_results)):
            nominal = self.normalized_results[i]['unfolded']
            syst_up_total = np.zeros_like(nominal)
            syst_down_total = np.zeros_like(nominal)
            
            for syst in self.systematics:
                if 'Down' in syst:
                    if syst.startswith('herwig'):
                        ## This is for taking just the difference
                        syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_down = np.abs(syst_down - nominal)
                        print("Systematic Down:", syst, "Diff Down:", diff_down)
                        syst_down_total += diff_down**2

                        ## This is for fethcing uncertainty from non-closure of herwig
                        # diff_down = herwig_unc[i] * nominal
                        # syst_down_total += diff_down**2
                    else:    
                        syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_down = np.abs(syst_down - nominal)
                        print("Systematic Down:", syst, "Diff Down:", diff_down)
                        syst_down_total += diff_down**2
                else:
                    if syst=='nominal':
                        continue
                    if syst.startswith('herwig'):
                        print("New Herwig adopted")
                        ## Difference way
                        syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_up = np.abs(syst_up - nominal)
                        print("Systematic Up:", syst, "Diff Up:", diff_up)
                        syst_up_total += diff_up**2

                        ## Fetching uncertainty from non-closure of herwig
                        # diff_up = herwig_unc[i] * nominal
                        # syst_up_total += diff_up**2
                    else:    
                        syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_up = np.abs(syst_up - nominal)
                        print("Systematic up:", syst, "Diff up:", diff_up)
                        syst_up_total += diff_up**2
            stat_unc = self.stat_unc_pt_binned[i] * nominal
                
            syst_up_total += stat_unc**2
            syst_down_total += stat_unc**2
            # Take sqrt of sum of squares for total uncertainty
            total_up_unc = np.sqrt(syst_up_total)
            total_down_unc = np.sqrt(syst_down_total)
            self.normalized_results[i]['syst_unc'] = {
            'up': total_up_unc,
            'down': total_down_unc
            }
            self.normalized_results[i]['stat_unc'] = stat_unc

    def plot_systematic_fraction(self, syst_name='all', show=True):
        # Plot the systematic uncertainties as a fraction of the nominal unfolded result
        
        for i, result in enumerate(self.normalized_results):
            if not hasattr(self, 'syst_fraction_dicts'):
                self.syst_fraction_dicts = []
            plt.figure(figsize=(12, 8))
            pt_bin = result['pt_bin']
            nominal = result['unfolded']
            total_syst_up = result['syst_unc']['up']
            total_syst_down = result['syst_unc']['down']


            # Group systematics by prefix
            jes_up = []
            jes_down = []
            jer_up = []
            jer_down = []
            ele_up = []
            ele_down = []
            mu_up = []
            mu_down = []
            print
            for syst_name in self.normalized_systematics[i]['unfolded']:
                if syst_name.startswith('JES'):
                    #print("Processing JES systematic:", syst_name)
                    if 'Down' in syst_name:
                        jes_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    else:
                        jes_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                elif syst_name.startswith('ele'):
                    if 'Down' in syst_name:
                        ele_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    else:
                        ele_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                elif syst_name.startswith('mu'):
                    if 'Down' in syst_name:
                        mu_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    else:
                        mu_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                elif syst_name.startswith('JER'):
                    #print("Processing JER systematic:", syst_name)
                    if 'Down' in syst_name:
                        jer_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    else:
                        jer_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
            
            #print("JES up variations:", jes_up)
            #print("JER up variations:", jer_up)

            # Combine grouped uncertainties in quadrature
            # Dictionary to store fractional systematic uncertainties by name
            # Store the systematic fraction dict as a class attribute for each pt bin
            
            syst_fraction_dict = {}

            # JES group
            if jes_up:
                print("Plotting JES group")
                jes_up_total = np.sqrt(np.sum([diff**2 for diff in jes_up], axis=0))
                jes_up_frac = np.abs(jes_up_total / nominal)
                hep.histplot(jes_up_frac, self.bins.gen_rho_edges_by_pt[i], label="JES ", ls = '--')
                syst_fraction_dict['JESUp'] = jes_up_frac
            if jes_down:
                jes_down_total = np.sqrt(np.sum([diff**2 for diff in jes_down], axis=0))
                jes_down_frac = np.abs(jes_down_total / nominal)
                syst_fraction_dict['JESDown'] = jes_down_frac
            
            # JER group
            if jer_up:
                jer_up_total = np.sqrt(np.sum([diff**2 for diff in jer_up], axis=0))
                jer_up_frac = np.abs(jer_up_total / nominal)
                hep.histplot(jer_up_frac, self.bins.gen_rho_edges_by_pt[i], label="JER ", ls = '--')
                syst_fraction_dict['JERUp'] = jer_up_frac
                print("Plotting JER group")
            if jer_down:
                jer_down_total = np.sqrt(np.sum([diff**2 for diff in jer_down], axis=0))
                jer_down_frac = np.abs(jer_down_total / nominal)
                syst_fraction_dict['JERDown'] = jer_down_frac

            # Electron SFs group
            if ele_up:
                ele_up_total = np.sqrt(np.sum([diff**2 for diff in ele_up], axis=0))
                ele_up_frac = np.abs(ele_up_total / nominal)
                hep.histplot(ele_up_frac, self.bins.gen_rho_edges_by_pt[i], label="Electron SFs", ls = '--')
                syst_fraction_dict['ElectronSFUp'] = ele_up_frac
            if ele_down:
                ele_down_total = np.sqrt(np.sum([diff**2 for diff in ele_down], axis=0))
                ele_down_frac = np.abs(ele_down_total / nominal)
                syst_fraction_dict['ElectronSFDown'] = ele_down_frac

            # Muon SFs group
            if mu_up:
                mu_up_total = np.sqrt(np.sum([diff**2 for diff in mu_up], axis=0))
                mu_up_frac = np.abs(mu_up_total / nominal)
                hep.histplot(mu_up_frac, self.bins.gen_rho_edges_by_pt[i], label="Muon SFs", ls = '--')
                syst_fraction_dict['MuonSFUp'] = mu_up_frac
            if mu_down:
                mu_down_total = np.sqrt(np.sum([diff**2 for diff in mu_down], axis=0))
                mu_down_frac = np.abs(mu_down_total / nominal)
                syst_fraction_dict['MuonSFDown'] = mu_down_frac

            # Individual non-grouped systematics
            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                if syst.startswith('JES') or syst.startswith('ele') or syst.startswith('mu') or syst.startswith('JER'):
                    continue  # Already handled above
                if syst.endswith('Up'):
                    syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                    if syst.startswith('herwig'):
                        ## case when fetching uncertainty from non-closure of herwig
                        # diff_up = nominal*self.herwig_unc[i]  Using the closure uncertainty as the systematic variation for Herwig
                        ## regular
                        diff_up = syst_up - nominal
                    else:
                        diff_up = syst_up - nominal
                    syst_fraction_up = np.abs(diff_up / nominal)
                    syst_fraction_dict[f"{syst}"] = syst_fraction_up
                    
                    if syst.startswith('herwig'):
                        hep.histplot(syst_fraction_up, self.bins.gen_rho_edges_by_pt[i], label=f"Model Uncertainty", ls = '-.')
                    else:
                        label = f"{syst[:-2]}"
                        label_dic = {'pu':'Pileup', 'l1prefiring': 'L1 Prefiring', 'q2': r'Q$^2$ Scale', 'pdf': 'PDF', 'herwig': 'Model Unc.', 'isr': 'ISR', 'fsr': 'FSR'}
                        if label in label_dic:
                            label = label_dic[label]

                        hep.histplot(syst_fraction_up, self.bins.gen_rho_edges_by_pt[i], label=f"{label}")
                if syst.endswith('Down'):
                    syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                    if syst.startswith('herwig'):
                        ## case when fetching uncertainty from non-closure of herwig
                        # diff_down = nominal*self.herwig_unc[i]
                        ## regular
                        diff_down = syst_down - nominal
                    else:
                        diff_down = syst_down - nominal
                    syst_fraction_down = np.abs(diff_down / nominal)
                    syst_fraction_dict[f"{syst}"] = syst_fraction_down
            stat_fraction = self.stat_unc_pt_binned[i]  
            hep.histplot(stat_fraction, self.bins.gen_rho_edges_by_pt[i], label="Stat Unc", ls='--')

            # Calculate systematic fraction
            total_syst_fraction_up = np.abs(total_syst_up / np.abs(nominal))
            total_syst_fraction_down = np.abs(total_syst_down / np.abs(nominal))

            # Store total up/down uncertainties as well
            syst_fraction_dict['Total_Up'] = total_syst_fraction_up
            syst_fraction_dict['Total_Down'] = total_syst_fraction_down

            # Save the dictionary for this pt bin as a class attribute
            self.syst_fraction_dicts.append(syst_fraction_dict)
            result['syst_fraction_dict'] = syst_fraction_dict

            # # Calculate systematic fraction
            # total_syst_fraction_up = total_syst_up / nominal
            # total_syst_fraction_down = total_syst_down / nominal

            # # Store total up/down uncertainties as well
            # syst_fraction_dict['Total_Up'] = total_syst_fraction_up
            # syst_fraction_dict['Total_Down'] = total_syst_fraction_down

            # # Save the dictionary for this pt bin
            # result['syst_fraction_dict'] = syst_fraction_dict

            # Plot the systematic fraction
            hep.histplot(total_syst_fraction_up, 
                        self.bins.gen_rho_edges_by_pt[i], 
                        label=f"Total ", color = 'k', lw = 3)
            plt.yscale('log')
            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV", loc='center left', bbox_to_anchor=(1, 0.5))
            hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)
            # if self.groomed:
            #     plt.xlim(20,250)
                
            # else:
            #     plt.xlim(10,250)
            # # place the last x-tick at 250 and label it with infinity
            # edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)
            # ticks = edges.copy()
            # ticks[-1] = 250.0
            # # remove the third tick to reduce clutter (index 2) if it exists
            # if ticks.size > 2:
            #     ticks = np.delete(ticks, 2)
            # # disable minor ticks on the x-axis
            # plt.gca().tick_params(axis='x', which='minor', bottom=False, top=False)
            # # create human-readable labels for ticks and replace the last one with infinity
            # labels = [str(int(x)) if float(x).is_integer() else f"{x}" for x in ticks]
            # labels[-1] = r"$\infty$"
            # plt.xticks(ticks, labels)
            plt.ylim(10e-5, 1)
            if self.groomed:
                plt.xlim(-4.5,0)
            else:
                plt.xlim(-2.5, 0)
            plt.xlabel(r"Groomed Jet $\log_{10}(\rho^2)$" if self.groomed else r"Ungroomed Jet $\log_{10}(\rho^2)$")
            plt.ylabel("Fractional Uncertainty")
            save_path = f'./outputs/rho/uncertainties/summary_groomed_{i-1}.pdf' if self.groomed else f'./outputs/rho/uncertainties/summary_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)

    
    def plot_systematic_frac_indiv(self, syst_names=['JES', 'JER'], ylim=None, show=True):

        # First, collect all values to determine global y-range
        all_values = []
        for i, result in enumerate(self.normalized_results):
            syst_fraction_dict = result.get('syst_fraction_dict', {})
            for syst in syst_names:
                up_key = f"{syst}Up"
                down_key = f"{syst}Down"
                if up_key in syst_fraction_dict:
                    all_values.append(np.abs(syst_fraction_dict[up_key]))
                if down_key in syst_fraction_dict:
                    all_values.append(np.abs(syst_fraction_dict[down_key]))


        # Now plot with fixed y-range
        for i, result in enumerate(self.normalized_results):
            syst_fraction_dict = result.get('syst_fraction_dict', {})
            #plt.figure(figsize=(12, 8))
            pt_bin = result['pt_bin']
            for syst in syst_names:
                up_key = f"{syst}Up"
                down_key = f"{syst}Down"
                color_map = ['red', 'blue', 'green']
                color = color_map[syst_names.index(syst)] if syst in syst_names and syst_names.index(syst) < len(color_map) else None

                # Plot Up uncertainty (solid)
                label_dic = {'pu':'Pileup', 'l1prefiring': 'L1 Prefiring', 'q2': r'Q$^2$ Scale', 'pdf': 'PDF', 'herwig': 'Model Unc.'}
                if up_key in syst_fraction_dict:
                    hep.histplot(syst_fraction_dict[up_key][1:], self.bins.gen_rho_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Up", color=color, ls='-')
                # Plot Down uncertainty (dashed)
                if down_key in syst_fraction_dict:
                    hep.histplot(-syst_fraction_dict[down_key][1:], self.bins.gen_rho_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Down", color=color, ls='--')


                
            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"
            
            # if ylim is not None:
            #     plt.ylim(ylim)
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize = 15)#loc='center left', bbox_to_anchor=(1, 0.5))
            hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)

            if self.groomed:
                plt.xlim(-4.5,0)
                plt.xlabel(r"$\log_{10}(\rho^2)$, Groomed")
                save_path = f'./outputs/rho/uncertainties/{syst_names[0]}_groomed_{i-1}.pdf'
            else:
                plt.xlim(-2.5, 0)
                plt.xlabel(r"$\log_{10}(\rho^2)$, Ungroomed")
                save_path = f'./outputs/rho/uncertainties/{syst_names[0]}_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)

    def plot_herwig_systematic(self, show=True):
        flat_uncertainty = np.sqrt(np.diag(self.cov_data_herwig_np))/np.abs(self.y_unf_dict['herwigUp'])
        uncertainty_pt_binned = unflatten_gen_by_pt(flat_uncertainty, self.bins.gen_rho_edges_by_pt)
        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.bins.gen_rho_edges_by_pt)
        
        for i, result in enumerate(self.normalized_results):
            syst_fraction_dict = result.get('syst_fraction_dict', {})
            error_in_syst = uncertainty_pt_binned[i]*syst_fraction_dict['herwigUp']  # Uncertainty on relative uncertainty
            pt_bin = result['pt_bin']
            if 'herwigUp' in syst_fraction_dict:
                hep.histplot(syst_fraction_dict['herwigUp'], self.bins.gen_rho_edges_by_pt[i], yerr = error_in_syst, label=f"Model Unc.", color='g', ls='-')


            # Fit a polynomial to the herwigUp systematic fraction
            if 'herwigUp' in syst_fraction_dict:
                edges = np.array(self.bins.gen_rho_edges_by_pt[i], dtype=float)
                centers = 0.5 * (edges[:-1] + edges[1:])
                #centers[0] = -100000000  # Set the first center to a very large negative value to exclude it from the fit
                y = syst_fraction_dict['herwigUp']
                mask = np.isfinite(y) & (y > 0)
                if mask.sum() > 3:
                    degree = 2
                    coeffs = np.polyfit(centers[mask], y[mask], degree, w=1.0/np.where(error_in_syst[mask] > 0, error_in_syst[mask], 1e-10))
                    poly = np.poly1d(coeffs)
                    x_fit = np.linspace(centers[mask][1], centers[mask][-1], 200)
                    plt.plot(x_fit, poly(x_fit), 'b--', lw=2, label=f"Poly fit (deg {degree})")

            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
            hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)
            plt.ylim(0,0.5)
            if self.groomed:
                plt.xlim(-4.5, 0)
            else:
                plt.xlim(-2.5, 0)

            
            plt.xlabel(r"Groomed Jet $\log_{10}(\rho^2)$" if self.groomed else r"Ungroomed Jet $\log_{10}(\rho^2)$")
            plt.ylabel("Relative Uncertainty")
            save_path = f'./outputs/rho/uncertainties/herwig_groomed_{i-1}.pdf' if self.groomed else f'./outputs/rho/uncertainties/herwig_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)


            
    def plot_purity_stability(self, show=True):
        hep.style.use("CMS")
        #ignore first slice in sum bc it's underflow
        len_underflow = len(self.bins.gen_mass_edges_by_pt[0]) - 1

        purity = np.diag(self.mosaic_gen) / self.mosaic_gen[len_underflow:, :].sum(axis=0)
        stability = np.diag(self.mosaic_gen) / self.mosaic_gen[:, len_underflow:].sum(axis=1)

        purity_unflattened = unflatten_gen_by_pt(purity, self.bins.gen_mass_edges_by_pt)
        stability_unflattened = unflatten_gen_by_pt(stability, self.bins.gen_mass_edges_by_pt)

        hep.histplot(purity_unflattened[2], self.bins.gen_mass_edges_by_pt[2], label = "Purity")
        hep.histplot(stability_unflattened[2], self.bins.gen_mass_edges_by_pt[2], label = "Stability")
        plt.axhline(0.5, color='k', linestyle='--', linewidth=1)
        hep.cms.label(self.cms_label, data=False, lumi = 138, com = 13, fontsize = 20)
        plt.ylabel("Purity/Stability")
        plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
        plt.legend()
        plt.xlim(0,200)
        self._finalize_plot(save_path=f'./outputs/rho/unfold/purity_stability_ptslice_{"groomed" if self.groomed else "ungroomed"}.pdf', show=show)
        
        
        plt.stairs(purity, label = "Purity")
        plt.xlabel("Global Bin Number")
        plt.ylabel("Purity")
        #hep.cms.label(self.cms_label, data=False)
        plt.stairs(stability, label = "Stability")
        plt.xlabel("Global Bin Number")
        plt.ylabel("Stability")
        hep.cms.label(self.cms_label, data=False, lumi = 138, com = 13, fontsize = 20)
        plt.legend()
        self._finalize_plot(save_path=f'./outputs/rho/unfold/purity_stability_global_{"groomed" if self.groomed else "ungroomed"}.pdf', show=show)


    def plot_correlation(self, show=True):
        cov_matrix = self.cov_uncorr_np + self.cov_data_np
        std_devs = np.sqrt(np.diag(cov_matrix))

        # Avoid division by zero by replacing zeros with a small number
        std_devs[std_devs == 0] = 1e-10

        # Compute the outer product of standard deviations
        std_matrix = np.outer(std_devs, std_devs)

        # Compute correlation matrix
        corr_matrix = cov_matrix / std_matrix

        self.corr_matrix = corr_matrix
        ## colormap
        num_bins = 20
        bounds = np.linspace(-1, 1, num_bins + 1)  # 21 boundaries for 20 bins

        # Get the 'seismic' colormap
        base_cmap = plt.get_cmap("seismic", num_bins)
        colors = base_cmap(np.linspace(0, 1, num_bins))  # Extract colors
        
        # Force the bins for -0.1 to 0.1 to be white
        for i in range(len(bounds) - 1):
            if -0.1 <= bounds[i] <= 0.1:
                colors[i] = [1, 1, 1, 1]  # Set to white (RGBA)


            
        #import matplotlib.colors as mcolors
        # Create colormap and normalizer
        cmap =  mcolors.ListedColormap(colors)  # Discrete 'seismic' colormap
        norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize for discrete bins
        

        img = plt.imshow(corr_matrix, cmap = cmap, norm = norm, origin = 'lower')
        
        # ---- Add grid lines and labels for pt bins ----
        # Get bin structure from gen_mass_edges_by_pt and pt_edges
        ncols_by_gp = [len(e)-1 for e in self.bins.gen_rho_edges_by_pt]
        x_bounds = np.r_[0, np.cumsum(ncols_by_gp)]
        # Draw dashed lines at pt bin boundaries
        for x in x_bounds[1:-1]:
            plt.axvline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)
            plt.axhline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)
        # Optional: thin grid inside each block at every integer cell
        plt.xticks(np.arange(-0.5, corr_matrix.shape[1], 1), minor=True)
        plt.yticks(np.arange(-0.5, corr_matrix.shape[0], 1), minor=True)
        plt.grid(which="minor", color="w", alpha=0.15, lw=0.5)
        # Tick labels at block centers
        x_centers = (x_bounds[:-1] + x_bounds[1:] - 1) / 2.0
        pt_edges = self.pt_edges
        x_labels = [f"{int(pt_edges[i])}–{int(pt_edges[i+1]) if i+1 < len(pt_edges)-1 else '∞'}" for i in range(len(pt_edges)-1)]
        plt.xticks(x_centers, x_labels)
        plt.yticks(x_centers, x_labels, rotation=90, va="center")

        plt.xlabel(r"GEN $p_T$ (GeV)")
        plt.ylabel(r"GEN $p_T$ (GeV)")

        cbar = plt.colorbar(img, ticks=bounds, boundaries=bounds, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation (Groomed)" if self.groomed else "Correlation (Ungroomed)")
        hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)
        save_path = f'outputs/rho/unfold/correlation_groomed.pdf' if self.groomed else f'outputs/rho/unfold/correlation_ungroomed.pdf'
        self._finalize_plot(save_path=save_path, show=show)
    def plot_response_matrix(self, probability=True, log=False, show=True):
        fig, ax = self._plot_response_mosaic_cms(
            self.mosaic,
            reco_mass_edges_by_pt=self.bins.reco_rho_edges_by_pt,
            gen_mass_edges_by_pt=self.bins.gen_rho_edges_by_pt,
            reco_pt_edges=self.pt_edges,
            gen_pt_edges=self.pt_edges,
            probability = probability,
            mask_zeros=True,
            log=log,                              # set False for linear
            rlabel=f"Groomed, " if self.groomed else f"Ungroomed, ",
        )
        self._finalize_plot(show=show, fig=fig)

    def run_all_plots(self, show=False):
        self.plot_unfolded_fancy(show=show)
        self.plot_systematic_fraction(show=show)
        self.plot_herwig_systematic(show=show)
        self.plot_systematic_frac_indiv(["JES", "JER"], show=show)
        self.plot_systematic_frac_indiv(["JMS", "JMR"], show=show)
        self.plot_systematic_frac_indiv(["q2", "pdf", "pu", "l1prefiring"], show=show)
        self.plot_systematic_frac_indiv(["herwig"], show=show)
        self.plot_systematic_frac_indiv(["ElectronSF", "MuonSF"], show=show)
        self.plot_correlation(show=show)
        self.plot_unfolded(show=show)
        self.plot_response_matrix(probability=True, show=show)
        self.plot_folded(show=show)
        self.plot_bottom_line(show=show)
        self.plot_fakes_misses(show=show)
    def _plot_response_mosaic_cms(
        self,
        mosaic,
        reco_mass_edges_by_pt,   # list: per reco-pT slice, mass edges used (rows per block)
        gen_mass_edges_by_pt,    # list: per gen-pT slice,  mass edges used (cols per block)
        reco_pt_edges,           # e.g. [200, 290, 400, 13000]
        gen_pt_edges,            # e.g. [200, 290, 400, 13000]
        *,
        mask_zeros=True,
        probability=True,  # if True, normalize each block to sum to 1
        log=False,
        cmap="viridis",
        rlabel=None,             # e.g. "Ungroomed, Cond. = 5.6e+16"
        vmin=None, vmax=None,
        ax=None
    ):
        """
        Draw a CMS-style gridded 'flattened' response plot.
        - `mosaic` is your unpadded 2D array (blocks concatenated).
        - Mass bin counts per (reco pT, gen pT) block come from the edge lists.
        - Dashed grid lines drawn at pT-block boundaries.
        """
        if probability:
            # Normalize each column to sum to 1
            mosaic = mosaic / np.sum(mosaic, axis=0, keepdims=True)
            # ensure no NaNs
            mosaic = np.nan_to_num(mosaic, nan=0.0)
        
        # Compute all the singular values of the mosaic matrix
        # and condition number (ratio of largest to smallest singular value)
        # to check for numerical stability.

        singular_values = np.linalg.svd(mosaic, compute_uv=False)
        print("Singular values of the response mosaic:", singular_values)

        hep.style.use("CMS")
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 12))
        else:
            fig = ax.figure

        img = mosaic.copy()
        if mask_zeros:
            img = np.ma.masked_where(img == 0, img)

        # Choose norm
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else None

        # Plot
        im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                    vmin=None if log else vmin, vmax=None if log else vmax)

        # ---- Grid lines at pT block boundaries ----
        # Row/col counts per block:
        nrows_by_rp = [len(e)-1 for e in reco_mass_edges_by_pt]
        ncols_by_gp = [len(e)-1 for e in gen_mass_edges_by_pt]

        # Cumulative boundaries (in pixel/bin units)
        y_bounds = np.r_[0, np.cumsum(nrows_by_rp)]
        x_bounds = np.r_[0, np.cumsum(ncols_by_gp)]

        # Draw dashed lines
        for y in y_bounds[1:-1]:
            ax.axhline(y-0.5, color="r", ls="--", lw=2, alpha=0.6)
        for x in x_bounds[1:-1]:
            ax.axvline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)

        # Optional: thin grid inside each block at every integer cell
        ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)
        ax.grid(which="minor", color="w", alpha=0.15, lw=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # ---- Tick labels at pT bin centers ----
        # Use block centers for labels:
        x_centers = (x_bounds[:-1] + x_bounds[1:] - 1) / 2.0
        y_centers = (y_bounds[:-1] + y_bounds[1:] - 1) / 2.0
        # Label with pT edges (lower edges are fine; pick what you prefer)
        x_labels = [f"{int(gen_pt_edges[i])}–{int(gen_pt_edges[i+1]) if i+1 < len(gen_pt_edges)-1 else '∞'}" for i in range(len(gen_pt_edges)-1)]
        y_labels = [f"{int(reco_pt_edges[i])}–{int(reco_pt_edges[i+1]) if i+1 < len(reco_pt_edges)-1 else '∞'}" for i in range(len(reco_pt_edges)-1)]


        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_centers)
        ax.set_yticklabels(y_labels, rotation=90, va="center")

        ax.set_xlabel("GEN pT (GeV)")
        ax.set_ylabel("RECO pT (GeV)")

        # Diagonal block markers (optional; like your reference plot)
        # draw a red diagonal only within the matching (i,i) blocks
        for i in range(min(len(x_bounds)-1, len(y_bounds)-1)):
            x0, x1 = x_bounds[i]-0.5, x_bounds[i+1]-0.5
            y0, y1 = y_bounds[i]-0.5, y_bounds[i+1]-0.5
            ax.plot([x0, x1], [y0, y1], color="r", lw=1, alpha=0.7)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Probability" if probability else "Counts")

        # CMS label
        hep.cms.label(self.cms_label, data=False, rlabel=(rlabel + f"Cond. = {np.linalg.cond(mosaic):.2f}") if rlabel else f"Cond. = {np.linalg.cond(mosaic):.2f}")

        fig.tight_layout()
        if self.groomed:
            plt.savefig("outputs/rho/unfold/response_groomed.pdf")
        else:
            plt.savefig("outputs/rho/unfold/response_ungroomed.pdf")
        return fig, ax
    
    def _make_inputs_numpy(self,
        filenames = [
            "./inputs/rhoInputs/pythia_2016.pkl",
            "./inputs/rhoInputs/pythia_2016APV.pkl",
            "./inputs/rhoInputs/pythia_2017.pkl",
            "./inputs/rhoInputs/pythia_2018.pkl",
        ]):
        # Load pickled inputs and save as numpy arrays for faster loading in the future
        # Define a mapping from era to key in the response dictionary
        from collections import defaultdict
        print("------------- Making numpy inputs from pickled files -----------------")

        keymap = {
            "2016": "pythia_UL16NanoAODv9",
            "2016APV": "pythia_UL16NanoAODAPVv9",
            "2017": "pythia_UL17NanoAODv9",
            "2018": "pythia_UL18NanoAODv9",
        }
        response_dict = {}
        ratio_data_mc = [1.0, 1.0, 1.0, 1.0]  # Placeholder for data/MC ratios per pt bin if needed for scaling
        for filename in filenames:
            era = filename.split('_')[1].split('.')[0]
            print(f"Processing era: {era}")
            key = keymap[era]
            response_dict.setdefault('u', {})
            response_dict.setdefault('g', {})

            
            with open(filename, "rb") as f:
                data = pkl.load(f)
                # ensure top-level 'u' and 'g' keys exist
                
                if era == "2016APV" or era == "2016":
                    response_dict['u'][key] = data['response_matrix_rho_u'].project('ptreco','dataset', 'ptgen', 'mpt_gen',  'mpt_reco', 'systematic')
                    response_dict['g'][key] = data['response_matrix_rho_g'].project('ptreco','dataset', 'ptgen', 'mpt_gen',  'mpt_reco', 'systematic')
                    continue
                h_old = data['response_matrix_rho_u'].project('ptreco','dataset', 'ptgen', 'mpt_gen',  'mpt_reco', 'systematic')
                #h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
                response_dict['u'][key] = h_old

                h_old = data['response_matrix_rho_g'].project('ptreco','dataset', 'ptgen', 'mpt_gen',  'mpt_reco', 'systematic')
                #h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
                response_dict['g'][key] = h_old

                for i in range(4):
                    response_dict['u'][key].view()[i] *= ratio_data_mc[i]
                    response_dict['g'][key].view()[i] *= ratio_data_mc[i]

        correlation_dic = {
            'JES_AbsoluteMPFBias': 1,
            'JES_AbsoluteScale': 1,
            'JES_AbsoluteStat': 0,
            'JES_FlavorQCD': 1,
            'JES_Fragmentation': 1,
            'JES_PileUpDataMC': 0.5,
            'JES_PileUpPtBB': 0.5,
            'JES_PileUpPtEC1': 0.5,
            'JES_PileUpPtEC2': 0.5,
            'JES_PileUpPtHF': 0.5,
            'JES_PileUpPtRef': 0.5,
            'JES_RelativeFSR': 0.5,
            'JES_RelativeJEREC1': 0,
            'JES_RelativeJEREC2': 0,
            'JES_RelativeJERHF': 0.5,
            'JES_RelativePtBB': 0.5,
            'JES_RelativePtEC1': 0,
            'JES_RelativePtEC2': 0,
            'JES_RelativePtHF': 0.5,
            'JES_RelativeBal': 0.5,
            'JES_RelativeSample': 0,
            'JES_RelativeStatEC': 0,
            'JES_RelativeStatFSR': 0,
            'JES_RelativeStatHF': 0,
            'JES_SinglePionECAL': 1,
            'JES_SinglePionHCAL': 1,
            'JES_TimePtEta': 0,
            'JER': 0,
        }

        jes_sys_list = ['JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown',
                        'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp',
                        'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown',
                        'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
                        'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up',
                        'JES_RelativeJEREC1Down', 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown',
                        'JES_RelativePtBBUp', 'JES_RelativePtBBDown', 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 'JES_RelativePtEC2Down',
                        'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 
                        'JES_RelativeStatECUp', 'JES_RelativeStatECDown', 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown',
                        'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown', 'JERUp', 'JERDown']


        non_jes_sys_list = ['nominal', 'puUp', 'puDown', 'elerecoUp', 'elerecoDown',
                            'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp',
                            'murecoDown', 'muidUp', 'muidDown', 'mutrigUp', 'muisoUp', 'muisoDown','mutrigDown', 'pdfUp',
                            'pdfDown', 'q2Up', 'q2Down', 'l1prefiringUp', 'l1prefiringDown', 'isrUp', 'isrDown', 'fsrUp', 'fsrDown',
                            'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']


        non_jes_sys_list_up = [sys for sys in non_jes_sys_list if sys[-2:] == 'Up' ]
        non_jes_sys_list_down = [sys for sys in non_jes_sys_list if sys[-4:] == 'Down' ]

        jes_sys_list_up = [sys for sys in jes_sys_list if sys[-2:] == 'Up' ]
        jes_sys_list_down = [sys for sys in jes_sys_list if sys[-4:] == 'Down' ]

        sys_matrix_dic_up = {}

        groomed = self.groomed
        if not groomed:
            response = response_dict['u']
        else:
            response = response_dict['g']





        for sys in jes_sys_list_up:
            m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() +\
                response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()

            variance = np.array([response[era][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').variances() for era in ['pythia_UL16NanoAODv9', 'pythia_UL16NanoAODAPVv9', 'pythia_UL17NanoAODv9', 'pythia_UL18NanoAODv9']]).sum(axis = 0)
            
            m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() + \
                response['pythia_UL16NanoAODAPVv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            
            
            m_var_2016 = m_sys_2016 + m_nom_2017 + m_nom_2018
            m_var_2017 = m_nom_2016 + m_sys_2017 + m_nom_2018
            m_var_2018 = m_nom_2016 + m_nom_2017 + m_sys_2018
            
            
            rho = correlation_dic[sys[:-2]] ## correlation factor
            
            sigma_2016 = m_sys_2016 - m_nom_2016
            sigma_2017 = m_sys_2017 - m_nom_2017
            sigma_2018 = m_sys_2018 - m_nom_2018
            
            sigma_corr = rho*sigma_2016 + rho*sigma_2017 + rho*sigma_2018
            
            sigma_uncorr_2016 = (1-rho)*sigma_2016
            sigma_uncorr_2017 = (1-rho)*sigma_2017
            sigma_uncorr_2018 = (1-rho)*sigma_2018
            
            m_nom =  m_nom_2016 + m_nom_2017 + m_nom_2018
            m_corr = m_nom + sigma_corr
            
            m_uncorr_2016 = m_nom + sigma_uncorr_2016
            
            m_uncorr_2017 = m_nom + sigma_uncorr_2017

            m_uncorr_2018 = m_nom + sigma_uncorr_2018

            

            sys_matrix_dic_up[sys+'_corr'] = m_corr
            sys_matrix_dic_up[sys+'_uncorr_2016'] = m_uncorr_2016
            sys_matrix_dic_up[sys+'_uncorr_2017'] = m_uncorr_2017
            sys_matrix_dic_up[sys+'_uncorr_2018'] = m_uncorr_2018
            
        non_jes_sys_matrix_dic_up = {}
        for sys in non_jes_sys_list_up:
            sys_matrix_dic_up[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()

            non_jes_sys_matrix_dic_up[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()



        sys_matrix_dic_down = {}
        non_jes_sys_matrix_dic_down = {}
        for sys in jes_sys_list_down:
            m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() + \
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            
            m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() + \
                        response['pythia_UL16NanoAODAPVv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
            
            
            m_var_2016 = m_sys_2016 + m_nom_2017 + m_nom_2018
            m_var_2017 = m_nom_2016 + m_sys_2017 + m_nom_2018
            m_var_2018 = m_nom_2016 + m_nom_2017 + m_sys_2018
            
            
            rho = correlation_dic[sys[:-4]] ## correlation factor
            
            sigma_2016 = m_sys_2016 - m_nom_2016
            sigma_2017 = m_sys_2017 - m_nom_2017
            sigma_2018 = m_sys_2018 - m_nom_2018
            
            sigma_corr = rho*sigma_2016 + rho*sigma_2017 + rho*sigma_2018
            
            sigma_uncorr_2016 = (1-rho)*sigma_2016
            sigma_uncorr_2017 = (1-rho)*sigma_2017
            sigma_uncorr_2018 = (1-rho)*sigma_2018
            
            m_nom =  m_nom_2016 + m_nom_2017 + m_nom_2018
            m_corr = m_nom + sigma_corr
            
            m_uncorr_2016 = m_nom + sigma_uncorr_2016
            
            m_uncorr_2017 = m_nom + sigma_uncorr_2017

            m_uncorr_2018 = m_nom + sigma_uncorr_2018

            

            sys_matrix_dic_down[sys+'_corr'] = m_corr
            sys_matrix_dic_down[sys+'_uncorr_2016'] = m_uncorr_2016
            sys_matrix_dic_down[sys+'_uncorr_2017'] = m_uncorr_2017
            sys_matrix_dic_down[sys+'_uncorr_2018'] = m_uncorr_2018
            

        for sys in non_jes_sys_list_down:
            sys_matrix_dic_down[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()
        #                            + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values
            non_jes_sys_matrix_dic_down[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mpt_gen','ptreco','mpt_reco').values()

        if not groomed:
            response = response_dict['u']
        else:
            response = response_dict['g']

        sys_matrix_dic_up['nominal'] = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() +\
                response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()  +\
                    response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() +\
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()
        sys_matrix_dic_down['nominal'] = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() +\
                response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()  +\
                    response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values() +\
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco').values()

        prepend = "./inputs/rhoInputs/"
        #filename_herwig_1 = prepend + "herwig_all.pkl"
        filename_herwig_1 = prepend + "pythia_reweighted_all.pkl"
        with open(filename_herwig_1, "rb") as f:
            data_herwig = pkl.load(f)
            if not groomed:
                sys_matrix_dic_up['herwigUp'] = data_herwig['response_matrix_rho_u'].project('ptgen','mpt_gen','ptreco','mpt_reco').values()
                sys_matrix_dic_down['herwigDown'] = data_herwig['response_matrix_rho_u'].project('ptgen','mpt_gen','ptreco','mpt_reco').values()
            else:
                sys_matrix_dic_up['herwigUp'] = data_herwig['response_matrix_rho_g'].project('ptgen','mpt_gen','ptreco','mpt_reco').values()
                sys_matrix_dic_down['herwigDown'] = data_herwig['response_matrix_rho_g'].project('ptgen','mpt_gen','ptreco','mpt_reco').values()

        ## Merging the up and down dictionaries
        merged = dict(sys_matrix_dic_up)  # shallow copy of up
        for k, v in sys_matrix_dic_down.items():
            if k == "nominal":
                continue  # skip duplicate nominal from the down dict
            if k in merged:
                # if a key collides, store the down version with a suffix
                merged_key = f"{k}_down"
                # ensure uniqueness
                i = 1
                unique_key = merged_key
                while unique_key in merged:
                    unique_key = f"{merged_key}_{i}"
                    i += 1
                merged[unique_key] = v
            else:
                merged[k] = v
        self.sys_matrix_dic = merged
        print(f"✅ saved {len(merged)} in sys_matrix_dic")
    def _merge_eras(self, filenames = ["./inputs/rhoInputs/pythia_2016.pkl","./inputs/rhoInputs/pythia_2016APV.pkl","./inputs/rhoInputs/pythia_2017.pkl","./inputs/rhoInputs/pythia_2018.pkl",]):


        outputs = []
        for fname in filenames:
            with open(fname, "rb") as f:
                outputs.append(pkl.load(f))


        hist_keys = ['ptjet_rhojet_u_reco', 'ptjet_rhojet_g_reco', 'response_matrix_rho_u', 'response_matrix_rho_g', 'ptjet_rhojet_u_gen', 'ptjet_rhojet_g_gen',]
        out_dict = {}
        for i, output in enumerate(outputs):
            for key in hist_keys:
                if key in ['ptjet_rhojet_u_reco', 'ptjet_rhojet_g_reco']:

                    if i == 0:
                        out_dict[key] = output[key].project('ptreco', 'mpt_reco', 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptreco', 'mpt_reco', 'systematic')
                    #print(f"Processed {key} for file {i}, with sum {output[key].project('ptreco', 'mpt_reco', 'systematic').sum().value}")
                    #print("Current total sum:", out_dict[key].sum().value)
                elif key in ['response_matrix_rho_u', 'response_matrix_rho_g']:
                    if i == 0:
                        out_dict[key] = output[key].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco', 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptgen', 'mpt_gen', 'ptreco', 'mpt_reco', 'systematic')
                    #print(f"Processed {key} for file {i}")
                elif key in ['ptjet_rhojet_u_gen', 'ptjet_rhojet_g_gen']:
                    if i == 0:
                        out_dict[key] = output[key].project('ptgen', 'mpt_gen', 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptgen', 'mpt_gen', 'systematic')
                    #print(f"Processed {key} for file {i}")
            self.pythia_hists = out_dict
        out_filename = "./inputs/rhoInputs/pythia_all.pkl"
        with open(out_filename, "wb") as f:
            pkl.dump(out_dict, f)
    def _merge_eras_jk(
        self,
        filenames = [
            "./inputs/rhoInputs/jk_pythia_2016.pkl",
            "./inputs/rhoInputs/jk_pythia_2016APV.pkl",
            "./inputs/rhoInputs/jk_pythia_2017.pkl",
            "./inputs/rhoInputs/jk_pythia_2018.pkl",
        ] 
        ):
        outputs = []
        for fname in filenames:
            with open(fname, "rb") as f:
                outputs.append(pkl.load(f))
        hist_keys = [ 'response_matrix_rho_u', 'response_matrix_rho_g']
        out_dict = {}
        for i, output in enumerate(outputs):
            for key in hist_keys:
                if key in ['response_matrix_rho_u', 'response_matrix_rho_g']:
                    if i == 0:
                        out_dict[key] = output[key].project('jk','ptgen', 'mpt_gen', 'ptreco', 'mpt_reco', 'systematic')
                    else:
                        out_dict[key] += output[key].project('jk','ptgen', 'mpt_gen', 'ptreco', 'mpt_reco', 'systematic')





# sys_matrix_dic['herwigUp'] = resp_matrix_4d_herwig.project('ptgen','mgen','ptreco','mreco').values()
# sys_matrix_dic_down['herwigDown'] = resp_matrix_4d_herwig.project('ptgen','mgen','ptreco','mreco').values()
