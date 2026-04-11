from array import array
from pathlib import Path
import re

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

plt.rcParams["axes.prop_cycle"] = cycler(color=['#5790fc', '#f89c20', '#e42536', '#964a8b', '#9c9ca1', '#7a21dd'])

DEFAULT_MC_FILE = "./inputs/massInputs/mass_pythia_all.pkl"
DEFAULT_DATA_FILE = "./inputs/massInputs/mass_data_all.pkl"
#DEFAULT_HERWIG_FILE = "./inputs/massInputs/mass_reweight_pythia_all.pkl"
DEFAULT_HERWIG_FILE = "./inputs/massInputs/mass_herwig_all.pkl"
DEFAULT_JK_DATA_FILE = "./inputs/massInputs/mass_jk_data_all.pkl"

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
        self.edges = self.bins.mass_edges_reco
        self.edges_gen = self.bins.mass_edges_gen
        self.pt_edges = self.bins.pt_edges

    def _configure_systematics(self, do_syst):
        available_systematics = list(self.sys_matrix_dic.keys())
        self.systematics = available_systematics if do_syst else ["nominal"]

    def _load_pickle(self, filename):
        with open(filename, "rb") as handle:
            return pkl.load(handle)

    def _resolve_input_path(self, filename, *fallbacks):
        candidates = [filename, *fallbacks]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if path.exists():
                return str(path)
        candidate_list = ", ".join(str(Path(candidate)) for candidate in candidates if candidate is not None)
        raise FileNotFoundError(f"Could not find any of: {candidate_list}")

    def _finalize_plot(self, save_path=None, show=True, fig=None):
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            target_fig = fig if fig is not None else plt.gcf()
            target_fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        else:
            plt.close(fig if fig is not None else plt.gcf())

    def _cms_extra_label(self):
        return f" {self.cms_label}" if self.cms_label and not self.cms_label.startswith(" ") else self.cms_label

    def _observable_label(self):
        return "Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)"

    def _observable_short_label(self):
        return "Jet Mass (GeV), Groomed" if self.groomed else "Jet Mass (GeV), Ungroomed"

    def _observable_xlim(self, i_pt=None):
        if i_pt is not None:
            upper = float(self.bins.gen_mass_edges_by_pt[i_pt][-2])
        else:
            upper = max(float(edges[-2]) for edges in self.bins.gen_mass_edges_by_pt)
        if self.groomed:
            return (10.0, upper)
        else:
            return (20.0, upper)

    def _normalized_ylabel(self):
        return r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{dm\,dp_T} (GeV^{-1})$"

    def _histogram_keys(self):
        if self.groomed:
            return {
                "response": "response_matrix_g",
                "reco": "ptjet_mjet_g_reco",
                "gen": "ptjet_mjet_g_gen",
            }
        return {
            "response": "response_matrix_u",
            "reco": "ptjet_mjet_u_reco",
            "gen": "ptjet_mjet_u_gen",
        }

    def _prepare_jackknife_inputs(self, data2d_jk, mass_edges_reco, pt_edges, reco_mass_edges_by_pt):
        mosaic_2d_jk_list = []
        for i in range(10):
            reco_proj_jk = data2d_jk.project("jk", "ptreco", "mreco")[i, ...]
            h2d_jk = reco_proj_jk.values()
            h2d_jk_reordered, _ = reorder_to_expected_2d(h2d_jk, mass_edges_reco, pt_edges)
            mosaic_2d_jk_list.append(
                merge_mass_flat(h2d_jk_reordered, mass_edges_reco, reco_mass_edges_by_pt)
            )
        return mosaic_2d_jk_list

    def _prepare_truth_spectrum(self, gen2d, mass_edges_gen, pt_edges, gen_mass_edges_by_pt):
        if "systematic" in gen2d.axes.name:
            gen_proj = gen2d.project("ptgen", "mgen", "systematic")[:, :, "nominal"]
        else:
            gen_proj = gen2d.project("ptgen", "mgen")
        h2d_gen, _ = reorder_to_expected_2d(gen_proj.values(), mass_edges_gen, pt_edges)
        return merge_mass_flat(h2d_gen, mass_edges_gen, gen_mass_edges_by_pt)

    def _prepare_jackknife_response_matrices(
        self,
        response4d_jk,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        mosaic_jk_list = []
        nominal_jk = response4d_jk[{"systematic": "nominal"}]
        for i in range(10):
            proj_jk = nominal_jk[{"jk": i}].project("ptgen", "mgen", "ptreco", "mreco")
            resp_np_jk = proj_jk.values(flow=False)
            resp_np_jk_reordered, _ = reorder_to_expected(
                resp_np_jk,
                mass_edges_reco,
                pt_edges,
                mass_edges_gen,
            )
            mosaic_jk, _ = mosaic_no_padding(
                resp_np_jk_reordered,
                mass_edges_reco,
                mass_edges_gen,
                reco_mass_edges_by_pt,
                gen_mass_edges_by_pt,
            )
            mosaic_jk_list.append(mosaic_jk)
        return mosaic_jk_list

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
        reco_proj = input_data.project("ptreco", "mreco")
        self.h2d, _ = reorder_to_expected_2d(reco_proj.values(), mass_edges_reco, pt_edges)

        reco_proj_fakes = fakes.project("ptreco", "mreco")
        self.h2d_fakes, _ = reorder_to_expected_2d(reco_proj_fakes.values(), mass_edges_reco, pt_edges)

        reco_proj_misses = misses.project("ptgen", "mgen")
        self.h2d_misses, _ = reorder_to_expected_2d(reco_proj_misses.values(), mass_edges_gen, pt_edges)

        nominal_gen = resp_matrix_4d_gen[{"systematic": "nominal"}]
        proj_gen = nominal_gen.project("ptreco", "mreco", "ptgen", "mgen")
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
        resp_matrix_4d_herwig_gen,
        fakes_herwig,
        misses_herwig,
        mass_edges_reco,
        mass_edges_gen,
        pt_edges,
        reco_mass_edges_by_pt,
        gen_mass_edges_by_pt,
    ):
        resp_matrix_4d_syst = resp_matrix_4d_herwig[{"systematic": "nominal"}]
        h2d_herwig = resp_matrix_4d_syst.project("ptreco", "mreco").values(flow=False)
        self.h2d_herwig, _ = reorder_to_expected_2d(h2d_herwig, mass_edges_reco, pt_edges)

        reco_proj_fakes = fakes_herwig.project("ptreco", "mreco")
        self.h2d_fakes_herwig, _ = reorder_to_expected_2d(reco_proj_fakes.values(), mass_edges_reco, pt_edges)

        reco_proj_misses = misses_herwig.project("ptgen", "mgen")
        self.h2d_misses_herwig, _ = reorder_to_expected_2d(reco_proj_misses.values(), mass_edges_gen, pt_edges)

        self.fakes_2d_herwig = merge_mass_flat(self.h2d_fakes_herwig, mass_edges_reco, reco_mass_edges_by_pt)
        self.misses_2d_herwig = merge_mass_flat(self.h2d_misses_herwig, mass_edges_gen, gen_mass_edges_by_pt)

        nominal_herwig_gen = resp_matrix_4d_herwig_gen[{"systematic": "nominal"}]
        proj_herwig_gen = nominal_herwig_gen.project("ptreco", "mreco", "ptgen", "mgen")
        M_np_2d_herwig_gen, _ = reorder_to_expected(
            proj_herwig_gen.values(flow=False), mass_edges_gen, pt_edges, mass_edges_gen
        )
        self.mosaic_gen_herwig, _ = mosaic_no_padding(
            M_np_2d_herwig_gen,
            mass_edges_gen,
            mass_edges_gen,
            gen_mass_edges_by_pt,
            gen_mass_edges_by_pt,
        )

    def _response_matrix_for_systematic(self, syst, resp_matrix_4d_herwig, sys_matrix_dic):
        if syst in {"herwigUp", "herwigDown"}:
            resp_matrix_4d_syst = resp_matrix_4d_herwig[{"systematic": "nominal"}]
            proj = resp_matrix_4d_syst.project("ptreco", "mreco", "ptgen", "mgen")
            return sys_matrix_dic[syst], proj.variances(flow=False)
        return sys_matrix_dic[syst], None

    def _compute_fake_fraction(self, fakes_flat, matched_flat):
        total_reco = matched_flat + fakes_flat
        with np.errstate(divide="ignore", invalid="ignore"):
            fake_fraction = np.divide(
                fakes_flat,
                total_reco,
                out=np.zeros_like(fakes_flat, dtype=float),
                where=total_reco > 0,
            )
        return np.clip(fake_fraction, 0.0, 1.0)

    def _finalize_reco_views(self, mass_edges_reco, reco_mass_edges_by_pt):
        self.M_np_2d = self.M_np_2d_dict["nominal"]
        self.mosaic = self.mosaic_dict["nominal"]
        self.mosaic_2d = merge_mass_flat(self.h2d, mass_edges_reco, reco_mass_edges_by_pt)
        self.fake_fraction_2d = self._compute_fake_fraction(self.fakes_2d, self.mosaic.sum(axis=1))

        if "herwigUp" in self.systematics or "herwigDown" in self.systematics:
            self.mosaic_herwig_2d = merge_mass_flat(self.h2d_herwig, mass_edges_reco, reco_mass_edges_by_pt)
            self.fake_fraction_2d_herwig = self._compute_fake_fraction(
                self.fakes_2d_herwig,
                self.mosaic_herwig_2d,
            )

    def _load_data(self, filename_mc=DEFAULT_MC_FILE, filename_data=DEFAULT_DATA_FILE, filename_herwig=DEFAULT_HERWIG_FILE, filename_jk_data=DEFAULT_JK_DATA_FILE):
        print("------------- Adding inputs to unfolder -----------------")
        self._merge_eras()
        self._merge_eras_jk()
        output_pythia = self.pythia_hists
        output_pythia_jk = self.pythia_hists_jk
        print("Keys in pythia file:", output_pythia.keys())
        filename_data = self._resolve_input_path(filename_data, "./inputs/massInputs/data_all.pkl")
        filename_jk_data = self._resolve_input_path(filename_jk_data, "./inputs/massInputs/jk_data_all.pkl")
        filename_herwig = self._resolve_input_path(
            filename_herwig,
            "./inputs/massInputs/herwig_all.pkl",
        )

        output_herwig = self._load_pickle(filename_herwig)
        output_data = self._load_pickle(filename_data)
        output_jk_data = self._load_pickle(filename_jk_data)
        print("Keys in jk data file:", output_jk_data.keys())

        sys_matrix_dic = self.sys_matrix_dic
        keys = self._histogram_keys()

        pythia4d = output_pythia[keys["response"]]
        pythia4d_jk = output_pythia_jk[keys["response"]]
        herwig4d = output_herwig[keys["response"]]
        pythia2d = output_pythia[keys["reco"]]
        herwig2d = output_herwig[keys["reco"]]
        data2d = output_data[keys["reco"]]
        data2d_jk = output_jk_data[keys["reco"]]
        pythia_gen2d = output_pythia[keys["gen"]]
        herwig_gen2d = output_herwig[keys["gen"]]

        pythia4d_gen = rebin_hist(pythia4d.copy(), 'mreco',self.edges_gen )
        herwig4d_gen = rebin_hist(herwig4d.copy(), 'mreco',self.edges_gen )

        resp_matrix_4d_gen = pythia4d_gen

        reco_mass_edges_by_pt = self.bins.reco_mass_edges_by_pt
        gen_mass_edges_by_pt = self.bins.gen_mass_edges_by_pt
        pt_edges = self.bins.pt_edges
        mass_edges_reco = self.bins.mass_edges_reco
        mass_edges_gen = self.bins.mass_edges_gen

        fakes = pythia2d.project('ptreco', 'mreco', 'systematic')[:, :, 'nominal'] + (-1)*pythia4d.project('ptreco', 'mreco', 'systematic')[:, :, 'nominal']
        fakes_herwig = herwig2d.project('ptreco', 'mreco', 'systematic') + (-1)*herwig4d.project('ptreco', 'mreco', 'systematic')
        self.fakes = fakes
        self.fakes_herwig = fakes_herwig

        misses = pythia_gen2d.project('ptgen', 'mgen', 'systematic')[:, :, 'nominal'] + (-1)*pythia4d.project('ptgen', 'mgen', 'systematic')[:, :, 'nominal']
        misses_herwig = herwig_gen2d.project('ptgen', 'mgen', 'systematic') + (-1)*herwig4d.project('ptgen', 'mgen', 'systematic')
        self.misses = misses
        self.misses_herwig = misses_herwig
        self.y_true_herwig = self._prepare_truth_spectrum(
            herwig_gen2d,
            mass_edges_gen,
            pt_edges,
            gen_mass_edges_by_pt,
        )

        self.mosaic_dict = {}
        self.M_np_2d_dict = {}
        resp_matrix_4d = pythia4d
        resp_matrix_4d_herwig = herwig4d
        input_data = data2d


        self.input_data = input_data
        self.pythia_2d = pythia2d
        self.pythia_4d = pythia4d

        print("Loaded pkl files and rebinned histograms.")

        print("Processing jk inputs...")
        self.mosaic_2d_jk_list = self._prepare_jackknife_inputs(
            data2d_jk,
            mass_edges_reco,
            pt_edges,
            reco_mass_edges_by_pt,
        )
        self.mosaic_jk_list = self._prepare_jackknife_response_matrices(
            pythia4d_jk,
            mass_edges_reco,
            mass_edges_gen,
            pt_edges,
            reco_mass_edges_by_pt,
            gen_mass_edges_by_pt,
        )

        for syst in self.systematics:
            if syst == 'nominal':
                #print("Processing nominal systematic:", syst)
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
                #print("Processing Herwig systematic:", syst)
                self._prepare_herwig_inputs(
                    resp_matrix_4d_herwig,
                    herwig4d_gen,
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
        self._finalize_reco_views(mass_edges_reco, reco_mass_edges_by_pt)
        self.y_unf_jk_input_list = []
        self.y_unf_jk_matrix_list = []
        #print("h2d shape:", self.h2d.shape)
        #print("reco_mass_edges_by_pt:", reco_mass_edges_by_pt)
        #print("len reco_mass_edges_by_pt:", len(reco_mass_edges_by_pt))
    def plot_fakes_misses(self, show=True):
        title_list = ["",r"200 $<$ $p_T$ $<$ 290 GeV", r"290 $<$ $p_T$ $<$ 400 GeV", r"400 $<$ $p_T$ $< \, \infty$  GeV"]

        fakerate = self.fakes_2d/self.mosaic_2d

        fakerate = np.nan_to_num(fakerate, nan=0.0)# posinf=0.0, neginf=0.0)
        efficiency = 1 - (self.misses_2d/(self.misses_2d + self.mosaic.sum(axis=0)))
        efficiency_pt_binned = unflatten_gen_by_pt(efficiency, self.bins.gen_mass_edges_by_pt)
        fakerate_pt_binned = unflatten_gen_by_pt(fakerate, self.bins.reco_mass_edges_by_pt)

        for i in range(4):
            plt.stairs(1-fakerate_pt_binned[i], self.bins.reco_mass_edges_by_pt[i], label = f"Fake rate", lw = 1.5)
            plt.stairs(efficiency_pt_binned[i],self.bins.gen_mass_edges_by_pt[i], label = f"Efficiency", lw = 1.5)
            plt.legend(title = title_list[i])
            plt.xlabel(self._observable_short_label())
            plt.xlim(*self._observable_xlim(i))
            plt.ylim(0,1.05)
            hep.cms.label("Internal", data = False, lumi = 138, fontsize = 20)
            if self.groomed:
                save_path = f"./outputs/mass/fakerates_groomed_{i-1}.pdf"
            else:
                save_path = f"./outputs/mass/fakerates_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show)

    def _compute_stat_unc(self):
        for i in range(10):
            meas_flat = self.mosaic_2d_jk_list[i]
            self._perform_unfold(
                systematic='nominal',
                closure=self.closure,
                herwig_closure=self.herwig_closure,
                meas_flat=meas_flat,
                do_jk=True,
                jk_target="input",
            )

        for i in range(10):
            resp_np = self.mosaic_jk_list[i]
            self._perform_unfold(
                systematic='nominal',
                closure=self.closure,
                herwig_closure=self.herwig_closure,
                resp_np=resp_np,
                do_jk=True,
                jk_target="matrix",
            )

        jk_scale = np.sqrt(10.0 / 9.0)
        input_std = jk_scale * np.std(self.y_unf_jk_input_list, axis=0)
        matrix_std = jk_scale * np.std(self.y_unf_jk_matrix_list, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.input_stat_unc_frac = np.abs(
                np.divide(input_std, self.y_unf, out=np.zeros_like(input_std), where=self.y_unf != 0)
            )
            self.matrix_stat_unc_frac = np.abs(
                np.divide(matrix_std, self.y_unf, out=np.zeros_like(matrix_std), where=self.y_unf != 0)
            )

        self.stat_unc_frac = np.sqrt(self.input_stat_unc_frac**2 + self.matrix_stat_unc_frac**2)
        self.input_stat_unc_pt_binned = unflatten_gen_by_pt(self.input_stat_unc_frac, self.bins.gen_mass_edges_by_pt)
        self.matrix_stat_unc_pt_binned = unflatten_gen_by_pt(self.matrix_stat_unc_frac, self.bins.gen_mass_edges_by_pt)
        self.stat_unc_pt_binned = unflatten_gen_by_pt(self.stat_unc_frac, self.bins.gen_mass_edges_by_pt)

    def _select_measured_spectrum(self, closure, herwig_closure, meas_flat):
        if meas_flat is None:
            if closure:
                meas_flat = self.mosaic.sum(axis=1)
            else:
                meas_flat = self.mosaic_2d
            if herwig_closure:
                meas_flat = self.mosaic_herwig_2d
        return meas_flat

    def _apply_fake_correction(self, meas_flat, systematic, closure, herwig_closure):
        if closure or herwig_closure:
            return meas_flat

        fake_fraction = self.fake_fraction_2d
        if systematic in {"herwigUp", "herwigDown"} and hasattr(self, "fake_fraction_2d_herwig"):
            fake_fraction = self.fake_fraction_2d_herwig

        corrected = np.asarray(meas_flat, dtype=float) * (1.0 - fake_fraction)
        return np.clip(corrected, 0.0, None)

    def _build_root_binning(self):
        truth_root = ROOT.TUnfoldBinning("truth")
        reco_root = ROOT.TUnfoldBinning("reco")

        truth_signal = truth_root.AddBinning("signal")
        reco_primary = reco_root.AddBinning("primary")

        for i, edges in enumerate(self.bins.gen_mass_edges_by_pt):
            truth_node = truth_signal.AddBinning(f"pt{i}")
            truth_node.AddAxis("mass", len(edges) - 1, array("d", edges), False, False)

        for i, edges in enumerate(self.bins.reco_mass_edges_by_pt):
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

    def _store_unfold_result(self, systematic, do_jk, jk_target, unfold, h_meas, h_true):
        h_unfold = unfold.GetOutput("unfold")
        h_folded = unfold.GetFoldedOutput("folded")

        y_meas, ye_meas = self._th1_to_arrays(h_meas)
        y_true, ye_true = self._th1_to_arrays(h_true)
        x_folded, _ = self._th1_to_arrays(h_folded)
        y_unf, ye_unf = self._th1_to_arrays(h_unfold)

        if do_jk and systematic == "nominal":
            if jk_target == "matrix":
                self.y_unf_jk_matrix_list.append(y_unf)
            else:
                self.y_unf_jk_input_list.append(y_unf)
            return

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

    def _perform_unfold(self, systematic = 'nominal', closure = False, herwig_closure = False, meas_flat = None, do_jk = False, resp_np = None, jk_target = "input"):
        if resp_np is None:
            resp_np = self.mosaic_dict[systematic]
        meas_flat = self._select_measured_spectrum(closure, herwig_closure, meas_flat)
        meas_flat = self._apply_fake_correction(meas_flat, systematic, closure, herwig_closure)

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
        unfold.DoUnfold(0.0) # No regularization 
        self._store_covariances(unfold, systematic)
        self._store_unfold_result(systematic, do_jk, jk_target, unfold, h_meas, h_true)
        
        
    
    def plot_L(self, show=True):
        lMatrix = self.L
        #try plotting the L matrix root way
        c = ROOT.TCanvas("c", "L-curve Matrix", 800, 600)
        lMatrix.Draw("colz")
        c.SaveAs("outputs/mass/unfold/L_matrix_root.png")
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
        self._finalize_plot(save_path="outputs/mass/unfold/L_matrix_matplotlib.png", show=show)
        
        


    
    def _th1_to_arrays(self,h):
        nb = h.GetNbinsX()                       # bin numbers
        x  = np.arange(1, nb + 1)
        y  = np.array([h.GetBinContent(int(i)) for i in x])
        ye = np.array([h.GetBinError(int(i))   for i in x])
        return  y, ye
    def plot_folded(self, show=True):
        folded_pt_binned = unflatten_gen_by_pt(self.x_folded, self.bins.reco_mass_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.bins.reco_mass_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.bins.reco_mass_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths_reco = np.diff(self.bins.reco_mass_edges_by_pt[i])
            # two-panel plot: main + ratio
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            plt.sca(ax_top)
            hep.histplot(folded_pt_binned[i]/bin_widths_reco/folded_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], label='Folded', color='#e42536', alpha=0.8, ls='dotted', lw=3, ax=ax_top)
            hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], color='k', ls='--', alpha=1, label='Measured Data', ax=ax_top)

            # ratio (Measured / Folded)
            edges = np.array(self.bins.reco_mass_edges_by_pt[i], dtype=float)
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
            plt.xlabel(self._observable_label())

            # switch back to top axes so subsequent plotting (reco_mc, legend, labels) goes to the main panel
            plt.sca(ax_top)
            #hep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title) 
            
            if self.groomed:
                #plt.xlim(0,250)
                plt.xlim(*self._observable_xlim(i))
                hep.cms.label(self.cms_label, data = True,lumi = 138, com = 13, fontsize = 20)
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(*self._observable_xlim(i))
                hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
            save_path = f"./outputs/mass/unfold/folded_groomed_{i-1}.pdf" if self.groomed else f"./outputs/mass/unfold/folded_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
    
    def plot_jk(self, show= True):
        # Outputs
        n_pt_bins = len(self.pt_edges) - 1
        jk_pt_binned = [
            unflatten_gen_by_pt(sample, self.bins.gen_mass_edges_by_pt)
            for sample in self.y_unf_jk_input_list
        ]

        for pt_index in range(n_pt_bins):
            fig, ax = plt.subplots()
            for jk_index, unfolded_pt_binned in enumerate(jk_pt_binned):
                ax.stairs(
                    unfolded_pt_binned[pt_index],
                    self.bins.gen_mass_edges_by_pt[pt_index],
                    label=f"JK sample {jk_index}",
                    alpha=0.6,
                )

            pt_low = int(self.pt_edges[pt_index])
            pt_high = self.pt_edges[pt_index + 1]
            pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

            ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
            
            ax.set_ylabel("Unfolded entries")
            if self.groomed:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./outputs/mass/unfold/jk_outputs_groomed_pt{pt_index-1}.pdf"
            else:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./outputs/mass/unfold/jk_outputs_ungroomed_pt{pt_index-1}.pdf"

            plt.sca(ax)
            hep.cms.label(self.cms_label, data=True, lumi=138, com=13, fontsize=20)
            plt.tight_layout()
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        # Inputs
        for pt_index in range(n_pt_bins):
            fig, ax = plt.subplots()
            for jk_index, mosaic_2d_jk in enumerate(self.mosaic_2d_jk_list):
                reco_pt_binned_jk = unflatten_gen_by_pt(self.mosaic_2d_jk_list[jk_index], self.bins.reco_mass_edges_by_pt)
                ax.stairs(
                    reco_pt_binned_jk[pt_index],
                    self.bins.reco_mass_edges_by_pt[pt_index],
                    label=f"JK sample {jk_index}",
                    alpha=0.6,
                )

            pt_low = int(self.pt_edges[pt_index])
            pt_high = self.pt_edges[pt_index + 1]
            pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

            ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
            
            ax.set_ylabel("Entries")
            if self.groomed:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./outputs/mass/unfold/jk_inputs_groomed_pt{pt_index-1}.pdf"
            else:
                ax.set_xlim(*self._observable_xlim(pt_index))
                ax.set_xlabel(self._observable_label())
                save_path = f"./outputs/mass/unfold/jk_inputs_ungroomed_pt{pt_index-1}.pdf"

            plt.sca(ax)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20)
            self._finalize_plot(save_path=save_path, show=show, fig=fig)



    def plot_bottom_line(self, show=True):
        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.bins.gen_mass_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.bins.gen_mass_edges_by_pt)

        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.bins.reco_mass_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.bins.reco_mass_edges_by_pt)
        
        #now plot the ratio of unfolded to true and measured to reco mc in the same axis, just the ratio plot (no main panel)
        for i in range(len(self.pt_edges)-1):
            # two-panel plot: main + ratio
            error = self.normalized_results[i]['stat_unc']/self.normalized_results[i]['unfolded']
            fig, ax = plt.subplots(figsize=(12, 9))
            bin_widths = np.diff(self.bins.gen_mass_edges_by_pt[i])
            unfolded = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
            true = true_pt_binned[i]/bin_widths/true_pt_binned[i].sum()
            
            ratio_unf_true = np.divide(unfolded, true, out=np.full_like(unfolded, np.nan), where=true != 0)
            
            bin_widths_reco = np.diff(self.bins.reco_mass_edges_by_pt[i])
            measured = measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum()
            reco_mc = reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum()
            ratio_meas_reco = np.divide(measured, reco_mc, out=np.full_like(measured, np.nan), where=reco_mc != 0)
            
            ax.axhline(1.0, color='gray', ls='--')
            hep.histplot(ratio_unf_true, self.bins.gen_mass_edges_by_pt[i], yerr = np.abs(error),label='Unfolded / True', color='k', ls='--')
            hep.histplot(ratio_meas_reco, self.bins.reco_mass_edges_by_pt[i],  label='Measured / Reco_MC', color='#e42536', ls=':')
            ax.set_ylabel('Ratio')
            ax.set_xlim(self.bins.gen_mass_edges_by_pt[i][0], self.bins.gen_mass_edges_by_pt[i][-1])
            ax.set_ylim(0.5, 1.5)
            plt.xlim(*self._observable_xlim(i))
            plt.xlabel(self._observable_label())
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title) 
            hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
            save_path = f"./outputs/mass/bottom_line_groomed_{i-1}.pdf" if self.groomed else f"./outputs/mass/bottom_line_ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_unfolded_fancy(self, log=False, show=True):
        markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
        npt = len(self.pt_edges)-1
        title_list = ["",r"200$<$$p_T$$<$290 GeV", r"290 $<$$p_T$$<$400 GeV", r"400 $<$$p_T$$< \, \infty$  GeV"]
        true_herwig_pt_binned = unflatten_gen_by_pt(self.y_true_herwig, self.bins.gen_mass_edges_by_pt)
        for i in range(npt):
            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 10)
            )
            bin_widths = np.diff(self.bins.gen_mass_edges_by_pt[i])
            herwig_norm = true_herwig_pt_binned[i] / bin_widths / true_herwig_pt_binned[i].sum()
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            rho_edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            plt.sca(ax_main)
            plt.stairs( unfolded + syst_up,
                self.bins.gen_mass_edges_by_pt[i],
                baseline = unfolded - syst_down,
                fill = True, color = "yellowgreen" , label = r"Syst. $\oplus$ Stat. Unc.")
            plt.stairs( unfolded + stat_unc,
                self.bins.gen_mass_edges_by_pt[i],
                baseline = unfolded - stat_unc,
                fill = True, color = "darkgreen" , label = "Stat. Unc.")
            plt.stairs(self.normalized_results[i]['true'], self.bins.gen_mass_edges_by_pt[i], label = 'PYTHIA8', color = 'b', ls = 'dotted', lw = 3)
            plt.stairs(herwig_norm, self.bins.gen_mass_edges_by_pt[i], label='HERWIG7', color='r', ls='dashdot', lw=2)
            plt.plot(centers, unfolded, color='k', lw=0, marker=markers[i], markersize=8, label='Unfolded')

            plt.legend(title = title_list[i], fontsize=14, title_fontsize=15)
            hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
            plt.ylabel(self._normalized_ylabel())

            # Ratio Plot
            plt.sca(ax_ratio)
            plt.axhline(1.0, color='gray', ls='--')
            ratio_pythia = np.divide(unfolded, self.normalized_results[i]['true'])
            ratio_herwig = np.divide(unfolded, herwig_norm)
            stat_frac = np.divide(stat_unc, unfolded, out=np.zeros_like(stat_unc), where=unfolded != 0)
            total_frac_up = np.divide(syst_up, unfolded, out=np.zeros_like(syst_up), where=unfolded != 0)
            total_frac_down = np.divide(syst_down, unfolded, out=np.zeros_like(syst_down), where=unfolded != 0)

            plt.stairs(1.0 + total_frac_up, self.bins.gen_mass_edges_by_pt[i], baseline=1.0 - total_frac_down, fill=True, color="yellowgreen", label=r"Syst. $\oplus$ Stat. Unc.")
            plt.stairs(1.0 + stat_frac, self.bins.gen_mass_edges_by_pt[i], baseline=1.0 - stat_frac, fill=True, color="darkgreen", label="Stat. Unc.")
            plt.stairs(ratio_pythia, self.bins.gen_mass_edges_by_pt[i], color='b', ls='dotted', lw=2, label='Data / PYTHIA8')
            plt.stairs(ratio_herwig, self.bins.gen_mass_edges_by_pt[i], color='r', ls='dashdot', lw=2, label='Data / HERWIG7')
            plt.ylim(0, 2)
            plt.xlabel(self._observable_label())
            plt.ylabel(r"$\frac{Data}{Simulation}$")
            plt.xlim(*self._observable_xlim(i))
            if self.closure:
                save_path = f"./outputs/mass/closure_groomed_{i-1}.pdf" if self.groomed else f"./outputs/mass/closure_ungroomed_{i-1}.pdf"
            else:
                save_path = f"./outputs/mass/unfold/groomed_{i-1}.pdf" if self.groomed else f"./outputs/mass/unfold/ungroomed_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)
        
        # Now also plot a summary plot, with all of them together, but shifted on y axis for visibility

        for i in range(1, npt):
            exponent = 2 * i - 1
            scale = 10 ** exponent
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)
            bin_widths = np.diff(self.bins.gen_mass_edges_by_pt[i])
            herwig_norm = true_herwig_pt_binned[i] / bin_widths / true_herwig_pt_binned[i].sum()

            y_syst_up = scale * (unfolded + syst_up)
            y_syst_down = scale * (unfolded - syst_down)
            y_syst_down = np.maximum(y_syst_down, scale * unfolded * 1e-1)
            y_stat_up = scale * (unfolded + stat_unc)
            y_stat_down = scale * (unfolded - stat_unc)
            plt.stairs(scale * np.array(self.normalized_results[i]['true'], dtype=float), self.bins.gen_mass_edges_by_pt[i], label='PYTHIA8', color='b', ls='dotted', lw=3)
            plt.stairs(scale * herwig_norm, self.bins.gen_mass_edges_by_pt[i], label='HERWIG7', color='r', ls='dashdot', lw=2)
            plt.stairs(y_syst_up, self.bins.gen_mass_edges_by_pt[i], baseline=y_syst_down, fill=True, color="yellowgreen", label=r"Syst. $\oplus$ Stat. Unc.", alpha = 0.8)
            plt.stairs(y_stat_up, self.bins.gen_mass_edges_by_pt[i], baseline=y_stat_down, fill=True, color="darkgreen", label="Stat. Unc.")
            rho_edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            plt.plot(centers, scale * unfolded, label=rf'$10^{{{exponent}}}$ x {title_list[i]}', color='k', lw=0, marker=markers[i])
            

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
        plt.legend(fontsize=13)
        plt.xlabel(self._observable_label())
        plt.ylabel(self._normalized_ylabel())
        hep.cms.label(self.cms_label, data = True, lumi = 138, com = 13, fontsize = 20)
        plt.xlim(*self._observable_xlim())

        save_path = f"./outputs/mass/groomed_summary.pdf" if self.groomed else f"./outputs/mass/ungroomed_summary.pdf"
        self._finalize_plot(save_path=save_path, show=show)

    def plot_unfolded_summary_linear(self, show=True):
        markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
        npt = len(self.pt_edges) - 1
        title_list = ["", r"200$<$$p_T$$<$290 GeV", r"290 $<$$p_T$$<$400 GeV", r"400 $<$$p_T$$< \, \infty$  GeV"]

        fig = plt.figure(figsize=(12, 10))
        for i in range(1, npt):
            exponent = 2 * i - 1
            scale = 10 ** exponent
            unfolded = np.array(self.normalized_results[i]['unfolded'], dtype=float)
            syst_up = np.array(self.normalized_results[i]['syst_unc']['up'], dtype=float)
            syst_down = np.array(self.normalized_results[i]['syst_unc']['down'], dtype=float)
            stat_unc = np.array(self.normalized_results[i]['stat_unc'], dtype=float)

            y_syst_up = scale * (unfolded + syst_up)
            y_syst_down = scale * (unfolded - syst_down)
            y_stat_up = scale * (unfolded + stat_unc)
            y_stat_down = scale * (unfolded - stat_unc)

            plt.stairs(scale * np.array(self.normalized_results[i]['true'], dtype=float), self.bins.gen_mass_edges_by_pt[i], label='PYTHIA7', color='b', ls='dotted', lw=3)
            plt.stairs(y_syst_up, self.bins.gen_mass_edges_by_pt[i], baseline=y_syst_down, fill=True, color="yellowgreen", label=r"Syst. $\oplus$ Stat. Unc.", alpha=0.8)
            plt.stairs(y_stat_up, self.bins.gen_mass_edges_by_pt[i], baseline=y_stat_down, fill=True, color="darkgreen", label="Stat. Unc.")
            rho_edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)
            centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            plt.plot(centers, scale * unfolded, label=rf'$10^{{{exponent}}}$ x {title_list[i]}', color='k', lw=0, marker=markers[i])

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        for h, l in zip(handles, labels):
            if l == "" or l in seen:
                try:
                    h.set_label('_nolegend_')
                except Exception:
                    pass
            else:
                seen.add(l)
        plt.legend(fontsize=15)
        plt.xlabel(self._observable_label())
        plt.ylabel(self._normalized_ylabel())
        hep.cms.label(self.cms_label, data=True, lumi=138, com=13, fontsize=20)
        plt.xlim(*self._observable_xlim())

        save_path = f"./outputs/mass/groomed_summary_linear.pdf" if self.groomed else f"./outputs/mass/ungroomed_summary_linear.pdf"
        self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_herwig_pythia_comparison(self, show=True):
        """
        One plot per pT bin with three panels:
          - Main panel  : normalized gen-level jet mass for PYTHIA8 and HERWIG7.
          - Middle panel: gen-level model uncertainty (HERWIG7 - PYTHIA8) / PYTHIA8.
          - Bottom panel: post-unfolding model uncertainty, i.e. envelope of
                          |unfolded_herwigUp - nominal| / nominal and
                          |unfolded_herwigDown - nominal| / nominal.
        """
        true_herwig_pt_binned = unflatten_gen_by_pt(self.y_true_herwig, self.bins.gen_mass_edges_by_pt)
        npt = len(self.pt_edges) - 1
        title_list = [
            "",
            r"$p_T$ 200 – 290 GeV",
            r"$p_T$ 290 – 400 GeV",
            r"$p_T$ 400 – $\infty$ GeV",
        ]

        ratio_ymax = 0.5

        def _annotate_overflow(ax, values, edges, ymax, color, fontsize=11):
            """Write the numeric value above the clip line for bins that exceed ymax."""
            centers = 0.5 * (edges[:-1] + edges[1:])
            for x, y in zip(centers, values):
                if np.isfinite(y) and y > ymax:
                    ax.text(x, ymax - 0.02, f"{y:.2f}",
                            ha='center', va='top', fontsize=fontsize,
                            color=color, clip_on=True)

        for i in range(npt):
            bin_widths = np.diff(self.bins.gen_mass_edges_by_pt[i])
            edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)

            pythia_norm = np.array(self.normalized_results[i]['true'], dtype=float)
            herwig_raw = true_herwig_pt_binned[i]
            herwig_norm = herwig_raw / bin_widths / herwig_raw.sum()

            # Unsigned gen-level difference
            gen_model_diff = np.abs(np.divide(
                herwig_norm - pythia_norm, pythia_norm,
                out=np.full_like(herwig_norm, np.nan), where=pythia_norm != 0,
            ))

            # Post-unfolding model uncertainty: envelope of herwigUp / herwigDown
            syst_frac = self._build_syst_fraction_dict(i)
            herwig_up_frac   = syst_frac.get('herwigUp',   np.zeros_like(pythia_norm))
            herwig_down_frac = syst_frac.get('herwigDown', np.zeros_like(pythia_norm))
            model_unc_unfolded = np.maximum(herwig_up_frac, herwig_down_frac)

            # PYTHIA8 / HERWIG7 ratio (gen-level, normalized)
            pythia_over_herwig = np.divide(
                pythia_norm, herwig_norm,
                out=np.full_like(pythia_norm, np.nan), where=herwig_norm != 0,
            )

            fig, (ax_main, ax_ph, ax_ratio, ax_model) = plt.subplots(
                4, 1, sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1, 1]},
                figsize=(10, 14),
            )

            # Main panel
            hep.histplot(pythia_norm, edges, label='PYTHIA8', color='#5790fc', ls='dotted', lw=3, ax=ax_main)
            hep.histplot(herwig_norm, edges, label='HERWIG7', color='#e42536', ls='dashdot', lw=2, ax=ax_main)
            ax_main.set_ylabel(self._normalized_ylabel())
            pt_title = title_list[i] if i < len(title_list) else ""
            ax_main.legend(title=pt_title, fontsize=14, title_fontsize=15)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20, ax=ax_main)

            # PYTHIA8 / HERWIG7 ratio panel
            ax_ph.axhline(1.0, color='gray', ls='--', lw=1)
            hep.histplot(pythia_over_herwig, edges, color='#5790fc', ls='dotted', lw=2, ax=ax_ph)
            ax_ph.set_ylabel('P8 / H7', fontsize=13)
            ax_ph.set_ylim(0.5, 1.5)
            ax_ph.set_yticks([0.5, 1.0, 1.5])

            # Gen-level model uncertainty panel: |HERWIG7 - PYTHIA8| / PYTHIA8
            ax_ratio.axhline(0.0, color='gray', ls='--', lw=1)
            hep.histplot(gen_model_diff, edges, color='#e42536', ls='dashdot', lw=2, ax=ax_ratio)
            ax_ratio.set_ylabel('Rel. Unc. GEN', fontsize=13)
            ax_ratio.set_ylim(0, ratio_ymax)
            ax_ratio.set_yticks([0.0, 0.25, 0.5])
            _annotate_overflow(ax_ratio, gen_model_diff, edges, ratio_ymax, color='#e42536')

            # Post-unfolding model uncertainty panel
            ax_model.axhline(0.0, color='gray', ls='--', lw=1)
            hep.histplot(model_unc_unfolded, edges, color='#7a21dd', ls='solid', lw=2, ax=ax_model)
            ax_model.set_ylabel('Rel. Unc. Unfolded', fontsize=13)
            ax_model.set_ylim(0, ratio_ymax)
            ax_model.set_yticks([0.0, 0.25, 0.5])
            ax_model.set_xlabel(self._observable_label())
            _annotate_overflow(ax_model, model_unc_unfolded, edges, ratio_ymax, color='#7a21dd')

            for ax in (ax_main, ax_ph, ax_ratio, ax_model):
                ax.set_xlim(*self._observable_xlim(i))

            ax_model.tick_params(axis='x', pad=8)
            ax_model.tick_params(axis='y', pad=6)
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.1)
            suffix = "groomed" if self.groomed else "ungroomed"
            save_path = f"./outputs/mass/herwig_pythia_comparison_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_unfolded(self, log=False, show=True):

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.bins.gen_mass_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.bins.reco_mass_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.bins.reco_mass_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.bins.gen_mass_edges_by_pt)
        true_herwig_pt_binned = unflatten_gen_by_pt(self.y_true_herwig, self.bins.gen_mass_edges_by_pt) 
        #error_pt_binned = unflatten_gen_by_pt(self.ye_unf, self.bins.gen_mass_edges_by_pt)
        self.normalized_herwig = []
        #print("Herwig pt Binned", true_herwig_pt_binned)
        self.herwig_closure_unc = []
        for i in range(len(self.pt_edges)-1):
            yerr = self.normalized_results[i]['syst_unc']['up']
            bin_widths = np.diff(self.bins.gen_mass_edges_by_pt[i])
            bin_widths_reco = np.diff(self.bins.reco_mass_edges_by_pt[i])
            #self.normalized_herwig.append(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum())
            if self.herwig_closure:
                hep.histplot(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum(), self.bins.gen_mass_edges_by_pt[i], color='#964a8b', label = 'Herwig', alpha = 0.7, ls = 'dotted')
            else:
                hep.histplot(true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(), self.bins.gen_mass_edges_by_pt[i], color='#5790fc', label = 'PYTHIA', alpha = 0.8, ls = 'dotted', lw = 3)
            hep.histplot(unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), self.bins.gen_mass_edges_by_pt[i], label = 'Unfolded Herwig' if self.herwig_closure else 'Unfolded', color = 'k', ls = '--' )

            

            #hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], color = 'k', ls= '--', alpha= 0.5, label = 'Meas' )
            #dhep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.bins.reco_mass_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f" {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title, fontsize = 18) 
            
            if self.groomed:
                #plt.xlim(0,250)
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
            save_path = f"./outputs/mass/unfold/unfolded_basic_groomed_{i-1}.pdf" if self.groomed else f"./outputs/mass/unfold/unfolded_basic_ungroomed_{i-1}.pdf"
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
                plt.stairs(herwig_closure_unc, self.bins.gen_mass_edges_by_pt[i], label = 'Closure Unc (|Herwig - Unfolded| / Herwig)', color='#7a21dd', ls = 'dotted')
                if not self.groomed:
                    plt.xlim(*self._observable_xlim(i))
                else:
                    plt.xlim(*self._observable_xlim(i))
                plt.ylim(0, 1)
                plt.xlabel(self._observable_label())
                #plt.legend()
                save_path = f"./outputs/mass/unfold/herwig_closure_unc_groomed_{i-1}.pdf" if self.groomed else f"./outputs/mass/unfold/herwig_closure_unc_ungroomed_{i-1}.pdf"
                self._finalize_plot(save_path=save_path, show=show)
        # Save uncertainty in a file for later use
        if self.herwig_closure:
            if self.groomed:
                np.save("./inputs/massInputs/herwig_closure_unc_mass_groomed.npy", self.herwig_closure_unc)
            else:
                np.save("./inputs/massInputs/herwig_closure_unc_mass_ungroomed.npy", self.herwig_closure_unc)
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
        gen_mass_bin_edges_by_pt = self.bins.gen_mass_edges_by_pt
        reco_mass_bin_edges_by_pt = self.bins.reco_mass_edges_by_pt

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, gen_mass_bin_edges_by_pt)
        #print("Unfolded pt binned:", unfolded_pt_binned)
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
        #herwig_unc = np.load("./inputs/massInputs/herwig_closure_unc_mass_groomed.npy") if self.groomed else np.load("./inputs/massInputs/herwig_closure_unc_mass_ungroomed.npy")
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
                        #print("Systematic Down:", syst, "Diff Down:", diff_down)
                        syst_down_total += diff_down**2

                        ## This is for fethcing uncertainty from non-closure of herwig
                        # diff_down = herwig_unc[i] * nominal
                        # syst_down_total += diff_down**2
                    else:    
                        syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_down = np.abs(syst_down - nominal)
                        #print("Systematic Down:", syst, "Diff Down:", diff_down)
                        syst_down_total += diff_down**2
                else:
                    if syst=='nominal':
                        continue
                    if syst.startswith('herwig'):
                        #print("New Herwig adopted")
                        ## Difference way
                        syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_up = np.abs(syst_up - nominal)
                        #print("Systematic Up:", syst, "Diff Up:", diff_up)
                        syst_up_total += diff_up**2

                        ## Fetching uncertainty from non-closure of herwig
                        # diff_up = herwig_unc[i] * nominal
                        # syst_up_total += diff_up**2
                    else:    
                        syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                        diff_up = np.abs(syst_up - nominal)
                        #print("Systematic up:", syst, "Diff up:", diff_up)
                        syst_up_total += diff_up**2
            input_stat_unc = self.input_stat_unc_pt_binned[i] * nominal
            matrix_stat_unc = self.matrix_stat_unc_pt_binned[i] * nominal
            stat_unc = np.sqrt(input_stat_unc**2 + matrix_stat_unc**2)
                
            syst_up_total += stat_unc**2
            syst_down_total += stat_unc**2
            # Take sqrt of sum of squares for total uncertainty
            total_up_unc = np.sqrt(syst_up_total)
            total_down_unc = np.sqrt(syst_down_total)
            self.normalized_results[i]['syst_unc'] = {
            'up': total_up_unc,
            'down': total_down_unc
            }
            self.normalized_results[i]['input_stat_unc_frac'] = self.input_stat_unc_pt_binned[i]
            self.normalized_results[i]['matrix_stat_unc_frac'] = self.matrix_stat_unc_pt_binned[i]
            self.normalized_results[i]['stat_unc_frac'] = self.stat_unc_pt_binned[i]
            self.normalized_results[i]['input_stat_unc'] = input_stat_unc
            self.normalized_results[i]['matrix_stat_unc'] = matrix_stat_unc
            self.normalized_results[i]['stat_unc'] = stat_unc

    def plot_statistical_fraction(self, show=True):
        for i, result in enumerate(self.normalized_results):
            plt.figure()
            pt_bin = result['pt_bin']
            input_stat_fraction = result['input_stat_unc_frac']
            matrix_stat_fraction = result['matrix_stat_unc_frac']

            hep.histplot(
                input_stat_fraction[1:],
                self.bins.gen_mass_edges_by_pt[i][1:],
                label="Input statistical Uncertainty",
                ls="--",
            )
            hep.histplot(
                matrix_stat_fraction[1:],
                self.bins.gen_mass_edges_by_pt[i][1:],
                label="Matrix uncertainty",
                ls="-.",
            )

            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
            hep.cms.label(self.cms_label, data=True, lumi=138, com=13, fontsize=20)
            plt.xlim(*self._observable_xlim(i))
            plt.xlabel(self._observable_label())
            plt.ylabel("Fractional Uncertainty")
            ax = plt.gca()
            ax.tick_params(axis='x', pad=8)
            ax.tick_params(axis='y', pad=8)
            plt.subplots_adjust(left=0.16, bottom=0.15)
            save_path = (
                f'./outputs/mass/uncertainties/stat_fraction_groomed_{i-1}.pdf'
                if self.groomed
                else f'./outputs/mass/uncertainties/stat_fraction_ungroomed_{i-1}.pdf'
            )
            self._finalize_plot(save_path=save_path, show=show)

    def plot_systematic_fraction(self, syst_name='all', show=True, log=True):
        self._plot_systematic_fraction_summary(grouped=False, show=show, log=log)

    def _get_systematic_group_name(self, syst_name):
        syst_lower = syst_name.lower()
        if syst_lower.startswith("jes") or syst_lower.startswith("jer"):
            return "Jet Energy"
        if syst_lower.startswith("jms") or syst_lower.startswith("jmr"):
            return "Jet Mass"
        if syst_lower.startswith("ele") or syst_lower.startswith("mu"):
            return "Lepton SFs"
        if syst_lower.startswith("isr") or syst_lower.startswith("fsr"):
            return "Parton Shower"
        if (
            syst_lower.startswith("pu")
            or syst_lower.startswith("pdf")
            or syst_lower.startswith("q2")
            or syst_lower.startswith("l1prefiring")
        ):
            return "Other Theory"
        return None

    def _get_systematic_label(self, syst_name):
        base_name, _ = self._split_systematic_variation(syst_name)
        label_map = {
            "pu": "Pileup",
            "l1prefiring": "L1 Prefiring",
            "q2": r"Q$^2$ Scale",
            "pdf": "PDF",
            "herwig": "Model Uncertainty",
            "isr": "ISR",
            "fsr": "FSR",
            "jms": "JMS",
            "jmr": "JMR",
        }
        return label_map.get(base_name.lower(), base_name)

    def _split_systematic_variation(self, syst_name):
        match = re.match(r"^(.*?)(Up|Down)(?:_.*)?$", syst_name)
        if match:
            return match.group(1), match.group(2)
        return syst_name, None

    def _get_systematic_summary_name(self, syst_name, grouped=False):
        if grouped:
            group_name = self._get_systematic_group_name(syst_name)
            if group_name is not None:
                return group_name
            return self._get_systematic_label(syst_name)

        base_name, _ = self._split_systematic_variation(syst_name)
        base_lower = base_name.lower()
        summary_map = {
            "elereco": "Electron RECO",
            "eleid": "Electron ID",
            "eletrig": "Electron Trigger",
            "mureco": "Muon RECO",
            "muid": "Muon ID",
            "mutrig": "Muon Trigger",
            "muiso": "Muon ISO",
            "pu": "Pileup",
            "pdf": "PDF",
            "q2": "Q2 Scale",
            "l1prefiring": "L1 Prefiring",
            "herwig": "Model Uncertainty",
        }
        if base_lower.startswith("jes"):
            return "JES"
        if base_lower.startswith("jer"):
            return "JER"
        if base_lower.startswith("isr"):
            return "ISR"
        if base_lower.startswith("fsr"):
            return "FSR"
        if base_lower.startswith("jmr"):
            return "JMR"
        if base_lower.startswith("jms"):
            return "JMS"
        return summary_map.get(base_lower, self._get_systematic_label(syst_name))

    def _build_syst_fraction_dict(self, pt_index):
        result = self.normalized_results[pt_index]
        nominal = result["unfolded"]
        total_syst_up = result["syst_unc"]["up"]
        total_syst_down = result["syst_unc"]["down"]
        syst_fraction_dict = {}

        for syst_name, syst_unfolded in self.normalized_systematics[pt_index]["unfolded"].items():
            diff = syst_unfolded - nominal
            syst_fraction = np.abs(np.divide(diff, nominal, out=np.zeros_like(diff), where=nominal != 0))
            syst_fraction_dict[syst_name] = syst_fraction

        stat_fraction = self.stat_unc_pt_binned[pt_index]
        total_syst_fraction_up = np.abs(np.divide(total_syst_up, np.abs(nominal), out=np.zeros_like(total_syst_up), where=np.abs(nominal) != 0))
        total_syst_fraction_down = np.abs(np.divide(total_syst_down, np.abs(nominal), out=np.zeros_like(total_syst_down), where=np.abs(nominal) != 0))

        syst_fraction_dict["Stat Unc"] = stat_fraction
        syst_fraction_dict["Total_Up"] = total_syst_fraction_up
        syst_fraction_dict["Total_Down"] = total_syst_fraction_down
        return syst_fraction_dict

    def _group_syst_fraction_dict(self, syst_fraction_dict, grouped=True):
        grouped_fraction_dict = {}
        accumulators = {}

        for syst_name, syst_fraction in syst_fraction_dict.items():
            if syst_name in {"Stat Unc", "Total_Up", "Total_Down"}:
                continue

            target_name = self._get_systematic_summary_name(syst_name, grouped=grouped)
            _, variation = self._split_systematic_variation(syst_name)
            target_key = f"{target_name}Down" if variation == "Down" else f"{target_name}Up"

            if target_key not in accumulators:
                accumulators[target_key] = np.zeros_like(syst_fraction)
            accumulators[target_key] += syst_fraction**2

        for target_key, target_sum in accumulators.items():
            grouped_fraction_dict[target_key] = np.sqrt(target_sum)

        grouped_fraction_dict["Stat Unc"] = syst_fraction_dict["Stat Unc"]
        grouped_fraction_dict["Total_Up"] = syst_fraction_dict["Total_Up"]
        grouped_fraction_dict["Total_Down"] = syst_fraction_dict["Total_Down"]
        return grouped_fraction_dict

    def _plot_systematic_fraction_summary(self, grouped=False, show=True, log=True):
        self.syst_fraction_dicts = []
        linear_ymax = 0.5
        grouped_legend_order = [
            "Jet Energy",
            "Jet Mass",
            "Parton Shower",
            "Lepton SFs",
            "Other Theory",
            "Model Uncertainty",
            "Stat Unc",
            "Total",
        ]
        summary_style_map = {
            "JES": {"color": "#1f77b4", "ls": "-"},
            "JER": {"color": "#ff7f0e", "ls": "-"},
            "Pileup": {"color": "#d62728", "ls": "-"},
            "Electron RECO": {"color": "#9467bd", "ls": "-"},
            "Electron ID": {"color": "#8c564b", "ls": "-"},
            "Electron Trigger": {"color": "#e377c2", "ls": "-"},
            "Muon RECO": {"color": "#7f7f7f", "ls": "-"},
            "Muon ID": {"color": "#bcbd22", "ls": "-"},
            "Muon Trigger": {"color": "#17becf", "ls": "-"},
            "Muon ISO": {"color": "#4c78a8", "ls": "-"},
            "PDF": {"color": "#f58518", "ls": "-"},
            "Q2 Scale": {"color": "#54a24b", "ls": "-"},
            "L1 Prefiring": {"color": "#eeca3b", "ls": "-"},
            "ISR": {"color": "#b279a2", "ls": "-"},
            "FSR": {"color": "#ff9da6", "ls": "-"},
            "JMR": {"color": "#9d755d", "ls": "-"},
            "JMS": {"color": "#bab0ab", "ls": "-"},
            "Jet Energy": {"color": "#1f77b4", "ls": "-"},
            "Jet Mass": {"color": "#2ca02c", "ls": "-"},
            "Parton Shower": {"color": "#d62728", "ls": "-"},
            "Lepton SFs": {"color": "#9467bd", "ls": "-"},
            "Other Theory": {"color": "#8c564b", "ls": "-"},
            "Model Uncertainty": {"color": "#7f3c8d", "ls": "-."},
            "Stat Unc": {"color": "#4c78a8", "ls": ":"},
            "Total": {"color": "k", "ls": "-", "lw": 3},
        }

        for i, result in enumerate(self.normalized_results):
            fig = plt.figure(figsize=(12, 8))
            pt_bin = result["pt_bin"]
            syst_fraction_dict = self._build_syst_fraction_dict(i)
            result["syst_fraction_dict"] = syst_fraction_dict
            self.syst_fraction_dicts.append(syst_fraction_dict)

            plot_fraction_dict = self._group_syst_fraction_dict(syst_fraction_dict, grouped=grouped)
            rho_edges = np.asarray(self.bins.gen_mass_edges_by_pt[i], dtype=float)
            rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])

            plotted_labels = set()
            for syst_name, syst_fraction in plot_fraction_dict.items():
                label, variation = self._split_systematic_variation(syst_name)
                if syst_name in {"Stat Unc", "Total_Up", "Total_Down"} or variation == "Down":
                    continue

                if label in plotted_labels:
                    continue

                style = summary_style_map.get(label, {"ls": "-"})
                hep.histplot(
                    syst_fraction,
                    self.bins.gen_mass_edges_by_pt[i],
                    label=label,
                    **style,
                )
                plotted_labels.add(label)

            hep.histplot(
                plot_fraction_dict["Stat Unc"],
                self.bins.gen_mass_edges_by_pt[i],
                label="Stat Unc",
                **summary_style_map["Stat Unc"],
            )
            hep.histplot(
                plot_fraction_dict["Total_Up"],
                self.bins.gen_mass_edges_by_pt[i],
                label="Total",
                **summary_style_map["Total"],
            )

            if not log:
                ax = plt.gca()
                for x_pos, y_val in zip(rho_centers, np.asarray(plot_fraction_dict["Total_Up"], dtype=float)):
                    if y_val > linear_ymax:
                        ax.text(
                            x_pos,
                            linear_ymax - 0.015,
                            f"{y_val:.2f}",
                            ha="center",
                            va="top",
                            fontsize=13,
                            clip_on=True,
                        )

            if log:
                plt.yscale("log")
            if pt_bin[1] == float("inf") or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            if grouped:
                label_to_handle = dict(zip(labels, handles))
                ordered_labels = [label for label in grouped_legend_order if label in label_to_handle]
                ordered_handles = [label_to_handle[label] for label in ordered_labels]
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    title=rf"$p_T$  {pt_bin_label} GeV",
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    borderaxespad=0.0,
                )
            else:
                ax.legend(
                    title=rf"$p_T$  {pt_bin_label} GeV",
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    borderaxespad=0.0,
                )
            hep.cms.label(self._cms_extra_label(), data=True, lumi=138, com=13, fontsize=20, ax=ax)
            if log:
                plt.ylim(10e-5, 1)
            else:
                plt.ylim(0, linear_ymax)
            plt.xlim(*self._observable_xlim(i))
            #plt.xlim(0,200)
            ax.set_xlabel(self._observable_label())
            ax.set_ylabel("Fractional Uncertainty")
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)

            if grouped:
                if log:
                    save_path = (
                        f"./outputs/mass/uncertainties/summary_grouped_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./outputs/mass/uncertainties/summary_grouped_ungroomed_{i-1}.pdf"
                    )
                else:
                    save_path = (
                        f"./outputs/mass/uncertainties/summary_grouped_linear_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./outputs/mass/uncertainties/summary_grouped_linear_ungroomed_{i-1}.pdf"
                    )
            else:
                if log:
                    save_path = (
                        f"./outputs/mass/uncertainties/summary_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./outputs/mass/uncertainties/summary_ungroomed_{i-1}.pdf"
                    )
                else:
                    save_path = (
                        f"./outputs/mass/uncertainties/summary_linear_groomed_{i-1}.pdf"
                        if self.groomed
                        else f"./outputs/mass/uncertainties/summary_linear_ungroomed_{i-1}.pdf"
                    )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def plot_systematic_fraction_grouped(self, show=True, log=True):
        self._plot_systematic_fraction_summary(grouped=True, show=show, log=log)

    def _resolve_raw_systematic_pair(self, syst_name):
        if syst_name in self.systematics:
            base_name, variation = self._split_systematic_variation(syst_name)
            if variation == "Up":
                return syst_name, f"{base_name}Down"
            if variation == "Down":
                return f"{base_name}Up", syst_name

        candidates = [name for name in self.systematics if name != "nominal"]
        matched_up = None
        matched_down = None

        for candidate in candidates:
            base_name, variation = self._split_systematic_variation(candidate)
            if base_name.lower() != syst_name.lower():
                continue
            if variation == "Up":
                matched_up = candidate
            elif variation == "Down":
                matched_down = candidate

        return matched_up, matched_down

    def plot_nominal_minus_variation(self, syst_names=["pu"], show=True):
        for i, result in enumerate(self.normalized_results):
            fig, ax = plt.subplots(figsize=(12, 8))
            nominal = result["unfolded"]
            pt_bin = result["pt_bin"]

            for syst in syst_names:
                up_key, down_key = self._resolve_raw_systematic_pair(syst)
                label = self._get_systematic_label(syst)

                if up_key in self.normalized_systematics[i]["unfolded"]:
                    up_shift = nominal - self.normalized_systematics[i]["unfolded"][up_key]
                    hep.histplot(
                        up_shift,
                        self.bins.gen_mass_edges_by_pt[i],
                        label=f"{label}: nominal - Up",
                        ls="-",
                    )

                if down_key in self.normalized_systematics[i]["unfolded"]:
                    down_shift = nominal - self.normalized_systematics[i]["unfolded"][down_key]
                    hep.histplot(
                        down_shift,
                        self.bins.gen_mass_edges_by_pt[i],
                        label=f"{label}: nominal - Down",
                        ls="--",
                    )

            if pt_bin[1] == float("inf") or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"

            ax.axhline(0.0, color="k", lw=1, alpha=0.5)
            ax.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize=15)
            hep.cms.label(self._cms_extra_label(), data=True, lumi=138, com=13, fontsize=20, ax=ax)
            #ax.set_xlim(*self._observable_xlim(i))
            ax.set_xlim(0,200)
            ax.set_xlabel(self._observable_label())
            ax.set_ylabel("Nominal - Variation")
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", pad=8)

            save_stub = "_".join(str(name) for name in syst_names)
            save_path = (
                f"./outputs/mass/uncertainties/nominal_minus_{save_stub}_groomed_{i-1}.pdf"
                if self.groomed
                else f"./outputs/mass/uncertainties/nominal_minus_{save_stub}_ungroomed_{i-1}.pdf"
            )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    
    def plot_systematic_frac_indiv(self, syst_names=['JES', 'JER'], ylim=None, show=True):
        def build_plot_fraction_dict(raw_syst_fraction_dict):
            plot_fraction_dict = self._group_syst_fraction_dict(raw_syst_fraction_dict, grouped=False)

            electron_keys = [key for key in plot_fraction_dict if key.startswith("Electron ")]
            if electron_keys:
                electron_up = [plot_fraction_dict[key] for key in electron_keys if key.endswith("Up")]
                electron_down = [plot_fraction_dict[key] for key in electron_keys if key.endswith("Down")]
                if electron_up:
                    plot_fraction_dict["ElectronSFUp"] = np.sqrt(np.sum([value**2 for value in electron_up], axis=0))
                if electron_down:
                    plot_fraction_dict["ElectronSFDown"] = np.sqrt(np.sum([value**2 for value in electron_down], axis=0))

            muon_keys = [key for key in plot_fraction_dict if key.startswith("Muon ")]
            if muon_keys:
                muon_up = [plot_fraction_dict[key] for key in muon_keys if key.endswith("Up")]
                muon_down = [plot_fraction_dict[key] for key in muon_keys if key.endswith("Down")]
                if muon_up:
                    plot_fraction_dict["MuonSFUp"] = np.sqrt(np.sum([value**2 for value in muon_up], axis=0))
                if muon_down:
                    plot_fraction_dict["MuonSFDown"] = np.sqrt(np.sum([value**2 for value in muon_down], axis=0))

            return plot_fraction_dict

        def resolve_syst_keys(syst_fraction_dict, syst):
            alias_bases = [syst]
            summary_name = self._get_systematic_summary_name(syst, grouped=False)
            if summary_name not in alias_bases:
                alias_bases.append(summary_name)
            label_name = self._get_systematic_label(syst)
            if label_name not in alias_bases:
                alias_bases.append(label_name)

            for base_name in alias_bases:
                up_key = f"{base_name}Up"
                down_key = f"{base_name}Down"
                if up_key in syst_fraction_dict or down_key in syst_fraction_dict:
                    return up_key, down_key

            return f"{syst}Up", f"{syst}Down"

        # First, collect all values to determine global y-range
        all_values = []
        for i, result in enumerate(self.normalized_results):
            raw_syst_fraction_dict = result.get('syst_fraction_dict', {})
            syst_fraction_dict = build_plot_fraction_dict(raw_syst_fraction_dict)
            for syst in syst_names:
                up_key, down_key = resolve_syst_keys(syst_fraction_dict, syst)
                if up_key in syst_fraction_dict:
                    all_values.append(np.abs(syst_fraction_dict[up_key]))
                if down_key in syst_fraction_dict:
                    all_values.append(np.abs(syst_fraction_dict[down_key]))


        # Now plot with fixed y-range
        for i, result in enumerate(self.normalized_results):
            raw_syst_fraction_dict = result.get('syst_fraction_dict', {})
            syst_fraction_dict = build_plot_fraction_dict(raw_syst_fraction_dict)
            #plt.figure(figsize=(12, 8))
            pt_bin = result['pt_bin']
            for syst in syst_names:
                up_key, down_key = resolve_syst_keys(syst_fraction_dict, syst)
                color_map = ['#e42536', '#5790fc', '#964a8b']
                color = color_map[syst_names.index(syst)] if syst in syst_names and syst_names.index(syst) < len(color_map) else None
                if up_key not in syst_fraction_dict and down_key not in syst_fraction_dict:
                    print(
                        f"[DEBUG] Neither '{up_key}' nor '{down_key}' found in syst_fraction_dict "
                        f"for pt bin {pt_bin}. Available keys: {list(syst_fraction_dict.keys())}"
                    )

                # Plot Up uncertainty (solid)
                label_dic = {'pu':'Pileup', 'l1prefiring': 'L1 Prefiring', 'q2': r'Q$^2$ Scale', 'pdf': 'PDF', 'herwig': 'Model Unc.'}
                if up_key in syst_fraction_dict:
                    hep.histplot(syst_fraction_dict[up_key][1:], self.bins.gen_mass_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Up", color=color, ls='-')
                # Plot Down uncertainty (dashed)
                if down_key in syst_fraction_dict:
                    hep.histplot(-syst_fraction_dict[down_key][1:], self.bins.gen_mass_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Down", color=color, ls='--')


                
            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"
            
            # if ylim is not None:
            #     plt.ylim(ylim)
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize = 15)#loc='center left', bbox_to_anchor=(1, 0.5))
            hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)

            if self.groomed:
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
                save_path = f'./outputs/mass/uncertainties/{syst_names[0]}_groomed_{i-1}.pdf'
            else:
                plt.xlim(*self._observable_xlim(i))
                plt.xlabel(self._observable_label())
                save_path = f'./outputs/mass/uncertainties/{syst_names[0]}_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)

    def plot_herwig_systematic(self, show=True):
        flat_uncertainty = np.sqrt(np.diag(self.cov_data_herwig_np))/np.abs(self.y_unf_dict['herwigUp'])
        uncertainty_pt_binned = unflatten_gen_by_pt(flat_uncertainty, self.bins.gen_mass_edges_by_pt)
        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.bins.gen_mass_edges_by_pt)
        
        for i, result in enumerate(self.normalized_results):
            syst_fraction_dict = result.get('syst_fraction_dict', {})
            error_in_syst = uncertainty_pt_binned[i]*syst_fraction_dict['herwigUp']  # Uncertainty on relative uncertainty
            pt_bin = result['pt_bin']
            if 'herwigUp' in syst_fraction_dict:
                hep.histplot(syst_fraction_dict['herwigUp'], self.bins.gen_mass_edges_by_pt[i], yerr = error_in_syst, label=f"Model Unc.", color='#964a8b', ls='-')


            # Fit a polynomial to the herwigUp systematic fraction
            if 'herwigUp' in syst_fraction_dict:
                edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)
                centers = 0.5 * (edges[:-1] + edges[1:])
                #centers[0] = -100000000  # Set the first center to a very large negative value to exclude it from the fit
                y = syst_fraction_dict['herwigUp']
                mask = np.isfinite(y) & (y > 0)
                if mask.sum() > 3:
                    degree = 2
                    coeffs = np.polyfit(centers[mask], y[mask], degree, w=1.0/np.where(error_in_syst[mask] > 0, error_in_syst[mask], 1e-10))
                    poly = np.poly1d(coeffs)
                    x_fit = np.linspace(centers[mask][1], centers[mask][-1], 200)
                    plt.plot(x_fit, poly(x_fit), color='#5790fc', ls='--', lw=2, label=f"Poly fit (deg {degree})")

            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{pt_bin[0]}–∞"
            else:
                pt_bin_label = f"{pt_bin[0]}–{pt_bin[1]}"
            plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
            hep.cms.label(self.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)
            plt.ylim(0,0.5)
            plt.xlim(*self._observable_xlim(i))
            plt.xlabel(self._observable_label())
            plt.ylabel("Relative Uncertainty")
            save_path = f'./outputs/mass/uncertainties/herwig_groomed_{i-1}.pdf' if self.groomed else f'./outputs/mass/uncertainties/herwig_ungroomed_{i-1}.pdf'
            self._finalize_plot(save_path=save_path, show=show)


            
    def plot_purity_stability(self, show=True):
        hep.style.use("CMS")
        #ignore first slice in sum bc it's underflow
        len_underflow = len(self.bins.gen_mass_edges_by_pt[0]) - 1

        diagonal = np.diag(self.mosaic_gen)
        purity_denom = self.mosaic_gen[len_underflow:, :].sum(axis=0)
        stability_denom = self.mosaic_gen[:, len_underflow:].sum(axis=1)

        purity = np.divide(
            diagonal,
            purity_denom,
            out=np.zeros_like(diagonal, dtype=float),
            where=purity_denom != 0,
        )
        stability = np.divide(
            diagonal,
            stability_denom,
            out=np.zeros_like(diagonal, dtype=float),
            where=stability_denom != 0,
        )

        purity_unflattened = unflatten_gen_by_pt(purity, self.bins.gen_mass_edges_by_pt)
        stability_unflattened = unflatten_gen_by_pt(stability, self.bins.gen_mass_edges_by_pt)

        title_list = [
            "",
            r"200 $<$ $p_T$ $<$ 290 GeV",
            r"290 $<$ $p_T$ $<$ 400 GeV",
            r"400 $<$ $p_T$ $< \, \infty$ GeV",
        ]

        for i in range(len(self.pt_edges) - 1):
            fig, ax = plt.subplots(figsize=(12, 8))
            hep.histplot(
                purity_unflattened[i],
                self.bins.gen_mass_edges_by_pt[i],
                label="Purity",
                ax=ax,
            )
            hep.histplot(
                stability_unflattened[i],
                self.bins.gen_mass_edges_by_pt[i],
                label="Stability",
                ax=ax,
            )
            ax.axhline(0.5, color='k', linestyle='--', linewidth=1)
            ax.set_ylabel("Purity/Stability")
            ax.set_xlabel(self._observable_short_label())
            ax.set_xlim(*self._observable_xlim(i))
            ax.set_ylim(0.0, 1.05)
            ax.legend(title=title_list[i])
            plt.sca(ax)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20)

            save_path = (
                f'./outputs/mass/unfold/purity_stability_groomed_{i-1}.pdf'
                if self.groomed
                else f'./outputs/mass/unfold/purity_stability_ungroomed_{i-1}.pdf'
            )
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.stairs(purity, np.arange(len(purity) + 1), label="Purity")
        ax.stairs(stability, np.arange(len(stability) + 1), label="Stability")
        ax.set_xlabel("Global Bin Number")
        ax.set_ylabel("Purity/Stability")
        ax.set_ylim(0.0, 1.05)
        plt.sca(ax)
        hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=20)
        ax.legend()
        self._finalize_plot(
            save_path=f'./outputs/mass/unfold/purity_stability_global_{"groomed" if self.groomed else "ungroomed"}.pdf',
            show=show,
            fig=fig,
        )

    def plot_purity_stability_herwig(self, show=True):
        """Overlay Pythia8 vs Herwig7 purity & stability to diagnose generator dependence."""
        hep.style.use("CMS")
        suffix = "groomed" if self.groomed else "ungroomed"
        len_underflow = len(self.bins.gen_mass_edges_by_pt[0]) - 1

        def _purity_stability(mosaic):
            diagonal = np.diag(mosaic)
            purity_denom = mosaic[len_underflow:, :].sum(axis=0)
            stability_denom = mosaic[:, len_underflow:].sum(axis=1)
            purity = np.divide(diagonal, purity_denom,
                               out=np.zeros_like(diagonal, dtype=float), where=purity_denom != 0)
            stability = np.divide(diagonal, stability_denom,
                                  out=np.zeros_like(diagonal, dtype=float), where=stability_denom != 0)
            return (
                unflatten_gen_by_pt(purity, self.bins.gen_mass_edges_by_pt),
                unflatten_gen_by_pt(stability, self.bins.gen_mass_edges_by_pt),
            )

        purity_py, stability_py = _purity_stability(self.mosaic_gen)
        purity_hw, stability_hw = _purity_stability(self.mosaic_gen_herwig)

        title_list = [
            "",
            r"200 $<$ $p_T$ $<$ 290 GeV",
            r"290 $<$ $p_T$ $<$ 400 GeV",
            r"400 $<$ $p_T$ $< \, \infty$ GeV",
        ]

        for i in range(len(self.pt_edges) - 1):
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            fig.subplots_adjust(wspace=0.05)

            for ax, (pur_py, stab_py, pur_hw, stab_hw, metric) in zip(
                axes,
                [
                    (purity_py[i], None, purity_hw[i], None, "Purity"),
                    (None, stability_py[i], None, stability_hw[i], "Stability"),
                ],
            ):
                vals_py = pur_py if metric == "Purity" else stab_py
                vals_hw = pur_hw if metric == "Purity" else stab_hw
                edges = self.bins.gen_mass_edges_by_pt[i]
                hep.histplot(vals_py, edges, label="Pythia8", ax=ax, color="steelblue")
                hep.histplot(vals_hw, edges, label="Herwig7", ax=ax, color="darkorange", linestyle="--")
                ax.axhline(0.5, color="k", linestyle=":", linewidth=1.2, label="0.5 threshold")
                ax.set_xlabel(self._observable_short_label(), fontsize=14)
                ax.set_xlim(*self._observable_xlim(i))
                ax.set_ylim(0.0, 1.05)
                ax.set_title(metric, fontsize=14)
                ax.legend(title=title_list[i], fontsize=12)

            axes[0].set_ylabel("Purity / Stability", fontsize=14)
            hep.cms.label(self.cms_label, data=False, lumi=138, com=13, fontsize=18, ax=axes[0])

            save_path = f"./outputs/mass/unfold/purity_stability_herwig_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)


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
        ncols_by_gp = [len(e)-1 for e in self.bins.gen_mass_edges_by_pt]
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
        save_path = f'outputs/mass/unfold/correlation_groomed.pdf' if self.groomed else f'outputs/mass/unfold/correlation_ungroomed.pdf'
        self._finalize_plot(save_path=save_path, show=show)
    def plot_response_matrix(self, probability=True, log=False, show=True):
        fig, ax = self._plot_response_mosaic_cms(
            self.mosaic,
            reco_mass_edges_by_pt=self.bins.reco_mass_edges_by_pt,
            gen_mass_edges_by_pt=self.bins.gen_mass_edges_by_pt,
            reco_pt_edges=self.pt_edges,
            gen_pt_edges=self.pt_edges,
            probability = probability,
            mask_zeros=True,
            log=log,                              # set False for linear
            rlabel=f"Groomed, " if self.groomed else f"Ungroomed, ",
        )
        self._finalize_plot(show=show, fig=fig)

    def plot_uncertainty_heatmap(self, show=True):
        """
        2D heatmap of fractional uncertainties: rows = systematic groups,
        columns = mass bins, color = uncertainty magnitude in %.
        Gives an at-a-glance budget matrix showing which source dominates
        in which part of the spectrum. One figure per pT bin.
        """
        group_order = [
            "Jet Energy",
            "Jet Mass",
            "Parton Shower",
            "Lepton SFs",
            "Other Theory",
            "Model Uncertainty",
            "Stat Unc",
            "Total",
        ]

        for i, result in enumerate(self.normalized_results):
            syst_fraction_dict = self._build_syst_fraction_dict(i)
            grouped_dict = self._group_syst_fraction_dict(syst_fraction_dict, grouped=True)

            # Compute per-group envelope: max(Up, Down) for each group
            envelopes = {}
            seen_bases = set()
            for key in grouped_dict:
                if key in {"Stat Unc", "Total_Up", "Total_Down"}:
                    continue
                base, variation = self._split_systematic_variation(key)
                if base not in seen_bases:
                    seen_bases.add(base)
                    up = grouped_dict.get(f"{base}Up", np.zeros_like(grouped_dict[key]))
                    down = grouped_dict.get(f"{base}Down", np.zeros_like(grouped_dict[key]))
                    envelopes[base] = np.maximum(up, down)

            envelopes["Stat Unc"] = grouped_dict["Stat Unc"]
            envelopes["Total"] = np.maximum(
                grouped_dict.get("Total_Up", np.zeros(1)),
                grouped_dict.get("Total_Down", np.zeros(1)),
            )

            # Build matrix in the predefined group order, skipping absent groups
            rows = [g for g in group_order if g in envelopes]
            matrix = np.array([envelopes[row] for row in rows]) * 100.0  # → percent

            edges = np.array(self.bins.gen_mass_edges_by_pt[i], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])

            # Trim columns below the observable lower limit:
            # ungroomed starts at 20 GeV (20-30 bin), groomed at 10 GeV (10-20 bin)
            x_lo, _ = self._observable_xlim(i)
            start_col = int(np.searchsorted(edges[:-1], x_lo, side='left'))
            centers = centers[start_col:]
            matrix = matrix[:, start_col:]

            n_rows, n_cols = matrix.shape

            fig_w = max(10, n_cols * 0.9 + 3)
            fig_h = n_rows * 0.8 + 2.5
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            vmax = 50.0

            im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                           vmin=0.0, vmax=vmax, origin='upper')

            # Annotate every cell with its numeric value
            for r in range(n_rows):
                for c in range(n_cols):
                    val = matrix[r, c]
                    text_color = 'white' if val > vmax * 0.65 else 'black'
                    ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                            fontsize=8, color=text_color, fontweight='bold')

            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels([f"{c:.0f}" for c in centers],
                               rotation=45, ha='right', fontsize=9)
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(rows, fontsize=15)

            # Visual separators: dashed blue before stat unc, solid black before total
            if "Stat Unc" in rows:
                ax.axhline(rows.index("Stat Unc") - 0.5, color='steelblue', lw=1.5, ls='--')
            if "Total" in rows:
                ax.axhline(rows.index("Total") - 0.5, color='black', lw=2.0)

            # Aligned colorbar: make_axes_locatable ensures it matches the heatmap height exactly
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.15)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Fractional Uncertainty (%)", fontsize=11)

            pt_bin = result['pt_bin']
            if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
                pt_bin_label = f"{int(pt_bin[0])}–∞ GeV"
            else:
                pt_bin_label = f"{int(pt_bin[0])}–{int(pt_bin[1])} GeV"

            ax.set_xlabel(self._observable_label(), fontsize=12)
            ax.set_title(rf"Uncertainty budget  |  $p_T$: {pt_bin_label}", fontsize=13, pad=8)
            hep.cms.label(self._cms_extra_label(), data=True, lumi=138, com=13,
                          fontsize=16, ax=ax)

            plt.tight_layout()
            suffix = "groomed" if self.groomed else "ungroomed"
            save_path = f"./outputs/mass/uncertainties/heatmap_{suffix}_{i-1}.pdf"
            self._finalize_plot(save_path=save_path, show=show, fig=fig)

    def run_all_plots(self, show=False):
        self.plot_unfolded_fancy(show=show)
        self.plot_unfolded_summary_linear(show=show)
        self.plot_statistical_fraction(show=show)
        self.plot_systematic_fraction(show=show)
        self.plot_systematic_fraction_grouped(show=show)
        self.plot_systematic_fraction_grouped(show=show, log=False)
        self.plot_herwig_systematic(show=show)
        self.plot_systematic_frac_indiv(["JES", "JER"], show=show)
        self.plot_systematic_frac_indiv(["JMS", "JMR"], show=show)
        self.plot_systematic_frac_indiv(["q2", "pdf", "pu", "l1prefiring"], show=show)
        self.plot_systematic_frac_indiv(["herwig"], show=show)
        self.plot_systematic_frac_indiv(["ElectronSF", "MuonSF"], show=show)
        self.plot_systematic_frac_indiv(["isr", "fsr"], show=show)
        self.plot_correlation(show=show)
        self.plot_uncertainty_heatmap(show=show)
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
        #print("Singular values of the response mosaic:", singular_values)

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
            plt.savefig("outputs/mass/unfold/response_groomed.pdf")
        else:
            plt.savefig("outputs/mass/unfold/response_ungroomed.pdf")
        return fig, ax
    
    def _make_inputs_numpy(self,
        filenames = [
            "./inputs/massInputs/mass_pythia_2016.pkl",
            "./inputs/massInputs/mass_pythia_2016APV.pkl",
            "./inputs/massInputs/mass_pythia_2017.pkl",
            "./inputs/massInputs/mass_pythia_2018.pkl",
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
            era = Path(filename).stem.replace("mass_pythia_", "", 1)
            print(f"Processing era: {era}")
            key = keymap[era]
            response_dict.setdefault('u', {})
            response_dict.setdefault('g', {})

            
            with open(filename, "rb") as f:
                data = pkl.load(f)
                # ensure top-level 'u' and 'g' keys exist
                # debug: check what keys are in the loaded data
                print(f"keys in data for {era}: {list(data.keys())}")
                if era == "2016APV" or era == "2016":
                    response_dict['u'][key] = data['response_matrix_u'].project('ptreco','dataset', 'ptgen', 'mgen',  'mreco', 'systematic')
                    response_dict['g'][key] = data['response_matrix_g'].project('ptreco','dataset', 'ptgen', 'mgen',  'mreco', 'systematic')
                    continue
                h_old = data['response_matrix_u'].project('ptreco','dataset', 'ptgen', 'mgen',  'mreco', 'systematic')
                #h_new = group(h_old, oldname="dataset", newname="dataset", grouping=dict(grouping))
                response_dict['u'][key] = h_old

                h_old = data['response_matrix_g'].project('ptreco','dataset', 'ptgen', 'mgen',  'mreco', 'systematic')
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
            m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values() +\
                response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()

            variance = np.array([response[era][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').variances() for era in ['pythia_UL16NanoAODv9', 'pythia_UL16NanoAODAPVv9', 'pythia_UL17NanoAODv9', 'pythia_UL18NanoAODv9']]).sum(axis = 0)
            
            m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values() + \
                response['pythia_UL16NanoAODAPVv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            
            
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
            sys_matrix_dic_up[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()

            non_jes_sys_matrix_dic_up[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()



        sys_matrix_dic_down = {}
        non_jes_sys_matrix_dic_down = {}
        for sys in jes_sys_list_down:
            m_nom_2016 = response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values() + \
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_nom_2017 = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_nom_2018 = response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            
            m_sys_2016 = response['pythia_UL16NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values() + \
                        response['pythia_UL16NanoAODAPVv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_sys_2017 = response['pythia_UL17NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            m_sys_2018 = response['pythia_UL18NanoAODv9'][..., sys].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
            
            
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
            sys_matrix_dic_down[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()
        #                            + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values
            non_jes_sys_matrix_dic_down[sys] = response['pythia_UL16NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL16NanoAODAPVv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL17NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()\
                                    + response['pythia_UL18NanoAODv9'][{'systematic':sys}].project('ptgen','mgen','ptreco','mreco').values()

        if not groomed:
            response = response_dict['u']
        else:
            response = response_dict['g']

        sys_matrix_dic_up['nominal'] = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values() +\
                response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()  +\
                    response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values() +\
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()
        sys_matrix_dic_down['nominal'] = response['pythia_UL17NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values() +\
                response['pythia_UL18NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()  +\
                    response['pythia_UL16NanoAODv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values() +\
                        response['pythia_UL16NanoAODAPVv9'][..., 'nominal'].project('ptgen', 'mgen', 'ptreco', 'mreco').values()

        prepend = "./inputs/massInputs/"
        filename_herwig_1 = self._resolve_input_path(
            prepend + "mass_pythia_reweighted_all.pkl",
            prepend + "pythia_reweighted_all.pkl",
            prepend + "mass_herwig_all.pkl",
            prepend + "herwig_all.pkl",

        )
        with open(filename_herwig_1, "rb") as f:
            data_herwig = pkl.load(f)
            if not groomed:
                sys_matrix_dic_up['herwigUp'] = data_herwig['response_matrix_u'].project('ptgen','mgen','ptreco','mreco').values()
                sys_matrix_dic_down['herwigDown'] = data_herwig['response_matrix_u'].project('ptgen','mgen','ptreco','mreco').values()
            else:
                sys_matrix_dic_up['herwigUp'] = data_herwig['response_matrix_g'].project('ptgen','mgen','ptreco','mreco').values()
                sys_matrix_dic_down['herwigDown'] = data_herwig['response_matrix_g'].project('ptgen','mgen','ptreco','mreco').values()

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
    def _merge_eras(self, filenames = ["./inputs/massInputs/mass_pythia_2016.pkl","./inputs/massInputs/mass_pythia_2016APV.pkl","./inputs/massInputs/mass_pythia_2017.pkl","./inputs/massInputs/mass_pythia_2018.pkl",]):


        outputs = []
        for fname in filenames:
            with open(fname, "rb") as f:
                outputs.append(pkl.load(f))


        hist_keys = ['ptjet_mjet_u_reco', 'ptjet_mjet_g_reco', 'response_matrix_u', 'response_matrix_g', 'ptjet_mjet_u_gen', 'ptjet_mjet_g_gen',]
        out_dict = {}
        for i, output in enumerate(outputs):
            for key in hist_keys:
                if key in ['ptjet_mjet_u_reco', 'ptjet_mjet_g_reco']:

                    if i == 0:
                        out_dict[key] = output[key].project('ptreco', 'mreco', 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptreco', 'mreco', 'systematic')
                    #print(f"Processed {key} for file {i}, with sum {output[key].project('ptreco', 'mreco', 'systematic').sum().value}")
                    #print("Current total sum:", out_dict[key].sum().value)
                elif key in ['response_matrix_u', 'response_matrix_g']:
                    if i == 0:
                        out_dict[key] = output[key].project('ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
                    #print(f"Processed {key} for file {i}")
                elif key in ['ptjet_mjet_u_gen', 'ptjet_mjet_g_gen']:
                    if i == 0:
                        out_dict[key] = output[key].project('ptgen', 'mgen', 'systematic')
                    else:
                        out_dict[key] += output[key].project('ptgen', 'mgen', 'systematic')
                    #print(f"Processed {key} for file {i}")
            self.pythia_hists = out_dict
        out_filename = "./inputs/massInputs/mass_pythia_all.pkl"
        with open(out_filename, "wb") as f:
            pkl.dump(out_dict, f)
    def _merge_eras_jk(
        self,
        filenames = [
            "./inputs/massInputs/mass_jk_pythia_2016.pkl",
            "./inputs/massInputs/mass_jk_pythia_2016APV.pkl",
            "./inputs/massInputs/mass_jk_pythia_2017.pkl",
            "./inputs/massInputs/mass_jk_pythia_2018.pkl",
        ] 
        ):
        outputs = []
        for fname in filenames:
            with open(fname, "rb") as f:
                outputs.append(pkl.load(f))
        hist_keys = [ 'response_matrix_u', 'response_matrix_g']
        out_dict = {}
        for i, output in enumerate(outputs):
            for key in hist_keys:
                if key in ['response_matrix_u', 'response_matrix_g']:
                    if i == 0:
                        out_dict[key] = output[key].project('jk','ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
                    else:
                        out_dict[key] += output[key].project('jk','ptgen', 'mgen', 'ptreco', 'mreco', 'systematic')
        self.pythia_hists_jk = out_dict
        return out_dict





# sys_matrix_dic['herwigUp'] = resp_matrix_4d_herwig.project('ptgen','mgen','ptreco','mreco').values()
# sys_matrix_dic_down['herwigDown'] = resp_matrix_4d_herwig.project('ptgen','mgen','ptreco','mreco').values()
