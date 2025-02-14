import ROOT
import uproot
import numpy as np
import array as array
import math
import matplotlib.pyplot as plt
import pickle as pkl
import hist
import uproot
import tempfile
import mplhep as hep
import matplotlib.gridspec as gridspec
import mplhep as hep
hep.style.use("CMS")
#import statistics as st
ROOT.gStyle.SetOptStat(000000)
ROOT.gStyle.SetPalette(ROOT.kViridis)

from unfold_utils import binning
import matplotlib.colors as mcolors

import tempfile
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from unfold_utils.draw_tools import draw_colz_histogram
from unfold_utils.draw_tools import *




import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import hist
import mplhep as hep
hep.style.use("CMS")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kViridis)

from unfold_utils import binning
from unfold_utils.draw_tools import draw_colz_histogram

class Unfolder:
    def __init__(self, data_2d, resp_matrix_4d, fakes, misses,
                 systematics=None, backgrounds=None, closure=False, groomed=False,
                 is_uf=True, merge=False, normalised_xs=True, do_syst=True,
                 discard_low_bins=False, do_norm = False, regularisation="None"):
        # Store inputs
        self.data_2d = data_2d
        self.resp_matrix_4d = resp_matrix_4d
        self.fakes = fakes
        self.misses = misses
        self.systematics = systematics
        self.backgrounds = backgrounds
        self.closure = closure
        self.groomed = groomed
        self.is_uf = is_uf
        self.merge = merge
        self.normalised_xs = normalised_xs
        self.do_syst = do_syst
        self.discard_low_bins = discard_low_bins
        self.regularisation = regularisation
        self.do_norm = do_norm

        # Results containers
        self.output_pt_binned = None
        self.stat_pt_binned = None
        self.stat_mat_pt_binned = None
        self.cov_np = None
        self.tau = None

        # Set up binning and prepare response matrix
        self._setup_binning()
        self._prepare_response_matrix()

    def _setup_binning(self):
        """Extract bin edges and create the binning object."""
        self.nmbinsDet = len(self.resp_matrix_4d.axes['mreco'].centers)
        self.ptreco_center = self.resp_matrix_4d.axes['ptreco'].centers
        self.ptreco_width = self.resp_matrix_4d.axes['ptreco'].widths
        self.ptreco_edge = self.resp_matrix_4d.axes['ptreco'].edges

        self.ptgen_center = self.resp_matrix_4d.axes['ptgen'].centers
        self.ptgen_width = self.resp_matrix_4d.axes['ptgen'].widths
        self.ptgen_edge = self.resp_matrix_4d.axes['ptgen'].edges

        self.mreco_center = self.resp_matrix_4d.axes['mreco'].centers
        self.mreco_edge = self.resp_matrix_4d.axes['mreco'].edges
        self.mgen_center = self.resp_matrix_4d.axes['mgen'].centers
        self.mreco_width = self.resp_matrix_4d.axes['mreco'].widths
        self.mgen_width = self.resp_matrix_4d.axes['mgen'].widths
        self.mgen_edge = self.resp_matrix_4d.axes['mgen'].edges

        # Create the binning object using the provided bin edges
        self.bins = binning.binning(mbinsGen=self.mgen_edge, mbinsDet=self.mreco_edge,
                                    ptbinsGen=self.ptgen_edge, ptbinsDet=self.ptreco_edge)
        self.mbinsGen  = self.bins.mbinsGen
        self.mbinsDet  = self.bins.mbinsDet
        self.ptbinsGen = self.bins.ptbinsGen
        self.ptbinsDet = self.bins.ptbinsDet

        self.nmbinsGen  = self.bins.nmbinsGen
        self.nmbinsDet  = self.bins.nmbinsDet
        self.nptbinsGen = self.bins.nptbinsGen
        self.nptbinsDet = self.bins.nptbinsDet

    def _prepare_response_matrix(self):
        """Project the 4D response matrix to a 2D numpy array and prepare related quantities."""
        proj = self.resp_matrix_4d.project('ptgen', 'mgen', 'ptreco', 'mreco')
        self.M_np = proj.values(flow=False)
        self.M_np = self.M_np.reshape(self.M_np.shape[0]*self.M_np.shape[1],
                                        self.M_np.shape[2]*self.M_np.shape[3])

        # Underflow bins in pT gen (and discarding under/overflow in reco)
        h_np_underflow = self.resp_matrix_4d.project('ptgen', 'mgen', 'ptreco', 'mreco') \
                           .values(flow=True)[0, :, 1:-1, 1:-1].sum(axis=0)
        h_np_underflow = h_np_underflow.reshape(h_np_underflow.shape[0]*h_np_underflow.shape[1])
        self.underflow_frac = np.nan_to_num(h_np_underflow / self.M_np.sum(axis=0))

        # Errors on migration matrix bins
        M_np_error = proj.variances(flow=False)
        self.M_np_error = M_np_error.reshape(M_np_error.shape[0]*M_np_error.shape[1],
                                             M_np_error.shape[2]*M_np_error.shape[3])**0.5

        # True (generator-level) distribution
        self.htrue_np = self.M_np.sum(axis=1)

        # If systematics are provided, reshape them to match
        if self.systematics is not None:
            for sys_name in self.systematics:
                self.systematics[sys_name] = self.systematics[sys_name].reshape(
                    self.nptbinsGen*self.nmbinsGen, self.nptbinsDet*self.nmbinsDet)

        
            

        # Prepare the data histogram based on whether this is a closure test
        if self.closure:
            self.h_np = self.M_np.sum(axis=0)
            self.h_np_error = self.h_np**0.5
        else:
            reco_proj = self.data_2d.project('ptreco', 'mreco')
            self.h_np = reco_proj.values(flow=False)
            self.h_np_error = np.sqrt(self.h_np)
            self.h_np = self.h_np.reshape(self.h_np.shape[0]*self.h_np.shape[1])
            self.h_np_error = self.h_np_error.reshape(self.h_np_error.shape[0]*self.h_np_error.shape[1])
            # Account for underflow
            self.h_np = self.h_np *(1 - self.underflow_frac)
            # Correct for negatives
            self.h_np = np.where(self.h_np< 0, 0, self.h_np)

        # Miss and fake corrections
        self.miss_values = self.misses.project('ptgen', 'mgen').values().reshape(self.M_np.shape[0])
        self.fake_values = self.fakes.project('ptreco', 'mreco').values().reshape(self.M_np.shape[1])

        self.miss_frac = self.miss_values/ ( self.M_np.sum(axis = 1))
        
        if self.do_norm:
            for i in range(4):
                sum_i = self.h_np[i*self.nmbinsDet:(i+1)*self.nmbinsDet].sum()
                self.h_np[i*self.nmbinsDet:(i+1)*self.nmbinsDet] = self.h_np[i*self.nmbinsDet:(i+1)*self.nmbinsDet]/sum_i
                self.h_np_error[i*self.nmbinsDet:(i+1)*self.nmbinsDet] = self.h_np_error[i*self.nmbinsDet:(i+1)*self.nmbinsDet]/sum_i


                sum_i = self.M_np.sum(axis = 0)[i*self.nmbinsDet:(i+1)*self.nmbinsDet].sum()
                if (not self.closure) & (self.backgrounds!= None):
                    for bg in self.backgrounds:
                        self.backgrounds[bg][i*self.nmbinsDet:(i+1)*self.nmbinsDet] = self.backgrounds[bg][i*self.nmbinsDet:(i+1)*self.nmbinsDet]/sum_i

                self.M_np[:, i*self.nmbinsDet:(i+1)*self.nmbinsDet ] = self.M_np[:, i*self.nmbinsDet:(i+1)*self.nmbinsDet ]/sum_i

                for sys in self.systematics:
                    sum_i = self.systematics[sys].sum(axis = 0)[i*self.nmbinsDet:(i+1)*self.nmbinsDet].sum()
                    self.systematics[sys][:, i*self.nmbinsDet:(i+1)*self.nmbinsDet ]  = self.systematics[sys][:, i*self.nmbinsDet:(i+1)*self.nmbinsDet ]/sum_i
                            
                self.fake_values[ i*self.nmbinsDet:(i+1)*self.nmbinsDet ] /= (sum_i + self.fake_values[ i*self.nmbinsDet:(i+1)*self.nmbinsDet ])

        
            


        


            

    def create_root_objects(self):
        """Create and fill the ROOT histograms and migration matrices."""
        self.M = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(
            self.bins.genDist, self.bins.detDist, "M")
        self.h = self.bins.detDist.CreateHistogram("h")
        self.h_reco = self.bins.detDist.CreateHistogram("h_reco")
        self.htrue = self.bins.genDist.CreateHistogram("htrue")
        self.htruef = self.bins.genDist.CreateHistogram("htruef")

        # Fill the migration matrix
        for i in range(1, self.M.GetNbinsX()+1):
            for j in range(1, self.M.GetNbinsY()+1):
                self.M.SetBinContent(i, j, self.M_np[i-1][j-1])
                self.M.SetBinError(i, j, self.M_np_error[i-1][j-1])

        # If systematics exist, fill the additional matrices
        self.M_sys_dic = {}
        if self.systematics is not None:
            for sys_name in self.systematics:
                self.M_sys_dic[sys_name] = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(
                    self.bins.genDist, self.bins.detDist, "M"+sys_name)
                for i in range(1, self.M.GetNbinsX()+1):
                    for j in range(1, self.M.GetNbinsY()+1):
                        self.M_sys_dic[sys_name].SetBinContent(i, j,
                            self.systematics[sys_name][i-1][j-1])
                        self.M_sys_dic[sys_name].SetBinError(i, j,
                            self.M_np_error[i-1][j-1])

        # Fill the data and true histograms
        for i in range(1, self.h.GetNbinsX()+1):
            self.h.SetBinContent(i, self.h_np[i-1])
            self.h.SetBinError(i, self.h_np_error[i-1])
            self.h_reco.SetBinContent(i, self.M_np.sum(axis=0)[i-1])
        for i in range(1, self.htrue.GetNbinsX()+1):
            self.htrue.SetBinContent(i, self.htrue_np[i-1])

        # Create and fill the fake histogram
        self.fake_hist = self.h.Clone('fakes')
        self.fake_hist.Reset()
        if not self.closure:
            for i in range(1, self.M.GetNbinsX()):
                self.M.SetBinContent(i, 0, self.miss_values[i-1])
                if self.systematics is not None:
                    for sys_name in self.M_sys_dic:
                        self.M_sys_dic[sys_name].SetBinContent(i, 0, self.miss_values[i-1])
        for i in range(1, self.M.GetNbinsY()):
            self.fake_hist.SetBinContent(i, self.fake_values[i-1])



        orientation = ROOT.TUnfold.kHistMapOutputHoriz
        regMode = ROOT.TUnfold.kRegModeCurvature
        con = ROOT.TUnfold.kEConstraintArea
        mode = ROOT.TUnfoldDensity.kDensityModeBinWidth
        axisSteering = "*[UOB]"
        nScan = 50
        tauMin = 1e-8
        tauMax = 0.01
        logTauX = ROOT.MakeNullPointer(ROOT.TSpline)
        logTauY = ROOT.MakeNullPointer(ROOT.TSpline)
        lCurve = ROOT.MakeNullPointer(ROOT.TGraph)

        # Initialize the TUnfoldDensity object
        self.u = ROOT.TUnfoldDensity(self.M, orientation, regMode, con, mode,
                                     self.bins.genBin, self.bins.detBin,
                                     "signal", axisSteering)
        if self.systematics is not None:
            for sys in self.M_sys_dic:
                self.u.AddSysError(self.M_sys_dic[sys], sys, orientation,
                                   ROOT.TUnfoldDensity.kSysErrModeMatrix)
        self.u.SetInput(self.h)

        print("Working until background subtractions")
        if not self.closure:
            self.u.SubtractBackground(self.fake_hist, 'fakes')

        if not self.closure and self.backgrounds is not None:
            for bg in self.backgrounds.keys():
                bg_hist = self.h.Clone('bg_'+bg)
                bg_hist.Reset()
                for i in range(1, self.M.GetNbinsY()):
                    bg_hist.SetBinContent(i, self.backgrounds[bg][i-1])
                self.u.SubtractBackground(bg_hist, bg)


        print("Unfolder object created")

    def plot_response_matrix(self, probability_matrix  = True):
        """Plot the response matrix with overlayed bin boundaries."""
        # Prepare lines for visual separation of pT bins
        self.create_root_objects()
        th2f = self.u.GetProbabilityMatrix("Prob")

        with tempfile.NamedTemporaryFile(suffix=".root") as temp_file:
            # Create a new ROOT file and write the histogram to it
            root_file = ROOT.TFile(temp_file.name, "RECREATE")
            th2f.Write()
            root_file.Close()
    
            # Open the temporary file with uproot
            with uproot.open(temp_file.name) as file:
                # Extract the histogram
                hist = file[th2f.GetName()]
    
                # Get the bin contents as a numpy array
                hist_array = hist.values()
        condition_number = np.linalg.cond(hist_array)
        print("Condition number of the matrix ", condition_number)
        lines_x = []
        lines_y = []
        for pt in self.ptgen_edge[1:-1]:
            lines_x.append(self.bins.genDist.GetGlobalBinNumber(1, pt)-0.5)
        for pt in self.ptreco_edge[1:-1]:
            lines_y.append(self.bins.detDist.GetGlobalBinNumber(1, pt)-0.5)

        label_lines_x = []
        label_lines_y = []
        for pt in self.ptgen_edge[:-1]:
            label_lines_x.append(self.bins.genDist.GetGlobalBinNumber(70, pt))
        for pt in self.ptreco_edge[:-1]:
            label_lines_y.append(self.bins.detDist.GetGlobalBinNumber(70, pt))

        x_labels = np.array([str(int(x)) for x in self.ptgen_edge])
        if probability_matrix:
            ax = draw_colz_histogram(self.u.GetProbabilityMatrix("Prob"), use_log_scale=False)
        else:
            ax = draw_colz_histogram(self.M, use_log_scale=True)


       
            
        ax.set_title(" ")
        for x in lines_x:
            ax.axvline(x=x, color='black', linestyle='--')
        for y in lines_y:
            ax.axhline(y=y, color='black', linestyle='--')
        ax.plot([0.5, self.M.GetNbinsX()+0.5],
                [0.5, self.M.GetNbinsY()+0.5],
                color='red', linestyle='--', linewidth=1)
        ax.set_xticks(label_lines_x)
        ax.set_xticklabels(x_labels[:-1])
        ax.set_yticks(label_lines_y)
        ax.set_yticklabels(x_labels[:-1])
        ax.tick_params(axis='both', which='both', length=0)
        hep.cms.label("Preliminary", rlabel = f"Cond. number = {condition_number:.2f} ", fontsize = 20)
        ax.set_xlabel(r"GEN p$_{T}$ (GeV)")
        ax.set_ylabel(r"RECO $p_T$ (GeV)")

        self.matrix_fig = plt.gcf()
        plt.show()

    def perform_unfold(self, regularisation = None):

        if regularisation!= None:
            self.regularisation = regularisation
       
        
        if isinstance(regularisation, float):
            tau = regularisation
            self.regularisation = "Custom"
        """Run the unfolding procedure using TUnfoldDensity."""
        # Create ROOT objects and fill histograms/matrices
        self.create_root_objects()

        orientation = ROOT.TUnfold.kHistMapOutputHoriz
        regMode = ROOT.TUnfold.kRegModeCurvature
        con = ROOT.TUnfold.kEConstraintArea
        mode = ROOT.TUnfoldDensity.kDensityModeBinWidth
        axisSteering = "*[UOB]"
        nScan = 50
        tauMin = 1e-8
        tauMax = 0.01
        logTauX = ROOT.MakeNullPointer(ROOT.TSpline)
        logTauY = ROOT.MakeNullPointer(ROOT.TSpline)
        lCurve = ROOT.MakeNullPointer(ROOT.TGraph)
        
        # Regularisation choices
        if self.regularisation == "Custom":
            self.u.DoUnfold(tau)
        elif self.regularisation == "None":
            self.u.DoUnfold(0.000)
        elif self.regularisation == "ScanLcurve":
            i_best = self.u.ScanLcurve(nScan, tauMin, tauMax, lCurve, logTauX, logTauY)
        elif self.regularisation == "ScanTau":
            self.u.ScanTau(nScan, tauMin, tauMax, logTauX, ROOT.TUnfoldDensity.kEScanTauRhoAvg, 'signal')
        elif self.regularisation == "ScanSURE":
            logSURE = ROOT.MakeNullPointer(ROOT.TGraph)
            chi2 = ROOT.MakeNullPointer(ROOT.TGraph)
            lCurve = ROOT.MakeNullPointer(ROOT.TGraph)
            self.u.ScanSURE(nScan, tauMin, tauMax, logSURE, chi2, lCurve)
        else:
            print("Specify correct regularisation")
            return

        print("Regularisation USED", self.regularisation)
        print("Tau value", self.u.GetTau())
        self.tau = self.u.GetTau()

        # Get covariance matrices
        self.cov = self.u.GetEmatrixTotal("cov", "Covariance Matrix")
        self.cov_uncorr = self.u.GetEmatrixSysUncorr("cov_uncorr",
                                                     "Covariance Matrix from Uncorrelated Uncertainties")
        self.cov_uncorr_data = self.u.GetEmatrixInput("cov_uncorr_data",
                                                      "Covariance Matrix from Stat Uncertainties of Input Data")
        self.cov_total = self.u.GetEmatrixTotal('total', "Cov")

        # Get the unfolded output histogram
        self.o = self.u.GetOutput("o", "pythia", "signal", axisSteering, False)
        self.o_np = np.zeros(self.nmbinsGen * self.nptbinsGen)
        for i in range(1, self.o.GetNbinsX()+1):
            self.o_np[i-1] = self.o.GetBinContent(i)

        # Build a full covariance matrix in numpy
        self.cov_np = np.zeros((self.nmbinsGen * self.nptbinsGen, self.nmbinsGen * self.nptbinsGen))
        for i in range(1, self.o.GetNbinsX()+1):
            for j in range(1, self.o.GetNbinsX()+1):
                self.cov_np[i-1, j-1] = self.cov.GetBinContent(i, j)

        # Retrieve systematics uncertainties if available
        self.delta_sys_dic = {}
        if self.systematics is not None:
            for sys in self.systematics:

                osys = self.u.GetDeltaSysSource(sys, sys, 'hello', 'signal', axisSteering, 0)
                self.delta_sys_dic[sys] = np.zeros(self.o.GetNbinsX())
                for i in range(1, self.o.GetNbinsX()+1):
                    self.delta_sys_dic[sys][i-1] = osys.GetBinContent(i)


        # Get statistical uncertainties
        ematrix_input = self.u.GetEmatrixInput('name', 'name')
        ematrix_mat = self.u.GetEmatrixSysUncorr('name2', 'name2')
        self.stat = np.zeros(self.nmbinsGen * self.nptbinsGen)
        self.stat_mat = np.zeros(self.nmbinsGen * self.nptbinsGen)
        for i in range(1, self.o.GetNbinsX()+1):
            self.stat[i-1] = np.sqrt(ematrix_input.GetBinContent(i, i))
            self.stat_mat[i-1] = np.sqrt(ematrix_mat.GetBinContent(i, i))



        total_sys = np.zeros(self.o_np.shape)
        for sys in self.delta_sys_dic.keys():
            for i in range(len(self.o_np)):
                total_sys[i] += self.delta_sys_dic[sys][i]**2

        self.total_sys = total_sys**0.5

        self.total_error = np.sqrt(self.total_sys**2 + self.stat**2)

        # Bin results per pT bin (here assumed to be 4)
        self.output_pt_binned = []
        self.stat_pt_binned = []
        self.stat_mat_pt_binned = []
        self.total_error_pt_binned = []

        
        for i in range(0, 4):
            self.output_pt_binned.append(self.o_np[i*self.nmbinsGen:(i+1)*self.nmbinsGen])
            self.stat_pt_binned.append(self.stat[i*self.nmbinsGen:(i+1)*self.nmbinsGen])
            self.stat_mat_pt_binned.append(self.stat_mat[i*self.nmbinsGen:(i+1)*self.nmbinsGen])
            self.total_error_pt_binned.append(self.total_error[i*self.nmbinsGen:(i+1)*self.nmbinsGen] )
        
        
        self.total_sys = total_sys**0.5

    def plot_input(self):
        "plot the input vs the projection of response matrix"
        for i in range(4):
            #plt.stairs(self.M_np.sum(axis = 0)[i*self.nmbinsDet:(i+1)*self.nmbinsDet]/self.mreco_width, ls = '--')
            plt.stairs(self.h_np[i*self.nmbinsDet:(i+1)*self.nmbinsDet]/self.mreco_width, label = "Input")
            plt.legend()
            plt.show()
    def plot_covariance(self):
        """Plot the total covariance matrix."""
        plt.imshow(self.cov_np, origin='lower')
        plt.xlabel("Global Bin Number (Generator)")
        plt.ylabel("Global Bin Number (Generator)")
        plt.show()

    def plot_unfolded(self):
        """Plot the unfolded result compared to the truth for each pT bin."""
        plt.stairs(self.o_np)
        plt.show()
        for i in range(0, 4):
            plt.stairs(self.output_pt_binned[i] / self.mgen_width /
                       np.sum(self.output_pt_binned[i]),
                       self.mgen_edge, label="Unfolded DATA", color='k')
            # Modify the last bin edge if desired
            mgen_edge_mod = self.mgen_edge.copy()
            mgen_edge_mod[-1] = 300
            plt.stairs(self.htrue_np[i*self.nmbinsGen  :(i+1)*self.nmbinsGen] /
                       self.mgen_width / np.sum(self.htrue_np[i*self.nmbinsGen:(i+1)*self.nmbinsGen]),
                       self.mgen_edge, label='TRUE Pythia', color='r', linestyle='--')

            # plt.stairs(self.htrue_np[i*self.nmbinsGen   +1 :(i+1)*self.nmbinsGen] /
            #            self.mgen_width[1:] / np.sum(self.htrue_np[i*self.nmbinsGen + 1 :(i+1)*self.nmbinsGen]),
            #            self.mgen_edge[1:], label='TRUE Pythia', color='r', linestyle='--')
            mgen_center_mod = self.mgen_center.copy()
            mgen_center_mod[-1] = 250
            plt.errorbar(mgen_center_mod, self.output_pt_binned[i] / self.mgen_width / np.sum(self.output_pt_binned[i]), 
                         self.stat_pt_binned[i] / self.mgen_width / np.sum(self.output_pt_binned[i]), ls= "")
            plt.xlabel("GEN Mass (GeV)")
            plt.legend()
            plt.show()

    def get_results(self, return_stat=False):
        """
        Return the unfolded results.
          - If return_stat is True, return the stat. uncertainties and covariance matrix.
          - Otherwise, return only the unfolded (pt binned) result.
        """
        if return_stat:
            return self.stat_pt_binned, self.stat_mat_pt_binned, self.cov_np
        else:
            return self.output_pt_binned, self.total_error_pt_binned, self.stat_pt_binned

    def plot_systematic(self, systematic = None):
        """
        Return unfolded result, with a systematic variation
        """
        if systematic == None:
            systematic = list(self.delta_sys_dic.keys())[0]

            
        
            # plt.stairs(self.output_pt_binned[i] / self.mgen_width /
            #                np.sum(self.output_pt_binned[i]),
            #                self.mgen_edge, label="Unfolded DATA", color='k')
    
            # plt.stairs((self.delta_sys_dic[systematic][i*self.nmbinsGen  :(i+1)*self.nmbinsGen] + self.output_pt_binned[i])/
            #                self.mgen_width / np.sum(self.output_pt_binned[i]),
            #                self.mgen_edge, label='sys', color='r', linestyle='--')

            
        if systematic == 'total':
            # total_sys = np.zeros(self.o_np.shape)
            # for sys in self.delta_sys_dic.keys():
            #     for i in range(len(self.o_np)):
            #         total_sys[i] += self.delta_sys_dic[sys][i]**2
            # self.total_sys = total_sys**0.5

            # plt.stairs(total_sys)
            # plt.show()
            for i in range(4):
                plot_ratio(self.output_pt_binned[i] / self.mgen_width /
                               np.sum(self.output_pt_binned[i]),
                           (self.total_sys[i*self.nmbinsGen  :(i+1)*self.nmbinsGen] + self.output_pt_binned[i])/
                               self.mgen_width / np.sum(self.output_pt_binned[i]),
                           self.mgen_edge, label1 = "Nominal", label2 = systematic
                           
                          )
        else:
            for i in range(4):
                plot_ratio(self.output_pt_binned[i] / self.mgen_width /
                               np.sum(self.output_pt_binned[i]),
                           (self.delta_sys_dic[systematic][i*self.nmbinsGen  :(i+1)*self.nmbinsGen] + self.output_pt_binned[i])/
                               self.mgen_width / np.sum(self.output_pt_binned[i]),
                           self.mgen_edge, label1 = "Nominal", label2 = systematic
                           
                          )
                plt.show()

    def plot_systematic_frac(self):
        
        total_sys_jes = np.zeros(self.o_np.shape)
        for sys in self.delta_sys_dic.keys():
            if sys[:3] == "JES":
                for i in range(len(self.o_np)):
                    total_sys_jes[i] += self.delta_sys_dic[sys][i]**2
        total_sys_jes = total_sys_jes**0.5
        for i in range(4):
            for sys in self.delta_sys_dic.keys():
                if sys[:3] != "JES":
                    
                    plt.stairs(np.abs(self.delta_sys_dic[sys][i*self.nmbinsGen  :(i+1)*self.nmbinsGen])/np.abs(self.output_pt_binned[i]), self.mgen_edge , label = sys )

            plt.stairs(total_sys_jes[i*self.nmbinsGen  :(i+1)*self.nmbinsGen]/np.abs(self.output_pt_binned[i]), self.mgen_edge , label = "JESUp" )
            plt.stairs(self.total_sys[i*self.nmbinsGen  :(i+1)*self.nmbinsGen]/np.abs(self.output_pt_binned[i]), self.mgen_edge , label = "Total" )
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.ylim(0.00001,0.5)
            plt.yscale('log')
            plt.show()
                        

            
        
                

        
        
        



def unfold_using_matrix(data_2d, resp_matrix_4d, fakes, misses, 
                        systematics = None,
                        backgrounds = None, 
                        closure = False,
                        groomed = False,
                        is_uf = True,
                        merge = False,
                        normalised_xs = True,
                        do_syst = True,
                        discard_low_bins = False,
                        plot = False,
                        regularisation = "None",
                        return_tau = False,
                        return_stat = False
                        ):

    nmbinsDet =len( resp_matrix_4d.axes['mreco'].centers)
    ptreco_center = resp_matrix_4d.axes['ptreco'].centers
    ptreco_width = resp_matrix_4d.axes['ptreco'].widths
    ptreco_edge = resp_matrix_4d.axes['ptreco'].edges
    
    ptgen_center = resp_matrix_4d.axes['ptgen'].centers
    ptgen_width = resp_matrix_4d.axes['ptgen'].widths
    ptgen_edge = resp_matrix_4d.axes['ptgen'].edges
    mreco_center = resp_matrix_4d.axes['mreco'].centers
    mreco_edge = resp_matrix_4d.axes['mreco'].edges
    mgen_center = resp_matrix_4d.axes['mgen'].centers
    
    mreco_width = resp_matrix_4d.axes['mreco'].widths
    mgen_width = resp_matrix_4d.axes['mgen'].widths
    mgen_edge = resp_matrix_4d.axes['mgen'].edges
    

    bins = binning.binning(mbinsGen =mgen_edge,mbinsDet = mreco_edge, ptbinsGen = ptgen_edge, ptbinsDet = ptreco_edge )

    mbinsGen  =  bins.mbinsGen
    mbinsDet  =  bins.mbinsDet
    ptbinsGen =  bins.ptbinsGen
    ptbinsDet =  bins.ptbinsDet
    
    
    nmbinsGen  =  bins.nmbinsGen
    nmbinsDet  =  bins.nmbinsDet
    nptbinsGen =  bins.nptbinsGen
    nptbinsDet =  bins.nptbinsDet



    M_np = resp_matrix_4d.project('ptgen','mgen','ptreco','mreco').values(flow = False)
    M_np = M_np.reshape(M_np.shape[0]*M_np.shape[1], M_np.shape[2]*M_np.shape[3])

    h_np_underflow = resp_matrix_4d.project('ptgen','mgen','ptreco','mreco').values(flow = True)[0,:,1:-1,1:-1].sum(axis=0) # underflow bins in pt gen, and discarding the under and overflow bins for reco

    h_np_underflow = h_np_underflow.reshape(h_np_underflow.shape[0]* h_np_underflow.shape[1])
    print('shape', h_np_underflow.shape)
    underflow_frac = np.nan_to_num(h_np_underflow/M_np.sum(axis = 0))
    
    #plt.stairs(underflow_frac)
    #plt.show()
    
    M_np_error =resp_matrix_4d.project('ptgen','mgen','ptreco','mreco').variances(flow = False)
    M_np_error = M_np.reshape(M_np_error.shape[0]*M_np_error.shape[1], M_np_error.shape[2]*M_np_error.shape[3])**0.5
    

    htrue_np =  M_np.sum(axis = 1)
  
    if systematics!= None:
        for sys_name in systematics:
            systematics[sys_name] = systematics[sys_name].reshape( nptbinsGen*nmbinsGen, nptbinsDet*nmbinsDet )

    
    
    if closure:
        # h_np = M_np.sum(axis = 0)
        # h_np_error = M_np_error.sum(axis = 0)
        h_np = M_np.sum(axis = 0)
        #h_np_error = M_np_error.sum(axis = 0)
        h_np_error = h_np**0.5
        
        # for i in range(4):
        #     sum_i = h_np[i*nmbins:(i+1)*nmbinsDet].sum()
        #     h_np[i*nmbinsDet:(i+1)*nmbinsDet]/= (sum_i *0.000001)
        #     h_np_error[i*nmbinsDet:(i+1)*nmbinsDet]/= sum_i
        # h_np = h_np.reshape(h_np.shape[0]*h_np.shape[1])
        # h_np_error = h_np.reshape(h_np_error.shape[0]*h_np_error.shape[1])
        
    else:
        h_np = data_2d.project('ptreco','mreco').values(flow = False)
        h_np_error =  h_np**0.5
        # for i in range(4):
        #     sum_i =  1
        #     h_np[i]/= sum_i
        #     h_np_error/= sum_i
        h_np = h_np.reshape(h_np.shape[0]*h_np.shape[1])
        h_np_error = h_np_error.reshape(h_np_error.shape[0]*h_np_error.shape[1])

        h_np = h_np * (1 - underflow_frac)
        
        
       
        


        
        
        htrue_np =  M_np.sum(axis = 1)

    miss_values = misses.project('ptgen', 'mgen').values().reshape(M_np.shape[0])
    fake_values = fakes.project('ptreco', 'mreco').values().reshape(M_np.shape[1])

    
    print("Len of input", h_np.shape)
    print("Shape of matrix", M_np.shape)
    print("Len of fakes", fake_values.shape)
    print("Len of misses", miss_values.shape)
    ## Create the ROOT objects
    M = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(bins.genDist,bins.detDist,"M")
    
    h = bins.detDist.CreateHistogram("h")
    h_reco = bins.detDist.CreateHistogram("h_reco")
    
    htrue = bins.genDist.CreateHistogram("htrue")
    htruef = bins.genDist.CreateHistogram("htruef")

    ##### Fill the ROOT objects
    M_sys_dic = {}
    
    for i in range(1, M.GetNbinsX()+1):
        for j in range(1, M.GetNbinsY()+1):
            M.SetBinContent(i, j, M_np[i-1][j-1])
            M.SetBinError(i, j, M_np_error[i-1][j-1])
    if systematics!=None:
        for sys_name in systematics:
            M_sys_dic[sys_name ] = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(bins.genDist,bins.detDist,"M"+sys_name)
            for i in range(1, M.GetNbinsX()+1):
                for j in range(1, M.GetNbinsY()+1):
                    M_sys_dic[sys_name].SetBinContent(i, j, systematics[sys_name][i-1][j-1])
                    M_sys_dic[sys_name].SetBinError(i, j, M_np_error[i-1][j-1])
            
                    
    for i in range(1, h.GetNbinsX()+1):
        h.SetBinContent(i, h_np[i-1])
        h.SetBinError(i, h_np_error[i-1])
        h_reco.SetBinContent(i, M_np.sum(axis = 0)[i-1])
    for i in range(1, htrue.GetNbinsX()+1):
        htrue.SetBinContent(i, htrue_np[i-1])

    

    
    

    fake_hist = h.Clone('fakes')
    fake_hist.Reset()
    
    if not closure:
        for i in range(1, M.GetNbinsX()):
            M.SetBinContent(i, 0, miss_values[i-1])
            ## same for other matrices
            if systematics!= None:
                for sys_name in M_sys_dic:
                    M_sys_dic[sys_name].SetBinContent(i, 0, miss_values[i-1])
                        

    for i in range(1, M.GetNbinsY()):
        fake_hist.SetBinContent(i, fake_values[i-1])


    genBin = bins.genBin
    detBin = bins.detBin

    
    ### Plotting the response matrix
    if plot == True:
        lines_x = []
        lines_y = []
    
        for pt in ptgen_edge[1:-1]:
            lines_x.append(bins.genDist.GetGlobalBinNumber(1,pt)-0.5)
    
        for pt in ptreco_edge[1:-1]:
    
            lines_y.append(bins.detDist.GetGlobalBinNumber(1,pt)-0.5)
    
    
        # print(lines_x)
        # print(lines_y)
    
    
    
        label_lines_x = []
        label_lines_y = []
    
        for pt in ptgen_edge[0:-1]:
    
            label_lines_x.append(bins.genDist.GetGlobalBinNumber(70,pt))
    
        for pt in ptreco_edge[0:-1]:
    
            label_lines_y.append(bins.detDist.GetGlobalBinNumber(70,pt))
        # print(label_lines_x)
        # print(label_lines_y)
    
        arr = ptgen_edge
        x_labels = np.array([str(int(x)) for x in arr])
        #print(x_labels)
        
        ax = draw_colz_histogram(M, use_log_scale=True)
        ax.set_title(" ")
        for x in lines_x:
            ax.axvline(x=x, color='black', linestyle='--')
    
        # Draw horizontal dashed lines
        for y in lines_y:
            ax.axhline(y=y, color='black', linestyle='--')
        ax.plot([0.5, M.GetNbinsX()+0.5], [0.5, M.GetNbinsY()+0.5], color='red', linestyle='--', linewidth=1)    
    
        ax.set_xticks([])
        ax.set_yticks([])
    
        # Add new x-axis labels
        ax.set_xticks(label_lines_x)
        ax.set_xticklabels(x_labels[:-1])
    
        ax.set_yticks(label_lines_y)
        ax.set_yticklabels(x_labels[:-1])
        ax.tick_params(axis='both', which='both', length=0)
        hep.cms.label("Preliminary")
        ax.set_xlabel(r"GEN p$_{T}$ (GeV)")
        ax.set_ylabel(r"RECO $p_T$ (GeV)")

        matrix_fig = plt.gcf()
        plt.show()

    ## Unfold

    orientation = ROOT.TUnfold.kHistMapOutputHoriz
    regMode = ROOT.TUnfold.kRegModeCurvature

    # if regularisation == "None":
    #     regMode = ROOT.TUnfold.kRegModeNone
    con = ROOT.TUnfold.kEConstraintArea #ROOT.TUnfold.kEConstraintArea
    mode =  ROOT.TUnfoldDensity.kDensityModeBinWidth
    axisSteering =  "*[UOB]"
    
    nScan=50
    tauMin=0.00000001
    tauMax=0.01
    iBest=0
    
    logTauX = ROOT.MakeNullPointer(ROOT.TSpline)
    logTauY = ROOT.MakeNullPointer(ROOT.TSpline)
    lCurve = ROOT.MakeNullPointer(ROOT.TGraph)

    
    u = ROOT.TUnfoldDensity(M, orientation, regMode, con, mode, genBin, detBin, "signal", axisSteering)
    if systematics!=None:
        for sys in M_sys_dic:
            u.AddSysError(M_sys_dic[sys], sys, orientation,  ROOT.TUnfoldDensity.kSysErrModeMatrix)

            
    u.SetInput( h )



    
    if not closure:
        u.SubtractBackground(fake_hist, 'fakes')

    bg_hists = {}
    if not closure:
        print("Subtracting background")
        if backgrounds!=None:
            for bg in backgrounds.keys():
                bg_hists[bg] = h.Clone('fakes')
                bg_hists[bg].Reset()
                for i in range(1, M.GetNbinsY()):
    
                    bg_hists[bg].SetBinContent(i, backgrounds[bg][i-1])
                u.SubtractBackground(bg_hists[bg], bg)
            
    

    
    
    #u.ScanTau(nScan, tauMin, tauMax, logTauX, ROOT.TUnfoldDensity.kEScanTauRhoAvg, 'signal', )
    
    #u.ScanLcurve(nScan,tauMin,tauMax,lCurve,logTauX,logTauY)
    
    #i_best = u.ScanLcurve(nScan,tauMin,tauMax,lCurve,logTauX,logTauY)
    if regularisation == "None":
        u.DoUnfold(0.000)
    elif regularisation == "ScanLcurve":
        i_best = u.ScanLcurve(nScan,tauMin,tauMax,lCurve,logTauX,logTauY)
    elif regularisation == "ScanTau":
        u.ScanTau(nScan, tauMin, tauMax, logTauX, ROOT.TUnfoldDensity.kEScanTauRhoAvg, 'signal', )
    elif regularisation == "ScanSURE":
        logSURE = ROOT.MakeNullPointer(ROOT.TGraph)
        chi2 = ROOT.MakeNullPointer(ROOT.TGraph)
        lCurve = ROOT.MakeNullPointer(ROOT.TGraph)
        u.ScanSURE(nScan, tauMin, tauMax, logSURE , chi2, lCurve)
    else:
        print("Specify correct regularisation")
        return 0

    #u.DoUnfold(u.GetTau())
    print("Regularisation USED", regularisation)
    print("Tau value", u.GetTau())
    cov = u.GetEmatrixTotal("cov", "Covariance Matrix")
    #cov = u.GetEmatrixTotal("ematrix","Covariance Matrix", "signal", "*[]", False)
    
    
    cov_uncorr = u.GetEmatrixSysUncorr("cov_uncorr", "Covariance Matrix from Uncorrelated Uncertainties")
    cov_uncorr_data = u.GetEmatrixInput("cov_uncorr_data", "Covariance Matrix from Stat Uncertainties of Input Data")

    cov_total = u.GetEmatrixTotal('total', "Cov")


    o = u.GetOutput("o","pythia","signal", axisSteering, False)

    o_np = np.zeros(nmbinsGen*nptbinsGen)
    cov_np = np.zeros((nmbinsGen*nptbinsGen, nmbinsGen*nptbinsGen))
    for i in range(1, o.GetNbinsX()+1):
        o_np[i-1] = o.GetBinContent(i)
        
    # plt.stairs(h_np)
    # plt.stairs(M_np.sum(axis = 0), ls = 'dashed')
    
    # plt.show()
    # plt.stairs(M_np.sum(axis = 1), ls = 'dashed')
    # plt.stairs(o_np, ls = 'dotted')
    # plt.show()
    output_pt_binned = []

    stat_pt_binned = []
    stat_mat_pt_binned = []
    
    ematrix_input = u.GetEmatrixInput('name', 'name')
    ematrix_mat = u.GetEmatrixSysUncorr('name2', 'name2')

    total_cov_np = np.zeros((nmbinsGen*nptbinsGen, nmbinsGen*nptbinsGen) )
    for i in range(1, o.GetNbinsX()+1):
        for j in range(1, o.GetNbinsX()+1):
            total_cov_np[i-1, j-1] = cov_total.GetBinContent(i,j)
            
    if systematics!=None:
        e_sys = u.GetDeltaSysSource('JES_AbsoluteMPFBiasUp', 'puUp', 'hello', 'signal', axisSteering, 0)
        c = ROOT.TCanvas()
        e_sys.Draw()
        c.SaveAs('a.png')
        
        e_sys_np = np.zeros(o_np.shape)
        for i in range(1, o.GetNbinsX()+1):
            #print(e_sys.GetBinContent(i))
            e_sys_np[i-1] = e_sys.GetBinContent(i)
    if plot:
        plt.imshow(total_cov_np, origin = 'lower')
        plt.xlabel("Gloabl Bin Number (Generator)")
        plt.ylabel("Global Bin Number (Generator)")

        
        plt.show()
    total_error = np.diag(total_cov_np)**0.5


    
    sys_name_list = []
    
    for i in range(len(u.GetSysSources())):
        sys_name_list.append(u.GetSysSources()[i])



    delta_sys_dic = {}
    if systematics!= None:
        for sys in systematics:
            osys = u.GetDeltaSysSource(sys, sys, 'hello', 'signal', axisSteering, 0)
            delta_sys_dic[sys] = np.zeros(o.GetNbinsX())
            for i in range(1, o.GetNbinsX()+1):
                delta_sys_dic[sys][i-1] =  osys.GetBinContent(i)

    print(delta_sys_dic['JES_AbsoluteMPFBiasUp'])
    
    if plot:
        for sys_name in delta_sys_dic:
            if sys_name[:3]=='JES':
                plt.stairs(np.abs(delta_sys_dic[sys_name])/np.abs(delta_sys_dic['JES_AbsoluteMPFBiasUp']), label = sys_name)
        plt.xlabel("Global Bin Number (GEN level)")
        plt.ylabel("Relative Uncertainty")
        plt.ylim(0,0.2)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.show()

    

    total_sys = np.zeros( o.GetNbinsX())
    jes_total_sys = np.zeros(o.GetNbinsX())


    for sys in delta_sys_dic:
        for i in range(len(total_sys)):
            total_sys[i] += delta_sys_dic[sys][i]**2
        
    total_sys = total_sys**0.5

    
    for sys in delta_sys_dic :
        if sys[:3] == "JES":
            for i in range(len(total_sys)):
                jes_total_sys[i] += delta_sys_dic[sys][i]**2
            plt.stairs(jes_total_sys**0.5/np.abs(o_np))
    plt.ylim(0,1)
    plt.show()
                
    jes_total_sys = jes_total_sys**0.5

    if plot:
        plt.stairs(jes_total_sys/o_np)
        plt.ylim(0,1)
        plt.show()
    
    

    


    

    
    stat =  np.zeros(nmbinsGen*nptbinsGen )
    stat_mat = np.zeros(nmbinsGen*nptbinsGen)
    for i in range(1, o.GetNbinsX()+1):
        stat[i-1] = np.sqrt(ematrix_input.GetBinContent(i, i))
        stat_mat[i-1] = np.sqrt(ematrix_mat.GetBinContent(i, i))
        for j in range(1, o.GetNbinsX()+1):
            cov_np[i-1, j-1] = cov.GetBinContent(i, j)
    if plot:
        plt.stairs(total_error/o_np, label = 'Total')
        plt.stairs(stat/o_np, label = "data", lw = 2)
        plt.stairs(stat_mat/o_np, label = 'mat')
        if systematics!=None:
            plt.stairs(np.abs(total_sys)/o_np, label = 'Sys', lw = 2)
    
        plt.legend()
        plt.ylim(0,0.2)
        plt.show()

    for i in range(0,4):
        if plot:
            plt.stairs( o_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width/o_np[i*nmbinsGen:(i+1)*nmbinsGen].sum(), mgen_edge, label = "Unfolded DATA", color = 'k')
            # plt.stairs( o_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width, mgen_edge, label = "Unfolded DATA", color = 'k')
            mgen_center[-1] = 300
            
            plt.stairs(htrue_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width/htrue_np[i*nmbinsGen:(i+1)*nmbinsGen].sum(),mgen_edge,  label = 'TRUE Pythia', color = 'r', ls = '--')
            # plt.stairs(htrue_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width,mgen_edge,  label = 'TRUE Pythia', color = 'r', ls = '--')
            plt.xlabel("GEN Mass (GeV)")
            plt.legend()
            plt.show()

        output_pt_binned.append( o_np[i*nmbinsGen:(i+1)*nmbinsGen])
        stat_pt_binned.append( stat[i*nmbinsGen:(i+1)*nmbinsGen])
        stat_mat_pt_binned.append( stat_mat[i*nmbinsGen:(i+1)*nmbinsGen])
    
        
    if return_stat:
        return stat_pt_binned, stat_mat_pt_binned, cov_np
        
    elif plot:
        return output_pt_binned, matrix_fig
    
    elif not return_tau:
        return output_pt_binned
        
    
    else:
        return output_pt_binned, u.GetTau()
    
    