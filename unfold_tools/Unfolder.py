import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
#hep.style.use("CMS")
from matplotlib import colors, ticker
import pickle as pkl
from unfold_utils.integrate_and_rebin import *
from unfold_utils.merge_helpers import *
from unfold_tools import *
from unfold_tools import binning
import ROOT
import numpy as np
import matplotlib.colors as mcolors
class Unfolder:
    def __init__(self, groomed, closure = False):
        self.groomed = groomed
        self.closure = closure
        self.bins = binning.bin_edges(self.groomed)
        edges, edges_gen, pt_edges = self.bins.edges, self.bins.edges_gen, self.bins.pt_edges

        jes_sys_list = ['JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown',
                'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp',
                'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown',
                'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
                'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up',
                'JES_RelativeJEREC1Down', 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown',
                'JES_RelativePtBBUp', 'JES_RelativePtBBDown', 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 'JES_RelativePtEC2Down',
                'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 
                'JES_RelativeStatECUp', 'JES_RelativeStatECDown', 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown',
                'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown']


        non_jes_sys_list = ['nominal', 'puUp', 'puDown', 'elerecoUp', 'elerecoDown',
                            'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp',
                            'murecoDown', 'muidUp', 'muidDown', 'mutrigUp', 'muisoUp', 'muisoDown','mutrigDown', 'pdfUp',
                            'pdfDown', 'q2Up', 'q2Down', 'prefiringUp', 'prefiringDown', 
                            'JERUp', 'JERDown', 'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']


        self.systematics = jes_sys_list + non_jes_sys_list

        #self.systematics = ['nominal', 'JERUp', 'JERDown', 'JMSUp', 'JMSDown']
        self.pt_edges = pt_edges
        self.edges = edges
        self.edges_gen = edges_gen
        self._load_data(filename_mc = 'latest_pkl/0508/mc_0508_full.pkl', filename_data = "latest_pkl/0508/data_0508_full.pkl")
        self.y_unf_dict = {}
        self._perform_unfold(closure = self.closure)
        for syst in self.systematics:
            self._perform_unfold(systematic = syst, closure = self.closure)
        self._normalize_result()
        self._compute_total_systematic()


    def _normalize_result(self):
        self.normalized_results = []
        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.gen_mass_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.reco_mass_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.reco_mass_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.gen_mass_edges_by_pt)
        error_pt_binned = unflatten_gen_by_pt(self.ye_unf, self.gen_mass_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths = np.diff(self.gen_mass_edges_by_pt[i])
            bin_widths_reco = np.diff(self.reco_mass_edges_by_pt[i])
            result = {
                "true": true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(),
                "unfolded": unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(),
                "unfolded_err": error_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(),
                "measured": measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(),
                "reco_mc": reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(),
                "pt_bin": (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf'))
            }
            self.normalized_results.append(result)
        
        # Storing normalized results for systematics
        self.normalized_systematics = []
        # Prepare normalized_systematics as a list of dicts, one per pt bin
        self.normalized_systematics = []
        for i in range(len(self.pt_edges)-1):
            pt_bin = (self.pt_edges[i], self.pt_edges[i+1] if i+1 < len(self.pt_edges)-1 else float('inf'))
            unfolded = {}
            bin_widths = np.diff(self.gen_mass_edges_by_pt[i])
            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf_dict[syst], self.gen_mass_edges_by_pt)
                unfolded[syst] = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
            self.normalized_systematics.append({
            "pt_bin": pt_bin,
            "unfolded": unfolded
            })
    def _compute_total_systematic(self):
        print("Computing total systematic uncertainty...")
        # Compute total systematic uncertainty for each pt bin
        for i in range(len(self.normalized_results)):
            nominal = self.normalized_results[i]['unfolded']
            syst_up_total = np.zeros_like(nominal)
            syst_down_total = np.zeros_like(nominal)
            for syst in self.systematics:
                if syst.endswith('Up'):
                    syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                    diff_up = syst_up - nominal
                    print("Systematic Up:", syst, "Diff Up:", diff_up)
                    syst_up_total += diff_up**2
                elif syst.endswith('Down'):
                    syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                    diff_down = syst_down - nominal
                    print("Systematic Down:", syst, "Diff Up:", diff_down)
                    syst_down_total += diff_down**2
            # Take sqrt of sum of squares for total uncertainty
            total_up_unc = np.sqrt(syst_up_total)
            total_down_unc = np.sqrt(syst_down_total)
            self.normalized_results[i]['syst_unc'] = {
            'up': total_up_unc,
            'down': total_down_unc
            }

    def plot_systematic_fraction(self):
        # Plot the systematic uncertainties as a fraction of the nominal unfolded result
        plt.figure(figsize=(10, 8))
        for i, result in enumerate(self.normalized_results):

            pt_bin = result['pt_bin']
            nominal = result['unfolded']
            total_syst_up = result['syst_unc']['up']
            total_syst_down = result['syst_unc']['down']


            # Group systematics by prefix
            jes_up = []
            jes_down = []
            ele_up = []
            ele_down = []
            mu_up = []
            mu_down = []
            for syst_name in self.normalized_systematics[i]['unfolded']:
                if syst_name.startswith('JES'):
                    if syst_name.endswith('Up'):
                        jes_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    elif syst_name.endswith('Down'):
                        jes_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                elif syst_name.startswith('ele'):
                    if syst_name.endswith('Up'):
                        ele_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    elif syst_name.endswith('Down'):
                        ele_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                elif syst_name.startswith('mu'):
                    if syst_name.endswith('Up'):
                        mu_up.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)
                    elif syst_name.endswith('Down'):
                        mu_down.append(self.normalized_systematics[i]['unfolded'][syst_name] - nominal)

            # Combine grouped uncertainties in quadrature
            if jes_up:
                jes_up_total = np.sqrt(np.sum([diff**2 for diff in jes_up], axis=0))
                hep.histplot(np.abs(jes_up_total / nominal), self.gen_mass_edges_by_pt[i], label="JES Up")
            # if jes_down:
            #     jes_down_total = np.sqrt(np.sum([diff**2 for diff in jes_down], axis=0))
            #     hep.histplot(np.abs(jes_down_total / nominal), self.gen_mass_edges_by_pt[i], label="JES Down")
            if ele_up:
                ele_up_total = np.sqrt(np.sum([diff**2 for diff in ele_up], axis=0))
                hep.histplot(np.abs(ele_up_total / nominal), self.gen_mass_edges_by_pt[i], label="ele Up")
            # if ele_down:
            #     ele_down_total = np.sqrt(np.sum([diff**2 for diff in ele_down], axis=0))
            #     hep.histplot(np.abs(ele_down_total / nominal), self.gen_mass_edges_by_pt[i], label="ele Down")
            if mu_up:
                mu_up_total = np.sqrt(np.sum([diff**2 for diff in mu_up], axis=0))
                hep.histplot(np.abs(mu_up_total / nominal), self.gen_mass_edges_by_pt[i], label="mu Up")
            # if mu_down:
            #     mu_down_total = np.sqrt(np.sum([diff**2 for diff in mu_down], axis=0))
            #     hep.histplot(np.abs(mu_down_total / nominal), self.gen_mass_edges_by_pt[i], label="mu Down")

            # Plot individual non-grouped systematics
            for syst in self.systematics:
                if syst == 'nominal':
                    continue
                if syst.startswith('JES') or syst.startswith('ele') or syst.startswith('mu'):
                    continue  # Already handled above
                if syst.endswith('Up'):
                    syst_up = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                    diff_up = syst_up - nominal
                    syst_fraction_up = np.abs(diff_up / nominal)
                    hep.histplot(syst_fraction_up, self.gen_mass_edges_by_pt[i], label=f"{syst}")
                # if syst.endswith('Down'):
                #     syst_down = self.normalized_systematics[i]['unfolded'].get(syst, np.zeros_like(nominal))
                #     diff_down = syst_down - nominal
                #     syst_fraction_down = np.abs(diff_down / nominal)
                #     hep.histplot(syst_fraction_down, self.gen_mass_edges_by_pt[i], label=f"{syst}")
                    
                #syst_fraction_down = syst_down / nominal

                # Plot the systematic fraction
                
                # hep.histplot(syst_fraction_down, 
                #             self.gen_mass_edges_by_pt[i], 
                #             label=f"{syst} Down")
            # Calculate systematic fraction
            total_syst_fraction_up = total_syst_up / nominal
            total_syst_fraction_down = total_syst_down / nominal

            # Plot the systematic fraction
            hep.histplot(total_syst_fraction_up, 
                        self.gen_mass_edges_by_pt[i], 
                        label=f"Total ")
            plt.yscale('log')
            plt.legend(title=rf"$p_T$  {pt_bin[0]}-{pt_bin[1]} GeV", loc='center left', bbox_to_anchor=(1, 0.5))
            hep.cms.label("Preliminary", data=True)
            if self.groomed:
                plt.xlim(20,250)
                
            else:
                plt.xlim(10,250)
            plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            plt.tight_layout()
            plt.savefig(f'plots/uncertainties/summary_groomed_{i}.pdf' if self.groomed else f'plots/uncertainties/summary_ungroomed_{i}.pdf')
            plt.show()
            
        
    def plot_correlation(self):
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
        ncols_by_gp = [len(e)-1 for e in self.gen_mass_edges_by_pt]
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
        cbar.set_label("Correlation")
        hep.cms.label("Preliminary", data=True)
        plt.tight_layout()
        plt.savefig(f'plots/unfold/correlation_groomed.pdf' if self.groomed else f'plots/unfold/correlation_ungroomed.pdf')

        plt.show()


    



    def _load_data(self, filename_mc = 'latest_pkl/0508/mc_0508_full.pkl', filename_data = "latest_pkl/0508/data_0508_full.pkl"):
        print(filename_mc)
        with open(filename_mc, "rb") as f:
            output_pythia= pkl.load( f )

    
        with open(filename_data, "rb") as f:
            output_data = pkl.load( f )

        if not self.groomed:
            resp_matrix_4d = output_pythia['response_matrix_u']

            input_data = output_data['ptjet_mjet_u_reco']
            # fakes = output_pythia['fakes_u']
            # misses = output_miss['misses_u']
            # hist_bg = output_bg['response_matrix_u'].project('dataset','ptreco','mreco')
            # resp_matrix_pythia = resp_matrix_4d
            # resp_matrix_4d_herwig  = output_herwig['response_matrix_u'][{'systematic':['herwig']}]
            # fakes_herwig = output_herwig['fakes_u']
            # misses_herwig = output_herwig['misses_u']
        else:
            resp_matrix_4d = output_pythia['response_matrix_g']

            input_data = output_data['ptjet_mjet_g_reco']
            # fakes = output_pythia['fakes_g']
            # misses = output_miss['misses_g']
            # hist_bg = output_bg['response_matrix_u'].project('dataset','ptreco','mreco')
            # resp_matrix_pythia = resp_matrix_4d
            # resp_matrix_4d_herwig = output_herwig['response_matrix_g'][{'systematic':['herwig']}]
        
            # fakes_herwig = output_herwig['fakes_g']
            # misses_herwig = output_herwig['misses_g']

            

        npt = len(self.pt_edges) - 1
        nmreco = len(self.edges) -1
        resp_matrix_4d = rebin_hist(resp_matrix_4d, 'ptreco', self.pt_edges)
        resp_matrix_4d = rebin_hist(resp_matrix_4d, 'ptgen', self.pt_edges)

        resp_matrix_4d = rebin_hist(resp_matrix_4d, 'mreco',self.edges )

        resp_matrix_4d = rebin_hist(resp_matrix_4d, 'mgen',self.edges_gen )

        input_data = rebin_hist(input_data, 'mreco', self.edges)
        input_data = rebin_hist(input_data, 'ptreco', self.pt_edges)

        
        

        



        

        pt_edges        = self.bins.pt_edges
        mass_edges_reco = self.bins.mass_edges_reco
        mass_edges_gen  = self.bins.mass_edges_gen
        
        
    
        reco_mass_edges_by_pt = self.bins.reco_mass_edges_by_pt

        gen_mass_edges_by_pt = self.bins.gen_mass_edges_by_pt

        self.reco_mass_edges_by_pt = reco_mass_edges_by_pt
        self.gen_mass_edges_by_pt = gen_mass_edges_by_pt
        
        
        self.M_np_2d_dict = {}
        self.mosaic_dict = {}
        for syst in self.systematics:
            print(resp_matrix_4d.axes)
            resp_matrix_4d_syst = resp_matrix_4d[{'systematic':syst}]


            proj = resp_matrix_4d_syst.project('ptreco', 'mreco', 'ptgen', 'mgen')
            M_np = proj.values(flow=False)

            if syst == 'nominal':
                reco_proj = input_data.project('ptreco', 'mreco')
                h2d = reco_proj.values()
                self.h2d, perm_used = reorder_to_expected_2d(h2d, mass_edges_reco, pt_edges)
            
            # Ensure correct axis order first:
            self.M_np_2d_dict[syst], perm_used = reorder_to_expected(M_np, mass_edges_reco, pt_edges, mass_edges_gen)

            # Build the unpadded mosaic (no NaNs):
            self.mosaic_dict[syst], blocks = mosaic_no_padding(
                self.M_np_2d_dict[syst], mass_edges_reco, mass_edges_gen,
                reco_mass_edges_by_pt, gen_mass_edges_by_pt
            )

            
        self.M_np_2d = self.M_np_2d_dict['nominal']
        self.mosaic = self.mosaic_dict['nominal']
            
        self.mosaic_2d = merge_mass_flat(self.h2d,
                                    mass_edges_reco,
                                    reco_mass_edges_by_pt)


        del output_pythia, resp_matrix_4d, resp_matrix_4d_syst
    def _perform_unfold(self, systematic = 'nominal', closure = False):
        # ------------------------------------------------------------------
        # 1.  Provide your numpy inputs
        # ------------------------------------------------------------------
        # resp_np   : 2-D numpy array – rows = reco bins, cols = true(gen) bins
        # meas_flat : 1-D numpy array – reco distribution ( *same* row order as resp_np )

        # example placeholders – replace with your real arrays
        resp_np   = self.mosaic_dict[systematic]  # 2D numpy array (reco x true)
        if closure:
            meas_flat = self.mosaic.sum(axis = 1)
        else:
            meas_flat = self.mosaic_2d

        true_flat = self.mosaic.sum(axis = 0)
        n_reco, n_true = resp_np.shape
        assert len(meas_flat) == n_reco, "measured spectrum must have n_reco bins"

        # ------------------------------------------------------------------
        # 2.  Build ROOT histograms
        # ------------------------------------------------------------------
        # Response matrix:  x = truth  (columns),  y = reco  (rows)
        h_resp = ROOT.TH2D("resp", "response;truth bin;reco bin",
                        n_true,  0, n_true,
                        n_reco,  0, n_reco)
        for i_reco in range(n_reco):
            for j_true in range(n_true):
                h_resp.SetBinContent(j_true + 1, i_reco + 1, resp_np[i_reco, j_true])

        # Measured (reco) spectrum
        h_meas = ROOT.TH1D("meas", "measured;reco bin;entries", n_reco, 0, n_reco)
        for i_reco, val in enumerate(meas_flat, 1):
            h_meas.SetBinContent(i_reco, float(val))


        # True spectrum
        h_true = ROOT.TH1D("true", "sim;true bin;entries", n_true, 0, n_true)
        for i_true, val in enumerate(true_flat, 1):
            h_true.SetBinContent(i_true, float(val))

        # ------------------------------------------------------------------
        # 3.  Run TUnfold
        # ------------------------------------------------------------------
        unfold = ROOT.TUnfoldDensity(
            h_resp,
            ROOT.TUnfold.kHistMapOutputHoriz  # orientation matches (truth x, reco y)
        )

        # feed measured spectrum
        status = unfold.SetInput(h_meas)
        if status >= 10000:
            raise RuntimeError("TUnfold input had overflow/underflow – check your hist.")

        # Optional: scan L-curve to choose tau.  Quick-n-dirty: 20 points, auto range
        #unfold.ScanSURE(50, 0.000000001, 0.1, )
        #print("Chosen tau =", tau_best)
        # nPoint, tauMin, tauMax = 30, 1e-6, 1e+0   # coarse example range
        # g_logTauSURE = ROOT.TGraph()
        # g_df_chi2A   = ROOT.TGraph()
        # g_lCurve     = ROOT.TGraph()

        # tau_sure = unfold.ScanSURE(
        #     nPoint,
        #     tauMin,
        #     tauMax,
        #     g_logTauSURE,
        #     g_df_chi2A,
        #     g_lCurve,
        # )
        unfold.DoUnfold(0.0)
        if systematic == 'nominal':
            self.cov = unfold.GetEmatrixTotal("cov", "Covariance Matrix")
            self.cov_uncorr = unfold.GetEmatrixSysUncorr("cov_uncorr",
                                                        "Covariance Matrix from Uncorrelated Uncertainties")
            self.cov_uncorr_data = unfold.GetEmatrixInput("cov_uncorr_data",
                                                        "Covariance Matrix from Stat Uncertainties of Input Data")
            self.cov_total = unfold.GetEmatrixTotal('total', "Cov")

            #convert these to numpy arrays
            # Convert ROOT covariance matrices to numpy arrays using the shape of self.M_np_2d
            n_reco, n_true = self.mosaic.shape
            n_bins = n_true  # unfolded bins correspond to truth bins

            self.cov_np = np.zeros((n_bins, n_bins))
            self.cov_uncorr_np = np.zeros((n_bins, n_bins))
            self.cov_data_np = np.zeros((n_bins, n_bins))
            for i in range(1, n_bins + 1):
                for j in range(1, n_bins + 1):
                    self.cov_np[i-1, j-1] = self.cov.GetBinContent(i, j)
                    self.cov_uncorr_np[i-1, j-1] = self.cov_uncorr.GetBinContent(i, j)
                    self.cov_data_np[i-1, j-1] = self.cov_uncorr_data.GetBinContent(i, j)
        # ------------------------------------------------------------------
        # 4.  Grab the unfolded spectrum and covariance
        # ------------------------------------------------------------------
        h_unfold = unfold.GetOutput("unfold")          # TH1D with n_true bins
        #cov      = unfold.GetEmatrixInput()            # TH2D covariance of input
        #cov_out  = unfold.GetEmatrixOutput("cov_out")  # TH2D covariance of unfolded

        # quick printout
        #h_unfold.Print("all")
        y_meas, ye_meas = self._th1_to_arrays(h_meas)
        y_true, ye_true = self._th1_to_arrays(h_true)
        y_unf , ye_unf  = self._th1_to_arrays(h_unfold)
        
        if systematic == 'nominal':
            self.y_meas = y_meas
            self.ye_meas = ye_meas
            self.y_unf = y_unf
            self.ye_unf = ye_unf
            self.y_true = y_true
        else:
            self.y_unf_dict[systematic] = y_unf



    def _th1_to_arrays(self,h):
        nb = h.GetNbinsX()                       # bin numbers
        x  = np.arange(1, nb + 1)
        y  = np.array([h.GetBinContent(int(i)) for i in x])
        ye = np.array([h.GetBinError(int(i))   for i in x])
        return  y, ye

    def plot_unfolded(self, log = False):

        unfolded_pt_binned = unflatten_gen_by_pt(self.y_unf, self.gen_mass_edges_by_pt)
        measured_pt_binned = unflatten_gen_by_pt(self.y_meas, self.reco_mass_edges_by_pt)
        reco_mc_pt_binned = unflatten_gen_by_pt(self.mosaic.sum(axis = 1), self.reco_mass_edges_by_pt)
        true_pt_binned = unflatten_gen_by_pt(self.y_true, self.gen_mass_edges_by_pt)
        error_pt_binned = unflatten_gen_by_pt(self.ye_unf, self.gen_mass_edges_by_pt)
        for i in range(len(self.pt_edges)-1):
            bin_widths = np.diff(self.gen_mass_edges_by_pt[i])
            bin_widths_reco = np.diff(self.reco_mass_edges_by_pt[i])
            
            hep.histplot(true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(), self.gen_mass_edges_by_pt[i], color = 'b', label = 'PYTHIA', alpha = 0.5)
            hep.histplot(unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), self.gen_mass_edges_by_pt[i], yerr = error_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), label = 'Unfolded', color = 'k' )
            hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), self.reco_mass_edges_by_pt[i], color = 'k', ls= '--', alpha= 0.5, label = 'Meas' )
            hep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), self.reco_mass_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
            title = f"pT bin: {int(self.pt_edges[i])}-{int(self.pt_edges[i+1]) if i+1 < len(self.pt_edges)-1 else '∞'} GeV"
            plt.legend(title = title) 
            
            if self.groomed:
                plt.xlim(0,250)
                plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            #plt.ylim(0,0.02)
            if not self.groomed:
                plt.xlim(20,250)
                plt.xlabel("Groomed Jet Mass (GeV)" if self.groomed else "Ungroomed Jet Mass (GeV)")
            plt.show()    
    def plot_response_matrix(self, probability = True, log = False):
        fig, ax = self._plot_response_mosaic_cms(
            self.mosaic,
            reco_mass_edges_by_pt=self.reco_mass_edges_by_pt,
            gen_mass_edges_by_pt=self.gen_mass_edges_by_pt,
            reco_pt_edges=self.pt_edges,
            gen_pt_edges=self.pt_edges,
            probability = probability,
            mask_zeros=True,
            log=log,                              # set False for linear
            rlabel=f"Groomed, " if self.groomed else f"Ungroomed, ",
        )
        plt.show()
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
        hep.cms.label("Preliminary", data=False, rlabel=(rlabel + f"Cond. = {np.linalg.cond(mosaic):.2f}") if rlabel else f"Cond. = {np.linalg.cond(mosaic):.2f}")

        fig.tight_layout()
        if self.groomed:
            plt.savefig("plots/unfold/response_groomed.pdf")
        else:
            plt.savefig("plots/unfold/response_ungroomed.pdf")
        return fig, ax
