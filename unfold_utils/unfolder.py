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

def unfold_using_matrix(data_2d, resp_matrix_4d, fakes, misses, 
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

    M = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(bins.genDist,bins.detDist,"M")
    
    h = bins.detDist.CreateHistogram("h")
    h_reco = bins.detDist.CreateHistogram("h_reco")
    
    htrue = bins.genDist.CreateHistogram("htrue")
    htruef = bins.genDist.CreateHistogram("htruef")


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
  

    
    if closure:
        # h_np = M_np.sum(axis = 0)
        # h_np_error = M_np_error.sum(axis = 0)
        h_np = M_np.sum(axis = 0)
        h_np_error = M_np_error.sum(axis = 0)
        
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



    ##### Fill the ROOT objects
    for i in range(1, M.GetNbinsX()+1):
        for j in range(1, M.GetNbinsY()+1):
            M.SetBinContent(i, j, M_np[i-1][j-1])
                #M.SetBinError(i, j, M_np_error[i-1][j-1])
    for i in range(1, h.GetNbinsX()+1):
        h.SetBinContent(i, h_np[i-1])
        h.SetBinError(i, h_np_error[i-1])
        h_reco.SetBinContent(i, M_np.sum(axis = 0)[i-1])
    for i in range(1, htrue.GetNbinsX()+1):
        htrue.SetBinContent(i, htrue_np[i-1])

    

    
    miss_values = misses.project('ptgen', 'mgen').values().reshape(M_np.shape[0])
    fake_values = fakes.project('ptreco', 'mreco').values().reshape(M_np.shape[1])

    fake_hist = h.Clone('fakes')
    fake_hist.Reset()
    
    if not closure:
        for i in range(1, M.GetNbinsX()):
            M.SetBinContent(i, 0, miss_values[i-1])

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
    
    
        print(lines_x)
        print(lines_y)
    
    
    
        label_lines_x = []
        label_lines_y = []
    
        for pt in ptgen_edge[0:-1]:
    
            label_lines_x.append(bins.genDist.GetGlobalBinNumber(70,pt))
    
        for pt in ptreco_edge[0:-1]:
    
            label_lines_y.append(bins.detDist.GetGlobalBinNumber(70,pt))
        print(label_lines_x)
        print(label_lines_y)
    
        arr = ptgen_edge
        x_labels = np.array([str(int(x)) for x in arr])
        print(x_labels)
    
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
            
    # if do_syst:
    #     for sys in sys_list:
    #         u.AddSysError(resp_syst_dic[sys], sys, orientation,  ROOT.TUnfoldDensity.kSysErrModeMatrix)
    #u.AddSysError(M1, "Up", orientation, ROOT.TUnfoldDensity.kSysErrModeMatrix)
    #u.AddSysError(M2, "Dn", orientation, ROOT.TUnfoldDensity.kSysErrModeMatrix)
    
    
    #u.ScanTau(nScan, tauMin, tauMax, logTauX, ROOT.TUnfoldDensity.kEScanTauRhoAvg, 'signal', )
    
    #u.ScanLcurve(nScan,tauMin,tauMax,lCurve,logTauX,logTauY)
    
    #i_best = u.ScanLcurve(nScan,tauMin,tauMax,lCurve,logTauX,logTauY)
    if regularisation == "None":
        u.DoUnfold(0.00)
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

    u.DoUnfold(u.GetTau())
    print("Regularisation USED", regularisation)
    print("Tau value", u.GetTau())
    cov = u.GetEmatrixTotal("cov", "Covariance Matrix")
    #cov = u.GetEmatrixTotal("ematrix","Covariance Matrix", "signal", "*[]", False)
    
    
    cov_uncorr = u.GetEmatrixSysUncorr("cov_uncorr", "Covariance Matrix from Uncorrelated Uncertainties")
    cov_uncorr_data = u.GetEmatrixInput("cov_uncorr_data", "Covariance Matrix from Stat Uncertainties of Input Data")


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
    
    ematrix_input = u.GetEmatrixInput('name')
    ematrix_mat = u.GetEmatrixSysUncorr('name2')
    stat =  np.zeros(nmbinsGen*nptbinsGen )
    stat_mat = np.zeros(nmbinsGen*nptbinsGen)
    for i in range(1, o.GetNbinsX()+1):
        stat[i-1] = np.sqrt(ematrix_input.GetBinContent(i, i))
        stat_mat[i-1] = np.sqrt(ematrix_mat.GetBinContent(i, i))
        for j in range(1, o.GetNbinsX()+1):
            cov_np[i-1, j-1] = cov.GetBinContent(i, j)
   
    
    for i in range(0,4):
        if plot:
            plt.stairs( o_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width/o_np[i*nmbinsGen:(i+1)*nmbinsGen].sum(), mgen_edge, label = "Unfolded DATA", color = 'k')
            mgen_center[-1] = 300

            #plt.plot(mgen_center[2:], o_herwig_np[i*nmbinsGen:(i+1)*nmbinsGen][2:]/mgen_width[2:]/o_herwig_np[i*nmbinsGen:(i+1)*nmbinsGen][2:].sum(), 's', ls = '-', label = "HERWIG")
            #plt.plot(mgen_center, o_herwig_toy_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width/o_herwig_toy_np[i*nmbinsGen:(i+1)*nmbinsGen].sum(), marker = '^', ls = '--', label = "HERWIG TOY")
            #plt.plot(mgen_center, o_pythia_toy_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width/o_pythia_toy_np[i*nmbinsGen:(i+1)*nmbinsGen].sum(), 'v', ls = 'dotted', label = "PYTHIA TOY")
            #plt.plot(resp_syst_dic_np['herwig'].sum(axis = 1)[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width, 'o-', label = 'herwig gen')
            print(htrue_np.shape)
            plt.stairs(htrue_np[i*nmbinsGen:(i+1)*nmbinsGen]/mgen_width/htrue_np[i*nmbinsGen:(i+1)*nmbinsGen].sum(),mgen_edge,  label = 'TRUE Pythia', color = 'r')
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
    