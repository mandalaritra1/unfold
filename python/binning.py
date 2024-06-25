import ROOT
import uproot
import numpy as np
import array as array
import math
import matplotlib.pyplot as plt

#import statistics as st
ROOT.gStyle.SetOptStat(000000)


class binning():
    def __init__(self, mbinsGen = None, mbinsDet = None, ptbinsGen = None, ptbinsDet = None, flow = True):
        # mbinsGen  = array.array('d',[0, 10, 20, 40, 60, 80, 100, 150, 200, 13000] )
        # mbinsDet  = array.array('d', [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 6200, 13000])
        # ptbinsGen = array.array('d', [  170.,   200.,   260.,   350.,   460., 13000.])
        # ptbinsDet = array.array('d', [200, 260, 350, 460, 13000])
        # ptbinsDet = array.array('d', [  170.,   185.,   200.,   230.,   260.,   305.,   350.,   405.,
        #  460., 13000.])


        nmbinsGen  = len(mbinsGen) -1
        nmbinsDet  = len(mbinsDet) -1
        nptbinsGen = len(ptbinsGen) -1
        nptbinsDet = len(ptbinsDet) -1
        
        self.mbinsGen = mbinsGen
        self.mbinsDet = mbinsDet
        self.ptbinsGen = ptbinsGen
        self.ptbinsDet = ptbinsDet
        
        self.nmbinsGen = nmbinsGen
        self.nmbinsDet = nmbinsDet
        self.nptbinsGen = nptbinsGen
        self.nptbinsDet = nptbinsDet



        generatorBinning = ROOT.TUnfoldBinning("generator")
        ### Need coarser binning for signal
        signalDistribution = generatorBinning.AddBinning("signal")    
        signalDistribution.AddAxis("massgen",nmbinsGen,mbinsGen,
                                True, # needed for fakes
                                True # overflow bin
                                )
        signalDistribution.AddAxis("ptgen",nptbinsGen,ptbinsGen,
                                True, # needed for fakes
                                True # overflow bin
                                )

        # X axis : generator binning is Signal : mgen * ptgen and Background : mrec * ptrec

        xaxis =  array.array('d',[a for a in range(nmbinsGen *nptbinsGen + 1 )])
        nx = len(xaxis)-1

        genBin = generatorBinning



        detectorBinning = ROOT.TUnfoldBinning("detector") 
        detectorDistribution=detectorBinning.AddBinning("detectordistribution")
        detectorDistribution.AddAxis("mass",nmbinsDet,mbinsDet,
                                True, # no underflow bin (not reconstructed)
                                True # overflow bin
                                )
        detectorDistribution.AddAxis("pt",nptbinsDet,ptbinsDet,
                                True, # no underflow bin (not reconstructed)
                                True # overflow bin
                                )

        yaxis =  array.array('d',[a for a in range(nmbinsDet *nptbinsDet+1 )])
        ny = len(yaxis)-1

        detBin  = detectorBinning
        
        self.detBin = detectorBinning
        self.genBin = generatorBinning
        
        self.detDist = detectorDistribution
        self.genDist = signalDistribution
        