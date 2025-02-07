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
        


        mbinsDet[-2] = 250
        mbinsDet[-1] = 300.0
        mbinsGen[-1] = 300.0
        ptbinsGen[-1] = 10000
        ptbinsDet[-1] = 10000

        # mbinsGen = array.array('d', [0.0e+00, 1.0e+01, 2.0e+01, 4.0e+01, 6.0e+01, 8.0e+01, 1.0e+02, 1.5e+02, 2.0e+02, 3.0e+02])
        # mbinsDet = array.array('d', [0.00e+00, 5.00e+00, 1.00e+01, 1.50e+01, 2.00e+01, 3.00e+01,
        #            4.00e+01, 5.00e+01, 6.00e+01, 7.00e+01, 8.00e+01, 9.00e+01,
        #            1.00e+02, 1.25e+02, 1.50e+02, 1.75e+02, 2.00e+02, 250,
        #            3.0e+02])
        
        ptbinsGen = array.array('d', [  200.,   260.,   350.,   460., 10000])
        ptbinsDet = array.array('d', [  200.,   260.,   350.,   460.,10000])

        
        generatorBinning = ROOT.TUnfoldBinning("generator")
        ### Need coarser binning for signal
        signalDistribution = generatorBinning.AddBinning("signal")    
        signalDistribution.AddAxis("massgen",nmbinsGen,mbinsGen,
                                False, # needed for fakes
                                False # overflow bin
                                )
        signalDistribution.AddAxis("ptgen",4,ptbinsGen,
                                False, # needed for fakes
                                False# overflow bin
                                )

        # X axis : generator binning is Signal : mgen * ptgen and Background : mrec * ptrec



        genBin = generatorBinning



        detectorBinning = ROOT.TUnfoldBinning("detector") 
        detectorDistribution=detectorBinning.AddBinning("detectordistribution")
        detectorDistribution.AddAxis("mass",nmbinsDet,mbinsDet,
                                False, # no underflow bin (not reconstructed)
                                False # overflow bin
                                )
        detectorDistribution.AddAxis("pt",4,ptbinsDet,
                                False, # no underflow bin (not reconstructed)
                                False # overflow bin
                                )

        yaxis =  array.array('d',[a for a in range(nmbinsDet *nptbinsDet+1 )])
        ny = len(yaxis)-1

        detBin  = detectorBinning
        
        self.detBin = detectorBinning
        self.genBin = generatorBinning
        
        self.detDist = detectorDistribution
        self.genDist = signalDistribution
        