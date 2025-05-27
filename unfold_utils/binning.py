import ROOT
import uproot
import numpy as np
import array as array
import math
import matplotlib.pyplot as plt
import bisect

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

        

        print(nmbinsGen)
        print(mbinsGen)

        print(nmbinsDet)
        print(mbinsDet)


        #mbinsDet[-2] = 250
        #mbinsDet[-1] = 300.0
        #mbinsGen[-1] = 300.0


        
        
        #ptbinsGen[-1] = 600
        #ptbinsDet[-1] = 600
        #ptbinsDet[-1] = 10000

        # mbinsGen = array.array('d', [0.0e+00, 1.0e+01, 2.0e+01, 4.0e+01, 6.0e+01, 8.0e+01, 1.0e+02, 1.5e+02, 2.0e+02, 3.0e+02])
        # mbinsDet = array.array('d', [0.00e+00, 5.00e+00, 1.00e+01, 1.50e+01, 2.00e+01, 3.00e+01,
        #            4.00e+01, 5.00e+01, 6.00e+01, 7.00e+01, 8.00e+01, 9.00e+01,
        #            1.00e+02, 1.25e+02, 1.50e+02, 1.75e+02, 2.00e+02, 250,
        #            3.0e+02])
        
        #ptbinsGen = array.array('d', [  200.,   260.,   350.,   460., 10000])
        #ptbinsDet = array.array('d', [  200.,   260.,   350.,   460.,10000])

        ptbinsGen[-1] = 1000
        mbinsGen[-1] = 300
        
        generatorBinning = ROOT.TUnfoldBinning("generator")
        ### Need coarser binning for signal
        signalDistribution = generatorBinning.AddBinning("signal")    
        signalDistribution.AddAxis("massgen",nmbinsGen,mbinsGen,
                                False, # needed for fakes
                                False # overflow bin
                                )
        signalDistribution.AddAxis("ptgen", nptbinsGen ,ptbinsGen,
                                False, # needed for fakes
                                False# overflow bin
                                )


        pt_factors = [1.        , 3.22558071, 7.93939854]     ## 1/N_mc_truth
        
        # build a piecewise formula in x only
        pieces = []
        for i, f in enumerate(pt_factors):
            low, high = ptbinsGen[i], ptbinsGen[i+1]
            # note: x = pt, y = mass (we ignore y here)
            pieces.append(f"{f}*(y>={low} && y<{high})")
        formula = "+".join(pieces)
        
        # define TF2 over the full pt range (x) and full mass range (y)
        func2 = ROOT.TF2(
            "ptOnlyFunc",     # name
            formula,          # piecewise in x
            ptbinsGen[0],     # x-min = lowest pt
            ptbinsGen[-1],    # x-max = highest pt
            mbinsGen[0],      # y-min = lowest mass
            mbinsGen[-1]      # y-max = highest mass
        )
        
        # attach to your node—ROOT will evaluate func2(x(pt), y(m)) but return only your pt-based piece
        signalDistribution.SetBinFactorFunction(1.0, func2)
        
        for i in range(1,30):
            print(signalDistribution.GetBinFactor(i))
        


        genBin = generatorBinning



        detectorBinning = ROOT.TUnfoldBinning("detector") 
        detectorDistribution=detectorBinning.AddBinning("detectordistribution")
        detectorDistribution.AddAxis("mass",nmbinsDet,mbinsDet,
                                False, # no underflow bin (not reconstructed)
                                False # overflow bin
                                )
        detectorDistribution.AddAxis("pt", nptbinsDet ,ptbinsDet,
                                False, # no underflow bin (not reconstructed)
                                False # overflow bin
                                )

        yaxis =  array.array('d',[a for a in range(nmbinsDet *nptbinsDet+1 )])
        ny = len(yaxis)-1
        

        print("Detector Binning created")
        detBin  = detectorBinning
        
        self.detBin = detectorBinning
        self.genBin = generatorBinning
        
        self.detDist = detectorDistribution
        self.genDist = signalDistribution
        