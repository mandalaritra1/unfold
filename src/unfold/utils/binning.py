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

        # ################# pt factors ###########################
        pt_factors = [1.        , 3.22558071, 100]     ## 1/N_mc_truth
        
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

        # #############################################################

        # node = signalDistribution

        # m_pt_factors = np.array([[1.00000000e+00, 4.43960233e+00, 3.96006094e+00, 4.91562583e+00,
        #                         5.96291810e+00, 1.09242219e+01, 3.59080083e+01, 2.24050740e+02,
        #                         3.03434626e+03, 1.86257010e+04, 3.25241044e+05],
        #                        [3.60898295e+00, 1.56030193e+01, 1.33252650e+01, 1.79504050e+01,
        #                         2.10243958e+01, 2.53092300e+01, 3.51254342e+01, 6.29033193e+01,
        #                         1.57107448e+02, 5.50001232e+02, 1.82067644e+04],
        #                        [9.94255738e+00, 4.23724609e+01, 3.48858221e+01, 4.75986364e+01,
        #                         6.10242658e+01, 7.35455616e+01, 8.43982258e+01, 1.00128802e+02,
        #                         1.25559957e+02, 1.07197832e+02, 1.83120491e+02]])
        # pieces = []
        # for i_pt, row in enumerate(m_pt_factors):
        #     pt_lo, pt_hi = ptbinsGen[i_pt],   ptbinsGen[i_pt+1]
        #     for j_m, factor in enumerate(row):
        #         m_lo, m_hi = mbinsGen[j_m], mbinsGen[j_m+1]
        #         pieces.append(
        #             f"{factor}*(y>={pt_lo} && y<{pt_hi})*(x>={m_lo} && x<{m_hi})"
        #         )
        # formula = "+".join(pieces)
        
        # # define TF2 over full pt (x) and mass (y) ranges
        # f2 = ROOT.TF2(
        #     "binFactor2D", formula,
        #     ptbinsGen[0], ptbinsGen[-1],
        #     mbinsGen[0],  mbinsGen[-1]
        # )
        
        # # attach via the public method
        # signalDistribution.SetBinFactorFunction(1.0, f2)
        # print("Done: attached m–pt lookup via TF2")

        
        
        # for i in range(1,30):
        #     print(signalDistribution.GetBinFactor(i))
        



        
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
        