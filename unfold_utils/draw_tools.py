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

from unfold_utils import binning
import matplotlib.colors as mcolors

import tempfile
import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_colz_histogram(th2f, use_log_scale=False, vmin=None, vmax=None):
    # Create a temporary file to store the TH2F histogram
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

            # Get the bin edges for x and y axes
            x_edges = hist.axis(0).edges()
            y_edges = hist.axis(1).edges()

            # Create a meshgrid for plotting
            X, Y = np.meshgrid(x_edges, y_edges)

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(12, 10))

            # Use logarithmic color normalization if specified
            if use_log_scale:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                pcm = ax.pcolormesh(X, Y, hist_array.T, shading='auto', cmap='viridis', norm=norm)
            else:
                pcm = ax.pcolormesh(X, Y, hist_array.T, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)

            # Add color bar
            cbar = plt.colorbar(pcm, ax=ax, label='Counts')
            ax.set_xlabel(th2f.GetXaxis().GetTitle())
            ax.set_ylabel(th2f.GetYaxis().GetTitle())
            ax.set_title(th2f.GetTitle())

            # Return the axis object for further customization
            return ax