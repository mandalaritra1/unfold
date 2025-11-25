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




def plot_ratio(output_data, output_sys, mgen_edge, label1, label2):
    """
    Plots a main canvas (upper panel) with stairs plots for the unfolded data and its systematic variation,
    and a ratio plot (lower panel) of (data+sys)/data.

    Parameters:
      output_data : array-like
          The unfolded data (for a single category).
      sys_delta : array-like
          The systematic shift corresponding to the unfolded data.
      mgen_width : array-like
          The bin widths used for normalization.
      mgen_edge : array-like
          The bin edges used for the stairs plot.
    """
    # Normalize the data
    norm_factor = np.sum(output_data)
    y_data = output_data 
    y_sys = output_sys 

    # Compute the ratio (using np.divide to safely handle division by zero)
    ratio = np.divide(np.abs(y_sys-y_data), np.abs(y_data), out=np.ones_like(y_sys), where=y_data != 0)

    # Create a figure with two subplots (main and ratio)
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True,
                                              figsize=(8, 6),
                                              gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the unfolded data (main canvas)
    ax_main.stairs(y_data, mgen_edge, label=label1, color='k')
    # Plot the systematic variation (main canvas)
    ax_main.stairs(y_sys, mgen_edge, label=label2, color='r', linestyle='--')
    ax_main.set_ylabel("Normalized Rate")
    ax_main.legend(loc='best')

    # Plot the ratio (systematic over data)
    ax_ratio.stairs(ratio, mgen_edge, label="Ratio (sys/data)", color='b')
    ax_ratio.axhline(1, color='gray', linestyle='--')  # Reference line at 1
    ax_ratio.set_xlabel("Bin Edge")
    ax_ratio.set_ylabel("Ratio")

    ax_ratio.set_ylim(0,0.2)
    
    plt.tight_layout()
    plt.show()


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
            hist_array = np.ma.masked_where(hist_array <= 0, hist_array)
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