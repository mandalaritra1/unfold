import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import mplhep as hep


import mplhep as hep
hep.style.use('CMS')
import numpy as np
import matplotlib.pyplot as plt

def plot_data_mc_ratio_with_asymmetric_errors(
    bin_edges, data_counts, data_errors, mc_histograms, mc_stat_errors, mc_syst_errors_up = None, mc_syst_errors_down = None,
    labels = None, colors = None, stack=False, divide_by_binwidth=False, yscale = 'linear', rp_xlabel = None, normalize_to_data = False, rlabel = r"2018 ( L = 59.83 $fb^{-1}$)"
):
    """
    Plots a Data/MC ratio plot with binned ratios and asymmetric error bars for systematic uncertainties.

    Args:
        bin_edges: Array of bin edges (len = n_bins + 1).
        data_counts: Array of data bin contents.
        data_errors: Array of data statistical uncertainties.
        mc_histograms: List of MC bin contents for multiple histograms.
        mc_stat_errors: Array of MC statistical uncertainties (total for stacked histograms).
        mc_syst_errors_up: Array of MC systematic upper uncertainties (total for stacked histograms).
        mc_syst_errors_down: Array of MC systematic lower uncertainties (total for stacked histograms).
        labels: List of labels for the MC histograms.
        colors: List of colors for the MC histograms.
        stack: Whether to stack the MC histograms (default: False).
        divide_by_binwidth: Whether to divide all histograms by the bin width (default: False).
    """
    # Compute bin centers and widths
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    print("length of bin centers")
    print("length of data_counts", len(data_counts))
    
    
    # Normalise to total DATA if required
    if normalize_to_data:
        scale_to_data = np.sum(data_counts)/np.sum(mc_histograms)
        mc_histograms = [mc * scale_to_data for mc in mc_histograms]
        
    # Normalize by bin width if the option is enabled
    if divide_by_binwidth:
        data_counts = data_counts / bin_widths
        data_errors = data_errors / bin_widths
        mc_histograms = [mc / bin_widths for mc in mc_histograms]
        # mc_stat_errors = mc_stat_errors / bin_widths
        # mc_syst_errors_up = mc_syst_errors_up / bin_widths
        # mc_syst_errors_down = mc_syst_errors_down / bin_widths
    
    
        
    # Combine MC histograms if stacking is enabled
    if stack:
        mc_stack = np.sum(mc_histograms, axis=0)
    else:
        mc_stack = mc_histograms[0]  # Use the first histogram as the base
        
    

    # Data/MC ratio and errors
    ratio = data_counts / mc_stack
    ratio_errors_data = data_errors / mc_stack
    
    # ratio_errors_mc_stat = mc_stat_errors / mc_stack
    # ratio_errors_mc_total_up = np.sqrt(mc_stat_errors**2 + mc_syst_errors_up**2) / mc_stack
    # ratio_errors_mc_total_down = np.sqrt(mc_stat_errors**2 + mc_syst_errors_down**2) / mc_stack

    # Start the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, )

    # --- Top plot: Data and MC ---
    ax1.errorbar(bin_centers, data_counts, xerr=bin_widths / 2, yerr=data_errors, fmt="k.", label="Data", zorder=10)

    # Plot MC histograms
    if stack:
        # Stacked histogram
        bottom = np.zeros_like(mc_histograms[0])
        for mc, label, color in zip(mc_histograms, labels, colors):
            ax1.bar(bin_centers, mc, width=bin_widths, bottom=bottom, color=color, alpha=1, label=label, edgecolor="none")
            bottom += mc
    else:
        # Overlay histograms
        for mc, label, color in zip(mc_histograms, labels, colors):
            ax1.bar(bin_centers, mc, width=bin_widths, color=color, alpha=1, label=label, edgecolor="black")

    ax1.set_yscale(yscale)
    ax1.set_ylabel("# Events per GeV" if not divide_by_binwidth else "# Events / GeV")
    #ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Bottom plot: Data/MC ratio ---
    ax2.errorbar(bin_centers, ratio, xerr=bin_widths / 2, yerr=ratio_errors_data, fmt="k.")

    # Add binned MC uncertainties (statistical and asymmetric total)
    # ax2.bar(
    #     bin_centers,
    #     2 * ratio_errors_mc_stat,
    #     width=bin_widths,
    #     bottom=1 - ratio_errors_mc_stat,
    #     color="gray",
    #     alpha=0.8,
    #     label="MC Stat.",
    #     edgecolor="none",
    #     linewidth=0.5,
    # )
    # for i in range(len(bin_centers)):
    #     ax2.bar(
    #         bin_centers[i],
    #         ratio_errors_mc_total_up[i] + ratio_errors_mc_total_down[i],
    #         width=bin_widths[i],
    #         bottom=1 - ratio_errors_mc_total_down[i],
    #         color="black",
    #         alpha=0.2,
    #         label=r"MC Stat. $ \bigoplus$ Syst." if i == 0 else "",
    #         edgecolor="none",
    #         linewidth=0.5,
    #     )

    ax2.axhline(1, color="black", linestyle="--", linewidth=1)  # Reference line at ratio=1
    ax2.set_ylim(0.5, 1.5)
    ax2.set_ylabel("Data/MC")
    if rp_xlabel == None:
        ax2.set_xlabel("Unknown")
    else:
        ax2.set_xlabel(rp_xlabel)
    #ax2.grid(True, linestyle="--", linewidth=0.5)

    # Consolidate legend in the top plot
    handles0, labels0 = ax1.get_legend_handles_labels()
    handles, labels = ax2.get_legend_handles_labels()

    for i in range(len(handles)):
        handles0.append(handles[i])
        labels0.append(labels[i])
    ax1.legend(handles0, labels0, fontsize = 15)
    hep.cms.label(ax = ax1, data = True, label = "Preliminary", rlabel = rlabel)
    # Show and save plot
    plt.tight_layout()
    
    return fig, ax1, ax2


    

class AsymmetricErrorPlot:
    def __init__(self):
        self.masses = []
        self.values = []
        self.values_data = []
        self.values_herwig = []
        self.widths = []
        self.stat_unc_up = []
        self.stat_unc_down = []
        self.syst_unc_up = []
        self.syst_unc_down = []

    def add_data_points(self, x_data, y_data, y2_data, y3_data, x_widths):
        self.masses = x_data
        self.values_data = y2_data
        self.values = y_data
        self.widths = x_widths
        
        if y3_data!= None:
            self.values_herwig = y3_data
        self.stat_unc_up = [0] * len(x_data)
        self.stat_unc_down = [0] * len(x_data)
        self.syst_unc_up = [0] * len(x_data)
        self.syst_unc_down = [0] * len(x_data)

    def add_stat_unc(self, stat_unc_up, stat_unc_down):
        print(len(stat_unc_up))
        print(len(self.masses))
        if len(stat_unc_up) != len(self.masses) or len(stat_unc_down) != len(self.masses):
            raise ValueError("Length of statistical uncertainties must match the number of data points.")
        self.stat_unc_up = stat_unc_up
        self.stat_unc_down = stat_unc_down

    def add_syst_unc(self, syst_unc_up, syst_unc_down):
        if len(syst_unc_up) != len(self.masses) or len(syst_unc_down) != len(self.masses):
            raise ValueError("Length of systematic uncertainties must match the number of data points.")
        self.syst_unc_up = syst_unc_up
        self.syst_unc_down = syst_unc_down

    def plot(self, xlabel='Mass (GeV)', ylabel='N / dG / dM_jet', custom_text='pT 170-200 GeV', pt_text = 'pt not defined', lumi_text = None):
        
        # Convert lists to numpy arrays for easier manipulation
        masses = np.array(self.masses)
        values = np.array(self.values)
        values_data = np.array(self.values_data)
        stat_unc_up = np.array(self.stat_unc_up)
        stat_unc_down = np.array(self.stat_unc_down)
        syst_unc_up = np.array(self.syst_unc_up)
        syst_unc_down = np.array(self.syst_unc_down)
        widths = np.array(self.widths)
        print('widths before', widths)
        
        widths[-1] = widths[-2]
        print('widths after', widths)
        masses[-1] = masses[-2] + widths[-1]
        
        #values_herwig = np.array(self.values_herwig)
        # Calculate total uncertainty as a combination of statistical and systematic uncertainties
        total_unc_up = np.sqrt(stat_unc_up**2 + syst_unc_up**2)
        total_unc_down = np.sqrt(stat_unc_down**2 + syst_unc_down**2)

        left_edges = masses - widths / 2
        right_edges = masses + widths / 2
        
        # Construct the full bin edges array
        bin_edges = np.concatenate(([left_edges[0]], right_edges))

#         # Create figure and grid spec layout
#         fig = plt.figure(figsize=(10, 8))
        
#         #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
#         ax1 = plt.subplot()

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        # Plot main data with error bars
        for (mass, value, stat_err_up, stat_err_down, total_err_up, total_err_down, width) in zip(masses, values_data, stat_unc_up, stat_unc_down, total_unc_up, total_unc_down, widths):
            #Total uncertainties
            ax1.add_patch(plt.Rectangle((mass - width / 2, value - total_err_down), width, total_err_up + total_err_down, edgecolor='cyan', facecolor='cyan', label='Total Unc.'))
            # Statistical uncertainties
            #ax1.add_patch(plt.Rectangle((mass - width / 2, value - stat_err_down), width, stat_err_up + stat_err_down ,facecolor='none', hatch='/////', label='Stat. Unc.'))
            ax1.add_patch(plt.Rectangle((mass - width / 2, value - stat_err_down), width, stat_err_up + stat_err_down ,edgecolor='b', facecolor='b', label='Stat. Unc.'))

        print("masses len", len(masses))
        print(" len values_data", len(values_data))
        ax1.scatter(masses, values_data, label='Unfolded Mass', color='black', marker='o')
        ax1.scatter(masses, values, label='MC Truth Pythia', color='red', marker = '*')
        #ax1.scatter(masses, values_herwig, label='Unfolded using Herwig Matrix', color='g', marker = 'o')
        
        ax1.set_xticks(bin_edges)  # Set tick positions
        xtick_labels = [f"{edge:.0f}" if i % 2 == 0 else "" for i, edge in enumerate(bin_edges)]
        
        xtick_labels[-1] = r'$\infty$'
        xtick_labels[-2] = int(bin_edges[-2])
        print('xtick labels', xtick_labels)
        ax1.set_xticklabels(xtick_labels)
        ax1.xaxis.set_minor_locator(plt.NullLocator())
        ax2.xaxis.set_minor_locator(plt.NullLocator())
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        
        handles.append(plt.Line2D([0], [0], color='none', label=custom_text))
        handles.append(plt.Line2D([0], [0], color='none', label=custom_text))
        labels.append(custom_text)
        labels.append(pt_text )

# Add legend with the custom entry

        by_label = dict(zip(labels, handles))
        leg = ax1.legend(by_label.values(), by_label.keys(), fontsize = 15)
        #leg = ax1.legend(by_label.values(), by_label.keys(), fontsize = 15)
        leg.get_texts()[-1].set_fontweight('bold')
        leg.get_texts()[-2].set_fontweight('bold')
        # Set labels and text
        ax1.set_ylabel(ylabel)
        #ax1.set_xlabel(xlabel)
        ax1.set_xlim(0,bin_edges[-2]+widths[-1])

        #ax1.text(0.50, 0.50, custom_text, ha='center', va='center', transform=ax1.transAxes, fontsize=22)
        closure = False
        if closure:
            hep.cms.label("Preliminary", ax = ax1, data = 0, rlabel = r"$138 fb^{-1}$ (2018)")
        else:
            hep.cms.label("Preliminary", ax = ax1, data = 1, rlabel = r"$138 fb^{-1}$ (2018)")
        # if lumi_text!=None:
        #     hep.cms.lumitext(str(lumi_text), ax = ax1)
        # else:
        #     hep.cms.lumitext('13 TeV', ax = ax1)
        #ax1.set_yscale('log')
        # Ratio plot
        
        # Ratio plot
        ratio = values / values_data
        print("Ratio", ratio)
        
        
        ratio_unc_stat = ratio * np.sqrt((stat_unc_up / values))**2 
        ratio_unc = ratio * np.sqrt((total_unc_up / values)**2 )
        
        total_ratio_unc_up = ratio_unc
        total_ratio_unc_down = ratio_unc
        
        print("Stat Ratio", ratio_unc_stat)
        print("Ratio Total Unc", ratio_unc)
        # Plot ratio with error bars
        ax2.plot(masses, ratio, marker='.', color='black', label='Stat. Unc.', linestyle = 'none')
        
        for (mass, rat, total_err_up, total_err_down, ratio_unc_stat, width) in zip(masses, ratio, total_ratio_unc_up, total_ratio_unc_down, ratio_unc_stat, widths):
            ax2.add_patch(plt.Rectangle((mass - width / 2, rat - total_err_down), width, total_err_up + total_err_down, edgecolor='cyan', facecolor='cyan'))
            # Statistical uncertainties
            ax2.add_patch(plt.Rectangle((mass - width / 2, rat - ratio_unc_stat), width, 2*ratio_unc_stat ,edgecolor='b', facecolor='b', label='Stat. Unc.'))
            
        ax2.axhline(1, color='red', linestyle='--')
        ax2.set_ylabel('MC/Data')
        ax2.set_xlabel(xlabel)
        ax2.set_ylim(0,5)
        ax2.set_xticklabels(xtick_labels)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.05)
        # plt.show()
        return ax1, ax2, fig