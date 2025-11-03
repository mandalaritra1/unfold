
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from .binning import compress_open_ended_last_bin, compress_edges_by_pt

def plot_response_matrix(h_resp, title="Response matrix"):
    import ROOT
    c = ROOT.TCanvas()
    h_resp.Draw("colz")
    c.SetTitle(title)
    c.Draw()
    return c

def th1_to_arrays(h):
    nb = h.GetNbinsX()
    x  = np.arange(1, nb + 1)
    y  = np.array([h.GetBinContent(int(i)) for i in x], dtype=float)
    ye = np.array([h.GetBinError(int(i))   for i in x], dtype=float)
    return x, y, ye

def stairs_normed(y, edges, edges_true, **kwargs):
    y = np.asarray(y, dtype=float)
    bw = np.diff(edges_true)
    print(bw)
    denom = y.sum()
    if denom <= 0:
        denom = 1.0
    vals = y / (bw * denom)
    return plt.stairs(vals, edges, **kwargs)

def plot_inputs_by_pt(reco_pt_binned, reco_edges_by_pt, xlim=None, title="Input (reco) by pT bin"):
    edges_disp = compress_edges_by_pt(reco_edges_by_pt)
    for i, (y, edges) in enumerate(zip(reco_pt_binned, edges_disp), 1):
        plt.figure(figsize=(6,4))
        stairs_normed(y, edges)
        plt.xlabel("m/pT (reco)")
        plt.ylabel("1/σ dσ/d(m/pT)")
        plt.title(f"{title} — pT bin {i}")
        if xlim:
            plt.xlim(*xlim)
        plt.tight_layout()
        plt.show()

def plot_unfold_vs_true_by_pt(unfold_pt, true_pt, gen_edges_by_pt, xlim=None, title="Unfolded vs Truth"):
    edges_disp = compress_edges_by_pt(gen_edges_by_pt)
    for i, (y_u, y_t, edges, edges_true) in enumerate(zip(unfold_pt, true_pt, edges_disp,gen_edges_by_pt), 1):
        plt.figure(figsize=(6,4))
        stairs_normed(y_t, edges, edges_true,label="truth", linestyle="-")
        stairs_normed(y_u, edges, edges_true,label="unfolded", linestyle="--")
        plt.xlabel("m (gen)")
        plt.ylabel("1/σ dσ/dm")
        plt.title(f"{title} — pT bin {i}")
        if xlim:
            plt.xlim(*xlim)
        plt.legend()
        plt.tight_layout()
        plt.show()
