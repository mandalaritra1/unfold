import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
#hep.style.use("CMS")
from matplotlib import colors, ticker
import pickle as pkl
from unfold_utils.integrate_and_rebin import *
import itertools



# ------------------------------------------------------------------
# helpers
def _subset_positions(old, new, atol=1e-9):
    old, new = np.asarray(old, float), np.asarray(new, float)
    ok = np.isclose(new[:, None], old[None, :], atol=atol).any(axis=1)
    if not np.all(ok):
        raise ValueError(f"Edges {new[~ok]} not subset of original.")
    pos = np.searchsorted(old, new)
    for k, e in enumerate(new):
        m = np.isclose(old, e, atol=atol)
        if m.any():
            pos[k] = np.flatnonzero(m)[0]
    return pos

def _merge_1d(vec, old_edges, new_edges):
    pos = _subset_positions(old_edges, new_edges)
    return np.array([vec[s:e].sum() for s, e in zip(pos[:-1], pos[1:])])

    
def unflatten_gen_by_pt(flat_array, gen_mass_edges_by_pt, *,
                        return_dict=False, label_fmt="pt{}"):
    """
    Reverse the concatenation we did when we flattened truth-mass columns.

    Parameters
    ----------
    flat_array : 1-D numpy array
        The unfolded (or any) spectrum in the order
        [ gen_pt bin 0, gen_pt bin 1, … ] where each block’s length is
        len(gen_mass_edges_by_pt[i]) - 1.
    gen_mass_edges_by_pt : list of 1-D sequences
        gen_mass_edges_by_pt[i] is the **mass** edge list for gen-pT slice i.
    return_dict : bool, default False
        • False → return a list `[col0, col1, …]`
        •  True → return a dict  { label_fmt.format(i): col_i }
    label_fmt : str
        Format string for dict keys when ``return_dict=True``.
        Example: f"pt{i}"  or  f"{pt_edges[i]}–{pt_edges[i+1]} GeV".

    Returns
    -------
    list OR dict
        Each item/entry is a 1-D numpy array whose length equals
        len(gen_mass_edges_by_pt[i]) − 1.  The order matches the original
        pT slices.
    """
    import numpy as np

    chunks = []
    offset = 0
    for edges in gen_mass_edges_by_pt:
        n = len(edges) - 1
        chunk = np.asarray(flat_array[offset:offset + n], dtype=float)
        chunks.append(chunk)
        offset += n

    if offset != len(flat_array):
        raise ValueError("flat_array length does not match edge specification")

    if return_dict:
        return {label_fmt.format(i): col for i, col in enumerate(chunks)}

    return chunks
# ------------------------------------------------------------------
# main routine
def merge_mass_flat(h2d,
                         mass_edges_reco,
                         reco_mass_edges_by_pt):
    """
    Parameters
    ----------
    h2d : ndarray (nRecoMassFine, nRecoPt)
    mass_edges_reco : fine reco-mass edges
    reco_mass_edges_by_pt : list (len = nRecoPt) of edge arrays,
        one per pT column (coarser edges must be subsets).

    Returns
    -------
    flattened : 1-D ndarray
        Column-by-column concatenation of the merged histogram.
        Length  = sum_i (len(edges_i) - 1)
    """
    n_pt = h2d.shape[1]
    assert len(reco_mass_edges_by_pt) == n_pt, "need one edge list per pT column"

    cols_flat = []
    for i in range(n_pt):
        coarse_col = _merge_1d(h2d[:, i], mass_edges_reco, reco_mass_edges_by_pt[i])
        cols_flat.append(coarse_col)

    return np.concatenate(cols_flat)      # 1-D result

    



def show_integer_grid(img2d, title=None):
    """
    img2d: your h_flat_2d (2D numpy array). NaNs are fine.
    Draws:
      • grid lines at every integer bin edge
      • a colorbar with integer ticks (0,1,2,3,...)
    """
    nrows, ncols = img2d.shape

    # Integer range for the colorbar (ignore NaNs)
    vmin = int(np.floor(np.nanmin(img2d)))
    vmax = int(np.ceil(np.nanmax(img2d)))

    # Discrete normalization so each integer has its own color band
    boundaries = np.arange(vmin - 0.5, vmax + 1.5, 1)
    norm = colors.BoundaryNorm(boundaries, ncolors=plt.get_cmap("viridis").N, clip=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(img2d, origin="lower", aspect="auto", cmap="viridis", norm=norm)
    if title:
        ax.set_title(title)

    # Grid lines at every integer bin boundary (between pixels)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", linestyle="--", linewidth=0.5, color="k", alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Major ticks at integer bin centers
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)

    # # Colorbar with integer ticks
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_ticks(np.arange(vmin, vmax + 1, 1))
    # cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    return fig, ax


import numpy as np
import itertools

# ---- 1) Auto-detect axis order so we really rebin mass along mass axes ----
def reorder_to_expected(H, mass_edges_reco, pt_edges, mass_edges_gen):
    expected = [
        len(mass_edges_reco) - 1,  # reco_mass
        len(pt_edges)       - 1,   # reco_pt
        len(mass_edges_gen) - 1,   # gen_mass
        len(pt_edges)       - 1,   # gen_pt
    ]
    for perm in itertools.permutations(range(4)):
        if [H.shape[p] for p in perm] == expected:
            return np.transpose(H, perm), perm
    raise ValueError(
        f"Could not match H.shape={H.shape} to expected {expected}. "
        "Check that you're passing the correct edges and that flow bins are excluded."
    )
    
def reorder_to_expected_2d(H2, mass_edges_reco, pt_edges):
    """
    Make sure `H2` is ordered [reco_mass, reco_pt].

    Parameters
    ----------
    H2 : ndarray
        2-D histogram whose axes might be swapped.
    mass_edges_reco : sequence
        Full fine reco-mass bin edges (length = nRecoMass + 1).
    pt_edges : sequence
        Reco-pT bin edges (length = nRecoPt + 1).

    Returns
    -------
    H2_reordered : ndarray
        Histogram with shape (nRecoMass, nRecoPt), axes = [reco_mass, reco_pt].
    permutation  : tuple
        Axis permutation applied to get there
        – `(0,1)` means no change
        – `(1,0)` means the original array was transposed.
    """
    expected = [
        len(mass_edges_reco) - 1,   # reco_mass
        len(pt_edges)       - 1,    # reco_pt
    ]

    for perm in itertools.permutations(range(2)):        # (0,1) and (1,0)
        if [H2.shape[p] for p in perm] == expected:
            return np.transpose(H2, perm), perm

    raise ValueError(
        f"Could not match H2.shape={H2.shape} to expected {tuple(expected)}. "
        "Check that the edges really describe this array and that flow bins are excluded."
    )
# ---- 2) Rebin helpers (subset merge) ----
def _check_edges_subset(old_edges, new_edges, atol=1e-9):
    old = np.asarray(old_edges, float)
    new = np.asarray(new_edges, float)
    ok = np.isclose(new[:, None], old[None, :], atol=atol).any(axis=1)
    if not np.all(ok):
        raise ValueError(f"New edges not subset of old: {new[~ok]}")
    pos = np.searchsorted(old, new)
    for i, e in enumerate(new):
        m = np.isclose(old, e, atol=atol)
        if m.any():
            pos[i] = np.flatnonzero(m)[0]
    return pos

def rebin_along_axis(arr, old_edges, new_edges, axis):
    pos = _check_edges_subset(old_edges, new_edges)
    a = np.moveaxis(arr, axis, 0)
    out = np.stack([a[s:e].sum(axis=0) for s, e in zip(pos[:-1], pos[1:])], axis=0)
    return np.moveaxis(out, 0, axis)

def mosaic_no_padding(H, mass_edges_reco, mass_edges_gen,
                      reco_mass_edges_by_pt, gen_mass_edges_by_pt):
    rows, blocks = [], {}
    for i, reco_edges_i in enumerate(reco_mass_edges_by_pt):
        row_blocks = []
        for j, gen_edges_j in enumerate(gen_mass_edges_by_pt):
            block = H[:, i, :, j]  # (reco_mass, gen_mass)
            block = rebin_along_axis(block, mass_edges_reco, reco_edges_i, axis=0)
            block = rebin_along_axis(block, mass_edges_gen,  gen_edges_j,  axis=1)
            row_blocks.append(block)
            blocks[(i, j)] = block
        rows.append(np.hstack(row_blocks))
    return np.vstack(rows), blocks