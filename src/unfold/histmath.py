"""Array helpers shared by the input loaders and the unfolding engine.

Everything here works on plain numpy arrays.  The one exception,
``rebin_hist``, rebins a ``hist.Hist`` object along one axis and is used by
the dijet/trijet input adapter.

Conventions
-----------
A "flat" spectrum is the concatenation of the per-pT-slice spectra, pT slice
by pT slice, each slice carrying its own (possibly different) bin edges.  A
"mosaic" is the response matrix in that flat layout: shape (n_reco, n_gen).
"""

from __future__ import annotations

import numpy as np


def _subset_positions(old, new, atol=1e-9):
    """Indices in ``old`` of every edge in ``new``; ``new`` must be a subset."""
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


def merge_1d(vec, old_edges, new_edges):
    """Sum the bins of ``vec`` (binned by ``old_edges``) into ``new_edges``."""
    pos = _subset_positions(old_edges, new_edges)
    return np.array([vec[s:e].sum() for s, e in zip(pos[:-1], pos[1:])])


def rebin_along_axis(arr, old_edges, new_edges, axis):
    """``merge_1d`` applied along one axis of an n-d array."""
    pos = _subset_positions(old_edges, new_edges)
    a = np.moveaxis(arr, axis, 0)
    out = np.stack([a[s:e].sum(axis=0) for s, e in zip(pos[:-1], pos[1:])], axis=0)
    return np.moveaxis(out, 0, axis)


def unflatten_gen_by_pt(flat_array, edges_by_pt, *, return_dict=False, label_fmt="pt{}"):
    """Split a flat spectrum back into one array per pT slice.

    ``edges_by_pt[i]`` is the edge list of slice ``i``; slice ``i`` therefore
    occupies ``len(edges_by_pt[i]) - 1`` consecutive entries of ``flat_array``.
    """
    chunks = []
    offset = 0
    for edges in edges_by_pt:
        n = len(edges) - 1
        chunks.append(np.asarray(flat_array[offset:offset + n], dtype=float))
        offset += n
    if offset != len(flat_array):
        raise ValueError("flat_array length does not match edge specification")
    if return_dict:
        return {label_fmt.format(i): col for i, col in enumerate(chunks)}
    return chunks


def flat_offsets(edges_by_pt):
    """(start index, bin count) of every pT slice in the flat layout."""
    counts = np.asarray([len(e) - 1 for e in edges_by_pt], dtype=int)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)
    return starts, counts


def merge_mass_flat(h2d, fine_edges, edges_by_pt):
    """Flatten a (fine observable bins, pT bins) array into the per-pT layout.

    Column ``i`` of ``h2d`` is merged from ``fine_edges`` onto
    ``edges_by_pt[i]`` and the merged columns are concatenated.
    """
    n_pt = h2d.shape[1]
    assert len(edges_by_pt) == n_pt, "need one edge list per pT column"
    return np.concatenate(
        [merge_1d(h2d[:, i], fine_edges, edges_by_pt[i]) for i in range(n_pt)]
    )


def reorder_to_expected(H, *_unused):
    """(gen_pt, gen_obs, reco_pt, reco_obs) -> (reco_obs, reco_pt, gen_obs, gen_pt).

    The producers always store the response with the gen axes first, so the
    reorder is a fixed transpose.  The extra positional arguments are accepted
    (and ignored) so old call sites that passed the edge lists keep working.
    """
    return np.transpose(H, (3, 2, 1, 0)), [3, 2, 1, 0]


def reorder_to_expected_2d(H2, obs_edges, pt_edges):
    """Return ``H2`` ordered as (observable, pT), transposing if needed."""
    expected = (len(obs_edges) - 1, len(pt_edges) - 1)
    if H2.shape == expected:
        return H2, (0, 1)
    if H2.shape == expected[::-1]:
        return np.transpose(H2), (1, 0)
    raise ValueError(
        f"Could not match H2.shape={H2.shape} to expected {expected}. "
        "Check that the edges really describe this array and that flow bins are excluded."
    )


def mosaic_no_padding(H, fine_reco_edges, fine_gen_edges, reco_edges_by_pt, gen_edges_by_pt):
    """Build the flat (n_reco, n_gen) response mosaic from a 4-d response.

    ``H`` has axes (reco_obs, reco_pt, gen_obs, gen_pt) on the fine edges.
    Each (reco pT slice i, gen pT slice j) block is merged onto that slice's
    analysis edges and the blocks are tiled.  Also returns the block dict.
    """
    rows, blocks = [], {}
    for i, reco_edges_i in enumerate(reco_edges_by_pt):
        row_blocks = []
        for j, gen_edges_j in enumerate(gen_edges_by_pt):
            block = H[:, i, :, j]
            block = rebin_along_axis(block, fine_reco_edges, reco_edges_i, axis=0)
            block = rebin_along_axis(block, fine_gen_edges, gen_edges_j, axis=1)
            row_blocks.append(block)
            blocks[(i, j)] = block
        rows.append(np.hstack(row_blocks))
    return np.vstack(rows), blocks


def reco_to_gen_rebin_1d(reco_vals, reco_edges, gen_edges):
    """Sum reco-binned values into gen bins (reco edges nest in gen edges).

    Works for counts and for variances (both add)."""
    reco_edges = np.asarray(reco_edges, float)
    gen_edges = np.asarray(gen_edges, float)
    reco_vals = np.asarray(reco_vals, float)
    out = np.zeros(len(gen_edges) - 1)
    for k in range(len(gen_edges) - 1):
        m = ((reco_edges[:-1] >= gen_edges[k] - 1e-9)
             & (reco_edges[1:] <= gen_edges[k + 1] + 1e-9))
        out[k] = reco_vals[m].sum()
    return out


def rebin_hist(h, axis_name, edges):
    """Rebin a ``hist.Hist`` along ``axis_name`` onto ``edges`` (a subset)."""
    import hist

    if isinstance(edges, int):
        return h[{axis_name: hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all(np.isclose(x, ax.edges).any() for x in edges):
        raise ValueError(
            f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}"
        )

    # If you rebin to a subset of the initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax

    storage = h.storage_type()
    hnew = hist.Hist(*axes, name=h.name, storage=storage)

    # Offset from the bin edge to avoid numeric issues
    offset = 0.5 * np.min(ax.edges[1:] - ax.edges[:-1])
    edge_idx = ax.index(np.asarray(edges) + offset)
    # Avoid going outside the range; reduceat adds the last index anyway
    if edge_idx[-1] == ax.size + ax.traits.overflow:
        edge_idx = edge_idx[:-1]
    if underflow:
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    take = range(new_ax.size + underflow + overflow)
    hnew.values(flow=flow)[...] = np.add.reduceat(
        h.values(flow=flow), edge_idx, axis=ax_idx).take(indices=take, axis=ax_idx)
    if isinstance(storage, hist.storage.Weight):
        hnew.variances(flow=flow)[...] = np.add.reduceat(
            h.variances(flow=flow), edge_idx, axis=ax_idx).take(indices=take, axis=ax_idx)
    return hnew
