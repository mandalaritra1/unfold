
import numpy as np

def build_matrices_and_vectors(output_pythia, output_data, groomed, pt_edges,
                               mass_edges_reco, mass_edges_gen):
    from unfold_utils.integrate_and_rebin import rebin_hist
    from unfold_utils.merge_helpers import reorder_to_expected, reorder_to_expected_2d

    if groomed:
        resp_key = 'response_matrix_g'
        data_key = 'ptjet_mjet_g_reco'
    else:
        resp_key = 'response_matrix_u'
        data_key = 'ptjet_mjet_u_reco'

    resp_matrix_4d = output_pythia[resp_key]
    input_data     = output_data[data_key]

    resp_matrix_4d = rebin_hist(resp_matrix_4d, 'ptreco', pt_edges)
    resp_matrix_4d = rebin_hist(resp_matrix_4d, 'ptgen',  pt_edges)
    resp_matrix_4d = rebin_hist(resp_matrix_4d, 'mreco',  mass_edges_reco)
    resp_matrix_4d = rebin_hist(resp_matrix_4d, 'mgen',   mass_edges_gen)
    input_data     = rebin_hist(input_data,     'mreco',  mass_edges_reco)
    input_data     = rebin_hist(input_data,     'ptreco', pt_edges)

    resp_matrix_4d = resp_matrix_4d[{'systematic':'nominal'}]

    proj   = resp_matrix_4d.project('ptreco', 'mreco', 'ptgen', 'mgen')
    H      = proj.values(flow=False)  # may be permuted
    H2d    = input_data.project('ptreco', 'mreco').values()

    H, perm_used  = reorder_to_expected(H,  mass_edges_reco, pt_edges, mass_edges_gen)
    H2d, perm2    = reorder_to_expected_2d(H2d, mass_edges_reco, pt_edges)

    return H, H2d, perm_used, perm2

def make_mosaic_and_vectors(H, H2d, mass_edges_reco, mass_edges_gen,
                            reco_mass_edges_by_pt, gen_mass_edges_by_pt,
                            closure=False):
    from unfold_utils.merge_helpers import mosaic_no_padding, merge_mass_flat

    mosaic, blocks = mosaic_no_padding(
        H, mass_edges_reco, mass_edges_gen,
        reco_mass_edges_by_pt, gen_mass_edges_by_pt
    )
    reco_flat = merge_mass_flat(H2d, mass_edges_reco, reco_mass_edges_by_pt)

    if closure:
        meas_flat = mosaic.sum(axis=1)
    else:
        meas_flat = reco_flat

    true_flat = mosaic.sum(axis=0)
    return mosaic, meas_flat, true_flat

def run_tunfold(mosaic, meas_flat):
    import ROOT
    n_reco, n_true = mosaic.shape
    assert len(meas_flat) == n_reco

    h_resp = ROOT.TH2D("resp", "response;truth bin;reco bin",
                       n_true,  0, n_true,
                       n_reco,  0, n_reco)
    for i_reco in range(n_reco):
        for j_true in range(n_true):
            h_resp.SetBinContent(j_true + 1, i_reco + 1, float(mosaic[i_reco, j_true]))

    h_meas = ROOT.TH1D("meas", "measured;reco bin;entries", n_reco, 0, n_reco)
    for i, val in enumerate(meas_flat, 1):
        h_meas.SetBinContent(i, float(val))

    unfold = ROOT.TUnfoldDensity(h_resp, ROOT.TUnfold.kHistMapOutputHoriz)
    status = unfold.SetInput(h_meas)
    if status >= 10000:
        raise RuntimeError("TUnfold input had overflow/underflow – check your hist.")

    g1 = ROOT.TGraph(); g2 = ROOT.TGraph(); g3 = ROOT.TGraph()
    unfold.ScanSURE(30, 1e-6, 1e0, g1, g2, g3)

    h_unfold = unfold.GetOutput("unfold")
    return h_resp, h_meas, h_unfold

def unflatten_by_pt(arr_flat, gen_mass_edges_by_pt):
    from unfold_utils.merge_helpers import unflatten_gen_by_pt
    return unflatten_gen_by_pt(arr_flat, gen_mass_edges_by_pt)
