#!/usr/bin/env python3
"""Recompute model covariances from saved normalized unfolds; never rerun TUnfold.

The comparison keeps central spectra, all stored variations and non-model
uncertainties fixed. Outputs are separate from the canonical production files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_uncertainty_state(arrays, grooming_mode, method):
    """Use production uncertainty methods on already unfolded/normalized arrays."""
    obj = Unfolder.__new__(Unfolder)
    obj._uses_prepared_inputs = True
    obj.groomed = grooming_mode == 'groomed'
    obj.pt_edges = arrays['pt_edges'].copy()
    edges = arrays['two_log10_rho_gen_edges']
    obj.gen_edges_by_pt = [edges.copy() for _ in obj.pt_edges[:-1]]
    npt, nbins = len(obj.pt_edges) - 1, len(edges) - 1
    mask = arrays['unfolded_reported_window_mask'].reshape(npt, nbins)
    obj._shown_gen_mask = lambda i: mask[i]
    obj.spec = SimpleNamespace(
        model_envelope=True, model_envelope_source='prepared_systematics',
        model_covariance_method=method,
        model_covariance_scope='global_templates' if method == 'enclosing_ellipsoid' else 'global_shown',
        bl_shown_floors_groomed=None, bl_shown_floors_ungroomed=None,
        xlim_lower_groomed=-3.5, xlim_lower_ungroomed=-2.5,
    )
    obj.stat_propagation = 'jacobian'
    obj.y_unf = arrays['unfolded'].copy()
    obj.systematics = tuple(str(x) for x in arrays['systematic_names'])
    obj.normalized_results = [{'unfolded': row.copy()} for row in arrays['normalized_result'].reshape(npt, nbins)]
    variations = arrays['systematic_normalized'].reshape(len(obj.systematics), npt, nbins)
    obj.normalized_systematics = [
        {'unfolded': {name: variations[j, i].copy() for j, name in enumerate(obj.systematics)}}
        for i in range(npt)]
    obj.norm_cov_input = arrays['norm_cov_input'].copy()
    obj.norm_cov_matrix = arrays['norm_cov_matrix'].copy()
    obj.norm_cov_stat = arrays['norm_cov_stat'].copy()
    # These are the saved Jacobian-propagated statistics, not a new estimate.
    obj._compute_normalized_stat_covariance = lambda: None
    obj._compute_total_systematic()
    return obj


def grouped_values(obj, i):
    return obj._group_syst_fraction_dict(obj._build_syst_fraction_dict(i), grouped=True)


def save_figure(fig, path, book):
    fig.savefig(path.with_suffix('.png'), dpi=100)
    fig.savefig(path.with_suffix('.pdf'))
    book.savefig(fig)
    plt.close(fig)


def stamp(fig, description):
    fig.supxlabel('2026-09-05 | unfold | saved Run-2 pair-split arrays | '+description, fontsize=12)


def plot_breakdown(old, new, i, channel, mode, path, book):
    before, after = grouped_values(old, i), grouped_values(new, i)
    edges = old.gen_edges_by_pt[i]
    visible = edges[:-1] >= (-3.5 if mode == 'groomed' else -2.5)
    ymax = 1.85 * max(np.nanmax(v['Total_Up'][visible]) for v in (before, after))
    fig, axes = plt.subplots(1, 2, figsize=(21.2, 10.6), layout='constrained', sharey=True)
    colors = [('Jet EnergyUp', 'Jet energy', '#5790fc', '-'),
              ('Jet MassUp', 'Jet mass', '#f89c20', '-'),
              ('Other TheoryUp', 'Other theory', '#9c9ca1', '-'),
              ('Shower ModelUp', 'Shower model', '#e42536', '-.'),
              ('Hadronization ModelUp', 'Hadronization model', '#7a21dd', '--'),
              ('Stat Unc', 'Stat unc.', 'black', ':')]
    hi = f'{old.pt_edges[i+1]:g}' if i+1 < len(old.pt_edges)-1 else r'$\infty$'
    for ax, values, label in zip(axes, (before, after), ('Before: binwise envelopes', 'After: enclosing-template covariance')):
        hep.histplot(values['Total_Up'], edges, histtype='fill', facecolor='0.88', edgecolor='black', lw=2, ax=ax, label='Total')
        for key, name, color, ls in colors:
            if key in values:
                hep.histplot(values[key], edges, histtype='step', yerr=False, ax=ax, label=name, color=color, ls=ls, lw=2.5)
        ax.set(xlim=(edges[:-1][visible][0], edges[-1]), ylim=(0, ymax),
               xlabel=rf'$2\log_{{10}}\rho$, {mode}', ylabel='Fractional uncertainty')
        ax.text(.04, .97, label, transform=ax.transAxes, va='top', fontsize=20)
        ax.legend(loc='upper left', bbox_to_anchor=(.02,.9), ncol=2, fontsize=18,
                  title=rf'{channel.capitalize()}, $p_T$ {old.pt_edges[i]:g}–{hi} GeV', title_fontsize=20)
        hep.cms.label('Internal', data=True, loc=0, ax=ax, rlabel=r'138 fb$^{-1}$ (13 TeV)', fontsize=24)
    stamp(fig, 'same central values and non-model bands')
    save_figure(fig, path, book)


def plot_matrices(old_matrix, new_matrix, nominal, mask, channel, mode, kind, path, book):
    idx = np.flatnonzero(mask)
    matrices = [matrix[np.ix_(idx, idx)] for matrix in (old_matrix, new_matrix)]
    if kind == 'correlation':
        converted = []
        for matrix in matrices:
            sigma = np.sqrt(np.clip(np.diag(matrix),0,None))
            denom = np.outer(sigma,sigma)
            converted.append(np.divide(matrix,denom,out=np.zeros_like(matrix),where=denom!=0))
        matrices = converted; bound=1.; cbarlabel='Model correlation'
    else:
        scale = np.outer(np.abs(nominal[idx]), np.abs(nominal[idx]))
        matrices = [np.divide(matrix,scale,out=np.zeros_like(matrix),where=scale!=0) for matrix in matrices]
        bound = max(np.max(np.abs(matrix)) for matrix in matrices)
        cbarlabel=r'Model covariance / $|y_i y_j|$'
    fig, axes = plt.subplots(1,2,figsize=(21.2,10.6),layout='constrained')
    edges=np.arange(len(idx)+1)
    for ax,matrix,label in zip(axes,matrices,('Before: selected FSR/model directions','After: both enclosing-template groups')):
        result=hep.hist2dplot(matrix,edges,edges,ax=ax,cmap='RdBu_r',cmin=-bound,cmax=bound,cbar=False)
        ax.set(xlabel=f'Reported gen bin\n{label}',ylabel='Reported gen bin',aspect='equal')
        hep.cms.label('Internal',data=True,loc=0,ax=ax,rlabel=f'{channel.capitalize()}, {mode}',fontsize=23)
    colorbar=fig.colorbar(result.pcolormesh,ax=axes,pad=.02,shrink=.85)
    colorbar.set_label(cbarlabel,fontsize=18)
    colorbar.ax.tick_params(labelsize=15)
    stamp(fig, 'model component only; identical color scale')
    save_figure(fig,path,book)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=Path('outputs/pairsplit_run2/PAIR_SPLIT_GROOMED_UNGROOMED_PLOT_BOOK_2026-08-27.inventory.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise FileExistsError('Completed comparison already exists; choose a new output directory')
    inventory=json.loads(args.inventory.read_text())
    report={'method':'origin-centered minimum-volume enclosing templates per group',
            'interpretation':'Template containment within each group, then independent-group covariance addition. Not a calibrated confidence region.',
            'inventory':str(args.inventory.resolve()),'inventory_sha256':file_hash(args.inventory),
            'nominal_and_nonmodel_unchanged':True,'runs':[]}
    root=Path(__file__).resolve().parents[2]
    report['source_sha256']={str(p.relative_to(root)):file_hash(p) for p in [Path(__file__).resolve(),root/'src/unfold/tools/model_covariance.py',root/'src/unfold/tools/unfolder_core.py']}
    report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    gallery=['# Model covariance comparison','',report['interpretation'],'']
    hep.style.use(hep.style.CMS)
    with PdfPages(args.output/'comparison.pdf') as book:
        for run in inventory['runs']:
            path=Path(run['artifact'])
            if file_hash(path)!=run['artifact_sha256']:
                raise ValueError(f'Canonical artifact hash mismatch: {path}')
            a=dict(np.load(path))
            channel,mode=run['channel'],run['grooming_mode'];stem=f'{channel}_{mode}'
            old=restore_uncertainty_state(a,mode,'selected_variation')
            new=restore_uncertainty_state(a,mode,'enclosing_ellipsoid')
            nominal=a['normalized_result'];mask=a['unfolded_reported_window_mask']
            # Verify the before panel against the canonical stored envelope.
            for key,source in [('model_ps_fraction','model_ps_frac'),('model_had_fraction','model_had_frac')]:
                np.testing.assert_allclose(np.concatenate([getattr(old,source)[i] for i in range(len(old.pt_edges)-1)]),a[key],rtol=1e-10,atol=1e-13)
            old_model=old._normalized_model_covariance(nominal)
            new_model=new._normalized_model_covariance(nominal)
            old_total=old.get_total_covariance();new_total=new.get_total_covariance()
            np.testing.assert_allclose(old_total,a['norm_cov_total'],rtol=1e-8,atol=1e-12)
            np.testing.assert_allclose(new_total-new_model,old_total-old_model,rtol=1e-9,atol=1e-12)
            w=np.zeros((len(new.pt_edges)-1,len(nominal)))
            nb=len(new.gen_edges_by_pt[0])-1
            for i in range(len(w)):
                w[i,i*nb:(i+1)*nb]=np.diff(new.gen_edges_by_pt[i])*new._shown_gen_mask(i)
            np.testing.assert_allclose(w@new_model,0,atol=1e-12)
            assert np.linalg.eigvalsh(new_model).min()>-1e-12
            np.testing.assert_array_equal(np.concatenate([r['unfolded'] for r in new.normalized_results]),nominal)
            arrays={'normalized_result':nominal,'pt_edges':new.pt_edges,'rho_edges':new.gen_edges_by_pt[0],
                    'reported_mask':mask,'norm_cov_stat':new.norm_cov_stat,
                    'old_model_covariance':old_model,'new_model_covariance':new_model,
                    'old_total_covariance':old_total,'new_total_covariance':new_total,
                    'new_ps_covariance':new.model_group_covariances['parton_shower'],
                    'new_had_covariance':new.model_group_covariances['hadronization']}
            for label,obj in [('old',old),('new',new)]:
                for i in range(len(w)):
                    for key,val in grouped_values(obj,i).items():arrays[f'{label}_pt{i}_{key}']=val
            np.savez_compressed(args.output/(stem+'.npz'),**arrays)
            newfrac=np.sqrt(np.clip(np.diag(new_model),0,None))/abs(nominal)
            oldfrac=a['model_total_fraction']
            record={'channel':channel,'mode':mode,'source_artifact':str(path),'source_sha256':file_hash(path),
                    'groups':new.model_covariance_diagnostics,
                    'max_normalization_residual':float(np.max(abs(w@new_model))),
                    'min_eigenvalue':float(np.linalg.eigvalsh(new_model).min()),
                    'reported_model_fraction_before':oldfrac[mask].tolist(),
                    'reported_model_fraction_after':newfrac[mask].tolist()}
            report['runs'].append(record)
            gallery += [f'## {channel.capitalize()} {mode}','']
            for i in range(len(w)):
                name=f'{stem}_pt{i}_breakdown'
                plot_breakdown(old,new,i,channel,mode,args.output/name,book)
                gallery += [f'![pT slice {i}]({name}.png)','']
            for kind in ['covariance','correlation']:
                name=f'{stem}_{kind}'
                plot_matrices(old_model,new_model,nominal,mask,channel,mode,kind,args.output/name,book)
                gallery += [f'![Model {kind}]({name}.png)','']
            print(stem,'ranks',{k:v['rank'] for k,v in new.model_covariance_diagnostics.items()},
                  'max model fraction before/after',float(oldfrac[mask].max()),float(newfrac[mask].max()),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'README.md').write_text('\n'.join(gallery))


if __name__=='__main__':main()
