#!/usr/bin/env python3
"""Compare normalized Vincia statistics using saved Run-2 arrays, without unfolding."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
import numpy as np

from unfold.tools.prediction_statistics import normalized_prediction_covariance


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=Path(
        'outputs/pairsplit_run2/PAIR_SPLIT_GROOMED_UNGROOMED_PLOT_BOOK_2026-08-27.inventory.json'))
    parser.add_argument('--measurement-directory', type=Path, default=Path(
        'outputs/pairsplit_run2/model_covariance_comparison_2026-09-05_v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new comparison output directory')
    args.output.mkdir(parents=True)
    inventory = json.loads(args.inventory.read_text())
    report = {
        'method': 'J diag(sumw2) J^T with the stored normalization window',
        'scope': 'Vincia saved arrays; measurement uses accepted enclosing model covariance',
        'limitation': 'No event-level or cross-pT covariance is recoverable from saved sumw2.',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(path): file_hash(path) for path in (
            Path(__file__), Path('src/unfold/tools/prediction_statistics.py'))},
        'runs': [],
    }
    hep.style.use(hep.style.CMS)
    with PdfPages(args.output/'comparison.pdf') as book:
        for run in inventory['runs']:
            if file_hash(run['artifact']) != run['artifact_sha256']:
                raise ValueError(f"Canonical artifact hash mismatch: {run['artifact']}")
            arrays = np.load(run['artifact'])
            channel, mode = run['channel'], run['grooming_mode']
            stem = f'{channel}_{mode}'
            measurement_path = args.measurement_directory/f'{stem}.npz'
            measurement = np.load(measurement_path)
            np.testing.assert_array_equal(arrays['normalized_result'], measurement['normalized_result'])
            measurement_covariance = measurement['new_total_covariance']
            offsets = arrays['mess_vincia_bin_offsets']
            edge_offsets = arrays['mess_vincia_gen_edge_offsets']
            pt_edges = arrays['pt_edges']
            blocks = []
            summary = {'channel': channel, 'grooming': mode, 'artifact': run['artifact'],
                       'artifact_sha256': run['artifact_sha256'],
                       'measurement': str(measurement_path),
                       'measurement_sha256': file_hash(measurement_path), 'slices': []}
            for i in range(len(offsets)-1):
                sl = slice(offsets[i], offsets[i+1])
                edges = arrays['mess_vincia_gen_edges_flat'][edge_offsets[i]:edge_offsets[i+1]]
                widths = np.diff(edges)
                counts = arrays['mess_vincia_sumw_flat'][sl]
                variance = arrays['mess_vincia_sumw2_flat'][sl]
                mask = arrays['mess_vincia_normalization_masks_flat'][sl].astype(bool)
                density = counts / widths / counts[mask].sum()
                np.testing.assert_allclose(density, arrays['mess_vincia_density_flat'][sl], rtol=1e-13)
                old_error = np.sqrt(variance) / widths / counts[mask].sum()
                np.testing.assert_allclose(old_error, arrays['mess_vincia_stat_unc_flat'][sl], rtol=1e-13)
                covariance = normalized_prediction_covariance(counts, np.diag(variance), widths, mask)
                new_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
                np.testing.assert_allclose(covariance @ (widths*mask), 0, atol=1e-12)
                assert np.linalg.eigvalsh(covariance).min() > -1e-12
                blocks.append(covariance)
                reported = arrays['unfolded_reported_window_mask'][sl].astype(bool)
                indices = np.arange(offsets[i], offsets[i+1])[reported]
                residual = arrays['normalized_result'][indices] - density[reported]
                measurement_block = measurement_covariance[np.ix_(indices, indices)]
                chi2 = []
                for pred_cov in (np.diag(old_error**2), covariance):
                    total = measurement_block + pred_cov[np.ix_(reported, reported)]
                    chi2.append(float(residual @ np.linalg.pinv(total, rcond=1e-10) @ residual))
                ndof = max(int(reported.sum())-1, 1)
                error_ratio = np.divide(new_error, old_error, out=np.ones_like(new_error), where=old_error>0)
                summary['slices'].append({'pt_low_GeV': float(pt_edges[i]), 'ndof': ndof,
                    'chi2_before': chi2[0], 'chi2_after': chi2[1],
                    'reported_error_ratio_min': float(error_ratio[reported].min()),
                    'reported_error_ratio_max': float(error_ratio[reported].max())})
                # Only overlay the two uncertainty curves: comparison helpers
                # do not supply the saved-covariance chi2 annotations needed here.
                fig, ax = plt.subplots(layout='constrained')
                fig.get_layout_engine().set(rect=(0,.035,1,.965))
                visible = (edges[:-1] >= (-3.5 if mode == 'groomed' else -2.5))
                start = np.flatnonzero(visible)[0]
                fractions = [np.divide(error, np.abs(density), out=np.full_like(error, np.nan),
                                       where=density!=0)[start:] for error in (old_error, new_error)]
                for values, color, label, style in zip(fractions, ('#5790fc', '#e42536'),
                        ('Before: fixed normalization', 'After: normalization Jacobian'), ('--', '-')):
                    hep.histplot(values, edges[start:], yerr=False, histtype='step',
                                 ax=ax, color=color, label=label, ls=style, lw=3)
                highest = max(np.nanmax(values) for values in fractions)
                ax.set(xlabel=rf'$2\log_{{10}}\rho$, {mode}', ylabel='Fractional MC statistical uncertainty',
                       ylim=(0, highest*1.7), xlim=(edges[start], edges[-1]))
                high_label = f'{pt_edges[i+1]:g}' if i+1<len(pt_edges)-1 else r'$\infty$'
                ax.legend(loc='upper left', bbox_to_anchor=(.02,.9), fontsize=20,
                          title=f'MESS+Vincia, {pt_edges[i]:g}–{high_label} GeV', title_fontsize=22)
                ax.text(.04,.70, rf'$\chi^2$ ({ndof} dof): {chi2[0]:.2f} $\to$ {chi2[1]:.2f}',
                        transform=ax.transAxes, fontsize=22)
                hep.cms.label('Internal', data=True, loc=0, ax=ax, rlabel=channel.capitalize())
                fig.text(.99,.006,'2026-09-08 | unfold | saved Run-2 arrays; central values unchanged',
                         ha='right', fontsize=10)
                name = f'{stem}_pt{i}'
                fig.savefig(args.output/f'{name}.png', dpi=110)
                fig.savefig(args.output/f'{name}.pdf')
                book.savefig(fig)
                plt.close(fig)
            full_covariance = np.zeros_like(measurement_covariance)
            for i, block in enumerate(blocks):
                full_covariance[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]] = block
            np.savez_compressed(args.output/f'{stem}.npz',
                density=arrays['mess_vincia_density_flat'],
                old_stat_error=arrays['mess_vincia_stat_unc_flat'],
                new_stat_error=np.sqrt(np.clip(np.diag(full_covariance), 0, None)),
                new_stat_covariance=full_covariance)
            report['runs'].append(summary)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['runs'], indent=2))


if __name__ == '__main__':
    main()
