"""Rebuild three historical figures from original code and verified public data.

These are new renders, not byte-identical replacements for missing cloud PNGs.
The authoritative historical numeric results remain in their original CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('results/rebuilt_archive_figures/.matplotlib').resolve()))

import numpy as np
import pandas as pd
import matplotlib

from scripts.analysis.run_v3_5_local_recovery_spatial_explanation import make_patch_explanation_figure
from scripts.analysis.run_v3_6_published_csd_benchmark import make_patch_figure
from scripts.analysis.summarize_v3_6_salt_marsh_precision_audit import make_figure
from src.real_data.published_csd_benchmarks import load_rindi_grids, compute_rindi_metrics
from src.real_data.recovery_spatial_models import (
    RecoveryModelConfig, block_recovery_field, merge_recovery_and_patch,
    out_of_fold_stress_predictions,
)
from src.real_data.tidal_marsh import (
    SITES, aggregate_recovery_units, load_tidal_marsh_site, summarize_patch_context,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--marsh-root', type=Path, default=Path('data/raw/tidal_marsh'))
    parser.add_argument('--rindi-root', type=Path, default=Path('data/raw/csd_benchmark/rindi_2018'))
    parser.add_argument('--summaries', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/rebuilt_archive_figures'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source_files = list(Path('src/real_data').glob('*.py')) + [
        Path('scripts/analysis/rebuild_archive_figures.py'),
        Path('scripts/analysis/run_v3_5_local_recovery_spatial_explanation.py'),
        Path('scripts/analysis/run_v3_6_published_csd_benchmark.py'),
        Path('scripts/analysis/summarize_v3_6_salt_marsh_precision_audit.py'),
    ]
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, matplotlib=matplotlib.__version__)
    inputs = {str(p): sha(p) for p in sorted(args.marsh_root.rglob('*')) if p.is_file()}
    inputs.update({str(p): sha(p) for p in sorted(args.rindi_root.glob('Exp_data_*.txt'))})
    provenance = dict(input_sha256=inputs, code_sha256={str(p): sha(p) for p in source_files}, versions=versions)
    cache = out/'fields_provenance.json'
    reuse = cache.exists() and json.loads(cache.read_text()) == provenance
    fields = {}
    for site in SITES:
        path = out/f'field_{site}_64m.csv'
        if reuse and path.exists():
            fields[site] = pd.read_csv(path)
        else:
            print(f'Rebuilding {site} 64 m field...', flush=True)
            data = load_tidal_marsh_site(args.marsh_root, site)
            units, _, _ = aggregate_recovery_units(data, [64.0])
            patch = summarize_patch_context(data, block_width_m=64.0, patch_resolution_m=1.0)
            rows = merge_recovery_and_patch(units[64.0], patch, min_losses=50, min_recoveries=10)
            predicted = out_of_fold_stress_predictions(rows, RecoveryModelConfig(alpha=0.1))
            fields[site] = block_recovery_field(rows, predicted)
            fields[site].to_csv(path, index=False)
            del data
    cache.write_text(json.dumps(provenance, indent=2))
    grids = load_rindi_grids(args.rindi_root)
    grid_metrics = compute_rindi_metrics(grids)
    make_patch_figure(grids, grid_metrics, out/'macroalgal_patch_explanation.png')
    if args.prepare_only:
        print('Fields and macroalgal figure prepared.', flush=True)
        return
    v35 = args.summaries/'v3_5_local_recovery_spatial_explanation'
    associations = pd.read_csv(v35/'patch_residual_associations.csv')
    associations = associations.loc[associations.block_width_m.eq(64.0)]
    make_patch_explanation_figure(fields, associations, out/'patch_spatial_explanation_64m.png')
    v36 = args.summaries/'v3_6_published_csd_benchmark'
    csv_paths = [v36/f'salt_marsh_{name}_audit.csv' for name in
                 ['resolution', 'regularization', 'scale', 'candidate_scale_regularization']]
    make_figure(*(pd.read_csv(p) for p in csv_paths), out/'salt_marsh_precision_audit.png')
    provenance['historical_result_sha256'] = {str(p): sha(p) for p in
                                             [v35/'patch_residual_associations.csv', *csv_paths]}
    provenance['outputs'] = {p.name: sha(p) for p in sorted(out.glob('*.png'))}
    provenance['status'] = 'regenerated_figures_not_original_png_bytes'
    (out/'rebuild_manifest.json').write_text(json.dumps(provenance, indent=2))
    print('Rebuilt three figures; original numeric summaries were not modified.', flush=True)


if __name__ == '__main__':
    main()
