"""Exploratory v3.7 spline comparison on the already frozen QIR pilot cohort.

Reuses the historical estimator, adapting its trials to equal block weights.
It is not a rerun of the original full-resolution v3.7 analysis.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.real_data.time_safe_survival import SurvivalModelConfig, fit_grouped_cloglog
from .events import compile_events, split_source_blocks
from .experiment import load_series, buffered_coordinates
from .validation import block_weights, fit_calibration_offset, probability, metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear_hazard(model, frame):
    x = model.preprocessing.transform(frame[['inundation_fraction', 'log_elapsed_years_start']])
    return np.column_stack([np.ones(len(x)), x]) @ model.result.params


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('results_summary/qir_phase1/marsh'))
    parser.add_argument('--output', type=Path, default=Path('results/qir_historical_baseline'))
    args = parser.parse_args()
    manifest = json.loads((args.pilot/'run.json').read_text())
    cfg, seed = manifest['config'], manifest['seed']
    for name, expected in manifest['code_sha256'].items():
        path = Path(name.replace('\\', '/'))
        if sha(path) != expected:
            raise ValueError(f'Original cohort code/config has changed: {path}')
    datasets = []
    for name, expected in manifest['input_sha256'].items():
        path = Path(name.replace('\\', '/'))
        if sha(path) != expected:
            raise ValueError(f'Pilot input hash mismatch: {path}')
        datasets.append(load_series(path))
    frames, ledgers = [], []
    for data in datasets:
        coords = buffered_coordinates(data, cfg['block_size_m'], max(cfg['scales_m']), cfg['pixels_per_block'], seed)
        rows, ledger = compile_events(data, coords, cfg['block_size_m'])
        frames.append(rows); ledgers.append(ledger)
    rows = pd.concat(frames, ignore_index=True)
    if len(rows) != manifest['n_intervals'] or sum(map(len, ledgers)) != manifest['n_episodes']:
        raise ValueError('Pilot cohort size mismatch')
    prior = pd.read_csv(args.pilot/'predictions.csv')
    grouped = rows.assign(inundation_fraction=rows.environment,
                          log_elapsed_years_start=np.log1p(rows.waiting_time),
                          interval_years=rows.delta_t)
    config = SurvivalModelConfig()
    scores, predictions, coefficients, fits = [], [], [], []
    for recorded in manifest['splits']:
        source, target = recorded['source'], recorded['target']
        split = split_source_blocks(rows, source, target, seed=seed)
        actual = {k: sorted(rows.iloc[v].block_id.unique()) for k, v in split.items()}
        if actual != recorded['split_blocks']:
            raise ValueError('Frozen block partition mismatch')
        tr, cal, te = (split[k] for k in ['train', 'calibration', 'test'])
        key = ['event_id', 'block_id', 'start_time', 'end_time', 'event']
        old = prior.loc[prior.source.eq(source) & prior.target.eq(target) & prior['mode'].eq('baseline'), key]
        expected = rows.iloc[te][key]
        pd.testing.assert_frame_equal(old.sort_values(key).reset_index(drop=True),
                                      expected.sort_values(key).reset_index(drop=True), check_dtype=False)
        train = grouped.iloc[tr].copy()
        train['n_at_risk'] = block_weights(train.block_id)*len(train)
        train['n_recovered'] = train.event*train.n_at_risk
        model = fit_grouped_cloglog(train, config=config)
        if not model.result.converged:
            raise RuntimeError(f'Historical spline did not converge: {source}')
        eta_cal = linear_hazard(model, grouped.iloc[cal])
        offset = fit_calibration_offset(eta_cal, rows.delta_t.to_numpy()[cal],
                                        rows.event.to_numpy()[cal], rows.block_id.to_numpy()[cal])
        eta = linear_hazard(model, grouped.iloc[te])
        test = rows.iloc[te]
        for calibrated in [False, True]:
            p = probability(eta+(offset if calibrated else 0.), test.delta_t.to_numpy())
            scores.append(dict(source=source, target=target, mode='historical_spline',
                               calibrated=calibrated, calibration_offset=offset if calibrated else 0.,
                               **metrics(test.event.to_numpy(), p, test.block_id.to_numpy())))
        pred = test[key].copy()
        pred['source'] = source; pred['target'] = target
        pred['probability'] = probability(eta+offset, test.delta_t.to_numpy())
        predictions.append(pred)
        coefficients.append(model.coefficients().assign(source=source, target=target))
        fits.append(dict(source=source, target=target, converged=model.result.converged,
                         iterations=model.result.iterations, calibration_offset=offset,
                         calibration_at_search_boundary=bool(abs(offset) > 4.999),
                         train_pressure_range=[float(train.inundation_fraction.min()), float(train.inundation_fraction.max())],
                         calibration_pressure_range=[float(grouped.iloc[cal].inundation_fraction.min()), float(grouped.iloc[cal].inundation_fraction.max())],
                         target_pressure_range=[float(grouped.iloc[te].inundation_fraction.min()), float(grouped.iloc[te].inundation_fraction.max())],
                         split_blocks=actual))
    previous = pd.read_csv(args.pilot/'metrics.csv')
    previous = previous.loc[previous.subset.isna() & previous.degradation.isna()].copy()
    result = pd.concat([previous, pd.DataFrame(scores)], ignore_index=True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out/'comparison.csv', index=False)
    pd.concat(predictions).to_csv(out/'predictions_spline.csv', index=False)
    pd.concat(coefficients).to_csv(out/'coefficients.csv', index=False)
    meta = dict(stage='exploratory_followup_after_viewing_original_pilot', config=asdict(config), fits=fits,
                frozen_pilot_manifest_sha256=sha(args.pilot/'run.json'), input_sha256=manifest['input_sha256'],
                code_sha256={str(p): sha(p) for p in [Path('src/qir_event/compare_historical_baseline.py'), Path('src/real_data/time_safe_survival.py')]},
                n_intervals=len(rows), n_episodes=sum(map(len, ledgers)),
                adaptation='same v3.7 spline family; fractional trials implement equal total block weights',
                limitations=['single fixed pilot and spline configuration', 'retrospective covariate availability',
                             'QIR residual still uses its neural M0; not yet trained on the spline baseline',
                             'no new independent confirmatory cohort or second system'])
    (out/'run.json').write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == '__main__':
    main()
