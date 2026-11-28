"""Orchestration. Test/forecast require a provenance-checked frozen selection."""
import gc
import importlib.util
import platform
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds
import scipy
import statsmodels

from .baselines import predict_baseline
from .common import digest, fingerprint, qcol, read_json, repair_quantiles, require, write_json
from .contracts import check_daily, check_period, seal, verify_run
from .features import aggregate_features, aggregate_history, period_features, period_training_origins
from .learners import fit_predict


class Inputs:
    def __init__(self, data, source_scope, sample_size=None):
        self.root = Path(data)
        self.source_scope = source_scope
        self.manifest = read_json(self.root / 'split_manifest.json')
        self.contract = read_json(self.root / 'feature_list.json')
        self.sales = pd.read_parquet(self.root / f'sales_{source_scope}.parquet')
        if sample_size is not None:
            self.sales = self.sales.sort_values('series_id').head(sample_size).reset_index(drop=True)
        self.ids = self.sales.series_id.tolist()
        require(len(self.ids) > 0 and len(set(self.ids)) == len(self.ids), 'Invalid leaf selection')
        self.mapping = pd.read_parquet(self.root / f'hierarchy_map_{source_scope}.parquet')
        self.mapping = self.mapping[self.mapping.series_id.isin(self.ids)].reset_index(drop=True)
        self.calendar = pd.read_parquet(self.root / 'calendar.parquet')
        self.origins = list(range(1353, 1437, 28)) + self.manifest['protocol']['training_origins']
        self.extra_cache = {}
        self.splits = self.manifest['splits']
        self.leaf_nodes = self.mapping[self.mapping.level.eq(12)].set_index('series_id').node_id
        self.feature_ds = ds.dataset(self.root / source_scope / 'features', format='parquet')
        self.label_ds = ds.dataset(self.root / source_scope / 'targets', format='parquet')

    def signature(self):
        # Frozen artifact manifest fingerprints all input data, then per-read checks verify files.
        return {'source_scope': self.source_scope, 'series_ids_sha256': fingerprint(self.ids),
                'split_manifest_sha256': digest(self.root / 'split_manifest.json'),
                'feature_list_sha256': digest(self.root / 'feature_list.json'),
                'artifact_manifest_sha256': digest(self.root / 'artifact_manifest.json'),
                'source_features_sha256': digest(self.root.parents[1] / '01_data.py')}

    def verify_inputs(self):
        expected = read_json(self.root / 'artifact_manifest.json')
        needed = ['split_manifest.json', 'feature_list.json', 'calendar.parquet',
                  f'sales_{self.source_scope}.parquet', f'hierarchy_map_{self.source_scope}.parquet']
        needed += [f'{self.source_scope}/{kind}/origin_{o}.parquet'
                   for kind in ['features', 'targets']
                   for o in self.manifest['protocol']['training_origins']]
        needed.append('prices.parquet')
        for relative in needed:
            # Member 1 creates Windows relative paths with backslashes on Windows.
            record = expected.get(relative, expected.get(relative.replace('/', '\\')))
            require(record is not None, f'Input not registered: {relative}')
            path = self.root / relative
            require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                    f'Input changed since Member 1 verification: {relative}')

    def leaf_frame(self, split, train):
        filt = ds.field('series_id').isin(self.ids)
        if train:
            filt = filt & (ds.field('origin_d') < split['origin_d']) & (
                ds.field('target_d') >= split['train_start_d']) & (
                ds.field('target_d') <= split['train_end_d'])
        else:
            filt = filt & (ds.field('origin_d') == split['origin_d'])
        columns = list(dict.fromkeys(self.contract['keys'] + self.contract['predictors']))
        features = self.feature_ds.to_table(columns=columns, filter=filt).to_pandas()
        if train:
            labels = self.label_ds.to_table(filter=filt).to_pandas()
            features = features.merge(labels, on=self.contract['keys'], validate='one_to_one')
            extra = []
            for origin in self.origins:
                if origin >= 1437 or origin >= split['origin_d'] or origin + 28 < split['train_start_d']:
                    continue
                part = self.extra_leaf_snapshot(origin)
                extra.append(part[part.target_d.between(split['train_start_d'], split['train_end_d'])])
            if extra:
                features = pd.concat([features] + extra, ignore_index=True)
            require(len(features) == len(self.ids) * 365 and features.units.notna().all(),
                    'Training labels must cover 365 days per leaf')
        require(not features.duplicated(self.contract['keys']).any(), 'Duplicate feature key')
        features['node_id'] = features.series_id.map(self.leaf_nodes)
        require(features.node_id.notna().all(), 'Unknown leaf node')
        return features

    def extra_leaf_snapshot(self, origin):
        """Extend snapshots backwards for calibration, using unchanged Member 1 feature code."""
        if origin in self.extra_cache:
            return self.extra_cache[origin]
        source = self.root.parents[1] / '01_data.py'
        spec = importlib.util.spec_from_file_location('m5_source_features', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(self, 'price_matrix'):
            prices = ds.dataset(self.root / 'prices.parquet').to_table(
                columns=['series_id', 'wm_yr_wk', 'sell_price'],
                filter=ds.field('series_id').isin(self.ids)).to_pandas()
            weeks = self.calendar.groupby('wm_yr_wk', sort=False).d_num.agg(['min', 'max'])
            self.price_matrix = prices.pivot(index='series_id', columns='wm_yr_wk', values='sell_price').reindex(
                index=self.ids, columns=weeks.index).to_numpy(dtype='float32')
            self.week_ends = (weeks['min'] + 6).to_numpy(dtype='int16')
        y = self.sales[[f'd_{d}' for d in range(1, 1942)]].to_numpy(dtype='int16')
        frame = module.build_features(self.sales, y, self.calendar, self.price_matrix, self.week_ends, origin)
        frame['units'] = y[np.repeat(np.arange(len(self.ids)), 28), frame.target_d.to_numpy() - 1]
        self.extra_cache[origin] = frame
        return frame


def make_split(origin, name=None):
    return {'origin_d': origin, 'split_id': name or f'calibration_{origin}',
            'train_start_d': origin - 364, 'train_end_d': origin,
            'target_start_d': origin + 1, 'target_end_d': origin + 28,
            'labels_available': origin + 28 <= 1941}


def export_splits(mode, splits, calibration_origins):
    """Fixed-config backtesting exports folds; it never invokes parameter search."""
    if mode == 'backtest':
        selected = [s for s in splits if s['split_id'] in
                    {'validation_1', 'validation_2', 'validation_3'}]
        require(len(selected) == 3, 'Backtest requires exactly three validation folds')
        return selected
    if mode == 'calibrate':
        return [make_split(o) for o in calibration_origins]
    selected = [s for s in splits if s['split_id'] == mode]
    require(len(selected) == 1, 'Unknown export mode or missing split')
    return selected


def aggregate_frames(inputs, nodes, y, split):
    parts = []
    for origin in inputs.origins:
        if origin >= split['origin_d']:
            continue
        frame = aggregate_features(nodes, y, inputs.calendar, origin)
        mask = frame.target_d.between(split['train_start_d'], split['train_end_d'])
        frame = frame.loc[mask].copy()
        if not len(frame):
            continue
        node_index = pd.Index(nodes.node_id).get_indexer(frame.node_id)
        frame['units'] = y[node_index, frame.target_d.to_numpy() - 1]
        parts.append(frame)
    train = pd.concat(parts, ignore_index=True)
    require(len(train) == len(nodes) * 365, 'Aggregate training window not covered')
    pred = aggregate_features(nodes, y, inputs.calendar, split['origin_d'])
    predictors = [c for c in pred if c not in {'origin_d', 'target_d', 'series_id'}]
    return train, pred, predictors


def train_daily(train, predict, predictors, config, selection, model_dir):
    point = fit_predict(train[predictors], train.units, predict[predictors], config,
                        selection['point'], 'poisson', model_dir / 'point')
    raw_quantiles = []
    for tau in config['daily_quantiles']:
        raw_quantiles.append(fit_predict(
            train[predictors], train.units, predict[predictors], config,
            selection['quantile'], 'quantile', model_dir / qcol(tau), alpha=tau))
    raw_quantiles = np.column_stack(raw_quantiles)
    quantiles, repairs = repair_quantiles(raw_quantiles)
    repairs['negative_point_values'] = int((point < 0).sum())
    return point, raw_quantiles, quantiles, repairs


def daily_scores(frame, y, nodes, split, taus):
    if split['origin_d'] + 28 > y.shape[1]:
        return []
    index = pd.Index(nodes.node_id)
    scale = np.abs(y[:, split['train_start_d'] - 1 + 7:split['origin_d']] -
                   y[:, split['train_start_d'] - 1:split['origin_d'] - 7]).mean(axis=1)
    scores = []
    for model, part in frame.groupby('model', sort=True):
        idx = index.get_indexer(part.node_id)
        actual = y[idx, part.target_d.to_numpy() - 1]
        error = np.abs(actual - part.yhat.to_numpy())
        eligible = scale[idx] > 0
        score = {'split_id': split['split_id'], 'origin_d': split['origin_d'],
                 'model': model, 'rows': len(part), 'actual_sum': float(actual.sum()),
                 'absolute_error_sum': float(error.sum()),
                 'wape': float(error.sum() / actual.sum()) if actual.sum() > 0 else None,
                 'mase': float(np.mean(error[eligible] / scale[idx][eligible])) if eligible.any() else None,
                 'mase_undefined_rows': int((~eligible).sum())}
        for tau in taus:
            if model == 'lightgbm':
                residual = actual - part[qcol(tau)].to_numpy()
                score[f'pinball_{tau}'] = float(np.maximum(tau * residual, (tau - 1) * residual).mean())
        scores.append(score)
    return scores


def period_predictions(inputs, nodes, y, split, config, selection, model_dir, metadata):
    parts, labels = [], []
    for origin in period_training_origins(inputs.origins, split):
        parts.append(period_features(nodes, y, inputs.calendar, origin))
        labels.append(y[:, origin:origin + 28].sum(axis=1))
    require(len(parts) > 0, 'No complete historical period labels')
    train = pd.concat(parts, ignore_index=True)
    target = np.concatenate(labels)
    predict = period_features(nodes, y, inputs.calendar, split['origin_d'])
    predictors = [c for c in predict if c not in {'series_id', 'origin_d'}]
    raw = []
    for tau in config['period_quantiles']:
        raw.append(fit_predict(train[predictors], target, predict[predictors], config,
                               selection['quantile'], 'quantile',
                               model_dir / f'period_{tau:.8f}', alpha=tau))
    raw = np.column_stack(raw)
    fixed, repair = repair_quantiles(raw)
    frame = pd.DataFrame({'node_id': np.repeat(nodes.node_id.to_numpy(), len(raw[0])),
                          'tau': np.tile(config['period_quantiles'], len(nodes)),
                          'raw_forecast': raw.ravel(), 'forecast': fixed.ravel(),
                          'period_start_d': split['origin_d'] + 1,
                          'period_end_d': split['origin_d'] + 28,
                          'model': 'lightgbm', 'reconciliation': 'unreconciled',
                          'distribution_method': 'direct_28day_quantile'})
    for key, value in metadata.items():
        frame[key] = value
    check_period(frame, nodes.node_id, split['origin_d'], config['period_quantiles'])
    return frame, repair


def run_group(inputs, split, level, config, selection, folder, context):
    started = time.monotonic()
    destination = folder / 'predictions' / split['split_id'] / f'L{level:02d}'
    destination.mkdir(parents=True, exist_ok=False)
    models = folder / 'models' / split['split_id'] / f'L{level:02d}'
    nodes, y = aggregate_history(inputs.sales, inputs.mapping, level)
    if level == 12:
        train, pred = inputs.leaf_frame(split, True), inputs.leaf_frame(split, False)
        predictors = inputs.contract['predictors']
    else:
        train, pred, predictors = aggregate_frames(inputs, nodes, y, split)
    point, qraw, quantiles, repairs = train_daily(train, pred, predictors, config, selection, models)
    metadata = {'run_id': folder.name, 'schema_version': config['schema_version'],
                'scope': context['scope'], 'source_scope': inputs.source_scope,
                'split_id': split['split_id'], 'level': level, 'origin_d': split['origin_d'],
                'config_hash': context['config_hash']}
    ml = pred[['node_id', 'origin_d', 'target_d', 'horizon']].copy()
    ml['series_id'] = pred.series_id.to_numpy() if level == 12 else None
    ml['model'], ml['method_used'], ml['status'] = 'lightgbm', 'lightgbm', 'ok'
    ml['fallback_reason'], ml['quantile_status'] = '', 'provided'
    ml['yhat'], ml['raw_yhat'] = np.maximum(point, 0), point
    for i, tau in enumerate(config['daily_quantiles']):
        ml[qcol(tau)], ml[f'raw_{qcol(tau)}'] = quantiles[:, i], qraw[:, i]
    outputs, events = [ml], []
    base = nodes.iloc[np.repeat(np.arange(len(nodes)), 28)].reset_index(drop=True)
    if 'series_id' not in base:
        base['series_id'] = None
    base['origin_d'] = split['origin_d']
    base['horizon'] = np.tile(np.arange(1, 29), len(nodes))
    base['target_d'] = split['origin_d'] + base.horizon
    for name in ['seasonal_naive', 'ets', 'tsb']:
        batch, methods, statuses, reasons = [], [], [], []
        for i, node in enumerate(nodes.node_id):
            result = predict_baseline(name, y[i, split['train_start_d'] - 1:split['origin_d']], config['tsb'])
            batch.extend(result.pop('prediction'))
            methods.extend([result['method_used']] * 28)
            statuses.extend([result['status']] * 28)
            reasons.extend([result['fallback_reason']] * 28)
            events.append(dict(result, node_id=node, model=name))
        frame = base.copy()
        frame['model'], frame['method_used'], frame['status'] = name, methods, statuses
        frame['fallback_reason'], frame['quantile_status'] = reasons, 'not_provided'
        frame['raw_yhat'] = np.asarray(batch)
        frame['yhat'] = np.maximum(frame.raw_yhat, 0)
        for tau in config['daily_quantiles']:
            frame[qcol(tau)] = np.nan
            frame[f'raw_{qcol(tau)}'] = np.nan
        outputs.append(frame)
    daily = pd.concat(outputs, ignore_index=True)
    for key, value in metadata.items():
        daily[key] = value
    check_daily(daily, nodes.node_id, split['origin_d'], config['daily_quantiles'])
    daily.to_parquet(destination / 'daily.parquet', index=False, compression='zstd')
    period, period_repairs = period_predictions(inputs, nodes, y, split, config, selection, models, metadata)
    period.to_parquet(destination / 'period.parquet', index=False, compression='zstd')
    scores = daily_scores(daily, y, nodes, split, config['daily_quantiles'])
    for score in scores:
        score['level'] = level
    write_json(destination / 'metrics.json', scores)
    if split['origin_d'] + 28 <= 1941:
        idx = pd.Index(nodes.node_id).get_indexer(daily.node_id)
        residuals = daily[['run_id', 'scope', 'source_scope', 'level', 'node_id', 'origin_d',
                           'target_d', 'horizon', 'model', 'config_hash', 'yhat']].copy()
        residuals['y_true'] = y[idx, daily.target_d.to_numpy() - 1]
        residuals['residual'] = residuals.y_true - residuals.yhat
        residuals['label_available_d'] = residuals.target_d
        residuals['role'] = 'historical_calibration' if split['split_id'].startswith('calibration_') else split['split_id']
        residuals['configuration_role'] = context['configuration_role']
        residuals.to_parquet(destination / 'residuals.parquet', index=False, compression='zstd')
    write_json(destination / 'baseline_events.json', events)
    write_json(destination / 'coverage.json', {'node_ids': nodes.node_id.tolist(),
                                             'leaf_count': len(inputs.ids)})
    write_json(destination / 'diagnostics.json', {
        'daily_quantile_repairs': repairs, 'period_quantile_repairs': period_repairs,
        'negative_baseline_values': int(((daily.model != 'lightgbm') & (daily.raw_yhat < 0)).sum()),
        'fallback_series': sum(event['status'] == 'fallback' for event in events),
        'train_rows': len(train), 'predictors': predictors,
        'elapsed_seconds': round(time.monotonic() - started, 3)})
    print(f"{split['split_id']} L{level:02d}: {len(nodes)} nodes, {len(train):,} train rows, "
          f"{time.monotonic()-started:.1f}s", flush=True)
    del train, pred, daily, period, y
    gc.collect()


def tune(inputs, config, folder):
    require(inputs.source_scope == 'development', 'Tune only on development scope')
    validation = [s for s in inputs.splits if s['split_id'].startswith('validation_')]
    require(len(validation) == 3, 'Three validation folds required')
    records = []
    for number, candidate in enumerate(config['candidates']):
        absolute_error = actual_sum = pinball_sum = rows = 0.0
        folds = []
        for split in validation:
            train, pred = inputs.leaf_frame(split, True), inputs.leaf_frame(split, False)
            point, _, quantiles, _ = train_daily(
                train, pred, inputs.contract['predictors'], config,
                {'point': candidate, 'quantile': candidate},
                folder / 'tuning_models' / f'candidate_{number}' / split['split_id'])
            labels = inputs.label_ds.to_table(filter=(ds.field('origin_d') == split['origin_d']) &
                                              ds.field('series_id').isin(inputs.ids)).to_pandas()
            truth = pred[inputs.contract['keys']].merge(labels, on=inputs.contract['keys'],
                                                       validate='one_to_one').units.to_numpy()
            require(np.isfinite(truth).all() and len(truth) == len(pred), 'Invalid validation labels')
            absolute_error += float(np.abs(truth - np.maximum(point, 0)).sum())
            actual_sum += float(truth.sum())
            for j, tau in enumerate(config['daily_quantiles']):
                error = truth - quantiles[:, j]
                pinball_sum += float(np.maximum(tau * error, (tau - 1) * error).sum())
            rows += len(truth)
            folds.append({'split_id': split['split_id'], 'rows': len(truth),
                          'absolute_error_sum': float(np.abs(truth - np.maximum(point, 0)).sum()),
                          'actual_sum': float(truth.sum())})
            print(f"tuning candidate {number+1}/{len(config['candidates'])} {split['split_id']} complete", flush=True)
            del train, pred
            gc.collect()
        records.append({'candidate': candidate, 'point_absolute_error': absolute_error,
                        'actual_sum': actual_sum, 'wape': absolute_error / actual_sum if actual_sum else None,
                        'mean_pinball': pinball_sum / (rows * len(config['daily_quantiles'])),
                        'folds': folds})
    point = min(records, key=lambda r: (r['point_absolute_error'], r['candidate']['num_leaves'], r['candidate']['rounds']))
    quantile = min(records, key=lambda r: (r['mean_pinball'], r['candidate']['num_leaves'], r['candidate']['rounds']))
    write_json(folder / 'tuning_results.json', records)
    return {'point': point['candidate'], 'quantile': quantile['candidate']}


def execute(args):
    config = read_json(args.config)
    require(config['daily_quantiles'] == [0.5, 0.9, 0.95, 0.99], 'Daily quantile contract changed')
    require(config['period_quantiles'] == sorted(set(config['period_quantiles'])), 'Period tau must be unique and sorted')
    require(all(0 < q < 1 for q in config['period_quantiles']), 'Invalid period tau')
    require(set(config['levels']) <= set(range(1, 13)) and len(config['levels']) == len(set(config['levels']))
            and config['levels'], 'Invalid/duplicate levels')
    inputs = Inputs(args.data, args.scope, config['smoke']['size'] if args.mode == 'smoke' else None)
    inputs.verify_inputs()
    folder = Path(args.output).resolve()
    require(not folder.exists(), 'Output folder already exists; choose a new run directory')
    folder.mkdir(parents=True)
    scope = 'smoke' if args.mode == 'smoke' else args.scope
    signature = inputs.signature()
    started = time.monotonic()
    if args.mode == 'smoke':
        candidate = {'num_leaves': config['smoke']['num_leaves'], 'rounds': config['smoke']['rounds']}
        selection = {'point': candidate, 'quantile': candidate}
        splits = inputs.splits + [make_split(o) for o in config['calibration_origins']]
    elif args.mode == 'validate':
        require(args.scope == 'development', 'Validation selection must use development scope')
        selection = tune(inputs, config, folder)
        splits = [s for s in inputs.splits if s['split_id'].startswith('validation_')]
    elif args.mode == 'calibrate' and args.selection is None:
        selection = {'point': config['candidates'][0], 'quantile': config['candidates'][0]}
        splits = [make_split(o) for o in config['calibration_origins']]
    else:
        require(args.selection is not None, 'Test/forecast/calibrate requires --selection')
        lock = read_json(args.selection)
        supplied_hash = lock.pop('selection_hash')
        require(fingerprint(lock) == supplied_hash, 'Frozen selection checksum mismatch')
        require(lock['scope'] == 'development' and lock['mode'] == 'validate', 'Smoke config cannot authorize test/forecast')
        verify_run(Path(args.selection).parent)
        require(lock['config_hash'] == fingerprint(config), 'Config changed after validation lock')
        # Full scope has different selected leaves, but input split/features/version must match.
        for key in ['split_manifest_sha256', 'feature_list_sha256', 'artifact_manifest_sha256', 'source_features_sha256']:
            require(lock['input_signature'][key] == signature[key], f'Frozen input mismatch: {key}')
        selection = lock['selected']
        splits = export_splits(args.mode, inputs.splits, config['calibration_origins'])
    lock = {'mode': args.mode, 'scope': scope, 'selected': selection,
            'config_hash': fingerprint(config), 'input_signature': signature,
            'validation_ids': ['validation_1', 'validation_2', 'validation_3'] if args.mode == 'validate' else []}
    lock['selection_hash'] = fingerprint(lock)
    write_json(folder / 'selected_config.json', lock)
    context = {'mode': args.mode, 'scope': scope, 'config': config, 'config_hash': fingerprint(config),
               'selected': selection, 'input_signature': signature, 'series_ids': inputs.ids,
               'configuration_role': ('predeclared' if args.mode == 'smoke' or
                                      (args.mode == 'calibrate' and args.selection is None)
                                      else 'selected_using_all_three_validation_windows'),
               'requested_splits': [s['split_id'] for s in splits],
               'selection_policy': '3-fold development selection; test/forecast never select parameters',
               'validation_outputs_policy': 'Selected-config validation outputs are retrospective; use independent test for unbiased scoring',
               'baseline_quantile_policy': 'unavailable; Member 4 calibrates from historical out-of-sample errors',
               'residual_policy': 'Filter label_available_d <= evaluation_origin; configuration selected later is not a forward validation claim',
               'period_policy': 'direct_28day_quantile; unreconciled only; never sum daily quantiles',
               'environment': {'python': platform.python_version(), 'system': platform.platform(),
                               'numpy': np.__version__, 'pandas': pd.__version__,
                               'pyarrow': pyarrow.__version__, 'lightgbm': lightgbm.__version__,
                               'statsmodels': statsmodels.__version__, 'scipy': scipy.__version__}}
    context['split_definitions'] = splits
    code_root = Path(__file__).resolve().parents[1]
    context['model_source_sha256'] = {
        p.relative_to(code_root).as_posix(): digest(p)
        for p in sorted((code_root / 'modeling').glob('*.py'))}
    context['model_source_sha256']['02_models.py'] = digest(code_root / '02_models.py')
    write_json(folder / 'run_context.json', context)
    for split in splits:
        for level in config['levels']:
            run_group(inputs, split, level, config, selection, folder, context)
    write_json(folder / 'run_summary.json', {'status': 'completed', 'mode': args.mode,
                                           'scope': scope, 'leaf_count': len(inputs.ids),
                                           'elapsed_seconds': round(time.monotonic() - started, 3)})
    seal(folder)
    result = verify_run(folder)
    write_json(folder / 'verification_report.json', result)
    seal(folder)
    print(result, flush=True)
