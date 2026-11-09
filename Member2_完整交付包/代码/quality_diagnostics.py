"""Descriptive quantile diagnostics. Never chooses parameters from test results."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def inspect(run, data, output):
    context = json.loads((run / 'run_context.json').read_text(encoding='utf-8'))
    scope = context['input_signature']['source_scope']
    records = []
    constants = []
    mapping = pd.read_parquet(data / f'hierarchy_map_{scope}.parquet')
    for split in context['requested_splits']:
        files = sorted((run / 'predictions' / split).glob('L*/daily.parquet'))
        for path in files:
            frame = pd.read_parquet(path)
            frame = frame[frame.model.eq('lightgbm')]
            origin = int(frame.origin_d.iloc[0])
            level = int(path.parent.name[1:])
            if origin + 28 > 1941:
                continue
            days = [f'd_{d}' for d in range(origin + 1, origin + 29)]
            sales = pd.read_parquet(data / f'sales_{scope}.parquet', columns=['series_id'] + days)
            sales = sales[sales.series_id.isin(context['series_ids'])]
            part = mapping[mapping.level.eq(level)][['series_id', 'node_id']]
            actual = part.merge(sales, on='series_id', validate='one_to_one').groupby('node_id')[days].sum()
            ids = actual.index.get_indexer(frame.node_id)
            if (ids < 0).any():
                raise ValueError('Prediction node absent from observed targets')
            truth = actual.to_numpy()[ids, frame.target_d.to_numpy() - origin - 1]
            for tau, col in zip([.5, .9, .95, .99], ['q_050', 'q_090', 'q_095', 'q_099']):
                q = frame[col].to_numpy()
                error = truth - q
                records.append({'split': split, 'level': level, 'kind': 'daily', 'tau': tau,
                                'rows': len(q), 'below_strict': float(np.mean(truth < q)),
                                'at_or_below': float(np.mean(truth <= q)),
                                'pinball': float(np.maximum(tau * error, (tau - 1) * error).mean())})
            period = pd.read_parquet(path.parent / 'period.parquet')
            totals = actual.sum(axis=1)
            for tau, group in period.groupby('tau', sort=True):
                y = totals.loc[group.node_id].to_numpy()
                q = group.forecast.to_numpy()
                error = y - q
                records.append({'split': split, 'level': level, 'kind': 'period', 'tau': float(tau),
                                'rows': len(q), 'below_strict': float(np.mean(y < q)),
                                'at_or_below': float(np.mean(y <= q)),
                                'pinball': float(np.maximum(tau * error, (tau - 1) * error).mean())})
        for path in sorted((run / 'models' / split).glob('L*/period_*/training.json')):
            info = json.loads(path.read_text(encoding='utf-8'))
            model = lgb.Booster(model_file=str(path.parent / 'model.txt'))
            first = model.dump_model(num_iteration=1)['tree_info'][0]['num_leaves']
            if model.num_trees() == 1 and first == 1:
                constants.append({'split': split, 'level': int(path.parent.parent.name[1:]),
                                  'model': path.parent.name, 'train_rows': info['train_rows'],
                                  'classification': 'constant_training_quantile'})
    result = {'source_run': str(run), 'scope': context['scope'],
              'configuration_role': context['configuration_role'], 'records': records,
              'constant_period_models': constants,
              'interpretation': ['Descriptive assessment only; this script never selects or updates models.',
                'Discrete demand: strict and weak coverage bracket ties; exact equality to tau is not required.',
                'Rows and nodes are dependent; no independent-sample confidence interval or significance claim.',
                'Validation exports after selection are retrospective; do not label them independent evidence.',
                'Constant models may be valid benchmarks but do not establish conditional or tail calibration.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'diagnostic_records': len(records), 'constant_period_models': len(constants)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.run, args.data, args.output)
