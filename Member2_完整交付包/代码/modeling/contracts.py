"""Independent output checks, including completeness and finite predictions."""
import numpy as np
import pandas as pd

from .common import digest, read_json, require, qcol, write_json

MODELS = ['seasonal_naive', 'ets', 'tsb', 'lightgbm']


def check_daily(frame, node_ids, origin, quantiles):
    keys = ['model', 'node_id', 'origin_d', 'target_d']
    require(not frame.duplicated(keys).any(), 'Duplicate daily primary key')
    require(set(frame.model) == set(MODELS), 'Model coverage differs')
    expected = pd.MultiIndex.from_product([sorted(node_ids), range(origin + 1, origin + 29)])
    for model, group in frame.groupby('model'):
        observed = pd.MultiIndex.from_frame(group[['node_id', 'target_d']]).sort_values()
        require(observed.equals(expected), f'Incomplete 28-day node coverage: {model}')
    require(frame.origin_d.eq(origin).all() and
            frame.horizon.eq(frame.target_d - origin).all(), 'Daily origin/horizon mismatch')
    require(np.isfinite(frame.yhat).all() and frame.yhat.ge(0).all(), 'Invalid point predictions')
    q = frame.loc[frame.model.eq('lightgbm'), [qcol(t) for t in quantiles]].to_numpy()
    require(np.isfinite(q).all() and (q >= 0).all() and (np.diff(q, axis=1) >= 0).all(),
            'Invalid/non-monotone quantile predictions')
    require(frame.loc[~frame.model.eq('lightgbm'), [qcol(t) for t in quantiles]].isna().all().all(),
            'Baseline quantiles must explicitly remain unavailable')


def check_period(frame, node_ids, origin, quantiles):
    require(not frame.duplicated(['node_id', 'tau']).any(), 'Duplicate period quantile')
    require(set(frame.node_id) == set(node_ids) and len(frame) == len(node_ids) * len(quantiles),
            'Incomplete period predictions')
    require(frame.origin_d.eq(origin).all() and frame.period_start_d.eq(origin + 1).all()
            and frame.period_end_d.eq(origin + 28).all(), 'Period origin mismatch')
    wide = frame.pivot(index='node_id', columns='tau', values='forecast').sort_index(axis=1)
    require(np.allclose(wide.columns.to_numpy(), quantiles), 'Period tau coverage mismatch')
    values = wide.to_numpy()
    require(np.isfinite(values).all() and (values >= 0).all() and (np.diff(values, axis=1) >= 0).all(),
            'Invalid/non-monotone period quantiles')


def seal(folder):
    files = {p.relative_to(folder).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'checksums.json'}
    write_json(folder / 'checksums.json', files)


def verify_run(folder):
    context = read_json(folder / 'run_context.json')
    checksums = read_json(folder / 'checksums.json')
    for relative, record in checksums.items():
        path = folder / relative
        require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                f'Output checksum mismatch: {relative}')
    daily_count = period_count = 0
    for path in sorted(folder.glob('predictions/*/L*/daily.parquet')):
        frame = pd.read_parquet(path)
        node_ids = read_json(path.parent / 'coverage.json')['node_ids']
        origin = int(frame.origin_d.iloc[0])
        check_daily(frame, node_ids, origin, context['config']['daily_quantiles'])
        check_period(pd.read_parquet(path.parent / 'period.parquet'), node_ids, origin,
                     context['config']['period_quantiles'])
        residual_path = path.parent / 'residuals.parquet'
        if origin + 28 <= 1941:
            require(residual_path.exists(), 'Missing known-label residuals')
            residuals = pd.read_parquet(residual_path)
            require(len(residuals) == len(frame), 'Residual coverage differs from predictions')
            joined = frame[['model', 'node_id', 'target_d', 'yhat']].merge(
                residuals, on=['model', 'node_id', 'target_d'], validate='one_to_one', suffixes=('_p', '_r'))
            require(len(joined) == len(frame) and np.isfinite(joined.y_true).all()
                    and joined.y_true.ge(0).all(), 'Invalid residual labels')
            require(np.allclose(joined.yhat_p, joined.yhat_r) and
                    np.allclose(joined.residual, joined.y_true - joined.yhat_r) and
                    joined.label_available_d.eq(joined.target_d).all(), 'Residual alignment/sign mismatch')
        else:
            require(not residual_path.exists(), 'Unknown future labels must not have residuals')
        daily_count += len(frame)
        period_count += len(pd.read_parquet(path.parent / 'period.parquet', columns=['node_id']))
    require(daily_count > 0, 'No forecast files to verify')
    expected_groups = len(context['requested_splits']) * len(context['config']['levels'])
    require(len(list(folder.glob('predictions/*/L*/daily.parquet'))) == expected_groups,
            'Some requested split/level groups are missing')
    return {'status': 'passed', 'files_checked': len(checksums),
            'daily_rows': daily_count, 'period_rows': period_count,
            'scope': context['scope'], 'requested_groups': expected_groups}
