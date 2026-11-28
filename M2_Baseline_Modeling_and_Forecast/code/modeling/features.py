"""Aggregate and period features built exclusively from origin-visible history."""
import numpy as np
import pandas as pd

from .common import require

IDENTIFIERS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
CALENDAR = ['wm_yr_wk', 'wday', 'month', 'year', 'event_name_1', 'event_type_1',
            'event_name_2', 'event_type_2', 'is_event']


def historical(y, origin):
    require(origin >= 90 and origin <= y.shape[1], 'History origin out of range')
    history = y[:, :origin]
    data = {f'sales_lag_{lag}': history[:, -lag].astype('float32')
            for lag in [1, 7, 14, 28, 56]}
    for window in [7, 28, 56, 90]:
        data[f'sales_mean_{window}'] = history[:, -window:].mean(axis=1).astype('float32')
        data[f'sales_std_{window}'] = history[:, -window:].std(axis=1).astype('float32')
    return pd.DataFrame(data)


def aggregate_features(nodes, y, calendar, origin):
    base = pd.concat([nodes.reset_index(drop=True), historical(y, origin)], axis=1)
    rows = base.iloc[np.repeat(np.arange(len(base)), 28)].reset_index(drop=True)
    rows['origin_d'] = np.int16(origin)
    rows['horizon'] = np.tile(np.arange(1, 29, dtype='int8'), len(base))
    rows['target_d'] = origin + rows.horizon.astype('int16')
    cal = calendar.set_index('d_num').loc[rows.target_d].reset_index(drop=True)
    for name in CALENDAR:
        if name == 'is_event':
            rows[name] = cal.event_name_1.notna().astype('int8')
        else:
            rows[name] = cal[name].fillna('None') if cal[name].dtype == object else cal[name]
    return rows


def period_features(nodes, y, calendar, origin):
    rows = pd.concat([nodes.reset_index(drop=True), historical(y, origin)], axis=1)
    cal = calendar.set_index('d_num').loc[origin + 1:origin + 28]
    require(len(cal) == 28, 'Incomplete known future calendar')
    rows['origin_d'] = np.int16(origin)
    rows['event_days_28'] = int(cal.event_name_1.notna().sum())
    rows['month_start'] = int(cal.month.iloc[0])
    rows['year_start'] = int(cal.year.iloc[0])
    for state in ['CA', 'TX', 'WI']:
        rows[f'snap_{state}_days_28'] = int(cal[f'snap_{state}'].sum())
    return rows


def period_training_origins(origins, split):
    # Entire 28-day block must lie within the declared training label window.
    return [o for o in origins if o < split['origin_d']
            and o + 1 >= split['train_start_d'] and o + 28 <= split['train_end_d']]


def aggregate_history(sales, mapping, level):
    part = mapping[mapping.level.eq(level)][['series_id', 'node_id']]
    require(len(part) == len(sales) and part.series_id.is_unique,
            'Hierarchy does not cover exactly the selected leaves')
    days = [f'd_{d}' for d in range(1, 1942)]
    merged = part.merge(sales[['series_id'] + days], on='series_id', validate='one_to_one')
    grouped = merged.groupby('node_id', sort=True)[days].sum()
    # int32 fits M5 sums; wider arithmetic is retained through aggregation.
    y = grouped.to_numpy(dtype='float64')
    nodes = pd.DataFrame({'node_id': grouped.index.to_numpy()})
    if level == 12:
        nodes['series_id'] = part.set_index('node_id').loc[nodes.node_id, 'series_id'].to_numpy()
    return nodes, y
