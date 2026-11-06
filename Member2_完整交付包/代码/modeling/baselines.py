"""365-day local baselines; no future values accepted by the interface."""
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .common import require


def seasonal_naive(history, horizon=28, season=7):
    history = np.asarray(history, dtype='float64')
    require(len(history) >= season, 'Insufficient history for seasonal naive')
    return np.resize(history[-season:], horizon)


def tsb(history, horizon=28, alpha=0.1, beta=0.1):
    """Initialize at first sale; p=1/(first-sale index+1); update remaining days."""
    history = np.asarray(history, dtype='float64')
    require(0 < alpha <= 1 and 0 < beta <= 1, 'TSB smoothing outside (0,1]')
    positive = np.flatnonzero(history > 0)
    if not len(positive):
        return np.zeros(horizon)
    first = positive[0]
    size, probability = history[first], 1.0 / (first + 1)
    for value in history[first + 1:]:
        probability += beta * (float(value > 0) - probability)
        if value > 0:
            size += alpha * (value - size)
    return np.full(horizon, probability * size)


def ets(history, horizon=28):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = ExponentialSmoothing(np.asarray(history, dtype='float64'), trend=None,
                                   seasonal='add', seasonal_periods=7,
                                   initialization_method='estimated').fit(
                                       optimized=True, use_brute=False)
    if not fit.mle_retvals.get('success', True):
        raise RuntimeError('ETS optimizer did not converge')
    return np.asarray(fit.forecast(horizon)), [str(w.message) for w in caught]


def predict_baseline(name, history, tsb_params=None):
    history = np.asarray(history, dtype='float64')
    require(history.ndim == 1 and len(history) >= 7 and np.isfinite(history).all()
            and (history >= 0).all(), 'Invalid baseline history')
    result = {'method_used': name, 'status': 'ok', 'fallback_reason': '', 'warnings': []}
    if not history.any():
        result['method_used'] = 'zero_history'
        result['prediction'] = np.zeros(28)
        return result
    try:
        if name == 'seasonal_naive':
            prediction = seasonal_naive(history)
        elif name == 'tsb':
            prediction = tsb(history, **(tsb_params or {}))
        elif name == 'ets':
            prediction, result['warnings'] = ets(history)
        else:
            raise ValueError(f'Unknown baseline: {name}')
        require(prediction.shape == (28,) and np.isfinite(prediction).all(), 'Invalid prediction')
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        if name not in {'ets', 'tsb'}:
            raise
        prediction = seasonal_naive(history)
        result.update(method_used='seasonal_naive', status='fallback',
                      fallback_reason=f'{type(exc).__name__}: {exc}')
    result['prediction'] = prediction
    return result
