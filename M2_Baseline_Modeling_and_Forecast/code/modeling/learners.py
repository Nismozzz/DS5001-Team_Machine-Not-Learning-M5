import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import require, write_json


class Encoder:
    """Training-only category dictionary; unseen values become missing."""
    def fit(self, frame):
        self.columns = list(frame.columns)
        self.categories = {c: sorted(frame[c].dropna().astype(str).unique().tolist())
                           for c in frame if frame[c].dtype == object or isinstance(
                               frame[c].dtype, pd.CategoricalDtype)}
        return self

    def transform(self, frame):
        result = frame[self.columns].copy()
        for col in self.columns:
            if col in self.categories:
                result[col] = pd.Categorical(result[col], categories=self.categories[col])
            else:
                result[col] = pd.to_numeric(result[col], errors='raise').astype('float32')
        return result

    def save(self, path):
        write_json(path, {'columns': self.columns, 'categories': self.categories})


def fit_predict(train, labels, predict, config, candidate, objective, folder, alpha=None):
    labels = np.asarray(labels, dtype='float32')
    require(len(train) == len(labels) and len(train) > 0 and np.isfinite(labels).all()
            and (labels >= 0).all(), 'Invalid LightGBM labels')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    encoder = Encoder().fit(train)
    encoder.save(folder / 'encoder.json')
    if not labels.any():
        write_json(folder / 'constant_model.json', {'prediction': 0, 'reason': 'all_zero_training_labels'})
        return np.zeros(len(predict))
    x = encoder.transform(train)
    xp = encoder.transform(predict)
    params = dict(config['lightgbm'], objective=objective, seed=config['seed'],
                  num_threads=config['threads'], num_leaves=candidate['num_leaves'])
    if alpha is not None:
        params['alpha'] = alpha
    model = lgb.train(params, lgb.Dataset(x, label=labels),
                      num_boost_round=candidate['rounds'])
    prediction = np.asarray(model.predict(xp, num_threads=config['threads']))
    require(np.isfinite(prediction).all(), 'Nonfinite LightGBM output')
    model.save_model(str(folder / 'model.txt'))
    write_json(folder / 'training.json', {'params': params, 'rounds': candidate['rounds'],
                                        'train_rows': len(train), 'predict_rows': len(predict)})
    del model, x, xp
    gc.collect()
    return prediction
