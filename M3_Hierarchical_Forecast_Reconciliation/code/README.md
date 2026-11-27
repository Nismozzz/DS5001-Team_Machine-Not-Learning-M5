# Code notes

## Implementation files

- `reconciliation.py`: hierarchy construction, external quantile reconciliation, top-down shares, MinT and diagnostic functions.
- `03_reconcile.py`: formal run entry point and output writing.
- `verify_module3.py`: re-reads the written Parquet files and checks row counts, constraints and hierarchical summation.
- `tests/test_reconcile.py`: CG, non-negativity and external-quantile tests on a synthetic hierarchy.

## Formal run

Install the dependencies at the repository root, then run:

```bash
python code/03_reconcile.py \
  --member2-root "/path/to/module2/full-handoff" \
  --output-root "prediction_outputs"
```

Optional arguments:

```bash
--splits validation_1 validation_2 validation_3 test forecast
--schemes unreconciled bottom_up top_down mint
--cg-tolerance 1e-8
--cg-max-iter 500
--keep-existing
```

By default the existing `预测结果/` under the target directory is deleted so that an older version is not mixed in. Do not point the output directory at the repository root.

## Tests

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

## Independent verification

```bash
PYTHONPATH=code python code/verify_module3.py \
  --module3-root "/path/to/module3/output" \
  --member2-root "/path/to/module2/full-handoff"
```

## Baseline quantile interface

Seasonal Naive, ETS and TSB quantiles are constructed by Module 4 from historical errors. After construction the interface must be called:

```python
from reconciliation import reconcile_quantile_block

reconciled_q, diagnostics = reconcile_quantile_block(
    catalog,
    scheme="mint",
    base_quantiles=node_by_quantile_columns,
    weights=node_min_weight,
    layout=(28, number_of_taus),
)
```

This interface avoids the final handoff problem where point forecasts are already reconciled but the quantiles are still incoherent.


