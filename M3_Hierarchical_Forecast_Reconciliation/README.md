# M5 hierarchical demand forecast reconciliation

This is the Module 3 code and acceptance package, publishable on GitHub. It covers 12-level demand forecast reconciliation for M5: unreconciled, bottom-up, top-down, and non-negative active-set MinT with diagonal error weights.

The repository contains only code, method documentation, lightweight diagnostics and the hierarchy contract; it does not contain the raw Kaggle data or the 901 MB full prediction tree. Full predictions are a local or private artifact that can be regenerated.

## Core capabilities

- A 12-level summation structure over 42,840 nodes.
- 30,490 item-store leaf series.
- A single exchange format for Seasonal Naive, ETS, TSB and LightGBM.
- Daily point forecasts, daily quantiles, 28-day point forecasts and direct 28-day cumulative quantiles.
- Non-negative active-set MinT, quantile monotonicity constraints, CG convergence diagnostics.
- A `reconcile_quantile_block()` interface for externally calibrated baseline quantiles.

## Quick start

Python 3.12, the Module 1 hierarchy contract, and Module 2's forecast handoff directories `final_backtest/`, `final_test/`, `final_forecast/` and `final_calibration/` are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python code/03_reconcile.py \
  --member2-root "/path/to/module2/full-handoff" \
  --output-root "prediction_outputs"
```

To run only some scenarios or schemes:

```bash
python code/03_reconcile.py \
  --member2-root "/path/to/module2/full-handoff" \
  --output-root "prediction_outputs" \
  --splits test forecast \
  --schemes unreconciled bottom_up top_down mint
```

To run the tests:

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

To verify the formal output independently:

```bash
PYTHONPATH=code python code/verify_module3.py \
  --module3-root "/path/to/module3/output" \
  --member2-root "/path/to/module2/full-handoff"
```

## Layout

| Path | Contents |
|---|---|
| `code/` | Reconciliation implementation, formal entry point, independent verification and unit tests |
| `reconciliation_notes.md` | Hierarchy, weights, shares, constraints, results and limitations |
| `hierarchy_map_final.parquet` | Full mapping from leaf series to the 12-level nodes |
| `docs/DATA_CONTRACT.md` | H2 input and H3 output field contract |
| `docs/ACCEPTANCE.md` | Acceptance matrix and open sign-off items |
| `docs/SECURITY_AND_PUBLICATION.md` | GitHub privacy and data publication notes |
| `artifacts/` | Lightweight hierarchy map, weights and diagnostics |
| `.github/workflows/ci.yml` | GitHub Actions unit tests |

## Verified results

LightGBM root-node WAPE on the independent test:

| Scheme | Daily WAPE | 28-day cumulative WAPE |
|---|---:|---:|
| Unreconciled | 8.914% | 6.912% |
| Bottom-up | **4.189%** | **3.403%** |
| Top-down | 8.914% | 6.912% |
| MinT | 5.919% | 4.963% |

The maximum hierarchical summation error over all reconciled point forecasts is zero; the number of negative point forecasts is zero; LightGBM daily and cumulative quantiles are both monotone; every MinT solve converged. Note: the 28-day cumulative quantile block does not aim for additivity level by level — quantile non-additivity means the block cannot equal each level's own quantile under any scheme (the top level is 35.4% below the sum over leaves at τ=0.95; store node CA_1 is 14.2% low; the largest node gap is 49.8%), and `bottom_up` is identical to `unreconciled` on that block; downstream consumers that need a quantile convention take leaf-level quantiles reconciled through `reconcile_quantile_block()`. See Section 9 of `reconciliation_notes.md`. Detailed results are in `artifacts/`.

## Data and licence

The raw M5 data is not included in this repository. Users must obtain it from Kaggle themselves and comply with its rules and licence. The applicable terms of use should also be checked before publishing derived results. No additional licence is granted by default; whether to add an open-source licence is a decision for the project owner or the team.


