# Data contract: Module 2 to Module 3, Module 3 to Module 4

## 1. Input directories

`--member2-root` should point at Module 2's full handoff root and must contain at least:

```text
final_backtest/predictions/validation_1/L01..L12/
final_backtest/predictions/validation_2/L01..L12/
final_backtest/predictions/validation_3/L01..L12/
final_test/predictions/test/L01..L12/
final_forecast/predictions/forecast/L01..L12/
final_calibration/predictions/calibration_1745/L01..L12/
final_calibration/predictions/calibration_1773/L01..L12/
final_calibration/predictions/calibration_1801/L01..L12/
```

Each level must provide:

- `coverage.json`: the node list of that level.
- `daily.parquet`: daily point forecasts for the four models and LightGBM daily quantiles.
- `period.parquet`: LightGBM direct 28-day cumulative quantiles.
- `residuals.parquet` for labelled scenarios: actual values, forecasts and residuals.

## 2. Key fields

| File | Join keys | Main fields |
|---|---|---|
| `daily.parquet` | `model, node_id, origin_d, target_d` | `horizon, yhat, q_050, q_090, q_095, q_099` |
| `period.parquet` | `model, node_id, tau` | `origin_d, forecast` |
| `residuals.parquet` | `model, node_id, target_d` | `horizon, yhat, y_true, residual, label_available_d` |

Residuals are defined as:

```text
residual = y_true - yhat
```

## 3. Output directories

```text
{output_root}/
  hierarchy_map_final.parquet
  hierarchy_nodes_final.parquet
  leaf_metadata_final.parquet
  mint_weights.parquet
  run_context.json
  verification_report.json
  artifact_manifest.json
  predictions/{split}/{scheme}/
    daily/{model}/L01..L12.parquet
    period_point/{model}/L01..L12.parquet
    period_quantile/L01..L12.parquet
  diagnostics/
```

Scheme names are fixed as `unreconciled`, `bottom_up`, `top_down` and `mint`. Model names are fixed as `seasonal_naive`, `ets`, `tsb` and `lightgbm`.

## 4. Downstream reading

- Daily point forecasts: align on `node_id, origin_d, target_d, horizon` and read `yhat`.
- 28-day point forecasts: align on `node_id, origin_d` and read `forecast`.
- 28-day quantiles: align on `node_id, origin_d, tau` and read `forecast`.
- In the `forecast` scenario `y_true` must be treated as missing; it must not be filled with 0 and used to compute metrics.

## 5. Baseline quantiles

The public package only delivers each scheme's handling of the upstream quantiles and does not promise that the quantile block is additive level by level (see Section 9 of `reconciliation_notes.md` for the non-additivity explanation). Seasonal Naive, ETS and TSB quantiles are constructed by Module 4 from historical rolling errors, and after construction `reconcile_quantile_block()` must be called to complete the reconciliation, which supplies the quantile input to the inventory simulation.

Example:

```python
reconciled_q, diagnostics = reconcile_quantile_block(
    catalog,
    scheme="bottom_up",
    base_quantiles=baseline_q,  # node x (28 * quantile_count)
    layout=(28, len(taus)),
)
```

## 6. Constraints

- All forecasts are non-negative.
- For the same node and horizon, quantiles are non-decreasing in `tau`.
- For all nodes of the same scheme, model and time point, point forecasts satisfy hierarchical summation; the 28-day cumulative quantile block does not (quantiles are not additive, see Section 9 of `reconciliation_notes.md`), and its convention is unified by the consumer-side interface `reconcile_quantile_block()`.
- Unreconciled output may violate summation, but this must be reported in the diagnostics.
- The final inventory input may only use a single combination of model, split, scheme and origin.


