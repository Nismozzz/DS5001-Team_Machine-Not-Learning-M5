# 模块 2 到模块 3、模块 3 到模块 4 的数据契约

## 1. 输入目录

`--member2-root` 应指向模块 2 的全量交接根目录，至少包含：

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

每层必须提供：

- `coverage.json`：本层节点清单。
- `daily.parquet`：四个模型的日点预测、LightGBM 日分位数。
- `period.parquet`：LightGBM 直接 28 天累计分位数。
- 有标签场景的 `residuals.parquet`：真实值、预测值和残差。

## 2. 关键字段

| 文件 | 连接键 | 主要字段 |
|---|---|---|
| `daily.parquet` | `model, node_id, origin_d, target_d` | `horizon, yhat, q_050, q_090, q_095, q_099` |
| `period.parquet` | `model, node_id, tau` | `origin_d, forecast` |
| `residuals.parquet` | `model, node_id, target_d` | `horizon, yhat, y_true, residual, label_available_d` |

残差定义为：

```text
residual = y_true - yhat
```

## 3. 输出目录

```text
{output_root}/
  hierarchy_map_final.parquet
  hierarchy_nodes_final.parquet
  leaf_metadata_final.parquet
  mint_weights.parquet
  run_context.json
  verification_report.json
  交付清单.json
  预测结果/{split}/{scheme}/
    daily/{model}/L01..L12.parquet
    period_point/{model}/L01..L12.parquet
    period_quantile/L01..L12.parquet
  诊断结果/
```

方案名固定为 `unreconciled`、`bottom_up`、`top_down`、`mint`。模型名固定为 `seasonal_naive`、`ets`、`tsb`、`lightgbm`。

## 4. 下游读取

- 日点预测：按 `node_id, origin_d, target_d, horizon` 对齐，读取 `yhat`。
- 28 天点预测：按 `node_id, origin_d` 对齐，读取 `forecast`。
- 28 天分位数：按 `node_id, origin_d, tau` 对齐，读取 `forecast`。
- `forecast` 场景的 `y_true` 必须视为缺失，不能填 0 后计算指标。

## 5. Baseline 分位数

公开包只调和上游实际提供的分位数。Seasonal Naive、ETS、TSB 的分位数由模块 4 从历史滚动误差构造，然后必须调用 `reconcile_quantile_block()` 再次调和。

示例：

```python
reconciled_q, diagnostics = reconcile_quantile_block(
    catalog,
    scheme="bottom_up",
    base_quantiles=baseline_q,  # node x (28 * quantile_count)
    layout=(28, len(taus)),
)
```

## 6. 约束

- 所有预测非负。
- 同一节点、同一 horizon 的分位数随 `tau` 非递减。
- 同一方案、模型、时间点的所有节点满足层级加总。
- 未调和结果允许违反加总，但必须在诊断中报告。
- 最终库存输入只能使用同一个模型、split、scheme、起点组合。

