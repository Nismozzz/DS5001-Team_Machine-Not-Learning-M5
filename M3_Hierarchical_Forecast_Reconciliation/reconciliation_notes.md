# Hierarchical structure and forecast reconciliation methods

Version: `m5-module03-v1`

## 1. Experimental scope

- The full 30,490 item-store leaf series of M5.
- 12 levels, 42,840 nodes, 365,880 leaf-series-to-node memberships.
- A 28-day forecast horizon.
- Three rolling validations: `origin_d=1829, 1857, 1885`.
- Independent test: `origin_d=1913`, target window `d_1914–d_1941`.
- Unlabelled forecast: `origin_d=1941`, target window `d_1942–d_1969`.

## 2. Where the hierarchy contract comes from

The public handoff package does not contain the large Parquet files produced by Module 1. The implementation reads each level's nodes from Module 2's node coverage list `coverage.json`, rebuilds the keys from item-store identifiers, and verifies them node by node.

The checks are:

1. Node counts per level match the Module 1 contract: `1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490`.
2. The key set derived from the 30,490 leaf series is identical to the upstream node ID set.
3. Every leaf series appears exactly once in each of the 12 levels.
4. `hierarchy_map_final.parquet` and `hierarchy_nodes_final.parquet` are written as reviewable H3 hierarchy artifacts.

## 3. The four schemes

| Scheme | Point forecasts | Daily quantiles | 28-day cumulative quantiles |
|---|---|---|---|
| `unreconciled` | independent forecasts at each level | LightGBM raw forecasts | upstream direct cumulative quantiles |
| `bottom_up` | leaf forecasts summed level by level | leaf quantiles summed level by level | upstream values passed through unchanged (identical to `unreconciled`; summing level by level does not remove quantile non-additivity either, see Section 9) |
| `top_down` | root forecast split by historical leaf shares, then summed | root quantiles split by daily shares, then summed | root cumulative quantiles split by 28-day shares, then summed |
| `mint` | non-negative diagonal MinT | non-negative MinT with monotonicity constraints | non-negative MinT on direct cumulative quantiles |

> Delivery note: the 28-day cumulative quantile block does not aim for "additivity level by level". Quantiles are not additive, so no summation or re-allocation can make every level's quantile equal its own quantile (the top level is 35.4% below the sum over leaves at τ=0.95). The four schemes handle this block as follows: `unreconciled` and `bottom_up` pass the upstream `period.parquet` values through unchanged (identical row by row, 701,523 rows with zero difference), `top_down` re-allocates by cumulative shares, and `mint` runs diagonal MinT; none of these changes the property that quantiles are not additive. Downstream consumers that need a quantile convention therefore always take leaf-level quantiles reconciled through `reconcile_quantile_block()`. The table above keeps the design definitions; see Section 9 for the explanation.

The unreconciled output keeps the hierarchical inconsistency produced by independent modelling and serves as the comparison baseline.

## 4. Top-down shares

Each forecast origin only uses labels with `label_available_d <= origin_d`.

1. Take the most recent 224 days of history.
2. Compute leaf-series demand shares by the weekday of the target day.
3. When a weekday has no history, fall back to shares over all history.
4. Only when all history is zero are equal leaf shares used.
5. Leaf series with zero historical sales get a zero share; non-zero leaves are renormalised.

The formal run never triggered the fallback rules. Shares by split and by horizon are kept in the full delivery package under `top_down_shares/`; the public package keeps only summary diagnostics.

## 5. MinT

The MinT-WLS approximation with diagonal error covariance:

```text
y_tilde = S (S' W^-1 S)^-1 S' W^-1 y
W = diag(sigma_1^2, ..., sigma_N^2)
```

where `S` is the aggregation matrix from leaf series to the 12-level nodes.

- Daily error variance: estimated from out-of-sample daily residuals at the three historical calibration origins.
- 28-day error variance: sum each calibration origin's 28-day point residuals first, then estimate across the three origins.
- Shrinkage: `0.8 * MSE_i + 0.2 * pooled_MSE`.
- Variance floor: `1e-8` of the pooled MSE.
- Solver: column-wise preconditioned conjugate gradient, tolerance `1e-8`, at most 500 iterations.

This is an explicit diagonal MinT, not a full-covariance MinT. A full 42,840-dimensional covariance cannot be estimated stably and is not a suitable dependency for a reproducible teaching project; the diagonal approximation keeps the explicit weight source of trace-minimising WLS.

## 6. Non-negativity and quantile constraints

Unconstrained MinT can produce negative demand. The procedure is:

1. Record the number of negative leaf and node values.
2. Point forecasts and the first quantile: fix leaves that violate non-negativity at the active-set lower bound of 0.
3. Re-solve WLS for the free leaves; repeat until no free leaf is below the bound.
4. For higher quantiles the bound is the previous quantile at the same horizon.
5. Re-aggregate from the leaves to all nodes.

The last step preserves hierarchical summation together with the non-negativity and monotonicity constraints. CG iterations, active-set rounds, initial negatives and final residuals are recorded in the diagnostics.

Daily quantiles and 28-day cumulative quantiles are two independent pipelines. The 28 daily quantiles are never summed and passed off as the 28-day cumulative quantiles; the latter come from upstream `period.parquet`.

## 7. Downstream reconciliation interface for baseline quantiles

Module 2's Seasonal Naive, ETS and TSB provide no quantiles, so Module 4 constructs them from historical rolling errors. To avoid a final inventory input where "point forecasts are coherent but quantiles are not", the public package provides:

```python
from reconciliation import reconcile_quantile_block

reconciled_q, diagnostics = reconcile_quantile_block(
    catalog,
    scheme="mint",                  # or bottom_up / top_down
    base_quantiles=node_by_columns,
    weights=weights,
    layout=(28, quantile_count),    # daily quantiles
)
```

Module 4 must call this interface after constructing baseline quantiles and before entering the inventory simulation.

## 8. Independent test results

LightGBM root-node point-forecast WAPE:

| Scheme | Daily forecast | 28-day cumulative |
|---|---:|---:|
| Unreconciled | 8.914% | 6.912% |
| Bottom-up | **4.189%** | **3.403%** |
| Top-down | 8.914% | 6.912% |
| MinT | 5.919% | 4.963% |

LightGBM item-store daily WAPE: Unreconciled/Bottom-up `72.087%`, Top-down `75.149%`, MinT `71.693%`.

LightGBM item-store 28-day WAPE: Unreconciled/Bottom-up `25.964%`, Top-down `31.889%`, MinT `26.157%`.

Bottom-up is best on high-level LightGBM forecasts in the independent test; MinT gives only a slight improvement to leaf-level daily point forecasts; top-down is usually harmful at leaf level. The ranking is not fully consistent across the three validation windows, so the final choice has to be decided by inventory cost.

## 9. Limitations

- Diagonal error variances are used; full high-level cross-covariance is not estimated.
- Weights come from historical calibration windows and use no future labels.
- The model configurations for the validation windows were chosen after the fact and suit retrospective diagnostics; the independent test is the main scoring window.
- The full forecasts are not in Git; a runner must regenerate them from a compliant Module 2 handoff package.
- The 28-day cumulative quantile block is handled per scheme (`unreconciled`/`bottom_up` pass upstream values through, `top_down` re-allocates by shares, `mint` runs MinT), but because quantiles are not additive the block does not satisfy level-by-level summation under any scheme; high-level quantile conclusions in this module therefore stop at leaf level, and high-level comparisons switch to the point-forecast convention.
- The final Module 4 sign-off and the cost results still have to be completed inside the course project.


