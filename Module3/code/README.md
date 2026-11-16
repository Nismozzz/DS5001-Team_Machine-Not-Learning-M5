# 代码说明

## 实现文件

- `reconciliation.py`：层级构建、外部分位数调和、top-down 比例、MinT 和诊断函数。
- `03_reconcile.py`：正式运行入口和输出写出。
- `verify_module3.py`：从已写 Parquet 重新读取并检查行数、约束和层级加和。
- `tests/test_reconcile.py`：合成层级上的 CG、非负和外部分位数测试。

## 正式运行

在仓库根目录安装依赖后运行：

```bash
python code/03_reconcile.py \
  --member2-root "/path/to/module2/full-handoff" \
  --output-root "prediction_outputs"
```

可选参数：

```bash
--splits validation_1 validation_2 validation_3 test forecast
--schemes unreconciled bottom_up top_down mint
--cg-tolerance 1e-8
--cg-max-iter 500
--keep-existing
```

默认会删除目标目录下已有的 `预测结果/`，避免混入旧版本。输出目录不要指向仓库根目录。

## 测试

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

## 独立核验

```bash
PYTHONPATH=code python code/verify_module3.py \
  --module3-root "/path/to/module3/output" \
  --member2-root "/path/to/module2/full-handoff"
```

## Baseline 分位数接口

Seasonal Naive、ETS、TSB 的分位数由模块 4 从历史误差构造。构造后必须调用：

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

该接口避免点预测已调和、分位数仍不连贯的最终交接问题。
