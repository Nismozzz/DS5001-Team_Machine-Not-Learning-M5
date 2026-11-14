# M5 分层需求预测调和

这是一个可公开托管在 GitHub 的模块 3 代码与验收包。内容覆盖 M5 的 12 层需求预测调和：未调和、自下而上、自上而下，以及带对角误差权重和非负主动集约束的 MinT。

本仓库只包含代码、方法文档、轻量诊断结果和层级契约，不包含原始 Kaggle 数据或 901 MB 的完整预测目录。完整预测是可重新生成的本地或私有交付物。

## 核心能力

- 42,840 个节点的 12 层加总关系。
- 30,490 条 item-store 叶序列。
- Seasonal Naive、ETS、TSB 和 LightGBM 的统一交换格式。
- 日点预测、日分位数、28 天点预测和直接 28 天累计分位数。
- 非负主动集 MinT、分位数单调约束、CG 收敛诊断。
- 面向 baseline 外部校准分位数的 `reconcile_quantile_block()` 接口。

## 快速开始

需要 Python 3.12、模块 1 的分层契约，以及模块 2 的预测交接目录 `final_backtest/`、`final_test/`、`final_forecast/`、`final_calibration/`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python code/03_reconcile.py \
  --member2-root "/path/to/module2/full-handoff" \
  --output-root "prediction_outputs"
```

只运行部分场景或方案：

```bash
python code/03_reconcile.py \
  --member2-root "/path/to/module2/full-handoff" \
  --output-root "prediction_outputs" \
  --splits test forecast \
  --schemes unreconciled bottom_up top_down mint
```

运行测试：

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

独立检查正式输出：

```bash
PYTHONPATH=code python code/verify_module3.py \
  --module3-root "/path/to/module3/output" \
  --member2-root "/path/to/module2/full-handoff"
```

## 目录

| 路径 | 内容 |
|---|---|
| `code/` | 调和实现、正式入口、独立校验和单测 |
| `reconciliation_notes.md` | 层级、权重、比例、约束、结果与限制 |
| `hierarchy_map_final.parquet` | 叶序列到 12 层节点的完整映射 |
| `docs/DATA_CONTRACT.md` | H2 输入与 H3 输出字段契约 |
| `docs/ACCEPTANCE.md` | 验收矩阵和未完成联签项 |
| `docs/SECURITY_AND_PUBLICATION.md` | GitHub 隐私与数据发布注意事项 |
| `artifacts/` | 轻量层级映射、权重和诊断结果 |
| `.github/workflows/ci.yml` | GitHub Actions 单元测试 |

## 已验证结果

独立 test 上，LightGBM 的 LightGBM 根节点 WAPE：

| 方案 | 日 WAPE | 28 天累计 WAPE |
|---|---:|---:|
| Unreconciled | 8.914% | 6.912% |
| Bottom-up | **4.189%** | **3.403%** |
| Top-down | 8.914% | 6.912% |
| MinT | 5.919% | 4.963% |

全部调和预测的最大层级加和误差为 0；负点预测数为 0；LightGBM 日/累计分位数均单调；全部 MinT 求解收敛。详细结果见 `artifacts/`。

## 数据与许可

原始 M5 数据没有包含在本仓库中。使用者必须自行从 Kaggle 获取数据并遵守其规则和许可。公开派生结果前也应检查相应使用条款。本仓库默认不授予额外许可证；是否添加开源许可证由项目所有者或团队决定。

