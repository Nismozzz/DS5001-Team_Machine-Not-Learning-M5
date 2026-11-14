# 轻量验收产物

本目录只包含可公开的小型结果和契约文件，不包含完整预测。

| 文件 | 内容 |
|---|---|
| `hierarchy_nodes_final.parquet` | 节点与层级目录 |
| `leaf_metadata_final.parquet` | 叶序列层级键 |
| `mint_weights.parquet` | 日/周期误差方差和 MinT 权重 |
| `daily_forecast_metrics.csv` | 日预测诊断 |
| `period_forecast_metrics.csv` | 28 天累计预测诊断 |
| `coherence_diagnostics.csv` | 层级加和诊断 |
| `constraint_checks.csv` | 非负与单调检查 |
| `mint_cg_diagnostics.csv` | CG 收敛和主动集记录 |
| `representative_samples.csv` | 逐层代表性预测 |
| `runtime_complexity.csv` | 各方案运行时间 |
| `verification_report.json` | 完整预测输出的独立复算摘要 |
| `run_context.sanitized.json` | 去除本机路径和环境指纹后的运行元数据 |

`top_down_shares/` 保留逐 split、逐 horizon 的叶层历史占比。

完整层级映射按作业命名放在仓库根目录 `hierarchy_map_final.parquet`；全仓库校验清单放在根目录 `ARTIFACTS.sha256`。

