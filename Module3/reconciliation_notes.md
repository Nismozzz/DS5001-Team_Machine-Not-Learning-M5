# 分层结构与预测调和方法

版本：`m5-module03-v1`

## 1. 实验范围

- M5 全量 30,490 条 item-store 叶序列。
- 12 层、42,840 个节点、365,880 个叶序列—节点成员关系。
- 28 天预测期。
- 三次 rolling validation：`origin_d=1829, 1857, 1885`。
- 独立 test：`origin_d=1913`，目标区间为 `d_1914–d_1941`。
- 无标签 forecast：`origin_d=1941`，目标区间为 `d_1942–d_1969`。

## 2. 层级契约来源

公开交接包中没有模块 1 生成的大型 Parquet。实现从模块 2 的节点覆盖清单 `coverage.json` 读取每层节点，再用 item-store 标识重建键值并逐节点校验。

校验包括：

1. 每层节点数与模块 1 契约一致：`1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490`。
2. 由 30,490 条叶序列派生的键集合与上游节点 ID 集合完全一致。
3. 每条叶序列恰在 12 个层级中出现一次。
4. 输出 `hierarchy_map_final.parquet` 和 `hierarchy_nodes_final.parquet` 作为可审阅的 H3 层级产物。

## 3. 四种方案

| 方案 | 点预测 | 日分位数 | 28 天累计分位数 |
|---|---|---|---|
| `unreconciled` | 上游各层独立预测 | LightGBM 原预测 | 上游直接累计分位数 |
| `bottom_up` | 叶预测逐层求和 | 叶分位数逐层求和 | 叶层直接累计分位数逐层求和 |
| `top_down` | 根预测按历史叶占比分配，再求和 | 根分位数按日占比分配，再求和 | 根累计分位数按 28 天占比分配，再求和 |
| `mint` | 非负对角 MinT | 非负 MinT 加单调约束 | 直接累计分位数的非负 MinT |

未调和结果保留独立建模产生的层级不一致，作为比较基准。

## 4. Top-down 比例

每个预测起点只使用 `label_available_d <= origin_d` 的标签。

1. 取最近 224 天历史。
2. 按目标日星期计算叶序列需求占比。
3. 某星期无历史时，回退到全部历史占比。
4. 全部历史仍为零时，才使用叶序列等额占比。
5. 历史销量为零的叶序列占比为零，非零叶重新归一化。

正式运行没有触发备用规则。逐 split、逐 horizon 的占比保存在完整交付包 `top_down_shares/`；公开包只保留汇总诊断。

## 5. MinT

使用对角误差协方差的 MinT-WLS 近似：

```text
y_tilde = S (S' W^-1 S)^-1 S' W^-1 y
W = diag(sigma_1^2, ..., sigma_N^2)
```

其中 `S` 是叶序列到 12 层节点的加总矩阵。

- 日误差方差：由三个历史 calibration origin 的样本外日残差估计。
- 28 天误差方差：先把每个 calibration origin 的 28 天点残差加总，再跨三个 origin 估计。
- 收缩：`0.8 * MSE_i + 0.2 * pooled_MSE`。
- 方差下限：pooled MSE 的 `1e-8`。
- 求解：逐列预处理共轭梯度，容差 `1e-8`，最多 500 次。

这是显式的对角 MinT，不是完整协方差 MinT。完整 42,840 维协方差无法稳定估计，也不适合作为可复现教学项目依赖；对角近似保留 trace-minimizing WLS 的明确权重来源。

## 6. 非负与分位数约束

无约束 MinT 可能产生负需求。处理流程：

1. 记录负叶值和负节点数量。
2. 点预测与第一分位数：把违反非负约束的叶值固定在主动集下界 0。
3. 对自由叶重新求解 WLS；重复，直到所有自由叶值不低于下界。
4. 更高分位数的下界为同一 horizon 的上一分位数。
5. 从叶层重新汇总到全部节点。

最后一步保留层级加和与非负/单调约束。CG 迭代、主动集轮数、初始负值和最终残差记录在诊断文件中。

日分位数和 28 天累计分位数是两条独立流程。没有把 28 个日分位数相加冒充 28 天累计分位数；后者来自上游 `period.parquet`。

## 7. Baseline 分位数的后续调和接口

模块 2 的 Seasonal Naive、ETS、TSB 不提供分位数，模块 4 会从历史滚动误差构造。为避免最终库存输入出现“点预测一致、分位数不一致”，公开包提供：

```python
from reconciliation import reconcile_quantile_block

reconciled_q, diagnostics = reconcile_quantile_block(
    catalog,
    scheme="mint",                  # 或 bottom_up / top_down
    base_quantiles=node_by_columns,
    weights=weights,
    layout=(28, quantile_count),    # 日分位数
)
```

模块 4 必须在构造 baseline 分位数之后、进入库存仿真之前调用该接口。

## 8. 独立 test 结果

LightGBM 根节点点预测 WAPE：

| 方案 | 日预测 | 28 天累计 |
|---|---:|---:|
| Unreconciled | 8.914% | 6.912% |
| Bottom-up | **4.189%** | **3.403%** |
| Top-down | 8.914% | 6.912% |
| MinT | 5.919% | 4.963% |

LightGBM item-store 日 WAPE：Unreconciled/Bottom-up `72.087%`，Top-down `75.149%`，MinT `71.693%`。

LightGBM item-store 28 天 WAPE：Unreconciled/Bottom-up `25.964%`，Top-down `31.889%`，MinT `26.157%`。

Bottom-up 在独立 test 的高层 LightGBM 预测上最好；MinT 只对叶层日点预测有轻微改善；Top-down 对叶层通常不利。排序在三个验证窗口并非完全一致，最终选择必须由库存成本决定。

## 9. 限制

- 使用对角误差方差，没有估计高层完整交叉协方差。
- 权重来自历史 calibration 窗口，不使用未来标签。
- 验证窗口的模型配置是事后选定，适合回顾性诊断；独立 test 是主要评分窗口。
- 完整预测没有进入 Git；运行者必须使用合规的模块 2 交接包重新生成。
- 最终模块 4 联签和成本结果仍需在课程项目内部完成。

