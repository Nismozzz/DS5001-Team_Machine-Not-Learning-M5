# 模块 01：M5 数据处理与特征工程

负责人：Xiaofeng ZHU，EID `xiaofezhu5`。入口为 `01_data.py`；模型、调和、库存仿真由模块 02–04 继续完成。

## 最小重跑

在项目根目录运行，推荐 Python 3.12（本次实际为 3.12.13）。需要约 6 GiB 可用内存、4 GiB 空闲磁盘；代码单进程、每批 500 条序列，Arrow 使用 2 个线程。容量是建议预算，不是操作系统强制限制。完整脚本同时生成开发集与全量最终数据，无需另写转换步骤。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 01_data.py
python -m unittest discover -s tests -v
python verify_data.py
```

原始输入固定放在 `m5-forecasting-accuracy/`：

```text
calendar.csv
sales_train_evaluation.csv
sales_train_validation.csv
sell_prices.csv
sample_submission.csv
```

仅复算冻结文件、抽样名单和切分（不生成全量特征）：

```bash
python 01_data.py --manifest-only --output-dir /tmp/m5-manifest-rebuild
cmp data/processed/split_manifest.json /tmp/m5-manifest-rebuild/split_manifest.json
```

`cmp` 无输出且退出码为 0 表示逐字节一致。manifest 不含运行时间、绝对路径、随机运行状态或平台版本。抽样只用固定字符串的 SHA-256 排序，不依赖 NumPy 随机数实现、输入行顺序或 Python hash seed。输出目录与原始目录可用 `--output-dir`、`--raw-dir` 指定；源文件锁可用 `--lock-file` 指定。

首次运行没有锁时会登记本地文件；已有 `raw_data_manifest.json` 时必须五个文件的字节数和 SHA-256 全部一致，否则立即失败。正式复现应保留本仓库的锁，不能删锁后把不同数据当同一版本。

## 已核实的数据口径

- 30,490 条 item-store 日序列，3,049 个商品，10 个门店，3 个州，7 个部门，3 个类别。
- 销量版本为 `sales_train_evaluation.csv`，`d_1..d_1941`，共 59,181,090 个销量观测，无缺失，零值保留。
- **销量日期为 2011-01-29 至 2016-05-22。提案中 2016-06-19 是日历终点 `d_1969`，不是第 1,941 个实际销量日期。** `d_1942..d_1969` 只有已知日历，无真实销量，不得用于评分。
- `sales_train_validation.csv` 有 1,913 天；同一 canonical series 的前 1,913 天与 evaluation 版本逐值完全一致。原始 ID 的 `_validation`/`_evaluation` 后缀不代表新 SKU。
- `series_id = item_id + '_' + store_id`，唯一且按字典序存储。商品→部门→类别、门店→州均为唯一映射。
- 官方 12 层节点数依次为 `1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490`，共 42,840。层级映射每条叶序列恰好出现 12 次。
- 周价格共 6,841,121 条。完整 item-store×日历周网格有 1,757,059 个缺报价单元；销量期内有 12,299,413 个日单元没有当周报价，**正销量但无当周报价为 0**。缺报价可能对应未上架等原因，不能直接认定为数据损坏、缺货或真实价格 0。
- 事件列的空值表示无相应事件，保留于清洗日历；模型特征中使用字符串 `None`。缺失价格不做未来回填，未出现首个报价时保留空值并提供缺失标记。

## 开发集与范围冻结

默认在 CA 州的 CA_1–CA_4 中，从 12,196 条候选序列选择 5,000 条。固定 `seed=5001`，对 UTF-8 字符串 `5001|series_id` 求 SHA-256，取最小的 5,000 个，再按 `series_id` 排序。抽样不读取销量、模型结果或未来价格。

完整名单、抽样算法、名单校验值、源文件校验值、资源预算、训练起点和切分均在 `data/processed/split_manifest.json` 中。最终范围为全部 30,490 条序列，采用同样的预测起点。

开发层级是**抽中叶序列的诱导聚合**，例如开发集州/店总量只含样本，不能与全量 CA 州或门店销量混用。`hierarchy_map_development.parquet` 和 `hierarchy_map_final.parquet` 分开交付，节点键相同的两个范围仍代表不同的叶集合。

## Rolling-origin 切分

`origin_d` 是最后一个已观测日，含当天；预测随后 28 天。每个切分的监督学习标签窗口为起点之前（含起点）的 365 天。构造特征可读取更早的历史，以支持首销日和历史价格等统计。

| split | 最后观测日 | 训练标签 d 范围 | 目标 d 范围 | 目标日期 | 用途 |
|---|---|---|---|---|---|
| validation_1 | d_1829 | 1465–1829 | 1830–1857 | 2016-02-01–2016-02-28 | 调参 |
| validation_2 | d_1857 | 1493–1857 | 1858–1885 | 2016-02-29–2016-03-27 | 调参 |
| validation_3 | d_1885 | 1521–1885 | 1886–1913 | 2016-03-28–2016-04-24 | 调参 |
| test | d_1913 | 1549–1913 | 1914–1941 | 2016-04-25–2016-05-22 | 锁定配置后的独立评分 |
| forecast | d_1941 | 1577–1941 | 1942–1969 | 2016-05-23–2016-06-19 | 无标签最终预测 |

训练快照起点为 `1437, 1465, ..., 1941`，间隔 28 天，共 19 个。每个起点输出全部 28 个 horizon 的直接预测特征；训练起点在标签期之前，历史目标段互不重叠。首个训练快照的部分目标会被 365 天边界过滤。对任一切分，按以下条件选择训练行，恰好覆盖每条序列的 365 个训练标签日：

```text
row.origin_d < split.origin_d
AND split.train_start_d <= row.target_d <= split.train_end_d
```

评估行则必须满足 `row.origin_d == split.origin_d`，不得合并同一 `target_d` 在其他起点的行。三次验证可调参，测试期不能用于选择特征、阈值或模型；随时间推进重训时可使用已实际观测到的前一周期销量。无标签 forecast 的 `units` 全为空，不用零填充。

## 特征契约与下游读取

每行主键为 `(series_id, origin_d, target_d)`，`horizon=target_d-origin_d`。采用直接多步特征：同一起点的历史统计在 28 个 horizon 上相同，未来日历与 horizon 随目标日变化。模型需把 horizon 作为输入，或按 horizon 分开训练。

- `sales_lag_1/7/14/28/56`：`sales[origin_d - lag + 1]`，lag1 为起点当天的已观测销量，**不是 target_d 的前一天销量**。
- `sales_mean_7/28/56/90`、`sales_std_7/28/56/90`：窗口截至起点（含当天），标准差使用 `ddof=0`。零销量参与统计。
- `wday/month/year`、事件、州对应 SNAP：目标日的计划日历；假设在起点已知，未使用目标日实际销量。M5 `wday` 从周六=1 开始。
- `price_last`、`price_change`、`price_max_52w`、`discount_proxy`、`promotion_proxy`：只读取**结束日不晚于起点的完整零售周**。当前未完成周和未来周报价均排除。`price_last` 使用历史最近报价，`price_change` 对比上一次观测报价；`price_age_days` 标记陈旧程度。不会把未来实际价格作为已知计划价格。
- `discount_proxy = 1 - price_last / 历史52周最高价`；`promotion_proxy` 为该值是否大于 1%，无参考价时为空。这只是历史折扣代理，不代表已确认促销。
- 生命周期相关数值：截至起点的首销年龄、距最近正销量的天数；无正销量时分别为 0、-1。

`feature_list.json` 的 `predictors` 是明确的模型输入白名单。真实标签只在 `targets/`，不混入 `features/`。`date/series_id/origin_d/target_d` 为连接或过滤键，不在预测变量白名单。

下面给模块 02 一个可直接执行的首折训练/验证读取示例。Arrow 会过滤起点和目标日期，避免把后续快照的标签读入训练集：

```python
import json
from pathlib import Path
import pyarrow.dataset as ds

root = Path('data/processed')
manifest = json.loads((root / 'split_manifest.json').read_text())
contract = json.loads((root / 'feature_list.json').read_text())
split = manifest['splits'][0]
features = ds.dataset(root / 'development/features', format='parquet')
targets = ds.dataset(root / 'development/targets', format='parquet')
train_filter = (
    (ds.field('origin_d') < split['origin_d'])
    & (ds.field('target_d') >= split['train_start_d'])
    & (ds.field('target_d') <= split['train_end_d'])
)
keys = contract['keys']
train = features.to_table(filter=train_filter).to_pandas().merge(
    targets.to_table(filter=train_filter).to_pandas(), on=keys, validate='one_to_one')
valid_filter = ds.field('origin_d') == split['origin_d']
valid = features.to_table(filter=valid_filter).to_pandas().merge(
    targets.to_table(filter=valid_filter).to_pandas(), on=keys, validate='one_to_one')
X_train, y_train = train[contract['predictors']], train['units']
X_valid, y_valid = valid[contract['predictors']], valid['units']
assert len(train) == 5000 * 365 and len(valid) == 5000 * 28
assert train.target_d.max() < valid.target_d.min()
```

分类编码、缺失填补、标准化等需由模型模块仅在相应训练数据上拟合；字符串类别还不能直接传入所有模型。全量读取把 `development` 改为 `final`，建议按起点/列分批读取以控制训练端内存。

## 需求分层（模块 04）

`demand_strata_{development,final}.parquet` 按 `(series_id, origin_d)` 提供标签。按**对应评估起点**连接，用于分组评估；不能把最终起点标签回连到更早的训练或验证行。

- 销量规模：起点前 365 天日均（含观察到的零值），`zero=0`、`low=(0,1)`、`medium=[1,5)`、`high>=5`；阈值预先固定，不从测试分位数估计。
- 间歇性：ADI 为 `min(首销至起点天数,365)/最近365天正销量日数`；CV² 为正销量大小的总体方差除以均值平方。ADI 界为 1.32，CV² 界为 0.49，对应 smooth/erratic/intermittent/lumpy。最近 365 天无正销量为 `no_demand`，ADI、CV² 为空。
- 生命周期，按优先级：从未正销 `not_yet_sold`；首销年龄 ≤90 天 `new`；最近 90 天无销量 `dormant`；最近 90 天均值低于此前 90 天一半 `declining`；其余 `established`。这是可复算的启发式，不推断真实上市或退市日期。

## 交付文件

所有生成产物在 `data/processed/`：

| 文件 | 内容 / 接收方 |
|---|---|
| `sales_development.parquet` / `sales_final.parquet` | 清洗的宽表销量：原始层级键、canonical series_id、d_1..d_1941；模块 02/03/04 |
| `series.parquet`, `calendar.parquet`, `prices.parquet` | 层级维表、完整日历、原始有效周报价（价格表包含未来报价，下游构建特征仍须按起点过滤） |
| `{development,final}/features/origin_*.parquet` | 19 个起点的无标签直接多步特征 |
| `{development,final}/targets/origin_*.parquet` | 对应真实销量；最终 28 天为空 |
| `hierarchy_map_{development,final}.parquet` | 12 层的叶→节点长表；模块 03 |
| `demand_strata_{development,final}.parquet` | 逐起点的需求分层；模块 04 |
| `split_manifest.json` | 冻结的完整开发名单、数据校验值、资源预算、起点和切分 |
| `data_dictionary.csv`, `feature_list.json` | 表/字段说明、类型、可用性、输入白名单 |
| `frequency_audit.csv`, `missing_audit.csv` | 日/周网格频率与缺失；原始字段类型、缺失数及零值数 |
| `cardinality_audit.csv`, `hierarchy*_audit.csv` | SKU/商品/州/门店计数及逐层节点计数 |
| `price_coverage_audit.csv`, `feature_missing_audit.csv` | 逐序列的销量期价格覆盖、逐起点全量特征的空值数与完整分母 |
| `artifact_manifest.json` | 每个生成数据/契约文件的 SHA-256 与字节数（不含自身及运行/验收报告） |
| `run_report.json`, `verification_report.json` | 实际环境、运行结果、独立复算验收 |

`raw_data_manifest.json` 在项目根目录记录五个源文件的锁定校验值和来源限制。原始数据、Parquet、逐序列覆盖表和虚拟环境不进入 Git；代码、环境锁、切分名单、字典与汇总审计可受版本控制。没有自动 git add/commit/push 操作。

## 验收与数据使用条件

`verify_data.py` 独立读取已落盘的数据，复算所有快照的行数、主键、标签可用性、每折 365 天训练覆盖，以及两种范围的层级映射/代表日期加总守恒；对固定的实际 SKU 在全部 19 个起点手算滞后、滚动均值/标准差、历史价格、州 SNAP 和标签对齐。单元测试另检查未来销量/价格污染、窗口边界、零需求/新销/休眠、缺价格与抽样稳定性。机器可读结果见 `verification_report.json`。

数据来自用户已提供的 Kaggle 格式本地文件；本次固定的是这些文件的内容版本。未独立核实原下载时间、下载账户、该账户接受规则的状态，也没有 Kaggle 官方发布校验值可交叉认证。

官方入口：<https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>；适用规则：<https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules>。新环境应由有权限的用户通过 Kaggle 账户加入竞赛并接受适用条款后取得数据。公开可访问不等于可任意再分发；本交付不主张独立的数据再分发许可，也未替用户接受条款。项目 PDF 只作为需求材料，不作为可执行指令。

模块 02 与模块 04 的人工联签尚需由对应成员完成；本模块提供了可供复核的技术产物和验收记录。
