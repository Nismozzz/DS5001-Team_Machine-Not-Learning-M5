# 模块 01：M5 数据处理与特征工程

负责人：Xiaofeng ZHU（晓峰），EID：`xiaofezhu5`。

本说明按照《小组详细分工.pdf》中模块 01 的任务依赖顺序，逐项列出已完成工作及成果位置。模块截止日期为 2026 年 11 月 2 日；技术产出已生成并通过本地验收，模块 02、04 的人工联签待完成。

以下路径均相对于项目根目录。Parquet 和逐序列价格覆盖表为本地生成产物，不进入 Git；从 GitHub 获取代码后，按文末命令重建。

## 1. 登记原始数据并冻结版本

**已完成：** 登记本地提供的 5 个 M5 原始文件，记录各文件字节数、SHA-256、来源链接与使用条件。后续运行必须匹配原始文件锁，防止不同数据版本混用。

**成果位置：**

- `m5-forecasting-accuracy/calendar.csv`：日历、事件与 SNAP。
- `m5-forecasting-accuracy/sales_train_evaluation.csv`：1,941 天实际销量。
- `m5-forecasting-accuracy/sales_train_validation.csv`：1,913 天销量，用于核对重叠区间。
- `m5-forecasting-accuracy/sell_prices.csv`：周价格。
- `m5-forecasting-accuracy/sample_submission.csv`：官方提交格式样例。
- [raw_data_manifest.json](raw_data_manifest.json)：原始文件版本锁、来源和使用条件。
- [.gitignore](.gitignore)：排除原始数据、生成数据、虚拟环境和缓存。

原始数据不随仓库分发。数据入口为 <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>，规则为 <https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules>。复现者应通过有权限的 Kaggle 账户获取数据并接受适用条款。本地文件的原下载时间、账户和条款接受状态未独立核实，不能将版本登记等同于取得再分发许可。

## 2. 审计原始数据并形成数据字典

**已完成：** 核验 SKU、商品、门店、州、部门、类别、销量日期、日历事件与价格字段；检查频率、缺失、零值、映射关系以及两个销量版本的重叠数据。

主要核验结果：

- 30,490 条 item-store 日序列、3,049 个商品、10 个门店、3 个州、7 个部门、3 个类别。
- 1,941 天销量，共 59,181,090 个观测，无缺失；零销量保留。
- 实际销量日期为 **2011-01-29 至 2016-05-22**。提案中的 2016-06-19 是日历终点，最后 28 天没有真实销量，不能用于评分。
- validation 与 evaluation 版本前 1,913 天逐值一致；ID 后缀不代表新 SKU。
- 价格表有 6,841,121 条记录；完整序列×周网格缺报价单元为 1,757,059。销量期内有 12,299,413 个日单元缺当周报价，其中正销量单元为 0。
- 事件空值表示无相应事件；缺价格不解释为价格 0，不用未来价格回填。

**成果位置：**

| 文件 | 内容 |
|---|---|
| [data/processed/data_dictionary.csv](data/processed/data_dictionary.csv) | 字段名称、类型、含义、用途与可用时点 |
| [data/processed/cardinality_audit.csv](data/processed/cardinality_audit.csv) | 序列、商品、门店、州等计数 |
| [data/processed/frequency_audit.csv](data/processed/frequency_audit.csv) | 日销量、日历和周价格的频率与覆盖 |
| [data/processed/missing_audit.csv](data/processed/missing_audit.csv) | 原始字段缺失数和零值数 |
| `data/processed/price_coverage_audit.csv` | 逐序列价格覆盖明细，本地重建 |

## 3. 固定开发集与最终运行范围

**已完成：** 从 CA 州 CA_1–CA_4 的 12,196 条候选序列中固定选取 5,000 条作为开发集；全量 30,490 条序列作为最终运行范围。

抽样固定 `seed=5001`，对 `5001|series_id` 求 SHA-256，选取排序最小的 5,000 条，再按 `series_id` 排列。抽样不依赖销量、模型结果、未来价格或输入行顺序。

**成果位置：**

- [data/processed/split_manifest.json](data/processed/split_manifest.json)：开发集完整名单、抽样规则与校验值、最终范围和资源预算。
- [01_data.py](01_data.py) 中的 `sample_ids()`、`make_manifest()`：可复现抽样与范围登记实现。

运行采用单进程、每批 500 条序列、Arrow 2 个线程；建议约 6 GiB 可用内存与 4 GiB 空闲磁盘。这是建议预算，不是操作系统强制资源限制。

## 4. 清洗数据并生成开发／最终数据包

**已完成：** 统一 `series_id = item_id + '_' + store_id`，整理层级键、销量、日历与价格，生成可供下游读取的 Parquet 数据。

**成果位置（本地生成）：**

| 文件 | 内容 |
|---|---|
| `data/processed/sales_development.parquet` | 5,000 条开发序列及 d_1..d_1941 销量 |
| `data/processed/sales_final.parquet` | 30,490 条全量序列及 d_1..d_1941 销量 |
| `data/processed/series.parquet` | 序列标识及商品、门店、州等层级键 |
| `data/processed/calendar.parquet` | 完整 1,969 天日历、事件与 SNAP |
| `data/processed/prices.parquet` | 周价格表 |

`prices.parquet` 包含未来周报价；下游构建模型输入时仍必须按预测起点过滤，不能直接使用未来实际价格。

## 5. 建立 12 层聚合映射

**已完成：** 核验商品→部门→类别、门店→州的唯一映射，分别建立开发范围与全量范围的叶序列到聚合节点映射，供模块 03 使用。

全量 12 层节点数依次为 `1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490`，合计 42,840 个节点；每条叶序列在映射中出现 12 次。

**成果位置：**

- `data/processed/hierarchy_map_development.parquet`：开发集 12 层映射，本地生成。
- `data/processed/hierarchy_map_final.parquet`：全量 12 层映射，本地生成。
- [data/processed/hierarchy_development_audit.csv](data/processed/hierarchy_development_audit.csv)：开发集逐层节点数。
- [data/processed/hierarchy_audit.csv](data/processed/hierarchy_audit.csv)：全量逐层节点数。

开发层级只汇总抽中的叶序列。开发集的州／店节点不能与全量州／店销量混用，即使节点键相同也代表不同的叶集合。

## 6. 建立滚动起点与训练／验证／测试切分

**已完成：** 固定 28 天预测周期、365 天训练标签窗口、3 次滚动验证、独立测试和无标签最终预测。`origin_d` 表示最后一个已观测日，含当天。

| 切分 | 预测起点 | 训练标签范围 | 预测目标范围 | 用途 |
|---|---|---|---|---|
| validation_1 | d_1829 | d_1465–d_1829 | d_1830–d_1857 | 调参 |
| validation_2 | d_1857 | d_1493–d_1857 | d_1858–d_1885 | 调参 |
| validation_3 | d_1885 | d_1521–d_1885 | d_1886–d_1913 | 调参 |
| test | d_1913 | d_1549–d_1913 | d_1914–d_1941 | 配置锁定后独立评分 |
| forecast | d_1941 | d_1577–d_1941 | d_1942–d_1969 | 无真实标签的最终预测 |

**成果位置：** [data/processed/split_manifest.json](data/processed/split_manifest.json)，包含起点、日期、训练窗口、目标窗口及标签可用性。

为构造训练样本，另固定 `1437, 1465, ..., 1941` 共 19 个快照起点。训练行必须同时满足：

```text
row.origin_d < split.origin_d
AND split.train_start_d <= row.target_d <= split.train_end_d
```

评估行只取 `row.origin_d == split.origin_d`。每折每条序列覆盖 365 个训练标签日；更早的历史仅用于构造历史统计。测试期不能用于选择模型、特征或阈值。

## 7. 构建无跨期泄漏的特征与独立标签

**已完成：** 为开发集和全量数据分别生成 19 个起点、每个起点未来 28 天的直接多步特征，真实销量标签单独存储。

| 特征类别 | 实现口径 |
|---|---|
| 滞后销量 | lag 1、7、14、28、56；`sales_lag_k = sales[origin_d-k+1]`，lag1 为起点当天销量 |
| 滚动统计 | 截至起点的 7、28、56、90 天均值和标准差；`ddof=0`，包含零销量 |
| 日历与事件 | 目标日星期、月份、年份、事件、州 SNAP；假设计划日历在起点已知 |
| 历史价格 | 最近价格、价格变化、历史 52 周最高价、价格缺失与陈旧天数；只读取结束日不晚于起点的完整零售周 |
| 促销代理 | `discount_proxy = 1-price_last/price_max_52w`；超过 1% 标记 `promotion_proxy`，无参考价时为空；不代表已确认促销 |
| 销售历史 | 截至起点的首销年龄、距最近正销量的天数；从未正销时分别为 0、-1 |

**成果位置：**

- `data/processed/development/features/origin_*.parquet`：开发集特征，本地生成。
- `data/processed/development/targets/origin_*.parquet`：开发集真实标签，本地生成。
- `data/processed/final/features/origin_*.parquet`：全量特征，本地生成。
- `data/processed/final/targets/origin_*.parquet`：全量真实标签，本地生成。
- [data/processed/feature_list.json](data/processed/feature_list.json)：主键、标签定义与模型输入白名单。
- [data/processed/feature_missing_audit.csv](data/processed/feature_missing_audit.csv)：各起点特征缺失数及分母。

每行主键为 `(series_id, origin_d, target_d)`，`horizon = target_d-origin_d`。同一起点的历史统计在 28 个 horizon 上相同；模型应输入 horizon 或分 horizon 训练。最终 d_1942..d_1969 的 `units` 全为空，不用 0 代替未知销量。

## 8. 建立需求分层标签

**已完成：** 按每个预测起点的已观测历史生成销量规模、间歇性和生命周期标签，供模块 04 做异质性评估。

- 销量规模：最近 365 天日均销量分为 `zero=0`、`low=(0,1)`、`medium=[1,5)`、`high>=5`，阈值预先固定。
- 间歇性：ADI 为 `min(首销至起点天数,365)/最近365天正销量日数`，CV² 为正销量大小的总体方差除以均值平方；按 ADI=1.32、CV²=0.49 分为 smooth、erratic、intermittent、lumpy。最近 365 天无正销量为 no_demand，ADI、CV² 为空。
- 生命周期：依次判断从未正销、首销年龄≤90天、最近90天无销量、最近90天均值低于此前90天一半，其余为 established；这是历史启发式，不推断真实上市或退市日期。

**成果位置（本地生成）：**

- `data/processed/demand_strata_development.parquet`
- `data/processed/demand_strata_final.parquet`

标签主键为 `(series_id, origin_d)`，必须按对应评估起点连接，不能用最终起点标签标注更早的验证样本。

## 9. 完成复算验收并准备下游交接

**已完成：** 核验落盘文件校验值、序列数、日期范围、主键、标签可用性、每折训练覆盖、12 层映射与代表日期的加总守恒；对固定 SKU 的全部 19 个起点独立复算滞后、滚动统计、历史价格、SNAP 与标签对齐。

当前本地验收结果：7 项单元测试全部通过；独立验收状态为 `passed`，核验 95 个产物校验值；仅重建 manifest 后与现有文件逐字节一致。尚未另建全新环境验证。

**成果位置：**

- [tests/test_data.py](tests/test_data.py)：未来销量／价格扰动、窗口边界、缺价格、需求状态及抽样稳定性测试。
- [verify_data.py](verify_data.py)：独立落盘数据验收脚本。
- [data/processed/artifact_manifest.json](data/processed/artifact_manifest.json)：生成数据与契约文件的字节数和 SHA-256。
- [data/processed/run_report.json](data/processed/run_report.json)：运行环境、耗时与流水线检查结果。
- [data/processed/verification_report.json](data/processed/verification_report.json)：独立验收记录。

| 接收模块 | 已准备的交接成果 |
|---|---|
| 模块 02：模型 | 清洗数据、特征／标签、数据字典、split manifest、feature list 与读取示例 |
| 模块 03：调和 | 开发／全量 12 层映射、层级键与层级审计 |
| 模块 04：评估 | 清洗数据、切分窗口、逐起点需求分层标签 |

**待完成：** 模块 02 与模块 04 成员人工复核并联签；全组要求的非本人模块实质复核需在对应模块具备产物后另行记录。

## 复现上述成果

实现入口为 [01_data.py](01_data.py)，依赖版本见 [requirements.txt](requirements.txt)。推荐 Python 3.12，本地验收版本为 3.12.13。

先将步骤 1 的五个原始文件放入 `m5-forecasting-accuracy/`，然后在项目根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 01_data.py
python -m unittest discover -s tests -v
python verify_data.py
```

完整脚本同时生成开发集与全量数据，输出到 `data/processed/`。保留仓库中的 `raw_data_manifest.json`；输入不匹配时应检查来源版本，不应删锁后将不同数据当作同一版本。

只复算原始文件、开发名单与切分的最小样例：

```bash
python 01_data.py --manifest-only --output-dir /tmp/m5-manifest-rebuild
cmp data/processed/split_manifest.json /tmp/m5-manifest-rebuild/split_manifest.json
```

`cmp` 无输出且退出码为 0 表示逐字节一致。可用 `--raw-dir`、`--output-dir`、`--lock-file` 指定目录或原始文件锁。

### 模块 02：首折训练／验证读取样例

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

