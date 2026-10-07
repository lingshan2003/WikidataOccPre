# CBDB：服务器分期 RGCN + GraphMask 流水线

默认从已经清洗好的 **92,168 条原始关系码三元组**开始，自动完成关系分组、7 个时期划分、PyG 构图、RGCN 训练、GraphMask 训练与测试报告、结果汇总。默认跑 **7 个时期 × multi_group/binary × seed 42 = 14 个独立模型**。也可从原始 SQLite 重新执行清洗。

## Git 同步与仓库根目录运行

代码和默认清洗输入均通过 Git 同步。所有命令在包含 `run.py`、`CBDB/`、`config/` 的仓库根目录执行，例如服务器已有的 `WikidataOccPre` checkout；不需要另建或进入 `cbdb_server` 目录。

```bash
# 当前已经在服务器的项目根目录。
git pull --ff-only
# 先激活已有的 PyTorch/PyG 环境。
python -m pip install -r CBDB/requirements.txt
python CBDB/install_sampling.py --run
```

默认输入为 Git 中的 `CBDB/data/person_relation_triples.tsv.gz`，约1.8MB。原来的 `external_data/` 被忽略，因此已将这份清洗表放入可提交的目录。修改须先在本地提交并推送到服务器使用的分支，然后服务器再 pull。原始数据库、训练产物和虚拟环境保持独立。

## 开跑前选择空闲 GPU

先查看服务器各卡的利用率、显存和占用进程，再选择可用卡。以下**假设确认物理2号卡空闲**；把 `2` 换为实际可用编号。脚本不会自动抢占或选择显卡。

```bash
nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

bash CBDB/run_pipeline.sh --check-env --device cuda:0
bash CBDB/run_pipeline.sh run all --device cuda:0
```

`CUDA_VISIBLE_DEVICES=2` 后，进程只看见所选的一张卡，它在进程内编号为 `cuda:0`。因此后面的 `--device` 保持 `cuda:0`。RGCN、GraphMask及其子进程继承相同环境变量，均使用这张卡。预检会打印 `CUDA_VISIBLE_DEVICES`、逻辑device与GPU名称。

`install_sampling.py` 根据当前 PyTorch 与其 CUDA 构建版本选择 PyG wheel 页面，只安装二进制包；默认不加 `--run` 时仅显示安装命令。匹配方法依据 [PyG 官方安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。请先安装适合服务器驱动的 PyTorch 构建；已有可用采样环境时可以直接跳过安装步骤。`--check-env` 实际运行一次 `NeighborLoader`，验证采样扩展，而不只检查能否 import。

指定虚拟环境解释器：

```bash
CBDB_PYTHON_BIN=/path/to/venv/bin/python bash CBDB/run_pipeline.sh run all --device cuda:0
```

## 预览、试跑和正式运行

```bash
# 只显示执行计划，标准库即可，不写数据。
bash CBDB/run_pipeline.sh plan all

# 真实的 1700+ 图、两种关系表示，CPU，小模型、1 轮。
# 输出独立放在 artifacts/cbdb_pipeline_v1_smoke。
bash CBDB/run_pipeline.sh run all --smoke

# 将 CPU 小规模试跑扩展到所有 7 期。
bash CBDB/run_pipeline.sh run all --smoke --periods all

# 正式运行：默认 RGCN 最多50轮，GraphMask每层3轮。
bash CBDB/run_pipeline.sh run all --device cuda:0

# 单期、单表示、多个模型种子；节点划分种子固定。
bash CBDB/run_pipeline.sh run all --periods 1100_1299 --representations multi_group \
  --seeds 42,43,44 --device cuda:0 --output-root artifacts/cbdb_1100_multi_3seeds
```

`--epochs`、`--graphmask-epochs`、`--workers` 可显式覆盖默认设置，优先于 `--smoke` 默认值。训练模式为 sampled，两个消息传递层使用 `15,10` 邻居采样。GraphMask 的 `auto` 延用检查点的采样规模，用 train 拟合、val 选择 probe、test 导出报告；默认保真约束为相对 macro-F1 差不超过 0.05，失败会保留日志并标记失败，不自动放宽约束。

配置文件是 `config/cbdb_pipeline_v1.json`。修改实验设置后使用新的 `--output-root`，避免旧结果被混入。单张 GPU 按作业顺序执行。OOM 时可在新配置中减小训练/GraphMask batch size、hidden dimension 或 fanouts，再用新输出目录执行。

## 分阶段与续跑

`run all` 完成全部阶段，也支持：

```bash
bash CBDB/run_pipeline.sh run group
bash CBDB/run_pipeline.sh run periods
bash CBDB/run_pipeline.sh run prepare
bash CBDB/run_pipeline.sh run train --device cuda:0
bash CBDB/run_pipeline.sh run graphmask --device cuda:0
bash CBDB/run_pipeline.sh run summarize
```

单跑阶段要求前置阶段已完成。解释器、输入和设置保持一致时，重复原命令会复用完成的阶段、重跑失败或中断的阶段；这属于阶段级续跑，失败的训练阶段会重新训练。文件大小、mtime、完整产物和执行参数用于判断复用，新 runner 不额外计算 SHA256。修改/删除完成产物或改动输入、训练设置时会拒绝混用，并提示使用新输出目录。日志和命令留存在对应阶段，根目录 `.pipeline.lock` 防止两个进程同时写同一批结果。

添加时期、表示或模型 seed 时，仍可复用已完成的相应产物；数据分期阶段固定导出全部7期，两种表示都保留。模型种子可以不同，但同一期两种表示的节点顺序、标签、特征、train/val/test 掩码必须一致，runner 会实际比较。

## 从 SQLite 重建

原始SQLite不进入Git。如服务器已有数据库，需要重新清洗时，沿用前面选择的GPU并指定 `--database`：

```bash
bash CBDB/run_pipeline.sh run all --from-sqlite \
  --database /data/cbdb_20260914.sqlite3 --device cuda:0 \
  --output-root artifacts/cbdb_from_sqlite_v1
```

原库使用只读 SQLite 连接。单独执行 `run clean` 会生成输出目录中的 flat 表；随后分阶段执行时也传 `--from-sqlite`，从该输出表接续。

## 数据与模型口径

| period ID | 时期 | 归期人物 | 实际构图人物 | 多组三元组 | 二分类三元组 |
|---|---|---:|---:|---:|---:|
| to_699 | 700年以前 | 2,345 | 2,031 | 3,762 | 3,736 |
| 700_899 | 700–899年 | 8,263 | 7,807 | 16,038 | 15,867 |
| 900_1099 | 900–1099年 | 3,163 | 2,771 | 12,311 | 11,619 |
| 1100_1299 | 1100–1299年 | 4,178 | 3,985 | 20,702 | 19,287 |
| 1300_1499 | 1300–1499年 | 3,113 | 2,875 | 12,942 | 12,535 |
| 1500_1699 | 1500–1699年 | 3,038 | 2,928 | 15,952 | 15,523 |
| 1700_plus | 1700年及以后 | 1,489 | 1,423 | 3,704 | 3,576 |

时期按“至少一个已知生卒年在区间内”归属，采用闭区间，不插补缺失年份。每条期内三元组要求两端均归该期；人物及边可以跨期重复，不能把各期当作互斥队列。每期留下全部归期人物 `eligible_nodes.tsv.gz` 与实际涉及期内关系的人物 `active_nodes.tsv.gz`。现有边表构图流程只将 active 人物纳入图，期内孤立人物保留在审计表。

multi_group 沿用12个关系大组，binary 为 inherited/acquired。稀有关系在某期可能完全没有边：该期 taxonomy/词表只记录实际出现的组，避免主程序因词表中不存在的组报错；完整12组仍在汇总中列出，无边/无采样消息与零保留率分开记录。多组各期实际基础组数依次为9、11、11、12、12、12、12，模型添加 `__rev` 后类型数翻倍。RGCN basis 数自动取配置值与该期实际类型数中的较小值。

职业沿用现有单层试验标签，仅写入 `occ_level1`，其它职业层和国家为空。每期按职业分层划分70/10/20，固定 split seed 20261006；该期不足3人的职业保持在图中，但标签置为 `-1`，不进入监督指标。验证/测试人物的职业特征为 unknown，训练和 GraphMask 前向隐藏当前 seed 自身的职业，保留现有项目的防泄漏协议。

实际构图后，700年前的职业分布为做官2,027、写作2、宗教1、艺术1；按每类至少3人的监督门槛，只剩“做官”一个监督类别。runner继续完成该期流程，同时显示 `single-class target`，汇总表标记 `target_status=single_class`。该期即使 macro-F1=1，也不能作为多职业预测效果；其余时期保留的监督类别数依次为4、7、9、8、6、7。

## 独立脚本

```bash
# 原库 -> 职业/年份合格、去重的原始关系三元组
python scripts/cbdb_build_flat_triples.py --database PATH --output-dir OUT

# 原始关系码 -> multi_group/binary，保留原码与来源支持
python scripts/cbdb_build_experiment_groups.py --input FLAT.tsv.gz --output-dir GROUPED

# 大组表 -> 7期诱导关系表 + 主实验接口CSV
python scripts/cbdb_split_periods.py --input-dir GROUPED --output-dir PERIODS

# 示例：构图与三阶段模型命令；pipeline中自动拼接路径和配置。
python run.py prepare --input PERIODS/1700_plus/multi_group/Q_R_Q_extended.csv.gz \
  --output-dir GRAPH --target-level 1 --min-class-count 3 --seed 20261006
python run.py train --data GRAPH/graph_data.pt --output-dir MODEL --model rgcn \
  --occupation-feature-levels 1 --auxiliary-features temporal --num-bases 14 \
  --tie-taxonomy PERIODS/1700_plus/multi_group/tie_taxonomy.json \
  --relation-taxonomy PERIODS/1700_plus/multi_group/relation_taxonomy.json --device cuda:0
python run.py graphmask-train --data GRAPH/graph_data.pt --checkpoint MODEL/best_model.pt \
  --output-dir MASK --num-neighbors auto --device cuda:0
python run.py graphmask-report --data GRAPH/graph_data.pt --checkpoint MODEL/best_model.pt \
  --probe MASK/graphmask_probe.pt --output-dir MASK/test_report --split test --device cuda:0
```

独立分期脚本默认拒绝覆盖非空目录；`--overwrite` 只接受自身生成的分期目录，并先 staging 再替换。原有细分关系脚本 `scripts/cbdb_group_relations.py` 保留作审计；模型流水线使用 `cbdb_build_experiment_groups.py` 的大组表示。

## 输出位置

默认根目录 `artifacts/cbdb_pipeline_v1/`：

```text
grouped/                         全局大组/二分类表与映射审计
periods/<period>/<representation>/  分期表、CSV、节点档、实际关系词表
graphs/<period>/<representation>/   graph_data.pt、nodes/edges.csv、划分与类别统计
models/<period>/<representation>/seed_<seed>/
                                 best_model.pt、metrics.json、test_predictions.csv
graphmask/<period>/<representation>/seed_<seed>/
                                 probe、validation.json、训练历史
  test_report/                   测试指标、每层关系保留率、root_top_edges.csv.gz
summary/matrix_summary.tsv       各作业状态、人数、类别数、RGCN/GraphMask指标
summary/relation_group_summary.tsv  各时期、表示、种子、层、关系组的保留率
summary/failures.json             失败详情
```

日志 `train.log`、`probe.log`、`report.log` 和相应 `*_command.sh` 记录具体执行过程。失败会返回非零退出码，剩余模型作业继续执行并汇总，便于服务器批跑后定位。GraphMask 指标中的 message observations 来自采样计算图，可以包含重复边观察；它与原始三元组数量不同。

## 本地验证

```bash
python3 -m unittest discover -s tests -p 'test_cbdb*.py'
```

30项自动测试通过。已核对全部7期、两种表示的导出数量与字段保留；CPU smoke已实际完成14/14条构图→RGCN→GraphMask→test报告链路，未放宽0.05保真约束。同命令重跑14/14复用完成产物，模型检查点mtime和大小均未变化。另验证过独立目录的1700+两种表示链路。从只读SQLite通过新runner重建的flat表也与当前清洗结果一致。

小模型和1轮指标仅用于检查管线能否执行；正式服务器运行使用默认训练配置。本地未验证CUDA正式训练。
