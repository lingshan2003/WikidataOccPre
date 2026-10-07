# DBpedia 分时期二分组／多关系组 RGCN + GraphMask

入口：`DBpedia/run_grouped_rgcn_graphmask_20y.sh`。实验配置：`config/dbpedia_grouped_rgcn_graphmask_20y_v1.json`。分组口径固定为 `config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json`，与前一轮审阅草案成员一致。

## 默认实验矩阵

现有八个时期：≤1500、1501–1900、1901–1920、1921–1940、1941–1960、1961–1980、1981–2000、≥2001。

每个时期从同一张原始谓词时期图独立生成两张图、训练两个 RGCN，然后各自训练 GraphMask：

| 表示 | 基础关系类型 | 模型有向关系 ID 数 | RGCN basis 数 |
| --- | --- | ---: | ---: |
| binary | inherited_ties / acquired_ties | 4 | 4 |
| multi_group | inherited, intimate_partnership, education_mentorship, professional_collaboration, influence_succession, religious_authority_recognition, other_acquired | 14 | 14 |

默认 **16 个独立 RGCN、16 个 GraphMask probe、16 份测试报告**。`--include-full` 可增加全图的两个模型，合计 18 套。原始 33 谓词基线仍由 `DBpedia/run_rgcn_graphmask_20y.sh` 运行，本入口只执行 binary / multi_group 矩阵。

两种表示均直接从原始时期图重编码，多组图不从二分组图生成。保留生成的反向消息类型；同一个 source/group/target 重复三元组会合并。节点、监督标签、train/val/test 划分、职业和时间特征保持一致，并在 collapse 后核验。记录每张图的 `edges_before`、`edges_after`、`duplicates_removed`。

RGCN 默认两层，隐藏维度 128，特征分支 64，dropout 0.2，采样 `15,10`，batch 512，最多 50 epoch，以验证 Macro-F1 早停（patience 6、min_delta 0.001），seed 42；使用 `occupation_feature_levels=1` 与 `auxiliary_features=temporal`，沿用 DBpedia 原始谓词实验的特征口径。basis 数针对压缩后的关系数设为 4/14，避免沿用原始 66 ID 模型的 30 bases；该值在 JSON 中显式记录。

GraphMask 默认 `num_neighbors=auto` 复用每个 RGCN 的 fanouts，batch 32、workers 0、每层 3 epoch、beta 0.03；验证集原模型和掩码后 Macro-F1 的相对差必须 ≤0.05。报告使用 test split，seed 42，top-k 50。不同表示的 GraphMask 都解释各自独立训练的冻结模型。

默认从所有训练阶段选择最稀疏且通过保真度的 GraphMask，可能选中仅启用 Layer 1 门控的版本。
针对五个 multi_group 时期的“仅从 Layer 0 门控已启用后选择”补充实验，见 [LAYER0_ENABLED_SUPPLEMENT.md](LAYER0_ENABLED_SUPPLEMENT.md)。该入口复用原 RGCN，另存 GraphMask 结果。

## 服务器运行

在已跑通原始 DBpedia RGCN / GraphMask 的同一仓库和 Python 环境中，先同步这些新增文件：

```text
DBpedia/run_grouped_rgcn_graphmask_20y.sh
DBpedia/grouped_pipeline.py
DBpedia/GROUPED_PIPELINE.md
config/dbpedia_grouped_rgcn_graphmask_20y_v1.json
config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json
```

并确认已有仓库文件也是当前版本：`run.py`、`cli.py`、`data/collapse_ties.py`、`data/collapse_relations.py`、`data/birth_cohort_artifacts.py`、`scripts/prepare_life_period_induced_artifacts.py`、`training/tie_taxonomy.py`、`training/train.py`、`training/graphmask_train.py`、`training/graphmask_report.py`、`training/graphmask/`，以及两个原有配置 `config/dbpedia_tie_taxonomy_v1.json` 和 `config/dbpedia_life_periods_20y_v1.json`。最好同步当前仓库代码整体，保留服务器的 data/artifacts/runs 目录。只上传 Bash 文件不足以运行。

默认需要已准备好的全图 `artifacts/dbpedia_2022_priority_v1/graph_data.pt` 及其 `nodes.csv`、`edges.csv`、`split_summary.json`。八张原始时期图位于 `artifacts/dbpedia_2022_priority_periods_20y_v1/`；已有的兼容图直接复用，缺失的时期才会由全图生成。本入口不重新抽取原始 TTL；如全图尚未准备，请先执行原入口的 `run data`。

```bash
export DBPEDIA_PYTHON_BIN="$(command -v python)"
nvidia-smi
# 根据实际空闲卡选择；这里的 2 只是示例物理卡号。
export CUDA_VISIBLE_DEVICES=2

bash DBpedia/run_grouped_rgcn_graphmask_20y.sh plan all
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run all
```

`CUDA_VISIBLE_DEVICES=2` 后，进程内仍用默认 `cuda:0`。脚本会检查 CUDA 和 PyG NeighborLoader；同一时间只运行一个 GPU 任务。

先跑一个时期验证环境：

```bash
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run all --periods 1901_1920
```

若需脱离终端后台执行，在仓库根目录：

```bash
mkdir -p runs/dbpedia_grouped_20y_v1
nohup bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run all \
  > runs/dbpedia_grouped_20y_v1/pipeline.log 2>&1 &
```

## 分阶段与重试

```bash
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run prepare
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run collapse
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run train
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run graphmask
bash DBpedia/run_grouped_rgcn_graphmask_20y.sh run summarize
```

每个阶段也支持 `plan`；`plan` 不写文件、不加载 torch，也不启动任务。`--periods` 支持逗号分隔的时期 ID；`--representations binary` 或 `--representations multi_group` 可只跑一种表示。参数需要放在 `plan|run` 和阶段名之后。

完整且输入／参数兼容的本流程产物自动跳过；某个任务失败会记录错误，继续其余时期／表示，最后返回非零退出码。GraphMask 若未通过保真度，不降低阈值，保留训练历史；随后重新执行相同命令会重试失败任务，已完成的任务跳过。训练器没有 epoch 级恢复，失败阶段从头重跑；已经完成的 RGCN 不会因为 GraphMask 失败而重训。测试报告生成中断时也可在同配置下重新执行 graphmask 阶段；曾标记完成而后来被删除／修改的产物会触发不兼容检查。

`pipeline_collapse.json`、`pipeline_train.json`、`pipeline_probe.json`、`pipeline_report.json` 保存每个阶段的命令、输入大小／修改时间、配置以及输出状态。已有无记录的文件、已完成后被修改的输出，或输入／参数与旧记录不一致时，拒绝复用；改用新的输出目录。这里不额外扫描 SHA256；原有重编码／训练／GraphMask 模块内建的溯源字段和 probe 来源检查继续工作。同一个 model root 有进程锁，进程退出或被终止会自动释放。

## 配置与覆盖

推荐复制实验 JSON 修改参数，并传 `--config config/你的配置.json`。改变采样、特征、seed 或训练／GraphMask 参数时，选择相应的新实验输出目录。常用环境变量：

| 环境变量 | 默认值／作用 |
| --- | --- |
| DBPEDIA_PYTHON_BIN | python；选择已安装 CUDA PyTorch / PyG 的解释器 |
| DBPEDIA_GROUP_CONFIG | 实验 JSON 路径 |
| DBPEDIA_GROUP_SOURCE_DATA | 全图 graph_data.pt 路径 |
| DBPEDIA_GROUP_PERIOD_ROOT | 已有原始谓词时期图根目录 |
| DBPEDIA_GROUP_RELATION_ROOT | artifacts/dbpedia_grouped_20y_v1 |
| DBPEDIA_GROUP_MODEL_ROOT | runs/dbpedia_grouped_20y_v1 |
| DBPEDIA_GROUP_GRAPHMASK_ROOT | runs_graphmask/dbpedia_grouped_20y_v1 |
| DBPEDIA_GROUP_PERIODS | all，或逗号分隔时期 ID |
| DBPEDIA_GROUP_REPRESENTATIONS | binary,multi_group |
| DBPEDIA_GROUP_INCLUDE_FULL | 0；设为 1 增加全图 |
| DBPEDIA_GROUP_DEVICE | cuda:0；也可传 --device |
| DBPEDIA_GROUP_SEED | 42 |
| DBPEDIA_GROUP_TRAIN_BATCH_SIZE | 512 |
| DBPEDIA_GROUP_TRAIN_WORKERS | 4 |
| DBPEDIA_GROUP_TRAIN_EPOCHS | 50 |
| DBPEDIA_GROUP_TRAIN_FANOUTS | 15,10 |
| DBPEDIA_GROUP_GRAPHMASK_BATCH_SIZE | 32 |
| DBPEDIA_GROUP_GRAPHMASK_WORKERS | 0 |
| DBPEDIA_GROUP_GRAPHMASK_EPOCHS_PER_LAYER | 3 |
| DBPEDIA_GROUP_GRAPHMASK_BETA | 0.03 |
| DBPEDIA_GROUP_GRAPHMASK_MAX_RELATIVE_F1_DIFF | 0.05；正式对照应保持一致 |

显存不足时可在开始新实验前降低 batch size。改变 RGCN 参数需要新 model / GraphMask roots；改变关系映射或原图还需要新 relation root。若仅改 GraphMask 参数，可复用相同 RGCN，选择新 GraphMask root。

## 结果目录与汇总

```text
artifacts/dbpedia_grouped_20y_v1/{时期}/{binary,multi_group}/
  graph_data.pt, nodes.csv, edges.csv, split_summary.json
  relation_collapse_manifest.json
  binary_tie_taxonomy.json 或 collapsed_tie_taxonomy.json

runs/dbpedia_grouped_20y_v1/{时期}/{表示}/seed_42/
  best_model.pt, metrics.json, test_predictions.csv

runs_graphmask/dbpedia_grouped_20y_v1/{时期}/{表示}/seed_42/
  graphmask_probe.pt, validation.json, training_history.json, manifest.json
  test_report/{test_metrics.json,relations_base.csv,relations_directed.csv,root_top_edges.csv.gz,manifest.json}

runs_graphmask/dbpedia_grouped_20y_v1/
  matrix_summary.tsv, matrix_summary.json
  relation_group_summary.tsv, relation_group_summary.json
  pipeline_failures.json
```

每个阶段目录另有命令 `.sh`、日志 `.log` 和状态 `.json`。`matrix_summary` 汇集各时期／表示的图规模、重编码删重数量、RGCN 测试 Macro-F1、GraphMask 采样原模型／掩码后的 Macro-F1、prediction agreement、retention 和验证保真度。训练器测试指标与 GraphMask 原模型指标可能使用不同采样图，分别保留，不能要求数值完全一致。

`relation_group_summary` 按时期／表示／层／组导出 GraphMask 留存率和保留消息占比，同时保留重编码后的原方向图边数和消息观测数。没有该组原图边时标记 `no_graph_edges`；有边但测试采样未观测到时标记 `no_sampled_messages`；这两种情况的留存率留空，不填成 0。当前 ≤1500 时期的职业合作组没有边，仍保留统一的 14 个模型关系 ID。

时期图按人物生命区间构造，不意味着关系发生在该时期；时期之间会重叠。这里的二分组是 inherited/acquired 关系类型重编码，不是人物二元组任务，也不是每个 acquired 小组各训练一个模型的消融矩阵。

## 本地验证记录

已实际生成八时期 × 两种表示的 16 张图，并逐张确认节点、标签、特征与三种 split mask 不变；第二次执行 collapse 全部正确跳过。在 ≤1500 时期用 CPU、小隐藏维度、一个 RGCN epoch、GraphMask 每层一个 epoch 跑通两种表示的完整链路，生成两份 probe/test report 和汇总，并验证无职业合作边时留存率留空。另有命令生成、真实 CLI 参数解析、阶段复用／参数变更拒绝、失败重试及缺失产物拒绝的单元测试。

这些是接口与流程验证，CPU 测试目录位于 `/private/tmp`，未写入正式模型输出目录；不代表服务器默认参数下的正式准确率、Macro-F1 或 GraphMask 稀疏性结果。CUDA 默认矩阵须在服务器执行后确认。
