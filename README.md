# Wikidata 人物职业预测

基于 Wikidata 人物关系网络的多关系图学习项目。项目以人物为节点、社会关系为带类型的边，在职业 Level 1/2/3 三个粒度上预测人物职业，并分析模型使用了哪些关系与邻居信息。

当前主线任务是**多关系人物图上的节点分类**；`link_prediction/` 是独立的异构图链接预测探索。项目研究的是职业与社会关系之间的预测关联，不把 attention、归因值或删边结果直接解释为社会因果关系。

## 目录

- [任务与数据](#任务与数据)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [模型与训练协议](#模型与训练协议)
- [实验控制与分析](#实验控制与分析)
- [命令索引](#命令索引)
- [项目结构](#项目结构)
- [测试与复现](#测试与复现)

## 任务与数据

原始数据文件为根目录下的 `Q_R_Q_extended.txt`。它实际是一个以边为行的 CSV 表，每行包含：

- 两个人物的 Wikidata Q-ID；
- 两人之间的关系；
- 两端人物的出生、死亡、国家和三层职业属性。

预处理会把重复的端点属性规范化为唯一人物表，审计冲突属性，为每条关系增加 `__rev` 反向关系，并保存 PyTorch Geometric 图对象。

当前完整导出的典型规模为：

| 项目 | 数量 |
| --- | ---: |
| 人物节点 | 334,099 |
| 原始人物关系边 | 691,503 |
| 加反向边并去重后的有向边 | 1,383,006 |
| 有向关系类型 | 80 |
| Level 3 类别（`min-class-count=20`） | 682 |

这些数字来自当前本地 artifact；更换数据后应以新生成的 `split_summary.json` 为准。

### 标签层级

- Level 1：粗粒度职业大类；
- Level 2：中等粒度职业类别；
- Level 3：细粒度职业类别。

三个层级分别准备数据、训练和评估。低频职业与无标签人物仍保留为图中的邻居，但其 `y=-1`，不参与分类损失和指标计算。

### 防止目标泄漏

职业既是预测目标，也是邻居信息，因此代码执行以下协议：

- 训练人物的 L1/L2/L3 职业可作为其他人物的邻居特征；
- 验证和测试人物的三层职业始终编码为 `__UNKNOWN__`；
- 每次 sampled forward 中，当前 seed 人物的三层职业会再次临时 mask；
- 人物关系边不会因职业标签被 mask 而删除；
- 验证和测试标签只用于计算指标，不能作为模型输入。

因此，本项目的默认任务不是“已知目标人物上层职业后预测其细粒度职业”。如需研究该任务，应单独定义条件式分类实验，不能与当前结果混报。

## 环境安装

建议使用 Python 3.9 或更高版本，并在独立虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

服务器使用 CUDA 时，建议先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与 CUDA 匹配的 PyTorch，再安装 `requirements.txt` 中的其余依赖。

`sentence-transformers` 只在 `occupation-embed` 工作流中使用，Matplotlib 只用于分析 notebook；主训练流程不需要下载语言模型。原始数据、预处理产物、模型 checkpoint 和分析报告均被 Git 忽略，不会随仓库提供。

## 快速开始

### 1. 准备图数据

把 `Q_R_Q_extended.txt` 放在项目根目录，然后为目标层级生成独立 artifact：

```bash
python run.py prepare \
  --input Q_R_Q_extended.txt \
  --output-dir artifacts/level3_hierarchy \
  --target-level 3 \
  --min-class-count 20 \
  --seed 42
```

主要输出如下：

| 文件 | 内容 |
| --- | --- |
| `graph_data.pt` | PyG 图、特征、标签、split mask 与 metadata |
| `nodes.csv` | 节点索引与 Wikidata Q-ID、规范化属性 |
| `edges.csv` | 有向关系边及 relation ID |
| `attribute_conflicts.csv` | 同一人物重复属性的冲突审计 |
| `class_stats.csv` | 目标职业频数 |
| `relation_stats.csv` | 关系频数 |
| `split_summary.json` | 数据规模、划分与预处理配置 |

同一层级的模型比较必须读取同一份 `graph_data.pt`。Level 1/2/3 默认各自按目标标签分层划分；若要跨层级比较完全相同的人物集合，需要另行固定共享 split。

### 2. 训练模型

```bash
python run.py train \
  --model rgat \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgat \
  --epochs 50 \
  --batch-size 512 \
  --num-layers 2 \
  --num-neighbors 15,10 \
  --hidden-dim 128 \
  --branch-dim 64 \
  --heads 4 \
  --seed 42 \
  --device auto
```

把 `--model` 改为 `rgcn` 或 `compgcn` 即可使用相同数据与训练协议比较模型。sampled 模式下，`--num-neighbors` 的 fan-out 数必须与 `--num-layers` 一致。

训练目录包含：

- `best_model.pt`：验证集选择出的 checkpoint 与完整配置；
- `metrics.json`：训练历史、选择指标、图扰动配置和最终指标；
- `test_predictions.csv`：测试预测；使用 `--skip-test` 时不生成。

默认以 validation loss 选择 checkpoint，采用 AdamW、梯度裁剪、`ReduceLROnPlateau` 和早停。测试集仅在最佳 checkpoint 确定后评估一次。探索配置时可加 `--skip-test`，锁定配置后再执行一次最终测试。

`--train-mode full` 只适用于不暴露职业输入的实验（可保留普通属性，或使用 `--feature-mode structural`），因为单次全图 forward 无法逐个隐藏训练 seed 自身的职业；确定性的验证/测试全图推理可使用 `--eval-mode full`。

### 3. 运行测试

```bash
python -m unittest discover -s tests -v
```

## 模型与训练协议

### 节点特征

| 特征 | 编码 | 默认可见性 |
| --- | --- | --- |
| `occupation_level1/2/3` | 独立类别 embedding | 仅训练人物可见，当前 seed 再 mask |
| `country` | 类别 embedding | 所有节点可见 |
| `temporal` | 出生年、死亡年、年龄及缺失标记 | 所有节点可见 |

`NodeFeatureEncoder` 按 schema 为每类特征建立独立分支，经 feature gate 和融合层生成初始节点表示。可用以下参数控制输入：

- `--occupation-feature-levels 1,2,3`：默认层级职业输入；
- `--occupation-feature-levels 3`：只使用邻居 Level 3；
- `--occupation-feature-levels none`：不使用职业特征；
- `--auxiliary-features country,temporal` 或 `none`；
- `--feature-mode structural`：所有人物共享一个常量特征，只保留拓扑和 relation type。

### 可用模型

| 模型 | 实现 | 说明 |
| --- | --- | --- |
| R-GCN | `models/rgcn.py` | `FastRGCNConv`/`RGCNConv` 关系卷积基线 |
| R-GAT | `models/rgat.py` | 多头关系注意力，可导出 attention 与归因 |
| CompGCN | `models/compgcn.py` | 组合源节点状态与 relation embedding |

三种模型共用数据、特征编码、采样、优化、checkpoint 选择和指标实现。主指标包括 Accuracy、Macro-F1、Weighted-F1、Macro-Precision 与 Macro-Recall。

### 固定职业语义向量

`occupation-embed` 从已准备的 categorical artifact 生成新文件，不覆盖输入，也不改变 split：

```bash
python run.py occupation-embed \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output artifacts/level3_hierarchy/graph_data_semantic.pt \
  --model-name intfloat/multilingual-e5-base \
  --device auto

python run.py train \
  --model rgcn \
  --data artifacts/level3_hierarchy/graph_data_semantic.pt \
  --output-dir runs/level3_rgcn_semantic \
  --occupation-representation semantic \
  --seed 42
```

只有训练人物可见的职业组合会被编码；validation、test 和当前 seed 使用共享 unknown 表示。模型 revision、prompt 指纹和来源 artifact 会写入 metadata/checkpoint。

## 实验控制与分析

### 关系和长尾控制

`train` 已支持以下受控实验：

- `--shuffle-relation-types`：保留拓扑和 relation 频数，随机重分配 relation ID；
- `--drop-relation-groups`：按语义组删除正向与反向关系；
- `--drop-relations`：删除指定原始关系及其反向边；
- `--match-random-drop-to-relation-groups`：匹配关系组规模的随机关系对删除；
- `--loss inverse_frequency|class_balanced|logit_adjusted`：长尾损失；
- `--train-root-sampling class_balanced`：类别均衡的训练 seed 采样。

所有运行时图扰动只作用于内存副本，不覆盖 `graph_data.pt`；实际删边数量、关系选择和随机 seed 会记录到输出配置。

### 图与预测诊断

```bash
python run.py diagnose \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir diagnostics/level3

python run.py diagnose \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --predictions runs/level3_rgat/test_predictions.csv \
  --output-dir diagnostics/level3_rgat
```

该命令只读取 artifact 和可选预测文件，导出图覆盖、连通性、职业可见性、关系同配性及按覆盖度分组的预测指标，不训练或改写模型。

### R-GAT 解释与关系分析

单节点候选边导出：

```bash
python run.py explain \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --checkpoint runs/level3_rgat/best_model.pt \
  --node-id Q1000023 \
  --output-dir explanations/Q1000023
```

批量分析入口的统计单位不同，使用时不要混淆：

| 命令 | 输出含义 |
| --- | --- |
| `attention-edge-report` | 以边为观测的 head 平均 attention |
| `attention-node-report` | 每个预测 root 内按关系/source L1 分组的原始 attention mass |
| `attention-rollout-report` | 两层 RGAT 的 typed two-hop attention product 候选分数 |
| `message-contribution-report` | `alpha × relation value vector` 的节点级消息幅值 |
| `gradient-attribution-report` | 对预测分数的 signed `alpha × gradient` 局部归因 |
| `relation-pair-ablation-report` | 指定职业—关系—职业 motif 与匹配随机对照的删边影响 |
| `relation-pair-sweep-report` | 一跳 RGAT 中所有实际 relation pair 的快速条件反事实扫描 |
| `attention-bootstrap` | 以预测 root 为重抽样单位的置信区间 |

`attention-report` 仅是 `attention-edge-report` 的兼容别名。所有关系名均按精确方向解释；例如 `child` 与 `child__rev` 不得自动合并。attention、rollout、message 和 gradient 是模型机制或局部敏感性描述，不能单独证明边删除效应，更不能证明社会因果关系；优先用配对扰动实验验证候选关系。

一跳节点权重表可从 `attention-node-report` 的输出构建：

```bash
python scripts/build_rgat_one_hop_node_weight_table.py \
  --input-dir runs_report/level1/rgat_l1_root_attention_all_relations/direct \
  --output-dir runs_report/level1/rgat_l1_root_attention_all_relations/direct/node_weight_tables
```

其中 `mean_a` 是“拥有该 pair 的目标节点内，匹配入边 attention mass 之和的节点平均”；`n` 是涉及的目标节点数，不是边数。`coverage` 和 `attention_share_of_Ot_budget` 应同时用于识别高权重但极低覆盖的稀有 pair。

分析 notebook 位于 `notebooks/rgat_l1_attention_message_heatmaps.ipynb`，用于读取已有 CSV 并生成 `visualization/` 下的热力图。底层分析均通过上表中的 `run.py` 子命令调用，不再保留绑定特定 L1 checkpoint 和固定目录的一次性 shell 包装。

正式三 seed 实验矩阵保留统一批处理入口：

```bash
bash scripts/run_report_experiments.sh plan all
bash scripts/run_report_experiments.sh run all
python scripts/summarize_report_runs.py --root runs_report/level3
```

第二个参数也可单独选择 `architecture`、`features`、`relations`、`longtail` 或 `coverage`；`run` 会跳过已经生成 `metrics.json` 的 seed 目录。

### 独立的链接预测探索

该分支构建 `Person --has_occupation--> Occupation` 异构图，使用 HGT 编码节点并以点积预测职业边：

```bash
python run.py link-prepare \
  --input Q_R_Q_extended.txt \
  --output-dir link_artifacts/level3 \
  --target-level 3 \
  --min-class-count 20 \
  --seed 42

python run.py link-train \
  --data link_artifacts/level3/hetero_graph.pt \
  --output-dir link_runs/level3_hgt \
  --device auto
```

链接预测与主线节点分类回答不同问题，结果不应直接横向比较。

## 命令索引

统一入口为：

```bash
python run.py <command> --help
```

| 类别 | 命令 |
| --- | --- |
| 数据 | `prepare`, `occupation-embed`, `link-prepare` |
| 训练 | `train`, `link-train` |
| 诊断/个例 | `diagnose`, `explain` |
| Attention | `attention-edge-report`, `attention-node-report`, `attention-rollout-report` |
| 归因/扰动 | `message-contribution-report`, `gradient-attribution-report`, `relation-pair-ablation-report`, `relation-pair-sweep-report` |
| 统计 | `attention-bootstrap` |

## 项目结构

```text
RGCN/
├── run.py                    # 统一命令入口
├── cli.py                    # 子命令路由
├── requirements.txt          # Python 依赖
├── data/                     # 原始表规范化、图准备、职业语义特征
├── models/                   # 特征编码器与 R-GCN/R-GAT/CompGCN
├── training/                 # 训练、诊断、attention、归因与扰动分析
├── link_prediction/          # 独立异构图链接预测分支
├── scripts/                  # 可复现实验与报告脚本
├── notebooks/                # 交互式结果分析
├── tests/                    # 合成图回归测试
└── legacy/                   # 仅供历史对照的旧基线
```

运行时目录不会提交到 Git：

- `artifacts/`、`link_artifacts/`：预处理数据；
- `runs/`、`link_runs/`、`runs_report/`：checkpoint、指标与批量报告；
- `diagnostics/`、`explanations/`：诊断和个例解释；
- `visualization/`：notebook 生成的图片。

`legacy/original_baseline/` 是项目接收时的旧实现，依赖当前仓库未提供的数据，并在 mini-batch 中重复做全图卷积。它只用于来源历史对照；新实验请使用 `python run.py train --model rgcn`。

## 测试与复现

- 修改数据处理、mask、采样、attention 或反事实逻辑后，至少运行完整 `unittest`；
- 正式模型比较应固定 artifact、split、seed 列表、模型选择指标和测试协议；
- 对比多个模型时不得混用 validation loss 与 Macro-F1 选择出的 checkpoint；
- 探索阶段使用 `--skip-test`，避免反复查看测试集；
- 报告关系结论时保留方向、root 覆盖数、多个 seed 的离散程度与匹配随机对照；
- 生成文件中的 manifest、checkpoint 配置和 `metrics.json` 是复现实验的主要依据。
