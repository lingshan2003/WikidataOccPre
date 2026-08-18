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
- `--match-random-drop-to-relation-groups`：按关系组的实际有向消息边数做随机删边；每个抽样单元是一条原始边及其生成反向边；
- `--loss inverse_frequency|class_balanced|logit_adjusted`：长尾损失；
- `--train-root-sampling class_balanced`：类别均衡的训练 seed 采样。

所有运行时图扰动只作用于内存副本，不覆盖 `graph_data.pt`；实际删边数量、关系选择和随机 seed 会记录到输出配置。

### 血缘家庭与后天社会关系审计

关系分类是可版本化的 JSON 入口。`config/tie_taxonomy_v1.json` 保留严格的
血缘家庭定义；明日 Level 1 审计使用
`config/tie_taxonomy_ascribed_family_v1.json`，将
`father`、`mother`、`child`、`sibling`、`relative`、`stepparent`、`godparent`、
`kinship_to_subject` 归入 `inherited`。其余当前 artifact 中的基础关系自动归入
`acquired`，但这里表示“其余社会关系”，并不宣称每一条关系都由主角自由选择。
配置只写基础关系名，生成的 `__rev` 反向边会自动获得相同类别。训练和报告会保存
配置内容哈希与完整解析结果，因此修改 JSON 后的结果不会与旧分类混淆。

```bash
# 删除一类关系，或删除相同数量的随机有向消息边作为图密度对照；随机删边始终成对保留原始边与其反向边。
python run.py train --model rgcn --data artifacts/level1_hierarchy/graph_data.pt \
  --output-dir runs/level1_without_inherited --tie-taxonomy config/tie_taxonomy_ascribed_family_v1.json \
  --drop-tie-groups inherited

python run.py train --model rgcn --data artifacts/level1_hierarchy/graph_data.pt \
  --output-dir runs/level1_random_matched_inherited --tie-taxonomy config/tie_taxonomy_ascribed_family_v1.json \
  --match-random-drop-to-tie-groups inherited
```

`diagnose --tie-taxonomy ...` 会额外导出 `tie_group_edge_summary.csv`、
`tie_group_homophily.csv`，并在节点报告中加入两类可见训练职业邻居计数与
`inherited_only`、`acquired_only`、`both`、`neither` 暴露分层。为锁定的 Level 1
正式矩阵使用：

```bash
bash scripts/run_tie_audit_experiments.sh plan
bash scripts/run_tie_audit_experiments.sh run matrix
bash scripts/run_tie_audit_experiments.sh run diagnose
bash scripts/run_tie_audit_experiments.sh run summarize
```

明日矩阵复用现有的 R-GCN/R-GAT 三 seed 完整图基线，绝不重跑这六个大实验；只在
相同的邻居职业特征协议下运行两类删边及其等规模随机对照（2 模型 × 4 条件 × 3 seed
= 24 次训练）。汇总按同一模型、同一 seed 与旧基线配对，报告指标差的均值和标准差。
如果服务器上每个配对实验都保留了 `test_predictions.csv`，汇总会自动报告 Macro-F1
差异的 paired test-node bootstrap 95% 区间；若缺少任一预测文件，则明确退回仅报告
同 seed 的均值和标准差。`best_model.pt` 会在报告中登记路径以便日后复现或做局部机制
分析，但明日的性能汇总不会重新训练或用它推理。所有结果表示模型依赖，不能解释为现实
职业因果效应。

### 生命时期异质性审计

时间审计按**生命区间与历史时期相交**分期，而不是按出生年唯一分箱。设人物生命区间为
`[birth_year, death_year]`，时期为 `[start, end]`；两者相交时，该人物进入该时期图。因此
跨越时期边界而仍存活的人会进入多个时期 artifact。默认配置为
`config/historical_life_periods_v2.json`，其内容哈希与日期规则都会写进 artifact 和报告。
双日期时使用完整生命区间；仅出生日期时进入出生所在时期，仅死亡日期时进入死亡所在时期。
只有两种日期都缺失、或死亡早于出生的人不进入时期图；不会凭空外推未观测的存活年份。

正式的时期比较必须使用**独立时期诱导子图**，而不是在完整图上只删去与某时期节点相连
的一部分边。对每一个时期，代码严格执行：

1. 选出生命区间与该时期相交的全部有效日期节点；
2. 仅保留两个端点都属于该时期的边；两个端点未共同处于该时期的边不进入该图；
3. 在该时期图内重新建立国家、时间和职业特征，并对该时期的监督标签重新做固定的
   70/10/20 分层 split；
4. 在这个 artifact 上新训练 `full` 基线，再训练删除 inherited、其实际有向删边量严格匹配的随机对照、删除
   acquired、其等量随机对照。

因此旧的六次完整图 R-GCN/R-GAT 基线**不能**用于这个问题。默认矩阵为 4 个时期 × 2 个
模型 × 5 个条件 × 3 个模型 seed = **120 次新训练**；其中 24 次是时期内 `full` 基线。
每个 artifact 会保存源数据哈希、生命周期时期配置哈希、日期完整性计数、节点同时属于几个
时期的计数、时期内节点/边数、跨时期删边数、局部罕见职业的处理和实际 split 大小。某职业
在一个时期中少于 3 人时，会在该时期中设为 `y=-1`（仍保留为图节点，但不进入三路监督
split）；这一点会明确列在 `split_summary.json`。

```bash
# 只打印 120 个任务，不训练。
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_tie_audit_experiments.sh plan all

# 第一步：只新建/核验四个时期 artifact；不训练 GNN。
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_tie_audit_experiments.sh run prepare

# 第二步：训练时期内 full 基线与四个消融/随机对照。可中断后直接重跑以续跑。
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_tie_audit_experiments.sh run matrix

# 第三步：仅汇总。它拒绝数据 artifact、时期配置、taxonomy 或测试节点不一致的比较。
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_tie_audit_experiments.sh run summarize
```

汇总目录 `period_induced_summary/` 的 `period_induced_condition_summary.csv` 同时给出
Accuracy、Macro-F1、Weighted-F1 的完整时期基线、各条件结果、同 seed 差值和 Macro-F1 的
测试节点 bootstrap 区间；`period_induced_relation_specificity_by_seed.csv` 给出
`F1(random matched) - F1(drop tie group)`。这些结论仍只表示模型依赖。Wikidata 图是静态的，
关系本身没有发生时间，不能据此宣称某个历史时期的关系造成职业结果。

### Acquired ties 的细分审计（R-GCN）

在二元审计确认 `acquired` 的重要性后，使用
`config/tie_taxonomy_acquired_subgroups_v1.json` 将其互斥细分为五个预设主组：
`intimate_partnership`（配偶/伴侣）、`education_mentorship`、
`professional_collaboration`、`influence_succession` 和
`religious_ordination`。`other_acquired` 是完整性保留的低频、异质残余桶，不默认进入主分析；
需要探索时可用环境变量显式加入。`inherited` 与此前 ascribed-family 定义完全一致。taxonomy
对 artifact 的全部基础关系做一次且仅一次覆盖，并把配置哈希及实际解析的成员写进每一次运行。

每一个子组都做两种同 seed 的反事实：直接删除该组的正、反向消息边，及从同一图均匀删除**完全相同数量**的
“原始边实例 + 生成反向边”单元。汇总中的 `random_minus_direct`（Macro-F1）即
`F1(等量随机删边) - F1(删子组)`；只有它为正，才支持该子组的影响超过一般图密度损失的模型依赖解释。

完整图复用已完成的 Level-1 R-GCN full baseline，不重训它；默认是
5 子组 ×（直接/随机）× 3 seeds = **30** 次新训练：

```bash
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_acquired_subgroup_ablation_experiments.sh plan matrix
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_acquired_subgroup_ablation_experiments.sh run matrix
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_acquired_subgroup_ablation_experiments.sh run summarize
```

时期结果继续使用独立诱导的时期图及其已完成的 R-GCN `full` baseline，绝不使用完整图的 cohort-local
实验替代。默认是 4 时期 × 5 子组 ×（直接/随机）× 3 seeds = **120** 次新训练：

```bash
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_acquired_subgroup_experiments.sh plan matrix
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_acquired_subgroup_experiments.sh run matrix
RGCN_PYTHON_BIN=.venv/bin/python \
  bash scripts/run_period_induced_acquired_subgroup_experiments.sh run summarize
```

两类汇总都输出 condition 指标、同 seed 的 `direct_minus_full`、`random_minus_full`、
`random_minus_direct` 以及可用时的 paired test-node bootstrap 区间；它们会拒绝 taxonomy、图 artifact、
随机删边数量不一致的比较，并在同时保存 prediction 文件时校验测试节点和真标签完全一致。

此前 `birth_cohort_tie_audit/` 的 96 次结果保留为**完整图上的局部出生队列节点删边敏感性分析**：
它训练时仍使用全局节点、全局边和全局 split，随后只对指定时期节点相连的部分关系边做干预。
它不能作为“时期内关系依赖”的主结果，也不能与本节的独立时期 baseline 混报。对已经完成的
全局模型 `test_predictions.csv` 做出生时期筛选同样可以作为描述性的全局模型分层，但不替代
上述重新建图、重新 split、重新训练的实验。

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

### GraphMask 消息与关系重要性

项目提供论文 [Interpreting Graph Neural Networks for NLP With Differentiable Edge
Masking](https://arxiv.org/abs/2010.00577) 的 amortized GraphMask 适配。它冻结已经训练好的
RGAT、RGCN 或 CompGCN，只在训练人物上拟合逐层 Hard-Concrete 消息门和替代基线；验证人物
用于选择满足保真约束的最稀疏 probe，测试人物只用于最终报告。

GraphMask 仍需要在服务器上进行一次轻量的第二阶段训练，并不是只加载原模型权重做一次推理：

```bash
python run.py graphmask-train \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --checkpoint runs/level3_rgat/best_model.pt \
  --output-dir runs_graphmask/level3_rgat/seed_42 \
  --num-neighbors auto \
  --seed 42 \
  --device cuda

python run.py graphmask-report \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --checkpoint runs/level3_rgat/best_model.pt \
  --probe runs_graphmask/level3_rgat/seed_42/graphmask_probe.pt \
  --output-dir runs_graphmask/level3_rgat/seed_42/test_report \
  --split test \
  --num-neighbors auto \
  --top-k 50 \
  --device cuda
```

`graphmask-train` 默认使用 `KL(original || masked) <= 0.03` 的约束进行优化，并在验证集
macro-F1 相对差异不超过 5% 的候选中选择消息保留率最低者。如果没有候选满足门槛，命令会
失败并保留 `training_history.json`，不会静默输出低保真的解释。`auto` 会从原 checkpoint 的
训练配置恢复 fanout；`full` 会对每个 root 使用完整 L-hop 邻域，计算成本更高。

报告文件的统计口径如下：

| 文件 | 含义 |
| --- | --- |
| `relations_directed.csv` | 按层和精确方向关系（如 `child`、`child__rev`）汇总 |
| `relations_base.csv` | 去掉 `__rev` 后合并正反向的原始关系汇总 |
| `root_top_edges.csv.gz` | 每个预测人物、每层 keep probability 最高的 Top-K 消息边 |
| `test_metrics.json` | 原模型/掩码模型指标、预测一致率、KL 和逐层保留率 |
| `manifest.json` | 数据、原 checkpoint、probe、fanout、采样种子和 Git 版本 |

关系表同时给出 observation 数、root 覆盖数、平均非零概率、hard retention rate 和 retained
share。不能仅凭罕见关系的高保留率判断其全局重要性。有限 fanout 的结果解释的是 manifest
记录的固定采样计算图；正式实验应为每个原模型 checkpoint 使用多个 `--seed` 独立训练 probe。

GraphMask 衡量的是冻结模型对消息和关系的预测依赖。它不能证明某种社会关系导致了职业，
也不能把被保留边直接解释为现实世界因果机制。核心实现改写自
[MichSchli/GraphMask](https://github.com/MichSchli/GraphMask) 的 MIT 许可代码，来源与许可证见
`training/graphmask/NOTICE.md`。

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
| 归因/扰动 | `message-contribution-report`, `gradient-attribution-report`, `relation-pair-ablation-report`, `relation-pair-sweep-report`, `graphmask-train`, `graphmask-report` |
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
