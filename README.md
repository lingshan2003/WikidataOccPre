# Wikidata 人物职业预测

这是一个基于 Wikidata 人物关系网络的应用型图学习项目。我们关心的问题是：

> 已知一个人与其他人的社会关系，以及部分相关人物已知的职业信息，能否预测该人物的职业？哪些关系对这种预测最有帮助？

项目首先将此问题建模为**多关系人物图上的节点职业分类**。现阶段关注的是职业和社会关系之间的预测关联，不将结果直接解释为因果关系或“职业继承”。

已实施实验及其结论见 [EXPERIMENTS_TRIED.md](EXPERIMENTS_TRIED.md)。

## 研究目标

1. **首要目标：提高职业预测性能。**
   在职业 Level 1、Level 2、Level 3 三个粒度上预测人物职业，并报告 Accuracy、Macro-F1、Weighted-F1 等指标。
2. **次要目标：识别有效社会关系。**
   判断“关系类型”相较于仅有图结构是否带来预测价值，并进一步识别哪些关系类型、哪些具体边对预测更重要。
3. **独立探索：职业 link prediction。**
   在不改变主线节点分类任务的前提下，另建异构 `Person--Occupation` 图，尝试预测缺失的职业边。该分支仍在探索中。

## 数据与图定义

原始文件为 `Q_R_Q_extended.txt`。每一行是一条：

```text
Node1 (Person), Relation, Node2 (Person),
Node1/Node2 的出生、死亡、国家、职业 Level 1/2/3 属性
```

当前导出中所有图节点都是人物；因此主图是**同构的人物节点图**，但边带有多种关系类型，是一个多关系图。

当前完整导出的典型规模为：

```text
人物节点：334,099
原始人物—人物边：691,503
加入反向边、去重后：1,383,006 条有向边
关系类型：80（包含反向关系）
```

预处理将边中心的原始 CSV 规范化为：

```text
Person 节点表 + 带 relation type 的 Person → Person 边表
```

每个原始关系均增加反向关系。例如 `parent_of` 会对应一个反向关系，使消息可以沿两个方向传播；图结构不会因职业标签的遮蔽而被删除。

## 任务定义：层级职业节点分类

职业标签分为三个层级：

```text
Level 1：粗粒度职业大类
Level 2：中等粒度职业类别
Level 3：细粒度职业类别
```

三个层级分别训练、分别评估。以 Level 3 为例：模型输出一个人物属于各个保留的 Level 3 职业类别的概率；默认 `min-class-count=20` 时，保留 682 个 Level 3 监督类别。低频职业和无标签人物仍保留为图中的邻居，只是不参与分类 loss 和评估。

### 关键防泄漏协议

职业既是预测目标，也是社会网络中极有价值的邻居信息。因此必须在“目标人物自身”与“其他已知人物”之间严格区分：

```text
训练人物：其 Level 1 / Level 2 / Level 3 职业可作为邻居节点特征
验证人物：三层职业均为 unknown
测试人物：三层职业均为 unknown
当前 forward 的 seed 人物：三层职业同时临时 mask 为 unknown
人物—人物关系边：始终保留
```

这意味着模型能够利用已知亲属、朋友或其他相关人物的完整职业层级，但永远不能读取被预测人物自己的 Level 1、Level 2 或 Level 3。

若未来希望研究“已知目标人物 Level 1/2、预测其 Level 3”的任务，应将其单独定义为**条件式细粒度职业分类**；它不能与主任务混合报告，因为目标自身的上层职业会大幅缩小 Level 3 候选空间。

## 当前节点特征

主线节点分类模型当前输入：

| 特征 | 类型 | 可见性 |
| --- | --- | --- |
| `occupation_level1` | 类别 embedding | 仅训练人物可见；seed 人物 mask |
| `occupation_level2` | 类别 embedding | 仅训练人物可见；seed 人物 mask |
| `occupation_level3` | 类别 embedding | 仅训练人物可见；seed 人物 mask |
| `country` | 类别 embedding | 所有节点可见 |
| `temporal` | 数值特征 | 出生年、死亡年、年龄及缺失标记；所有节点可见 |

特征编码器为 schema-driven 设计：每种类别、数值或未来文本向量特征各自经过独立分支，随后由 feature gate 和融合 MLP 组成节点初始表示。未来加入文本 embedding、教育经历等属性时，无需改写各个 GNN 模型。

## 数据预处理

所有新实验从 `run.py` 进入。

```bash
python run.py prepare \
  --input Q_R_Q_extended.txt \
  --output-dir artifacts/level3_hierarchy \
  --target-level 3 \
  --min-class-count 20 \
  --seed 42
```

`--target-level` 改为 `1`、`2` 或 `3`，分别生成对应的监督任务。请为每一层使用独立输出目录，例如：

```text
artifacts/level1_hierarchy/
artifacts/level2_hierarchy/
artifacts/level3_hierarchy/
```

预处理步骤如下：

1. 从每条边的两个端点属性构建唯一人物节点表，并导出属性冲突审计表；
2. 保留所有人物和关系边，加入反向边并去重；
3. 根据目标层级过滤可监督的职业类别；
4. 以人物为单位、按职业分层划分 train / validation / test，默认比例为 70% / 10% / 20%；
5. 按上述协议写入三层可遮蔽职业特征、Country 与时间特征；
6. 保存可复现的 PyG artifact 与 CSV 审计文件。

每个 artifact 的核心文件：

```text
graph_data.pt          # PyG 图、特征、掩码与 metadata
nodes.csv              # 人物索引到 Wikidata Q-ID 的映射
edges.csv              # 规范化后的关系边
class_stats.csv        # 目标职业频数
relation_stats.csv     # 关系频数
split_summary.json     # 数据规模、划分和特征协议
```

> 同一层级内的模型比较必须读取同一份 `graph_data.pt`。目前每个层级按其目标标签独立分层划分；若要严格比较三个层级的同一批人物，需要额外固定共享的人物 split manifest。

### 固定职业语义向量（可选实验）

`occupation-embed` 不重新准备图、不改变 split；它只为训练人物可见的
`(Level 1, Level 2, Level 3)` 职业组合生成固定的多语言语义向量。验证、测试
以及当前 seed 人物一律使用单独可学习的 `unknown` 表示，因此不会读取目标自身的
职业文本。

先安装额外依赖并从同一个 categorical artifact 生成语义副本：

```bash
python -m pip install -r requirements.txt

python run.py occupation-embed \
  --data artifacts/pure_rel_occ/level3/graph_data.pt \
  --output artifacts/pure_rel_occ/level3/graph_data_semantic.pt \
  --model-name intfloat/multilingual-e5-base --device cuda
```

编码器使用固定模板：`passage: A person's occupation hierarchy is Level 1: ...`
，并保存 prompt manifest、模型 revision 与 L2-normalised 向量表。训练语义版本时，
使用 `--occupation-representation semantic`；它与随机职业 embedding 对照时不应
同时读取 `occupation_level1/2/3`。

## 主训练管道

训练采用两层 NeighborLoader 邻居采样。每个 batch 的前 `batch_size` 个节点是当前监督 seed，loss 只在这些 seed 人物上计算；其余采样节点提供邻居信息。

```text
输入：采样人物子图 + 关系类型 + 节点特征
  ↓
独立特征 embedding / 数值编码 → feature gate → 融合表示
  ↓
两层关系型消息传递
  ↓
职业分类器
  ↓
CrossEntropy（仅 seed 节点）
```

优化器为 AdamW；默认以 validation loss 选择 checkpoint 和早停，并使用梯度裁剪与 ReduceLROnPlateau。测试集仅在选择出最佳 checkpoint 后评估一次。

### 可用模型

| 模型 | 文件 | 用途 |
| --- | --- | --- |
| R-GCN | `models/rgcn.py` | 关系卷积基线；默认 FastRGCNConv 以速度换显存 |
| R-GAT | `models/rgat.py` | 关系注意力模型；可导出 attention 候选边 |
| CompGCN | `models/compgcn.py` | 组合邻居节点表示与 relation embedding 的关系模型 |

三者共用相同数据、采样、优化器、特征编码和评估代码，因此可以进行公平的横向比较。

### 训练示例

默认输入邻居的完整 L1+L2+L3 职业层级：

```bash
python run.py train --model rgat \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgat_hierarchy \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --heads 4 \
  --occupation-feature-levels 1,2,3 \
  --num-workers 4 --patience 6 --seed 42 --device cuda
```

### L1 一跳 R-GAT 与关系权重表

`--num-layers` 必须与 `--num-neighbors` 的 fan-out 数一致。下面的 L1
对照只保留一层消息传递，因此每个预测只聚合一跳邻居；它与已有的
`runs_report/level1/rgat_baseline` 两层（两跳）结果使用相同的超参数和三条 seed。

```bash
bash scripts/run_l1_rgat_attention.sh plan
bash scripts/run_l1_rgat_attention.sh run
```

`run` 会训练 `rgat_one_hop/seed_{42,43,44}`，读取已有
`rgat_baseline/seed_{42,43,44}/best_model.pt`，并写入：

```text
runs_report/level1/rgat_l1_relation_attention/relation_attention_table.md
runs_report/level1/rgat_l1_relation_attention/relation_attention_table.csv
runs_report/level1/rgat_l1_relation_attention/l1_relation_attention_matrices.md
runs_report/level1/rgat_l1_relation_attention/l1_relation_attention_summary.csv
```

表格按测试集预测人物的**入边**汇总：每条关系的值是 RGAT 所有 head 的
attention alpha 先求均值、再在边上求均值，最后报告三条 seed 的均值 ± 标准差。
二跳模型会分别给出 `layer1` 和 `layer2` 两列；一跳模型只有 `layer1`。合成 self-loop
不计入任何真实关系。attention 是机制描述，不能直接当作关系的因果效应。

`l1_relation_attention_matrices.md` 仿照 OCI 的 Fig. 5：每个**有向**关系一张矩阵，行是
邻居的真实 L1，列是测试目标的真实 L1，单元格是该类入边的 mean attention ± 三 seed
标准差及边数。`father` 与 `father__rev` 不会被合并；若研究父到子的职业延续，必须先在
`edges.csv` 中核实并明确选择父→子的那一个有向标签。测试节点的真实 L1 仅用于导出后的
分组，模型输入仍保持 unknown 掩码。

将 `--occupation-matrix-relations` 设为 `all` 时，导出会一次性保存全部精确有向关系的
`relation × source L1 × target L1` 立方体（包括每种关系的 `__rev`）；之后应从 CSV 本地筛选，
无需为了新增关系再跑完整图。

若要让每个有 L1 标签的人都轮流成为预测 root，并在其完整一/二跳感受野中导出注意力，
不需要重训。以下脚本对一跳 checkpoint 自动使用 `-1`，对两跳 checkpoint 自动使用
`-1,-1`；每个 root 的自身职业仍会在该 batch 中 mask：

```bash
bash scripts/export_l1_full_receptive_attention.sh plan
bash scripts/export_l1_full_receptive_attention.sh run
```

默认输出位于 `runs_report/level1/rgat_l1_full_receptive_attention_all_relations/`，并一次性导出
全部关系的 L1 矩阵。两跳全邻域子图可能比训练时大得多；若显存不足，可先设置
`RGCN_L1_FULL_ATTENTION_BATCH_SIZE=8` 后再运行。

只使用邻居 Level 3 的受控对照：

```bash
python run.py train --model rgat \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgat_neighbor_l3_only \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --heads 4 \
  --occupation-feature-levels 3 \
  --num-workers 4 --patience 6 --seed 42 --device cuda
```

`--occupation-feature-levels none` 可运行不使用职业邻居信息的结构与普通属性基线。R-GCN 使用 `--model rgcn --rgcn-backend fast`；CompGCN 使用 `--model compgcn --compgcn-composition mult`。

纯关系职业语义的 Level 3 验证实验示例：

```bash
# 先重跑 categorical 对照；不跑 test，只以 validation Macro-F1 选 checkpoint。
python run.py train --model rgcn \
  --data artifacts/pure_rel_occ/level3/graph_data.pt \
  --output-dir runs/pure_rel_occ/level3_rgcn_categorical_reselected \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --occupation-representation categorical --occupation-feature-levels 1,2,3 \
  --auxiliary-features none --eval-mode sampled --early-stop-metric macro_f1 \
  --min-delta 0.001 --patience 6 --num-workers 4 --seed 42 --device cuda --skip-test

# 再跑唯一改变职业输入表示的 semantic 对照。
python run.py train --model rgcn \
  --data artifacts/pure_rel_occ/level3/graph_data_semantic.pt \
  --output-dir runs/pure_rel_occ/level3_rgcn_semantic \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --occupation-representation semantic --auxiliary-features none \
  --eval-mode sampled --early-stop-metric macro_f1 --min-delta 0.001 \
  --patience 6 --num-workers 4 --seed 42 --device cuda --skip-test
```

`--skip-test` 仅保存 validation 最优 checkpoint，不写测试指标；锁定配置后移除它，
对该设置执行一次最终测试。checkpoint 始终保存实际最优 validation 监控值，
`--min-delta` 只影响早停 patience。

### 下一轮优先实验（已支持，待服务器运行）

以下开关只修改运行时内存中的图，绝不改写输入的 `graph_data.pt`。实际删除的
关系、删边前后数量和置乱 seed 会写入 `metrics.json` 与 checkpoint；所有探索先加
`--skip-test`，锁定配置后才去掉它运行一次测试集。

**1. 纯关系/结构基线。** `structural` 为所有人物提供同一个可学习常量，不使用
职业、国家、时间或节点 ID。因此模型只能利用图拓扑和 relation type：

```bash
python run.py train --model compgcn \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_compgcn_structural \
  --feature-mode structural --occupation-feature-levels none --auxiliary-features none \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --num-workers 4 --seed 42 --device cuda --skip-test
```

**2. 关系语义置乱与消融。** `--shuffle-relation-types` 保留每条边的位置和每种
relation 的频数，但按 `--seed` 随机重新分配 relation ID。它应与完全相同、但不带
该开关的基线配对。`--drop-relation-groups` 会同时删除关系的正向与反向边；可用组为
`kinship`、`education_mentorship`、`professional_collaboration`、
`influence_succession`、`religious`、`other`。也可以用 `--drop-relations father,mother`
指定原始关系名。

```bash
# 关系类型是否提供超越纯拓扑的信号？
python run.py train --model rgcn \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgcn_relation_shuffled \
  --occupation-feature-levels 1,2,3 --auxiliary-features none \
  --shuffle-relation-types --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --num-workers 4 --seed 42 --device cuda --skip-test

# 对每个组分别运行；这里是亲属关系消融。
python run.py train --model rgcn \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgcn_without_kinship \
  --occupation-feature-levels 1,2,3 --auxiliary-features none \
  --drop-relation-groups kinship --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --num-workers 4 --seed 42 --device cuda --skip-test
```

亲属消融删掉的边数可能远大于其他关系组，因此还必须与**等数量随机删边**配对。
`--match-random-drop-to-relation-groups kinship` 先统计亲属关系包含的“原始关系 +
无向人物对”数量，再从全图均匀抽取相同数量的关系对；每一对的正向和反向边一起删除。
这保留了“删边规模”而不保留亲属语义：

```bash
python run.py train --model rgcn \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgcn_random_drop_matched_kinship \
  --occupation-feature-levels 1,2,3 --auxiliary-features none \
  --match-random-drop-to-relation-groups kinship \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --early-stop-metric macro_f1 --min-delta 0.001 \
  --num-workers 4 --seed 42 --device cuda --skip-test
```

若亲属消融明显差于这个随机删边对照，才有较强证据表明亲属边的价值超出单纯的图
连通性。`--random-edge-drop-pairs N` 也可用于手工指定要删除的关系对数量。随机删边
与关系消融或 relation type 置乱是独立对照，命令行不允许把它们混合。

对置乱模型导出 attention 时，程序会重放同一置乱；但其 relation 标签已经不对应
原始语义，因此不能用它做关系解释。

### 数据诊断：稀疏度、职业可见性与关系同配性

`diagnose` 不训练模型。它默认仅使用训练人物的真实职业标签来估计 relation-level
同配性，避免用验证/测试标签来描述训练时可见信号；并输出每个人的不同邻居数、连通
分量、可见训练职业邻居数及两跳覆盖。传入一次最终测试的预测 CSV 后，还会按这些分桶
报告 Accuracy 和 Macro-F1。

```bash
# 图和标签信号本身的诊断（可在训练前运行）
python run.py diagnose \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir diagnostics/level3_graph

# 需要先完成一次不带 --skip-test 的最终运行，才会生成 test_predictions.csv。
python run.py diagnose \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --predictions runs/level3_rgcn_baseline_macro/test_predictions.csv \
  --output-dir diagnostics/level3_baseline
```

输出包括：

```text
summary.json              # 点边比、连通分量、各 split 的直接/两跳职业覆盖
node_diagnostics.csv      # 每个节点的不同邻居数、分量大小和职业邻居覆盖
relation_homophily.csv    # 每种关系的职业相同率、相对独立基线 lift、互信息
prediction_strata.csv     # 可选：按度/职业邻居覆盖分桶的 Accuracy、Macro-F1
```

**3. L3 长尾训练。** 以下三种策略先逐一与同一基线比较，避免无法归因。它们均只用
训练 split 的类别频数，验证和测试标签从不参与权重或先验计算：
若以 Macro-F1 为比较重点，所有配对运行（包括普通 CrossEntropy 基线）都应加入
`--early-stop-metric macro_f1 --min-delta 0.001`。

```bash
# 有效样本数（class-balanced）损失
python run.py train --model rgcn --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgcn_class_balanced --loss class_balanced \
  --class-balanced-beta 0.9999 --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --early-stop-metric macro_f1 --min-delta 0.001 \
  --num-workers 4 --seed 42 --device cuda --skip-test

# logit-adjusted cross entropy（先验强度 tau）
python run.py train --model rgcn --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgcn_logit_adjusted --loss logit_adjusted \
  --logit-adjustment-tau 1.0 --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --rgcn-backend fast \
  --early-stop-metric macro_f1 --min-delta 0.001 \
  --num-workers 4 --seed 42 --device cuda --skip-test

# 每个 epoch 对训练 seed 作类别均衡的有放回采样
python run.py train --model rgcn --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_rgcn_balanced_roots --train-root-sampling class_balanced \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 --hidden-dim 128 --branch-dim 64 \
  --rgcn-backend fast --early-stop-metric macro_f1 --min-delta 0.001 \
  --num-workers 4 --seed 42 --device cuda --skip-test
```

旧的 `--class-weight` 仍可用，等价于 `--loss inverse_frequency`；不要和其他
`--loss` 选项混用。

**4. 真正的 full-batch 训练。** `--train-mode full` 在每个 epoch 对整图执行一次
forward/backward，并强制 `--eval-mode full`。为严格防泄漏，它不允许职业特征：一次
全图前向无法同时对每个训练人物遮蔽“自身职业”又把它暴露给其他训练 seed。优先尝试
CompGCN；FastRGCN 的全图计算已知可能 OOM。

```bash
python run.py train --model compgcn \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output-dir runs/level3_compgcn_full_structural \
  --train-mode full --eval-mode full --feature-mode structural \
  --occupation-feature-levels none --auxiliary-features none --num-neighbors=-1,-1 \
  --epochs 50 --hidden-dim 128 --branch-dim 64 --seed 42 --device cuda --skip-test
```

训练输出：

```text
runs/<experiment>/best_model.pt
runs/<experiment>/metrics.json
runs/<experiment>/test_predictions.csv
```

### 汇报用三 seed 重跑矩阵

`scripts/run_report_experiments.sh` 将历史尝试整理为五个可解释的问题，共 18 个条件：
模型架构、职业输入、关系语义/删边控制、长尾优化和邻域覆盖。每个条件固定同一 Level 3
artifact 并运行 seed `42/43/44`，输出统一放在 `runs_report/level3/<condition>/seed_<seed>/`。
它不会使用 `--skip-test`，因为该矩阵用于最终报告；每个 seed 都只按 validation 选
checkpoint，再测试一次。

先只打印全部 54 条命令，检查路径和 semantic artifact：

```bash
bash scripts/run_report_experiments.sh plan all
```

按组执行，便于从服务器断点重跑；已有 `metrics.json` 的 seed 会自动跳过：

```bash
bash scripts/run_report_experiments.sh run architecture
bash scripts/run_report_experiments.sh run features
bash scripts/run_report_experiments.sh run relations
bash scripts/run_report_experiments.sh run longtail
bash scripts/run_report_experiments.sh run coverage
```

全部完成后汇总每个条件的逐 seed 指标、均值与样本标准差：

```bash
python scripts/summarize_report_runs.py --root runs_report/level3
```

它会生成：

```text
runs_report/level3/report_seed_metrics.csv  # 每个 condition × seed 的最终 test 指标
runs_report/level3/report_summary.csv       # 每个 condition 的 mean/std，按 Macro-F1 排序
```

语义职业条件要求预先存在 semantic artifact；默认路径为
`artifacts/level3_hierarchy/graph_data_semantic.pt`。路径不同可在运行脚本前设置
`RGCN_REPORT_SEMANTIC_DATA`。同样可通过 `RGCN_REPORT_DATA` 和
`RGCN_REPORT_OUTPUT_ROOT` 覆盖普通 artifact 和输出目录。

若尚未生成这个 artifact，先在服务器执行一次：

```bash
python run.py occupation-embed \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --output artifacts/level3_hierarchy/graph_data_semantic.pt \
  --model-name intfloat/multilingual-e5-base --device cuda
```

## 关系重要性与解释

项目的次要目标不是只导出一张 attention 表，而是评估关系的真实预测贡献。

R-GAT 可先导出每个查询人物的 attention 候选边：

```bash
python run.py explain \
  --data artifacts/level3_hierarchy/graph_data.pt \
  --checkpoint runs/level3_rgat_hierarchy/best_model.pt \
  --node-id Q1000023 \
  --output-dir explanations \
  --num-neighbors 15,10
```

attention alpha 只是模型机制的候选信号，并不等于社会科学意义上的影响。后续应优先做：

1. 关系类型消融：遮蔽一类关系后观察验证/测试性能下降；
2. 关系类型置乱：保留图结构、打乱 relation type，检验关系语义是否有效；
3. 个体边删除：移除高 attention 边后观察目标职业概率或 logit 是否下降；
4. 按亲属、朋友、其他关系组分别报告贡献。

这些分析支持“预测关联和潜在关系贡献”的表述；若要主张因果影响，还需关系语义清洗、时间顺序和额外识别策略。

## 独立探索：异构图 Link Prediction

`link_prediction/` 是不影响主线节点分类的一条独立尝试。它将职业显式建成节点：

```text
Person --各种社会关系--> Person
Person --has_occupation--> Occupation
```

训练人物的 `has_occupation` 边进入消息图；验证/测试人物的职业边被移除并作为待预测链接。第一版使用 HGT encoder + DistMult decoder，并报告 MRR、Hits@1、Hits@3、Hits@10。

```bash
python run.py link-prepare --input Q_R_Q_extended.txt \
  --output-dir link_artifacts/level3 --target-level 3 --min-class-count 20 --seed 42

python run.py link-train --data link_artifacts/level3/hetero_graph.pt \
  --output-dir link_runs/level3_hgt --epochs 50 --batch-size 256 \
  --num-neighbors 15,10 --hidden-dim 128 --branch-dim 64 --heads 4 --device cuda
```

该分支的评估目标与节点分类不同，不能将 BCE loss、MRR 或 Hits@k 直接与节点分类 CrossEntropy、Accuracy 混为同一指标。

## 项目结构

```text
run.py                    # 唯一命令入口
cli.py                    # prepare/train/explain/link 子命令路由
data/
├── extended.py           # 原始扩展 CSV → 规范节点表与关系边表
└── prepare.py            # 节点分类 artifact、分割和层级职业掩码
models/
├── features.py           # 可扩展类别/数值/向量特征编码与融合
├── rgcn.py               # R-GCN
├── rgat.py               # R-GAT
├── compgcn.py            # CompGCN
└── __init__.py           # 模型注册表
training/
├── train.py              # 共享 NeighborLoader 训练、评估、早停
├── explain.py            # 单个人的 R-GAT attention 候选边导出
└── attention_report.py   # 多 checkpoint 的按关系 attention 汇总表
scripts/
└── run_l1_rgat_attention.sh  # L1 一跳训练 + 与两跳基线的关系权重对照
link_prediction/          # 独立异构职业 link-prediction 管道
legacy/original_baseline/ # 师姐留下的原始 R-GCN 代码，仅供追溯
RGAT_DESIGN.md            # R-GAT 与掩码协议的补充说明
```

## Legacy 代码

`legacy/original_baseline/` 保存交接时的节点分类实现，包括原始数据读取、全图 R-GCN 和早期特征消融。它不再是新实验入口，原因包括：全图计算成本高、测试集被用于选择 checkpoint、处理和训练逻辑耦合等。

新实验统一使用：

```bash
python run.py prepare ...
python run.py train --model rgcn|rgat|compgcn ...
python run.py explain ...
python run.py link-prepare ...
python run.py link-train ...
```


ssh -p 24191 root@connect.nmb1.seetacloud.com
