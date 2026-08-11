# 多关系图中“关系重要性”文献简报

更新：2026-08-11  
用途：明天 RGAT 权重分析汇报的相关工作、研究定位与下一步实验页。

## 一句话结论

现有的 raw attention、value-aware message magnitude、`attention × gradient` 已覆盖了三种**模型内部信号**；文献更强的下一步并不是再发明一套权重，而是把它们作为候选关系/路径的排序器，再以受控扰动、稀疏消息 mask 或最小反事实来验证“模型是否真的依赖它”。

对于多关系图，重要性至少应分成四类，不能混用：

| 问题 | 合适的量 | 现有状态 |
| --- | --- | --- |
| 模型把多少归一化预算分给关系？ | attention mass | 已完成，且有 node-first 聚合 |
| 该关系传来的表征有多大？ | relation-specific message vector / L2 norm | 已完成 |
| 微小改变该 attention 系数会使目标 logit 朝哪里变化？ | signed `alpha × gradient` | 已完成 |
| 去掉、替换或改写该关系/路径，模型预测是否仍保持？ | perturbation / mask / counterfactual fidelity | 可作为核心新增工作 |

> 注意：最后一项衡量的是**模型依赖（model reliance）**，不是“社会关系导致职业”的因果结论。

## 最相关论文：读什么、借什么

| 论文 | 核心思想与关系重要性单位 | 可直接借到本项目的内容 | 汇报时应保留的限制 |
| --- | --- | --- | --- |
| Wang et al., **HAN**, WWW 2019 | 两级 attention：节点级选择同一 meta-path 下的邻居，语义级对多个 meta-path 打权重。 | 将单关系 `r` 扩展到关系序列 `r1 → r2`；把 `child` 与 `child__rev` 保持为不同语义。是当前 attention 矩阵与两跳分析的架构先例。 | semantic attention 仍是模型内部权重，不足以证明删去该路径会改变预测。 |
| Yu et al., **R-HGNN**, 2021 | 逐关系学习表示，再做 cross-relation message passing 与层级 relation representation；实验同时报告“only this relation”和“without this relation”。 | 已有 `drop-relation-groups`、指定 relation 删除和 relation-label shuffle，可按它的范式升级为**单关系 / 移除单关系 / 完整图**三联对照，并按 L1 类别报告。 | 论文自己的消融也发现：单关系很弱、移除单关系往往影响不大；重要性常是关系组合而不是单关系。 |
| Li et al., **xPath**, AAAI 2023 | 对异构 GNN 的一个目标节点，区分“哪个源节点重要”与“它经由哪条 typed path 影响目标”；用 graph rewiring 测量路径影响，再做贪心搜索。 | 现有 `attention-rollout.py` 可继续当**候选路径生成器**，但可把 top `r1 → r2` 用局部可行重连后重跑 margin 来验证；输出 `source L1 × r1 × r2 × target L1`。 | 当前 rollout 是 `A2 @ A1` 的 attention-only 近似，忽略 relation value transform、GELU、LayerNorm 和 classifier，不能当最终路径归因。 |
| Mika et al., **HGExplainer**, WI-IAT 2023 | 面向异构图的 meta-path 图采样，联合互信息目标选择结构、语义与属性解释。 | 可将高分两跳关系序列汇总为职业预测的“语义路径库”，再做紧凑路径子图 case study。 | 主要面向 heterogeneous recommendation/link prediction；本项目是同构人物节点、带多关系边，应迁移思想而非照搬设置。 |
| Schlichtkrull et al., **GraphMask**, ICLR 2021 | 在每层消息上学习稀疏随机 gate；尽量屏蔽更多边，同时维持原预测，保留“不可被替代”的消息。 | 最匹配 message-level 分析：给每层 typed message 加 gate，按 `relation` 或 `(source L1, relation, target L1)` 汇总 gate 保留率；可与现有 message norm、gradient 排名做一致性检验。 | 原论文应用于 NLP 图；适配 RGAT 时要 mask **message value**，不要只把 alpha 设为零而破坏同一目标下的 softmax simplex。 |
| Luo et al., **PGExplainer**, NeurIPS 2020 | 训练参数化 explainer 为全部实例生成 edge mask，而非每个节点单独优化；适合 global / multi-instance explanation。 | 用 test root 的 layer-1 边训练 amortized mask，得到跨节点可比较的全局 relation ranking；DGL 还提供 heterogeneous PGExplainer 的实现先例。 | 最初是一般 GNN 方法；连续 edge mask 可能产生训练分布之外的图，需要与真实删边/可行重连对照。 |
| Lucic et al., **CF-GNNExplainer**, AISTATS 2022 | 找使预测改变的最小边删除集合，回答“什么最少的图改动会翻转该预测？” | 对正确分类的 test root，计算最小的 directed typed-edge 删除集合；再按 relation / relation pair / source L1 汇总“翻转频率”和“最小删除数”。 | 删除后的图可能不自然；对人物社会图需限制为现有边删除、同步处理人工加入的反向边，并明确这只是模型反事实。 |
| Amara et al., **GraphFramEx**, 2022 | 不只报一项 fidelity；区分 explanation 的必要性与充分性，并强调真实图上的评价与合成图不同。 | 为 attention / message / gradient / mask 四种排序统一报告：top-k 删除后的 prediction drop（necessity）、仅保留 top-k 的 prediction retention（sufficiency）、时间和稳定性。 | 全局的“删 relation 后重训”与局部 post-hoc explanation 是两个问题，不能混成一个 fidelity 分数。 |
| Degraeve et al., **R-GCN: The R Could Stand for Random**, 2022 | 随机参数的 RR-GCN 在一些节点分类与 link prediction 设置可接近训练 R-GCN，提醒性能不必然来自已学 relation weights。 | 让现有 relation-type shuffle 结果进入解释主线：先确认 relation labels 是否提供超越拓扑的增益，才讨论某个语义 relation 的重要性。 | RR-GCN 不是 RGAT，也不是本数据集的直接基线；它提供的是必要的“关系语义是否真正被使用”的质疑视角。 |
| Amara et al., **GInX-Eval**, NeurIPS 2023 XAI workshop | 指出直接删边会导致 distribution shift，使 fidelity 评价失真；提出 in-distribution explanation evaluation。 | 局部扰动应优先采用 relation-preserving / degree-aware rewiring，或至少报告普通删除与可行重连两种结果。 | 不能把任意删边后的 logit 降低直接解释为高质量解释；需报告扰动是否仍像原图。 |

链接（按上表顺序）：

1. [HAN](https://doi.org/10.1145/3308558.3313562)
2. [R-HGNN](https://arxiv.org/abs/2105.11122)
3. [xPath](https://ojs.aaai.org/index.php/AAAI/article/download/26040/25812)
4. [HGExplainer](https://doi.org/10.1109/WI-IAT59888.2023.00035)
5. [GraphMask](https://arxiv.org/abs/2010.00577)
6. [PGExplainer](https://papers.neurips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html)
7. [CF-GNNExplainer](https://proceedings.mlr.press/v151/lucic22a.html)
8. [GraphFramEx](https://arxiv.org/abs/2206.09677)
9. [RR-GCN](https://arxiv.org/abs/2203.02424)
10. [GInX-Eval](https://neurips.cc/virtual/2023/75169)

## 与当前项目的精确衔接

本项目不是一般的 heterogeneous graph：节点全是人物，异质性来自 80 个有向 relation type（含显式 `__rev`）。因此应把论文中的 meta-path 改写成这里的**有向关系序列**：

```text
source L1 --r1--> intermediate person --r2--> target person
```

其中 `source L1` 和 `target L1` 只用于导出后分组；test target 在模型输入中仍为 unknown。当前基础设施已经覆盖：

- `training/attention_node_report.py`：root 内的 attention mass 原始表；
- `training/message_contribution.py`：relation-specific value message 的向量范数；
- `training/gradient_attribution.py`：以 predicted/true margin 为标量目标的 signed `alpha × gradient`；
- `training/attention_rollout.py`：两跳 typed path 的 raw attention product；
- `training/relation_controls.py`：关系删除、配对的随机 relation-pair 删除、relation-type shuffle；
- `README.md`：已记录 relation group ablation 的等量随机删边对照原则。

这形成一个比较有说服力的定位：**先用 RGAT 内部机制提出关系/路径候选，再用语义与拓扑都受控的图扰动检验模型依赖。**

## 明天可以新增的三页“实质工作”

### 1. 相关工作页：从权重到可验证解释

```text
HAN / R-HGNN：relation 与 meta-path 的注意力或消融
                 ↓
当前项目：raw alpha + value message + alpha×gradient（关系、方向、职业对）
                 ↓
xPath / GraphMask / CF-GNNExplainer：路径扰动、稀疏消息 mask、最小反事实
                 ↓
GraphFramEx / GInX-Eval：必要性、充分性、分布内评价
```

建议口播：*“我们不将 attention 直接称作重要性，而把它作为假设生成器；关系是否重要必须通过预测保持/翻转来验证。”*

### 2. 新分析页：relation-path 的三证据一致性

以每个 `(source L1, r1, r2, target L1)` 为行，放置：

| 候选路径 | attention rollout mass | message / gradient 支持 | 受控重连后的 margin drop |
| --- | ---: | ---: | ---: |
| `Culture --child--> Culture --…--> Culture` | 已可导出 | 可补充 path-level 聚合 | 计划验证 |

不需要宣称 rollout 是精确因果归因；它只负责候选排序。新增的扰动列才是对模型依赖的验证。

### 3. 新实验页：一个轻量、一个完整

**P0：无需改变模型的 top-k 路径验证（优先做）**

1. 用现有两跳 rollout，在正确分类 test root 中为每个预测类别取得 top-k typed path sequence；
2. 对路径实例做两种干预：删除；在可行候选中做 relation-preserving、degree-aware rewiring；
3. 固定 checkpoint，仅重跑推理，报告 target-class margin 的均值变化、预测翻转率、每种 relation sequence 的覆盖数；
4. 比较 attention、message、gradient 三种排序所选 top-k 的 deletion / insertion 曲线。

### 已实现的全关系对一跳版本

命令 `relation-pair-sweep-report` 已实现为一跳 RGAT 的**全关系对**条件化反事实消融：在
每个 test root 上自动枚举实际出现的 `(source L1, exact directed relation, target L1)`，删掉
该 root 的全部匹配入边，并和该 root 中**相同数量、同为可见 source L1、但 relation 不同**
的入边删除做配对。它只需每个 batch 一次基础前向；随后利用一层 across-relation RGAT 的
attention 重归一化恒等式精确重算每个 motif 的目标输出。基础 logit 会与模型结果逐 batch
核对，误差超阈值即停止。

主要输出是固定 true-target margin 的 `pair_margin_drop` 与
`pair_minus_control_margin_drop`，而不是只看整体 Accuracy。`summary_across_seeds.csv` 可以
直接按后者排序，找出在控制 source L1、target L1、root 和删边数后仍最重要的关系对。

对全部 L1 relation pair 的服务器运行入口：

```bash
bash scripts/run_l1_relation_pair_ablation.sh plan
RGCN_PAIR_MAX_ROOTS=2000 bash scripts/run_l1_relation_pair_ablation.sh run  # quick pilot
bash scripts/run_l1_relation_pair_ablation.sh run                            # all eligible test roots
```

筛选特定结果时仍须在 `edges.csv` 核验如 `child` / `child__rev` 的精确方向。该命令只删除
root-directed message，不删除自动生成的反向边，因此其 estimand 是**有向 message 的模型
依赖**，而不是无向社会事实的删除效应。

**P1：GraphMask-style message gate（后续主工作）**

1. 冻结 RGAT；在每层 `alpha × W_r h_j` 后接稀疏 stochastic gate；
2. 优化“最大化屏蔽 + 预测分布不变”，并避免用后层 embedding，减少 hindsight bias；
3. 输出 relation/layer/pair 的 gate keep probability，和现有三种分数做 rank correlation；
4. 用 P0 的局部受控扰动评价 gate 排名是否最 faithful。

## 最小汇报实验矩阵

| 层级 | 对象 | 已有/待做 | 主要回答的问题 |
| --- | --- | --- | --- |
| 全局 | relation label shuffle | 已有代码，待整理结果 | relation 语义是否比纯拓扑更有用？ |
| 全局 | relation/group removal + matched random drop | 已有代码，待整理结果 | 某关系组的价值是否超出“少了同样多的边”？ |
| 局部 | alpha / message / gradient | 已有 | 一个 test root 的内部信息流指向哪里？ |
| 局部 | two-hop typed path | 已有 alpha-only 候选 | 哪些关系组合被模型聚合？ |
| 局部 | top-k deletion / rewiring | 新增 P0 | 三种分数谁更能预测模型依赖？ |
| 局部 | sparse message gate | 新增 P1 | 哪些 message 在不改变预测的前提下不可替代？ |

## 建议避免的表述

- 不说“attention 证明亲属关系导致职业相同”；说“模型在该输入、该 checkpoint 下分配了更多 attention / 对其局部敏感”。
- 不把 `child` 与 `child__rev` 合并；它们对应不同消息方向和不同社会语义。
- 不用未配对的关系删除比较不同规模 relation group；至少配等量随机删边。
- 不以单个漂亮 case study 代替总体统计；至少报告 root 覆盖数、三个 seed 的均值和标准差、排名稳定性。
- 不将普通删边 fidelity 当作唯一证据；删除可造成分布外邻域，最好补可行重连对照。

## 建议阅读顺序（约 90 分钟）

1. HAN：了解 meta-path importance 的经典语言；
2. xPath：直接对应两跳 typed-path 的下一步；
3. GraphMask：直接对应现有 message contribution 的验证型升级；
4. GraphFramEx + GInX-Eval：决定怎样报告而不夸大解释；
5. R-HGNN + RR-GCN：为现有消融与 relation shuffle 增加相关工作支撑。
