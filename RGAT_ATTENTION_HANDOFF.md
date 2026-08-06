# RGAT attention 工作交接（截至当前）

## 下一会话的目标

研究 L1 职业预测中，关系和来源职业如何被 RGAT 使用。已经完成一跳与两跳 RGAT 的全图注意力导出；下一优先级是把两跳模型从“第二层直接边 alpha”升级为 **two-hop attention rollout**，得到原始两跳来源人物对 root 的路径级贡献近似。

## 已完成的模型与实验

### 模型

- `RelationalGATClassifier` 已支持 `num_layers >= 1`。
- 一跳模型：`rgat_one_hop`，3 个 seed（42/43/44），一层 RGAT。
- 原有基线：`rgat_baseline`，3 个 seed（42/43/44），两层 RGAT。
- RGAT 有 4 heads，`concat=False`；每层为
  `LayerNorm(h + Dropout(GELU(RGATConv(h))))`，最后线性分类。
- 预测 root 的所有职业层级特征均在每个 forward 中掩码；邻居保留训练协议下可见的特征。

相关实现：

- `models/rgat.py`
- `training/train.py`（`--num-layers`）
- `training/attention_utils.py`（返回 alpha 与真实关系类型对齐，排除可能的合成 self-loop）

### 全感受野 attention report

已实现 `run.py attention-report` 与脚本：

```bash
bash scripts/export_l1_full_receptive_attention.sh run
```

它对一跳 checkpoint 使用 `--num-neighbors full` = `-1`，对两跳 checkpoint 使用 `-1,-1`；不重新训练。

完整输出目录：

```text
runs_report/level1/rgat_l1_full_receptive_attention_all_relations/
```

输出覆盖：

- 332,033 个拥有保留 L1 标签的节点；每个节点恰好一次作为 prediction root。
- 80 个精确有向关系（40 个原始关系和各自 `__rev`）。
- 6 个 checkpoint：一跳 3 个、两跳 3 个；两跳分别导出 Layer 1 和 Layer 2。
- manifest 显示 root 入边中的合成 self-loop 数为 0。
- `--occupation-matrix-relations all` 已实现；一次运行即可导出所有关系，而非只导出五种亲缘关系。

关键文件：

```text
l1_relation_attention_by_seed.csv       # seed 级的 relation × source L1 × target L1 聚合
l1_relation_attention_summary.csv       # 三 seed 均值、标准差、n、attention mass
relation_attention_by_seed.csv           # 不分 L1 的关系汇总
relation_attention_summary.csv
l1_relation_attention_matrices.md        # 80 个精确有向关系的矩阵 Markdown
attention_report_manifest.json
```

## 当前 alpha 的提取定义

对每个 checkpoint、每一层，代码：

1. 前向计算 RGAT，拿到每条边、每个 head 的 alpha。
2. 对 4 个 head 求均值。
3. 只保留“边的目标端是当前 prediction root”的真实 typed edge。
4. 使用 source 的真实 L1 与 root 的真实 L1 进行**事后分组**；这些 L1 标签只用于报告分组，不作为当前 root 的输入特征。

对关系 `r`、source L1 `s`、target L1 `t`，当前 CSV 的 `attention_mean` 是：

\[
\bar\alpha_{r,s,t}=
\frac{1}{|E_{r,s,t}|}
\sum_{e\in E_{r,s,t}}
\frac{1}{H}\sum_{h=1}^H\alpha_e^{(h)}.
\]

`attention_mass_per_target` 是：

\[
M_{r,s,t}=
\frac{1}{|V_t|}
\sum_{e\in E_{r,s,t}}
\frac{1}{H}\sum_h\alpha_e^{(h)}.
\]

前者是“典型边的相对注意力份额”；后者是“该 source-target cell 对典型 target 节点的累计注意力流量”。两者不能混用。

## “全边”与重复计权：已经解决和仍然存在的区别

### 已解决：没有因邻居采样漏边，也没有同一条 root 入边被跨 batch 重复累计

- 报告使用全邻居 fanout `-1`；不是 15/10 采样。
- 每个有 L1 标签的人恰好一次作为 root。
- 代码仅累计 `edge_index[1] < batch.batch_size` 的边，即“目标端为当前 root”的边。
- 因而，对一个原始有向边 `u -> v`：只要 `v` 是有 L1 标签的 root，它只会在 `v` 作为 root 时累计一次；不会因为 `v` 出现在其他人的邻域中而被再次累计。

### 仍然存在：edge-average 让高入度人物在一个 cell 内有更多影响

`attention_mean` 的除数是**边数**，所以并不会被“同一条边重复计算”污染；但它的统计单位是边，而不是人。

- 目标人物 A 在某 cell 中有 30 条不同的 `child` 入边，会贡献 30 个 alpha；B 只有 1 条，会贡献 1 个 alpha。
- 这不是 bug，也不是覆盖不全，而是 edge-level 均值的定义：高入度人物在 cell 均值中权重更大。
- 更关键的是每个 alpha 都在各自目标人物的完整入边集合内 softmax。不同 target 人物的入度、关系混合不同，即使 relation logit 相同，alpha 也可能不同。

因此全边导出保证了**覆盖完整**，但不自动让不同 L1 cell 的 `attention_mean` 成为无条件、可解释为“预测重要性”的统一标尺。

## 为什么当前 alpha 图不能直接称为预测重要性

本项目没有设定 `attention_mechanism`，PyG RGAT 默认是 `across-relation`：每条入边与该 root 的所有关系类型入边共同 softmax。

\[
\alpha_{j\to i}^{(r,h)}=
\frac{\exp(z_{j\to i}^{(r,h)})}
{\sum_{r'}\sum_{k\in\mathcal N_{r'}(i)}\exp(z_{k\to i}^{(r',h)})}.
\]

所以 alpha=0.30 是“该边在目标 i 的该 head 中占 30% 的局部注意力份额”，不是全局预测贡献为 0.30。

还需要注意：消息为约 `alpha * W_r h_j`，之后经过残差、GELU、LayerNorm、后续层与分类器。相同 alpha 的两条边传递的向量可能大小/方向不同；更高 alpha 不等于删边后 logit/F1 下降更多。

可使用当前 alpha 图做的表述：

> 模型在指定关系、source L1、target L1 的边上呈现的平均局部注意力份额。

不能只凭 alpha 做的表述：

> 某关系对职业继承的因果效应；或某 cell 对最终预测一定更重要。

## 当前结果与已做可视化

已做两个五关系的矩阵：source L1 为行、target L1 为列，保留 Culture、Discovery/Science、Leadership、Sports/Games（不画 Other）。

- 两跳主模型 Layer 2 的原始 mean alpha：`rgat-kinship-alpha.html`。
- 一跳 RGAT 的原始 mean alpha：`rgat-one-hop-kinship-alpha.html`。
- 之前的 attention mass 图：`rgat-attention-matrices.html`。

这三个可视化位于 Codex 线程可视化目录，而非项目 `runs_report`；其中原始数值可从 `l1_relation_attention_summary.csv` 重建。

已观察到的探索性模式（不应写成因果结论）：

- Science 目标的博士师承关系（`doctoral_advisor` / `doctoral_student` 及反向）累计 attention mass 约 39%–42%，与亲缘关系约 40% 相当。
- Culture 目标：教育关系约 17%–18%，`influenced_by` 约 4%。
- Leadership 目标：亲缘约 78%，`consecrator` 与反向约 16%。
- Sports/Games 目标：亲缘约 80%，教练/职业伙伴关系约 13%–14%。
- 用边数校正后的 source-target alpha lift，若干 Culture/Science -> Leadership 的 sibling/spouse/亲缘路径表现出稳定的高于列基线的平均 alpha；这仍是机制描述，非历史社会因果。

## 两跳解释的关键修正：不能把 Layer 2 alpha 叫作“二跳关系权重”

当前报告的两跳 Layer 2 alpha 仍然是**直接进入 root 的一跳边** `j -> i` 的 alpha，只是 `j` 的表示已经在 Layer 1 吸收了其邻居的信息。

它应命名为：

> 两层模型的第二层 contextualized direct-edge attention。

它不能直接回答：

> 原始两跳人物 `k -> j -> i` 对 root `i` 的贡献有多大？

## 下一优先级：two-hop Attention Rollout

目标是针对两层模型回推至原始输入人物节点：

1. 构建 Layer 1 与 Layer 2 的稀疏注意力矩阵。使用 destination-row convention：
   \(A^{(\ell)}_{i,j}=\alpha^{(\ell)}_{j\to i}\)。
2. 对每个 root，计算 raw two-hop path attention：

\[
R^{(2)} = A^{(2)}A^{(1)},\qquad
R^{(2)}_{i,k}=\sum_j A^{(2)}_{i,j}A^{(1)}_{j,k}.
\]

这里 `k -> j -> i` 是一条原始两跳路径；对不同中间节点 `j` 求和。

3. 用 rollout 分数 `R[i,k]`，而不是 Layer 2 的直接 alpha，按：
   - 两跳 source 的 L1；
   - root target 的 L1；
   - 需要时按 path relation pair `(r1: k->j, r2: j->i)`；
   - checkpoint / seed；
   进行聚合，输出与当前 Figure-5 风格相同的矩阵。

### 实现时必须改变什么

当前 exporter 对每层都只保存“目标是 root”的 alpha；这足够直接关系表，但不足 rollout。

新的 rollout exporter 必须在同一轮 full-neighbourhood forward 中额外保留：

- Layer 2：目标为 root 的边 `j -> i`；
- Layer 1：目标为这些中间节点 `j` 的边 `k -> j`；
- local/global node ID 映射（`batch.n_id`）；
- 每条路径的关系 pair `(r1,r2)`，或至少其 base relation；
- head 信息。

然后对每个 batch 做稀疏矩阵乘法/按路径 scatter-add。只在最终 root 维度汇总，避免 root 作为别人的邻居时重复计数。

### Rollout 的必要保留意见

`A2 @ A1` 是很有用的**attention-path approximation**，但不是当前 RGAT 的精确数值归因：

- 模型有 relation-specific value transform；实际消息不是单纯 alpha 的乘积。
- `concat=False` 让 head 在每层输出中混合；应尽量保留每个 head 后再按模型实际 head mixing 汇总，不能先随意平均后宣称精确 rollout。
- 残差、GELU 和 LayerNorm 使得标准 Transformer 式 rollout 不是严格等式。
- 可以同时输出：
  1. **raw path attention product** `A2 @ A1`（最透明）；
  2. **residual-aware heuristic rollout**，例如对每层使用 row-normalized `(I + A)`；
  但第二种必须明确为启发式。

最稳妥的命名：

> two-hop attention-path score / attention-rollout approximation。

不要命名为“真实两跳因果贡献”。

## 后续可做事项（优先顺序）

1. **实现 full-neighbourhood two-hop rollout exporter**，导出 raw `A2 @ A1` 的 source L1 × target L1 矩阵；必要时筛选 relation pair，避免 80² 个面板全部展示。
2. 为当前直接 alpha 增加 target-aware 输出：每个 root 内的 source-group alpha、同 root 匹配的比较、入度校正的 `d_i * alpha`。这需要新一次前向，因为当前 CSV 只有 cell 总和/边数，不保存 root-level alpha。
3. 选择要报告的关系组：亲缘五关系及反向、博士师承、教育、教练/职业伙伴、`consecrator`；全部 80 个关系保留为补充材料 CSV。
4. 做 cell/group-level edge ablation：删掉指定 `relation × source L1 × target L1`（必要时删除对应反向边），比较目标 L1 的 logit/probability 与 F1 变化。这才更接近“预测重要性”。
5. 如要作统计比较，按 root 配对并报告 seed 间不确定性；不要只按 cell 的 edge-average alpha 给出因果或继承强度结论。

## 关键终端命令

完整全关系的已有导出：

```bash
bash scripts/export_l1_full_receptive_attention.sh plan
RGCN_L1_FULL_ATTENTION_BATCH_SIZE=32 bash scripts/export_l1_full_receptive_attention.sh run
```

如果显存不足，使用 `16` 或 `8`。该脚本默认将结果写入：

```text
runs_report/level1/rgat_l1_full_receptive_attention_all_relations/
```
