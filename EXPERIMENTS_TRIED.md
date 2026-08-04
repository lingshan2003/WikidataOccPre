# 已尝试方向与实验结论

本文记录主线“人物关系是否有助于职业预测”的节点分类实验。它只列出已经实施的
方向；尚未运行的想法不应被误记为失败实验。

## 当前任务与共同协议

- 图：Wikidata 人物—人物多关系图，包含反向边。
- 任务：分别预测职业 Level 1、Level 2、Level 3；本文重点记录最困难的 Level 3。
- 防泄漏：验证/测试人物与当前训练 seed 人物的职业特征均为 `UNKNOWN`；只有已知训练邻居的职业可见。
- 纯关系职业实验：不输入 country、temporal 或人物文本，只输入职业表示与关系边。
- 主指标：Accuracy 与 Macro-F1。Level 3 类别长尾很强，不能只看 Accuracy。

## 已完成的实验

| 方向 | 具体做法 | 结果与结论 |
| --- | --- | --- |
| 三种关系 GNN | 对比 R-GAT、Fast R-GCN、CompGCN | Level 1 约 79% Accuracy，Level 2 约 68%，Level 3 约 43–44%。模型之间没有形成决定性差距；历史 L3 最好保存结果是 CompGCN 的 44.29% Accuracy、16.00% Macro-F1。 |
| 邻居职业作为输入 | 用可学习的 L1/L2/L3 类别 embedding 表示训练邻居职业；目标自身职业始终 mask | 这是当前主线输入协议。职业邻居确实是任务允许使用的核心信号，但尚未带来 L3 的明显突破。 |
| L3-only 对照 | 纯关系 R-GCN，只暴露邻居 Level 3 职业；sampled evaluation | Test：43.16% Accuracy、14.53% Macro-F1。 |
| 完整职业层级 | 纯关系 R-GCN，暴露邻居 L1+L2+L3 职业；sampled evaluation | Test：43.38% Accuracy、14.60% Macro-F1。相对 L3-only 只有极小增益，完整层级不是当前瓶颈的解法。 |
| 移除普通属性 | 去掉 country 与 temporal，形成“关系 + 职业”的干净实验 | 结果仍约 43%，没有显示普通属性是限制性能的主要因素。不同 artifact/split 的历史结果不可直接做小数点级比较。 |
| 固定职业语义向量 | 用 `intfloat/multilingual-e5-base` 编码训练邻居的 `(L1,L2,L3)` 职业描述；向量冻结，目标/val/test/seed 使用可学习 `UNKNOWN` | 服务器首轮 R-GCN 结果 Accuracy 仍约 0.43，Macro-F1 低约 2 个百分点。职业名称的文本语义没有提供额外有效信号。 |
| 扩大邻域 | CompGCN 使用 `--num-neighbors=-1,-1`，每个 mini-batch 的 seed 读取完整一跳与二跳邻居 | 未改善结果。因此原先 15/10 的采样上限不是主要障碍。此实验仍是两跳 mini-batch 训练，不是一次前向覆盖全图的 full-batch training。 |
| 完整图评估 | 支持 `--eval-mode full`，验证/测试在完整图上推理 | CompGCN 可以使用；FastRGCN 的 full-graph 前向曾因 relation-specific 中间张量需要约 84GB 显存而 OOM。该项用于减少评估采样噪声，不是特征或模型改进。 |
| checkpoint 与数据修正 | Macro-F1 严格最优 checkpoint 保存；`min_delta` 只控制早停。`Missing`/`Unknown`/空值统一为真实缺失，不参与监督、指标或职业输入 | 这些是可信度和防泄漏修正，不预期直接提高分数，但使后续比较有效。 |

## 结果解读

目前至少可以排除三种直觉解释：

1. **不是简单的关系 GNN 选型问题**：R-GAT、R-GCN、CompGCN 的差异有限。
2. **不是 15/10 采样漏掉了关键邻居**：完整一、二跳邻居没有带来增益。
3. **不是职业名称缺少语义初始化**：固定 E5 职业向量没有改善，且降低了 Macro-F1。

因此，Level 3 约 0.43 的平台更可能与细粒度职业长尾、关系对职业的有效信息有限，或两跳可传递信号本身不足有关。这个结论不等于“关系没有作用”，而是说目前的输入表示和两层传播尚未把关系信号转化为更高的细粒度预测性能。

## 可比性提醒

- 只比较同一 `graph_data.pt`、同一 split、同一 seed、同一 evaluation mode 下的结果。
- `sampled` 和 `full` evaluation 的指标不应做小数点级直接排名。
- 语义实验应与同 artifact、同 seed、同超参数的 categorical 实验配对比较。
- 探索阶段先使用 `--skip-test` 选 validation checkpoint；只有锁定配置后才运行一次最终 test。

## 已实现代码、尚未在服务器运行

以下能力已经进入训练入口，但尚无服务器实验结果；因此它们不应算作成功或失败尝试：

- 不输入职业、国家、时间或节点 ID 的纯关系/结构基线（共享常量节点特征）；
- relation type 全局置乱（保留拓扑和 relation 频数）、按关系组或关系名的消融，以及与关系组删边数量匹配的随机关系对删边对照；
- L3 长尾的有效样本数 class-balanced loss、logit-adjusted loss 与 class-balanced seed sampling；
- 真正的 full-batch 全图训练。该模式为避免职业泄漏，当前只支持不输入职业特征的设置。
- 图稀疏度、连通分量、可见训练职业邻居覆盖、relation-level 职业同配性和按覆盖分桶性能的离线诊断。

## 尚未实施，不应算作失败尝试

- 人物简介、教育经历等人物文本/外部属性；
- 独立的职业 link prediction 分支的系统性评估。
