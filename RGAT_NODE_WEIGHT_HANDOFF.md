# RGAT 节点级权重分析交接

更新时间：2026-08-07  
当前代码基线：`89281bb`（`main`）

## 1. 当前研究目标

分析一跳 RGAT 在预测目标节点职业时，模型如何在不同关系和来源职业之间分配 attention。

当前 pair 定义为：

```text
(O_s, R, O_t)
```

- `O_s`：源节点 `u` 的真实 L1 职业；
- `R`：图中的精确有向关系，例如 `child`；
- `O_t`：目标节点 `v` 的真实 L1 职业；
- `child` 与 `child__rev` 必须分别解释，不能自动合并。

当前只关注：

- 模型：`rgat_one_hop`；
- split：`test`；
- message-passing layer：第 1 层；
- seed：42、43、44；
- 模型训练输入包含 L1、L2、L3 层级职业特征。

## 2. 已确认的 RGAT attention 语义

项目使用 PyG `RGATConv`，当前模型没有覆盖以下默认参数：

- `attention_mechanism="across-relation"`；
- `flow="source_to_target"`；
- `attention_mode="additive-self-attention"`；
- `mod=None`。

因此 `edge_index[0]` 是源节点，`edge_index[1]` 是目标节点；同一目标节点的所有入边、所有关系共同参与 softmax。每个 head 下，一个目标节点的完整入边 alpha 之和约等于 1。

当前模型为 4 heads、`concat=False`。导出值 `attention_mass` 是先在 head 上平均，再在指定节点/pair 内对边求和。

## 3. 已拆分的两套分析入口

### 3.1 边级旧口径

文件：`training/attention_edge_report.py`  
命令：`python run.py attention-edge-report ...`

保留原有以边为观测的分析：

```text
attention_mean = matching edge alpha 总和 / matching edge 数
```

L1 pair 输出只包含边级 `attention_mean` 和 `edge_count`。先前混在其中的 `attention_mass_per_target` 已移除。

旧命令 `attention-report` 仍是 edge report 的兼容别名。

### 3.2 节点级原始导出

文件：`training/attention_node_report.py`  
命令：`python run.py attention-node-report ...`

它输出：

- `root_attention_roster_by_seed.csv.gz`：每个目标节点的总 attention budget；
- `root_direct_attention_sparse_by_seed.csv.gz`：每个目标节点内，按 source L1、关系和 source visibility 分组的 attention mass。

该入口只保存节点级原始量，不擅自定义跨节点最终指标。

共享 checkpoint、采样、设备和 CSV 工具已移至 `training/attention_common.py`，其中不包含聚合口径。

## 4. 当前节点优先权重表定义

生成脚本：

```text
scripts/build_rgat_one_hop_node_weight_table.py
```

对于 pair `p=(O_s,R,O_t)` 和目标节点 `v`，先定义：

```text
M_v(p) = v 的所有匹配 p 的入边 attention alpha 之和
```

计算时先合并同一目标节点、同一 pair 下的所有 `source_visibility` 行。因此 `visible_train`、`hidden_validation_or_test`、`missing_or_unknown` 不参与筛选或分层。

### 4.1 主表中的 `mean_a`

```text
mean_a(p) = sum_v M_v(p) / n(p)
```

其中只对至少拥有一条匹配边的目标节点求平均，且：

```text
n(p) = 拥有该 pair 的目标节点数
```

`n` 是节点数，不是边数。

它回答：

> 在测试集中，拥有该关系模式的目标节点中，该模式平均获得多少 attention mass？

### 4.2 覆盖率

```text
coverage(p) = n(p) / target_n(O_t)
```

### 4.3 全部目标节点平均分配

```text
mean_a_all_target(p) = sum_v M_v(p) / target_n(O_t)
                     = coverage(p) * mean_a(p)
```

没有该 pair 的目标节点在这里贡献零。

### 4.4 `O_t` 总 attention budget 占比

严格定义为：

```text
attention_share_of_Ot_budget(p)
  = pair 在所有 O_t 目标节点上的 attention mass 总和
    / 所有 O_t 目标节点的完整入边 attention budget 总和
```

分母从 roster 的 `total_attention_mass` 显式计算，不直接假设等于节点数。当前数据中每个目标节点总 mass 约为 1，因此该 share 与 `mean_a_all_target` 几乎相同。

指标的最终使用方式还没有决定。下一步需要讨论是以 `mean_a` 为主、以 budget share 为主，还是两者同时报告。

## 5. 当前数据是否足够

足够，不需要回服务器重新生成。

原始目录：

```text
runs_report/level1/rgat_l1_root_attention_all_relations/direct/
```

关键文件：

```text
root_direct_attention_sparse_by_seed.csv.gz
root_attention_roster_by_seed.csv.gz
attention_report_manifest.json
```

manifest 确认：

- `rgat_one_hop` 有 3 个 seed；
- 每个 seed 有 66,407 个 test target；
- 使用完整一跳邻域 `fanouts=[-1]`；
- synthetic self-loop 数为 0；
- 每个目标节点的总 attention mass 约为 1。

`source_visibility` 已全部合并。来源节点没有可用真实 L1 时没有丢弃，而是保留为 `__UNLABELED__`，从而保证全部 attention budget 被统计。

## 6. 已生成的权重表

输出目录：

```text
runs_report/level1/rgat_l1_root_attention_all_relations/direct/node_weight_tables/
```

文件：

```text
rgat_one_hop_l1_pair_node_weight_summary.csv   # 三 seed 汇总，996 行
rgat_one_hop_l1_pair_node_weight_by_seed.csv   # seed 明细，2988 行
rgat_one_hop_l1_pair_node_weight_manifest.json # 定义与完整性校验
```

汇总表的首要字段满足：

```text
source_l1,target_l1,relation,mean_a,n
```

并补充：

- `mean_a_seed_std`；
- `matching_edge_n`；
- `matching_edges_per_node`；
- `target_n`；
- `coverage`；
- `mean_a_all_target`；
- `attention_share_of_Ot_budget`；
- `attention_share_of_Ot_budget_seed_std`。

## 7. 示例：Culture → child → Culture

三 seed 汇总结果：

```text
source_l1                    Culture
target_l1                    Culture
relation                     child
mean_a                       0.3048657529
n                            3720
mean_a_seed_std              0.0690200932
matching_edge_n              4326
matching_edges_per_node      1.1629032258
target_n                     22539
coverage                     0.1650472514
mean_a_all_target            0.0503172546
attention_share_of_Ot_budget 0.0503172546
```

解释：

- 3,720 个 Culture test target 至少有一条 `Culture --child--> Culture` 匹配入边；
- 在这些涉及节点中，该 pair 平均获得 30.49% 的 attention；
- 它覆盖全部 Culture test targets 的 16.50%；
- 在全部 22,539 个 Culture targets 的总 attention budget 中，该 pair 占 5.03%。

## 8. 完整性校验

生成表时执行了以下校验：

- 所有 pair 的 mass 总和与 roster 的 typed attention budget 一致；
- 每个 seed、每个 `O_t` 下，所有 pair 的 budget share 加总约为 1；
- 最大 share 加总误差：`3.757778088697705e-09`；
- `Culture/Culture/child` 每个 seed 均为 3,720 个不同目标节点、4,326 条匹配边；
- 项目测试：24 项全部通过。

浮点 mass 总和与 typed budget 的最大绝对误差约为 `8.92e-05`，相对于两万量级 budget 可视为导出浮点累积误差。

## 9. 复现命令

```bash
.venv/bin/python scripts/build_rgat_one_hop_node_weight_table.py \
  --input-dir runs_report/level1/rgat_l1_root_attention_all_relations/direct \
  --output-dir runs_report/level1/rgat_l1_root_attention_all_relations/direct/node_weight_tables
```

脚本以流式方式读取约 402 万行 sparse CSV，不将原始文件整体加载进内存。

## 10. 下一步待讨论

1. 最终图表主指标采用 `mean_a`、`attention_share_of_Ot_budget`，还是并列报告。
2. `mean_a` 高但 `coverage` 极低的稀有 pair 如何呈现，是否设置最小 `n`。
3. 正式结果是否保留 `__UNLABELED__` 行；当前为完整性而保留。
4. 是否只展示原始关系、同时展示 `__rev`，或按研究语义人工选定方向。
5. seed 波动较大的 pair 如何报告，是否增加节点级 bootstrap 区间。
6. 下一阶段可制作固定 `O_t` 的 pair 排名、矩阵或热力图，但在指标口径最终确定前不要过度可视化。

## 11. 重要口径提醒

- 不要把 `mean_a` 描述为“测试集中所有节点的平均权重”；它是“拥有该 pair 的目标节点中的平均 attention mass”。
- 不要把 `n` 写成边数；边数是 `matching_edge_n`。
- 不要在节点平均前按匹配边数再除一次，否则会重新变成边级强度。
- budget share 的分母应写成显式的总 attention budget；只有在验证每节点总 mass 为 1 后才能简写为目标节点数。
- attention 是模型内部权重分配描述，不在本阶段扩展为因果重要性解释。
