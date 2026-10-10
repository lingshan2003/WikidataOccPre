# Freebase v3：仅构图的生命区间延伸对照

本版研究问题：将晚近出生、死亡年份没有记录的人物延伸到2026年，能否减少滑窗构图造成的孤立节点？对照为同一份 v2 基础数据、同一窗口下原有的“仅已知端点”规则。本次仅构图、关系分组和检查，**不执行 RGCN 或 GraphMask 训练**。

## 已确认的协议

- 出生年份 **≥1920（包含1920）**，且死亡年份确实缺失时，以 `[出生年份, 2026]` 作为构图生命区间。
- 有实际死亡年份者照常使用记录值；1920年以前出生者、只知死亡年份者、日期完全缺失者和无效生卒区间沿用旧规则。
- 不改写基础数据的 `death_year`，不把2026当作实际死亡日期输入模型。生成的节点表新增 `life_period_effective_death_year`、`life_period_death_year_imputed`；元数据明确标记 `life_period_membership_only`。
- 配置使用现有模块的严格比较 `birth > born_after`，因此写成 `born_after=1919`，对应这里的 `birth>=1920`。有效区间终点固定为2026，不随运行日期变化。
- 仍为1900–2000的101个中心年份，每年移动一次，闭区间 `[中心−20, 中心+20]`。窗口与生命区间相交即纳入，边必须两个端点都入选。
- 职业标签仍为 BHHT v2 的5个 L1。关系仍为 inherited、intimate_partnership、education_mentorship、influence_succession、other_acquired；保留正反向，共10个分组关系类型。
- 每个图仍按 seed=42 独立分层划分70%/10%/20%；局部类别少于3人时排除监督。训练标签可见范围与原特征处理一致。

基础节点/边与职业映射直接复用 v2，不需要另外上传大型数据。若规范化后的死亡年份缺失实际来自冲突或未来日期过滤，构图入口会拒绝将其视为缺失死亡时间，避免悄悄延伸错误记录。当前数据中符合规则的59,608人均为真正缺失死亡记录。

## Git同步后，在服务器执行

代码由Git管理；本次更新涉及新版入口、两份 v3 JSON、配置生成器、Freebase调度器的节点数估计，以及共享构图/审计模块。请将本次相关文件一起同步，不能只复制一份入口脚本。原来的原始导出表、职业映射及 v2 基础图保持原有布局。

服务器工作目录为：

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
conda activate wywikidata
```

这一步只用CPU，**不用指定 GPU**。入口使用当前环境的 `python`，依赖 NumPy、Pandas、SciPy、Torch、PyG；运行前会检查导入、来源记录、日期状态和磁盘空间。

先看计划，不构图：

```bash
bash Freebase/prepare_sliding_alive2026.sh plan
```

长时间构图请在共享 tmux 中运行，以便 SSH 断开后继续：

```bash
tmux new-session -A -s freebase_alive2026
```

进入会话后确认当前窗口空闲，将构图、日志查看和交互分别放在不同窗口。只在构图窗口执行一次：

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
conda activate wywikidata
bash Freebase/prepare_sliding_alive2026.sh run
```

这一个命令依次完成：检查/复用 v2 基础图 → 101个新生命窗口图 → 关系分组 → 精确构图检查 → 旧规则/新规则的拓扑对照。不会启动任何训练任务。

在另一个日志窗口，使用构图输出中显示的日志路径读取进度，例如 `tail -f artifacts/freebase_alive2026_construction_logs/<运行时间_PID>.log`。不要为了重新获取输出再次执行构图命令。

若需要先检查子集，使用：

```bash
bash Freebase/prepare_sliding_alive2026.sh run \
  --periods center_1900,center_1960,center_2000
```

后续运行全部时会复用兼容的已完成子图和分组图。运行锁防止重复写同一套输出；重跑会拒绝不兼容来源或配置。中断任务的临时目录不会被当作完整图，也不会删除已完成结果。

## 输出与检查

新版目录统一带有 `born_ge1920_alive2026_v3` 后缀，与旧版分开：

```text
artifacts/freebase_life_windows_1900_2000_pm20_step1_born_ge1920_alive2026_v3/
  center_YYYY/{graph_data.pt,nodes.csv,edges.csv,split_summary.json,...}
  construction_runs/<运行时间_PID>/
    manifest.json
    code_snapshot/...
    audit/{graph_audit.json,window_audit.tsv,checks.tsv,group_counts.tsv,
           topology_comparison.tsv,topology_comparison.json}
  latest_complete_construction.json

artifacts/freebase_grouped_sliding_1900_2000_pm20_step1_born_ge1920_alive2026_v3/
  center_YYYY/multi_group/{graph_data.pt,nodes.csv,edges.csv,...}

artifacts/freebase_alive2026_construction_logs/
  <运行时间_PID>.log
  <运行时间_PID>.exit_code
```

`latest_complete_construction.json` 指向最新成功运行的记录与审计目录，并列出该次实际处理的窗口。每次审计保存在独立运行目录，子集重跑不会覆盖先前101窗口的报告。

`manifest.json` 记录研究问题、对照、完整配置、seed、划分、Git提交/未提交状态及实际构图代码快照、主机、环境、命令、日志、结果路径、完成/失败状态和退出码。训练预算执行量固定记为0；配置中的 RGCN 最多50 epochs、GraphMask 每层3 epochs仅用于记录已有参数，尚未执行。

检查逐个比对原始节点身份和顺序、生命区间成员、完整诱导边与重编号、分组关系与反向边、去重结果、标签、互斥的数据划分、职业特征遮蔽和死亡年份缺失标记。基础数据现有的必要来源校验仍保留；一次多窗口构图只计算一次共享源图摘要，避免101次重复扫描。审计不额外计算大型数据或权重的哈希。

`topology_comparison.tsv` 将旧、新规则应用于相同源图，记录节点数、原始关系的有向消息边数、孤立节点数及比例、连通分量数、最大连通分量大小及比例、新增节点数和原先孤立而现在连接的节点数。这些是构图效果，不是模型性能。由于新增节点会改变局部划分，后续比较预测性能时还需要确定共同测试人物的评价方式；本版不做该训练实验。

构图成功后，下载最新审计目录和 `manifest.json` 即可分析拓扑变化，不必下载全部图和模型权重。

## 本地验收范围

2026-10-10已实际构建、分组并核查1900、1960、2000三个真实窗口；新版全部101个窗口尚待服务器构建，RGCN/GraphMask未开展。

| 中心年份 | 旧节点数 | 新节点数 | 旧孤立节点比例 | 新孤立节点比例 |
| --- | ---: | ---: | ---: | ---: |
| 1900 | 24,291 | 24,291 | 20.44% | 20.44% |
| 1960 | 72,412 | 79,786 | 12.46% | 6.50% |
| 2000 | 32,678 | 80,564 | 43.30% | 5.05% |

1900–1940的窗口按该规则不会变化。该假设改善了晚近窗口的连通性，但仍然不能消除所有孤立节点。
