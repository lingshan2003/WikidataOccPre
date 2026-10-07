# 五个时期的 Layer 0 启用后 checkpoint 补充实验

仅针对 **multi_group** 的 ≤1500、1501–1900、1941–1960、1981–2000、≥2001 五个时期。
复用 `artifacts/dbpedia_grouped_20y_v1` 的分组图和 `runs/dbpedia_grouped_20y_v1` 的冻结 RGCN，
只重跑这五个 GraphMask 训练及测试报告。原来的 GraphMask 结果保留。

## 选择规则

训练顺序及参数沿用原实验：先启用 Layer 1 门控训练 3 epoch，再启用 Layer 0 门控训练 3 epoch；
seed=42、batch=32、beta=0.03、fanouts=auto、验证 Macro-F1 相对差上限=0.05。
唯一改变是新增 `--checkpoint-selection all-layers-enabled`：只有两层门控都已启用后的 epoch 才能参与最终选择。
候选必须满足原保真度阈值，再选择**验证集整体 hard retention 最低**的版本。
这不是单独最小化 Layer 0 留存率，也不使用测试集选择 checkpoint。

如果这个阶段没有合格候选，任务明确失败并保留训练历史，不回退到只启用 Layer 1 的版本。
Layer 0 门控启用后仍可能保留 100%，本实验不强制删消息。
原入口默认 `any-stage`，继续允许从所有训练阶段选择，原实验含义不变。

旧训练器只保存最终选中的 `graphmask_probe.pt`，不保存每个 epoch 的权重。
因此需要按原参数重放 GraphMask 训练，不能仅凭 `training_history.json` 恢复某个 epoch 的权重；
也不要从旧 probe 接着训练，因为这会改变原训练轨迹。
相同 seed 有助于复现，但不同 CUDA／PyTorch／PyG 环境的重跑不保证得到完全相同的权重。

## 服务器运行

同步这些文件（或更新仓库）：

```text
training/graphmask_train.py
training/graphmask_report.py
DBpedia/grouped_pipeline.py
DBpedia/run_layer0_enabled_5periods.sh
DBpedia/compare_layer0_checkpoint_selection.py
DBpedia/LAYER0_ENABLED_SUPPLEMENT.md
config/dbpedia_multi_group_layer0_enabled_5periods_v1.json
```

在原来跑通实验的 Python 环境和服务器仓库根目录执行：

```bash
bash DBpedia/run_layer0_enabled_5periods.sh plan
bash DBpedia/run_layer0_enabled_5periods.sh run
```

脚本只运行 `graphmask` 阶段，固定五个时期与 multi_group，并强制不加入 full graph。
不会调用 RGCN 训练。成功后自动生成与原选择结果的对照表。
原有 RGCN／collapse 阶段记录的命令与输入必须仍兼容；使用原实验的 Python 解释器，保留原图与模型文件。
`DBPEDIA_PYTHON_BIN` 可指定解释器路径；没有设置时使用当前环境的 `python`。
此前若将该变量固定在其他环境，可先 `unset DBPEDIA_PYTHON_BIN`。

可沿用原来的 `CUDA_VISIBLE_DEVICES` 设置，进程内仍用 `cuda:0`。
新 JSON 的图、RGCN 参数及根目录与原实验相同；不要为这次选择对照另改 epochs、batch、beta、seed 或保真度阈值。
补充入口会清除遗留的 `DBPEDIA_GROUP_SEED`、`DBPEDIA_GROUP_TRAIN_*` 和 GraphMask 数值参数覆盖，
按 JSON 保持原训练设置，并固定新 GraphMask 根目录。源图／RGCN 路径和 device 的覆盖仍有效。
使用自定义路径时须相应修改 JSON、Bash 中的新根目录和比较／打包命令；打包快捷入口使用本文默认路径。

后台运行：

```bash
nohup bash DBpedia/run_layer0_enabled_5periods.sh run \
  > dbpedia_layer0_enabled_5periods.log 2>&1 &
```

失败后可执行同一条命令重试；完成且兼容的任务跳过。重新汇总与比较可执行：

```bash
bash DBpedia/run_layer0_enabled_5periods.sh summarize
```

## 新结果与下载

新根目录为 `runs_graphmask/dbpedia_multi_group_layer0_enabled_5periods_v1/`。
每个时期的 `multi_group/seed_42/` 包含 probe、validation、history、manifest 和 test_report。
`training_history.json` 中的 `eligible` 仍表示通过保真度检查；`selection_eligible` 另外考虑门控启用条件。
新 probe 与 manifest 的 `selected_checkpoint` 明确保存选中 epoch、策略和 `enabled_layers`。
测试报告 manifest 也记录实际启用的门控层。
新的 resolved config 存在新 GraphMask 根目录，保留原 RGCN 根目录中的实验配置。

根目录新增：

- `matrix_summary.tsv/json`、`relation_group_summary.tsv/json`：五个时期的新结果。
- `checkpoint_selection_comparison.tsv/json`：原选择／新选择的 epoch、Layer 0 启用状态、验证保真度、两层测试消息留存率及测试 Macro-F1。
- `layer0_group_comparison.tsv`：各关系组原／新 Layer 0 消息留存率与保留消息占比。
- `pipeline_failures.json`：应为空列表；只有五份报告都完成才表示本轮成功。

比较脚本根据旧 history 复现选择规则，并核对最终 validation；新结果还检查显式保存的启用状态。
比较前核对两次实验的源图／RGCN 路径、fanouts、seed，以及报告 split／采样设置。
表内留存率与占比都是 0–1 数值；报告统计对象是采样消息，不是去重后的全图边。
无观测关系组的留存率留空，不填成 0。

打包供本地绘图，保留原、新结果的 JSON/CSV/TSV 与配置，但不打包权重、大型节点／边 CSV 或逐根 top edges：

```bash
bash DBpedia/run_layer0_enabled_5periods.sh package
```

生成仓库根目录的 `dbpedia_layer0_enabled_5periods_visualization.tar.gz`。
在本地终端下载：

```bash
scp siruilai@dingqiyang:/mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre/dbpedia_layer0_enabled_5periods_visualization.tar.gz /Users/wangyue/RGCN/
```

如果 SSH 实际使用别名／IP／跳板机，沿用此前成功下载时的连接方式。

## 旧日志中的候选预览（不是新测试结果）

2026-10-06 收到的原实验训练历史中，五个时期都有启用 Layer 0 后且满足保真度的候选：

| 时期 | 按新规则应选的原日志 epoch | 验证整体消息留存率 | 验证 Macro-F1 相对差 |
| --- | ---: | ---: | ---: |
| ≤1500 | 5 | 61.70% | 3.01% |
| 1501–1900 | 6 | 68.41% | 0.05% |
| 1941–1960 | 6 | 66.43% | 0.03% |
| 1981–2000 | 5 | 75.52% | 0.48% |
| ≥2001 | 4 | 79.18% | 0.34% |

≤1500 原来已选 epoch 5，且 Layer 0 门控已启用但硬门控全部保留，故严格复现时该时期预计不变。
其余四个时期的原结果选中了仅启用 Layer 1 的阶段。
本地下载包没有权重；上述数值仅核对旧验证日志，新的测试集关系组占比必须由服务器重跑后生成。

## 本地验证

已通过 18 项 GraphMask／DBpedia 流程测试，覆盖早期与全层候选选择、无合格候选时明确失败、
checkpoint 重载后启用状态、原 RGCN 命令兼容，以及五期范围／输出根目录／参数固定。
另用真实两层 RGCN 小图在 CPU 上实际执行两种选择策略的 GraphMask 训练、验证与测试报告，
并运行比较脚本确认启用状态、结果表和无观测关系组的空值处理。
这些验证确认接口和选择逻辑；五个时期的正式 GPU 结果仍须在服务器运行后确认。
