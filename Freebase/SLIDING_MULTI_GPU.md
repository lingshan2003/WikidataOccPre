# Freebase v2：101 个逐年滑窗与多卡训练

默认配置为 `config/freebase_grouped_sliding_1900_2000_pm20_step1_v2.json`，入口为 `Freebase/run_sliding_multi_gpu.sh`。使用已经完成职业语义核查的 BHHT v2 标签：待审职业临时映射 Other，人物跨多个 L1 时按原职业数组顺序选第一个已有审核结果的职业；全部职业待审才使用临时 Other。仍为 5 个 L1 类别。

## 窗口与构图口径

- 中心年份 1900–2000，首尾均含，每次移动 1 年，共 **101** 个窗口。
- 每个窗口是闭区间 **[中心年份 − 20，中心年份 + 20]**。例如 `center_1900` 为 1880–1920，`center_1901` 为 1881–1921，`center_2000` 为 1980–2020。
- 这里“20 年”是前后各 20 年，实际跨度为 40 年、含 41 个整数年份。
- 延续原实验的生命区间口径：出生和死亡都已知时，保留有效生命区间与窗口相交的人物；只知一个日期时，仅在该日期落入窗口时纳入；两个日期都缺失或死亡早于出生时排除。
- 仅保留两个端点均在当前窗口中的边，重编号节点；保留基准职业与原关系 ID，生成正反向消息边。
- 每个子图独立分层划分 train/val/test = 70%/10%/20%，seed=42。少于 3 人的局部类别不参与监督。职业特征继续采用原实验的防标签泄漏处理。
- 默认只跑 `multi_group`：inherited、intimate_partnership、education_mentorship、influence_succession、other_acquired，正反向共 10 个关系类型。可用 `--representations binary,multi_group` 增加二分关系对照。

这些是人物生命区间窗口，关系本身没有事件时间过滤。相邻窗口高度重叠，人物可能在不同窗口进入不同的数据划分；结果适合比较窗口内独立模型，不应直接解释为跨时期预测或 101 次独立重复实验。

## 服务器运行

本地 `artifacts/freebase_sliding_multi_gpu_v2_code.tar.gz` 是代码包，`artifacts/freebase_sliding_multi_gpu_v2_inputs.tar.gz` 是基础输入包。代码包包含调度器及运行所需的共享模块；输入包包含两张原始导出表和 BHHT 职业映射。二者均不包含 2.2 GB 的本地滑窗图，服务器可自行重建。

将代码包上传服务器，并在服务器项目根目录解压：

```bash
tar -xzf freebase_sliding_multi_gpu_v2_code.tar.gz
```

如果服务器缺少上述输入，再上传并解压输入包：

```bash
tar -xzf freebase_sliding_multi_gpu_v2_inputs.tar.gz
```

服务器已经使用 `processed/05_final/` 布局时，也可以只从输入包取出新职业映射，从而继续使用原有原始表：

```bash
tar -xzf freebase_sliding_multi_gpu_v2_inputs.tar.gz \
  docs/freebase_bhht_semantic_review_2026-10-08/profession_l1_semantic_crosswalk.tsv
```

若已经构建过服务器 v2 基础图，请保留当时的输入布局、文件与环境覆盖设置，避免重复解压数据改变来源记录。代码更新和输入数据更新分开处理。

在服务器项目根目录执行，先激活环境；脚本直接使用当前环境的 `python`，不绑定本地或服务器 Python 路径。

```bash
conda activate wywikidata
export CUDA_VISIBLE_DEVICES=4,5
export FREEBASE_GROUP_DEVICE=cuda:0

# 查看 101 个窗口的构图、RGCN、GraphMask 命令
bash Freebase/run_sliding_multi_gpu.sh plan all

# 自动准备 v2 基础图和滑窗图，并开始多卡完整实验
bash Freebase/run_sliding_multi_gpu.sh run all
```

4、5 只是两张可用卡的示例，请改为当时允许使用的物理 GPU 编号。脚本不会扫描并自动占用其他卡。每个 worker 会把自己的 `CUDA_VISIBLE_DEVICES` 设成单张物理卡，因此训练命令均使用该进程内的逻辑 `cuda:0`。

使用一张卡时只需改为 `export CUDA_VISIBLE_DEVICES=4`。选定多张卡后，可通过 `--num-gpus 2` 只使用列表中的前两张；也可用 `--gpus 4,5` 显式覆盖环境中的池。

可先单独完成全部构图。这一步不要求 GPU，之后 `run all` 会复用已有子图：

```bash
bash Freebase/run_sliding_multi_gpu.sh run prepare
bash Freebase/run_sliding_multi_gpu.sh run all
```

先跑两个真实窗口确认服务器环境与显存配置：

```bash
bash Freebase/run_sliding_multi_gpu.sh run all --periods center_1900,center_1901
# 确认后使用同一配置跑全部；已完成的两个窗口会跳过兼容阶段
bash Freebase/run_sliding_multi_gpu.sh run all
```

默认 RGCN 最多 50 epochs，patience=6；GraphMask 每层 3 epochs，验证集允许相对 macro-F1 差异不超过 0.05。沿用第二版的其他参数与 GraphMask checkpoint 选择规则。具体参数以配置 JSON 为准。

如果某张卡显存不足，可在首次运行前设置较小 batch，例如：

```bash
export FREEBASE_GROUP_TRAIN_BATCH_SIZE=256
export FREEBASE_GROUP_GRAPHMASK_BATCH_SIZE=16
bash Freebase/run_sliding_multi_gpu.sh run all
```

训练后改变 batch、fanout、epochs 等参数会改变阶段合同，需要使用新的模型/GraphMask 输出根，不能将不同设置的结果混在同一实验中。

## 分配与恢复

主进程只初始化一次共享 v2 基础图与来源记录，随后启动每卡一个 worker。默认按预计节点数从大到小排列窗口，所有 worker 共用任务队列；空闲的卡领取下一个窗口，不预先固定哪张卡负责哪些年份。节点数只是计算量估计，实际分配随运行速度变化。`--schedule chronological` 可改为按中心年份排队。

每个 worker 在当前卡上串行执行一个窗口的：

```text
滑窗构图 → 关系分组 → RGCN → 校验本窗口 RGCN 权重
         → GraphMask 训练/验证 → test 报告 → 领取下一个窗口
```

GraphMask 明确读取同一窗口、同一关系表示、同一 seed 的 `best_model.pt`。RGCN 失败时不会继续运行该任务的 GraphMask，其他窗口继续执行。

重跑相同 `run all` 命令会复用配置和来源一致的已完成阶段，只重试未完成/失败阶段。切换物理 GPU 数量不会改变逻辑设备合同。worker 意外退出时，正在执行的窗口会被记录为 unfinished，本次运行返回非零；它不会在本次运行内自动重复领取，重跑即可恢复。Ctrl-C/SIGTERM 会停止 worker 及其训练子进程，保留恢复记录。运行锁防止多个调度器同时写相同的输出根；请勿同时从其他入口构建或修改共享基础图。

主进程在所有 worker 结束后统一生成汇总，避免多个 worker 同时覆盖文件。仅重新汇总不需要 GPU 池：

```bash
bash Freebase/run_sliding_multi_gpu.sh run summarize
```

单独执行 `run collapse`、`run train`、`run graphmask` 也支持多卡队列，但必须已经完成前置阶段。优先使用 `run all` 处理依赖和恢复。

## 输入与输出

生成 v2 基础图需要以下三份数据，以及包内的关系规则：

```text
external_data/freebase/descriptive_v2_local/05_final/nodes.csv
external_data/freebase/descriptive_v2_local/05_final/main_relation_facts.csv
docs/freebase_bhht_semantic_review_2026-10-08/profession_l1_semantic_crosswalk.tsv
```

服务器如果仍使用 `external_data/freebase/processed/05_final/` 的完整原始导出，适配器会在本地下载布局缺失时自动使用它；也可 `export FREEBASE_INPUT_DIR=/实际路径/05_final`。v2 不需要旧版人物职业审核表。旧版基础图不能替代 v2 标签图，脚本会生成独立的 v2 基础图。

新输出根均与旧版及原来 8 个时期实验分开：

| 内容 | 目录 |
| --- | --- |
| 共享 v2 基础图 | `artifacts/freebase_provisional_l1_v2/` |
| 101 个滑窗子图 | `artifacts/freebase_life_windows_1900_2000_pm20_step1_v2/center_YYYY/` |
| 分组后的图 | `artifacts/freebase_grouped_sliding_1900_2000_pm20_step1_v2/center_YYYY/multi_group/` |
| RGCN 权重与指标 | `runs/freebase_grouped_sliding_1900_2000_pm20_step1_v2/center_YYYY/multi_group/seed_42/` |
| GraphMask 权重与报告 | `runs_graphmask/freebase_grouped_sliding_1900_2000_pm20_step1_v2/center_YYYY/multi_group/seed_42/` |
| 每卡日志与队列状态 | GraphMask 根目录下 `multi_gpu_runs/<运行时间_PID>/` |
| 总体指标与分组指标 | GraphMask 根目录下 `matrix_summary.tsv`、`relation_group_summary.tsv` |

每次运行记录 `dispatch_plan.json`（GPU、队列顺序、节点数估计）和 `dispatch_status.json`（完成、未完成、worker 错误）。另有各阶段 `.log`、`pipeline_*.json`、`pipeline_failures.json` 便于定位失败。

全部实验完成后，供分析的最小包为 GraphMask 根目录的两张汇总表及 JSON、`pipeline_failures.json`，加上 101 个窗口的 RGCN `metrics.json`、GraphMask `validation.json`、`manifest.json` 与 `test_report/`。不必先下载大型图或模型权重。

如需改变中心年份或半窗，可用 `python Freebase/configure_sliding_windows.py --start-year 1900 --end-year 2000 --half-window 20 --step 1` 生成另一套配置，再通过 `--config` 指定。生成器拒绝覆盖内容不同的同名配置。

## 本地验证（2026-10-09）

本地已生成默认 101 个实际子图，共约 2.2 GB。逐个核对了生命区间成员、原始节点顺序、诱导边及重编号后的完整边张量、关系 ID、互斥的数据划分和调度节点数估计；全部保留 5 个职业类别。窗口节点数范围为 24,291–72,541，最大窗口为 `center_1962`。构图目录中有 `window_summary.tsv` 和 `window_validation.json`。

相关单元/回归测试共 65 项通过。另以两个 CPU worker 对 30 人的小型测试数据运行完整 RGCN→GraphMask→报告并验证重跑复用；该 smoke test 使用简化的训练参数，仅验证运行链路。真实数据的 101 个 RGCN/GraphMask 尚未训练，CUDA 环境与显存适配由服务器上的首次两个窗口运行验证。
