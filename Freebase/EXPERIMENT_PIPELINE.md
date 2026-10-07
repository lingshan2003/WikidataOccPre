# Freebase：临时单标签、时期图、RGCN 与 GraphMask

2026-10-06。按用户授权，对未取得唯一大类的 19,135 人暂取一个职业，生成独立实验版本。入口为 `Freebase/run_grouped_rgcn_graphmask_20y.sh`，默认配置为 `config/freebase_grouped_rgcn_graphmask_20y_v1.json`。流程依次执行源图适配、时期诱导图、关系组折叠、RGCN、GraphMask、测试集报告与矩阵汇总。

## 直接运行

在项目根目录激活服务器的 `wywikidata` 环境，然后手动指定设备。服务器有 6 张 RTX 4090；按用户 2026-10-07 提供的状态，0、1、2、3 号正在运行其他程序，以下示例选择物理 4 号卡，也可手动改为 5：

```bash
conda activate wywikidata
export CUDA_VISIBLE_DEVICES=4
export FREEBASE_GROUP_DEVICE=cuda:0

# 先打印全部步骤与输出路径；此命令不需要 PyTorch。
bash Freebase/run_grouped_rgcn_graphmask_20y.sh plan

# 完整实验：8 个时期，每个时期一个 RGCN 和一个 GraphMask。
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run all
```

入口直接使用激活环境后的 `python`，无需填写解释器路径。设备配置没有自动默认值；未设置 `FREEBASE_GROUP_DEVICE` 会在准备数据或启动任务前报错，`auto` 也不再接受。`CUDA_VISIBLE_DEVICES=4` 将物理 4 号卡暴露为进程内的 `cuda:0`，因此这两个编号不同是正常的。换用物理 5 号卡只需改为 `export CUDA_VISIBLE_DEVICES=5`，模型设备仍设为 `cuda:0`。每次脚本在一张卡上顺序运行所选时期，不自动调度多卡。

脚本开始训练前实际检查邻居采样、FastRGCN 消息替换、全开 mask 等价性和探针反向传播。训练环境需要仓库依赖，以及可用的 PyG 邻居采样后端（`pyg-lib` 或兼容的 `torch-sparse`）。

服务器已有 `wywikidata` 环境时，可在 tmux 中运行：

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
mkdir -p logs
tmux new -s freebase-20y
# 在 tmux 内激活环境、选择设备；pipefail 保留脚本失败的退出状态。
conda activate wywikidata
export CUDA_VISIBLE_DEVICES=4
export FREEBASE_GROUP_DEVICE=cuda:0
set -o pipefail
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run all \
  2>&1 | tee logs/freebase_20y_v1.log
```

重跑同一命令会复用兼容的完成步骤，中断的步骤会重新执行。输入、参数或完成输出发生变化时拒绝混用，提示使用新输出根目录。源图与时期图的源文件身份也会核验。更换训练参数时，至少另设模型和 GraphMask 根目录；更换标签或时期规则时，应给所有派生结果另开版本。

激活环境并完成上述 `export` 后，阶段、时期与对照可单独选择：

```bash
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run prepare
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run collapse
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run train
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run graphmask
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run summarize

# 只跑一个时期。
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run all --periods 1901_1920

# 可选：增加 inherited/acquired 二类关系对照，以及不切时期的全图对照。
bash Freebase/run_grouped_rgcn_graphmask_20y.sh run all \
  --representations binary,multi_group --include-full
```

`summarize` 要求所选组合报告齐全；部分完成时仍输出汇总并返回非零状态。某一时期训练/解释失败，会记录原因并继续处理其他时期，整次运行最终返回非零状态。

## 标签与原始关系

默认保持四个已有职业大类，增加一个临时 `Other` 类。不会更改原始职业数组或旧审核表。

| 标签处理 | 人数 | 确定规则 |
| --- | ---: | --- |
| 保留唯一大类候选 | 81,621 | 旧审核表与现有映射一致时原样保留 |
| 暂选一个已有映射的职业 | 18,003 | 按原数组顺序取第一个 `proposed` 值，并使用其 L1 |
| 暂时 `Other` | 1,132 | 全部值均无有效 L1 时取第一个原职业，训练标签暂为 `Other` |
| 总计 | 100,756 | 全图均有一个实验标签 |

数组顺序不代表职业的重要性；这些是用户授权的临时实验选择。全部候选映射也仍是初审草案。每个人的原职业、所选职业、原审核状态、选择理由与映射置信度均保存于 `artifacts/freebase_provisional_l1_v1/label_selection_audit.tsv`。`Other` 混合了四类之外和仍待审的职业，不是已经建立的第五个语义职业领域。

全图标签分布：Culture 73,062，Leadership 10,915，Discovery/Science 8,409，Sports/Games 7,238，Other 1,132。

如需把无法映射的原职业逐个作为独立类别，可另建配置，将 `source_prepare.label_policy` 改为 `mapped_first_raw_fallback`，并另设全部输出根目录。此时标签为 `Raw::<职业>`，少于 3 人的类别会退出监督集合，节点仍留在图中。

145,876 条原始关系事实先去除 47 条自环，再规范对称关系与影响谓词别名，得到 99,644 条基础关系边；添加生成反向边后为 199,288 条消息边。原始表保留。亲子方向按 `Children` 的父母→子女原向；`influenced_by` 统一为受影响者→影响者；学术导师、武术师承继续使用 `_raw` 方向，不额外推断师生方向。

## 关系组

参考主数据的 inherited/acquired 结构，覆盖当前 Freebase 的全部 9 种规范基础关系：

| 组 | 当前基础关系 | 上层性质 |
| --- | --- | --- |
| `inherited` | `child`, `sibling` | inherited ties |
| `intimate_partnership` | `partner`, `celebrity_romantic_relationship` | acquired |
| `education_mentorship` | `academic_advisor_raw`, `martial_arts_instructor_raw` | acquired |
| `influence_succession` | `influenced_by` | acquired |
| `other_acquired` | `peer`, `celebrity_friend` | acquired |

伴侣归 acquired；当前 Freebase 没有足够谓词建立独立的职业合作或宗教授任组。多组图为上述 5 组加各自反向，共 10 种 RGCN 消息类型。同一组内相同主客体的重复边折叠；每个方向保留独立关系类型。二类关系对照为 inherited/acquired 加各自反向，共 4 种类型。

## 时期与划分

窗口宽度 20 年、步长 20 年；这版默认窗口之间不重叠。按用户的整数年份边界，1500–1900 包含 1900，后续从 1901 开始。有效观测最晚为 2015 年，因此默认最后窗口为 2001–2020，不生成空的 2021 年以后图。

| 时期 | 节点数 | 分组前消息边 | 分组后消息边 | 有监督类别数 |
| --- | ---: | ---: | ---: | ---: |
| <1500 | 502 | 1,342 | 1,342 | 4 |
| 1500–1900 | 15,879 | 28,674 | 28,664 | 5 |
| 1901–1920 | 23,079 | 33,110 | 33,104 | 5 |
| 1921–1940 | 37,871 | 54,832 | 54,824 | 5 |
| 1941–1960 | 48,876 | 65,648 | 65,628 | 5 |
| 1961–1980 | 48,285 | 52,136 | 51,970 | 5 |
| 1981–2000 | 30,409 | 22,394 | 22,330 | 5 |
| 2001–2020 | 10,582 | 3,442 | 3,442 | 5 |

时期按生卒区间与窗口的交集确定。完整区间跨窗口的人可出现于多个独立实验；只有一个已知日期时只进入该日期窗口。两日期皆缺失或死亡早于出生不进入时期图；原始冲突/未来日期另外保留审计。各图只保留两端均在图中的关系，保留孤立节点；时期成员数不能直接相加当作总人数。关系缺少发生时间，因此这是生存时期诱导图，不是精确的关系事件动态图。

每个时期独立按类别重划 70%/10%/20% 的 train/val/test，seed=42；小于 3 人的时期局部类别不参与监督。`<1500` 有 1 人因此退出监督，其节点保留，监督划分为 352/49/100。比较各时期时应同时看图规模、活跃类别和关系支持量。

## 模型与解释口径

默认 RGCN：2 层 FastRGCN、hidden=128、branch=64、邻居采样 15/10、batch=512、最多 50 轮，按验证 Macro-F1 早停。特征使用职业 L1 与已知生卒年；L2/L3、国家字段缺失。只允许训练标签作为邻居职业输入；验证/测试职业置 unknown，每次前向也遮住当前被预测根节点的职业，避免直接读取目标标签。

默认 GraphMask：冻结 RGCN，每层训练 3 轮，beta=0.03；使用训练根节点训练探针，验证根节点选择探针，测试根节点出报告。验证集原模型与 masked 模型的相对 Macro-F1 差须不超过 5%，否则该组合记为失败。测试指标与训练时 sampled 评估的采样上下文可能不同，测试解释报告内的 original/masked 指标应成对比较。

关系组报告记录每层消息观测数、硬保留率、保留消息份额和平均保留概率。采样消息会重复出现，因此它们不是全图唯一边的删减率，也不是因果效应。未观测组显式标记 `no_sampled_messages` 或 `no_graph_edges`。

## 输出与服务器所需文件

| 输出根目录 | 内容 |
| --- | --- |
| `artifacts/freebase_provisional_l1_v1/` | 全图、选择/日期审计、规范基础边 |
| `artifacts/freebase_life_periods_20y_v1/<时期>/` | 时期源图、局部划分与成员审计 |
| `artifacts/freebase_grouped_20y_v1/<时期>/multi_group/` | 合并关系后的训练图与映射清单 |
| `runs/freebase_grouped_20y_v1/<时期>/multi_group/seed_42/` | RGCN 最优模型、指标、测试预测 |
| `runs_graphmask/freebase_grouped_20y_v1/<时期>/multi_group/seed_42/` | 探针、验证报告、训练历史、测试解释 |
| `runs_graphmask/freebase_grouped_20y_v1/matrix_summary.tsv` | 每个实验的完成状态、模型指标和解释保真度 |
| `runs_graphmask/freebase_grouped_20y_v1/relation_group_summary.tsv` | 时期 × 层 × 关系组的解释统计 |

服务器需要同步本次代码与配置，包括共享的 `DBpedia/grouped_pipeline.py`、`run.py`、`data/`、`models/`、`training/` 和时期准备脚本。只复制 shell 入口不够。最小输入按默认路径为：

```text
external_data/freebase/descriptive_v2_local/05_final/nodes.csv
external_data/freebase/descriptive_v2_local/05_final/main_relation_facts.csv
external_data/freebase/profession_l1_review_v1/person_l1_audit.tsv
docs/freebase_profession_review_2026-10-03/profession_l1_crosswalk_draft.tsv
Freebase/relation_rules.json
```

本地已具备这些输入。服务器若只有 `processed/05_final`，可将配置的 `source_prepare.input_dir` 改到该路径，其余输入也按实际路径修改。无需下载/重扫全量 `facts.txt`。可上传原始小表后在服务器重新准备；已生成的 `.pt` 及恢复记录绑定来源路径和文件时间戳，直接跨机器复制派生目录时不保证能复用恢复记录。

环境覆盖参数包括 `FREEBASE_GROUP_PERIODS`、`FREEBASE_GROUP_REPRESENTATIONS`、`FREEBASE_GROUP_DEVICE`、`FREEBASE_GROUP_TRAIN_BATCH_SIZE`、`FREEBASE_GROUP_TRAIN_WORKERS`、`FREEBASE_GROUP_TRAIN_EPOCHS`、`FREEBASE_GROUP_TRAIN_FANOUTS`、`FREEBASE_GROUP_GRAPHMASK_BATCH_SIZE`、`FREEBASE_GROUP_GRAPHMASK_EPOCHS_PER_LAYER`、`FREEBASE_GROUP_GRAPHMASK_BETA`。各根目录可用 `FREEBASE_GROUP_SOURCE_DATA`、`FREEBASE_GROUP_PERIOD_ROOT`、`FREEBASE_GROUP_RELATION_ROOT`、`FREEBASE_GROUP_MODEL_ROOT`、`FREEBASE_GROUP_GRAPHMASK_ROOT` 覆盖。`--config` 或 `FREEBASE_GROUP_CONFIG` 可指定新实验配置。

## 本机验证

已从真实输入生成完整 100,756 节点图、全部 8 个时期图及其多关系组版本。实际 CPU 贯通测试使用 `<1500` 的真实图，RGCN 2 轮、hidden=16、branch=8、采样 5/3，GraphMask 每层 1 轮；完成探针训练、100 个测试根节点的报告与矩阵汇总。验证相对 Macro-F1 差为 0，测试预测一致率为 1；短训练的硬消息保留率仍为 1，故这只是贯通验证，尚不支持关系重要性结论。正式配置的全部训练尚未在本机执行。

贯通测试配置与结果位于 `artifacts/freebase_pipeline_smoke_20y_v1/`，与正式训练输出分离。复跑：

```bash
# 本地短训练测试使用本地虚拟环境；服务器仍使用 wywikidata。
source .venv/bin/activate
export FREEBASE_GROUP_DEVICE=cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  bash Freebase/run_grouped_rgcn_graphmask_20y.sh run all \
  --config artifacts/freebase_pipeline_smoke_20y_v1/config.json

python -m unittest discover -s tests -p 'test_freebase*.py' -v
python -m unittest discover -s tests -p test_life_periods.py -v
```

Freebase 专用 20 项测试、时期边界 2 项测试与既有时期诱导图 2 项回归测试均通过。

2026-10-07 更新：设备必须由环境变量或显式 `--device` 指定，配置文件不再提供默认设备；服务器示例统一使用激活后的 `wywikidata` 与 `export`，不填写 Python 路径。更新后的 23 项 Freebase 测试通过，包含缺少设备时报错、禁止 auto、CUDA 可见编号传递及实际 CPU 预检；实际 shell 入口的计划模式与缺少设备的退出状态也已核验。
