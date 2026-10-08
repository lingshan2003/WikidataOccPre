# DBpedia 年度滑窗实验

默认配置暂定中心年 **1900–2000**，共 101 个窗口，每年移动一次。中心年 y 使用闭区间 `[y−20, y+20]`；例如 1900 → 1880–1920、1901 → 1881–1921。边界跨度为 40 年，包含 41 个整数年份。配置可在正式训练前生成其他中心年范围。

## 当前先试跑四个年份

先跑 **1900、1901、1950、2000**，包含相邻窗口、中期较大窗口及近期较稀疏窗口。使用 `config/dbpedia_grouped_sliding_pilot_4years_v1.json`，训练参数、日期规则、7组定义和输出根目录与正式年度矩阵一致；仅将任务限定到四窗。

在已配置 Python 环境的一个 tmux 会话中：

```bash
# 查看固定四窗的计划。
bash DBpedia/run_sliding_pilot.sh plan --gpus 4,5

# 构图 → 关系重编码 → 独立规则审计，通过后才训练四窗 RGCN/GraphMask。
bash DBpedia/run_sliding_pilot.sh run --gpus 4,5
```

也支持指定数量：`CUDA_VISIBLE_DEVICES=4,5 bash DBpedia/run_sliding_pilot.sh run --num-gpus 2`。只构图和审计、暂不训练时，将 `run` 换成 `prepare`；单独重查已生成的图用 `bash DBpedia/run_sliding_pilot.sh audit`。入口固定为四窗 multi_group，拒绝扩大范围的选项，并关闭继承的 `include_full`。

`prepare` / `run` 自动调用 `DBpedia/audit_sliding_window_artifacts.py`：直接从原始全图节点/边表重算窗口归属和诱导边，与保存的图进行逐项比较；同时检查本地索引、反向关系、分组去重、标签过滤、seed固定的类内70/10/20划分和验证/测试职业特征掩码。审计失败就停止，不进入训练。报告在 GraphMask 根目录的 `pilot_audit/{graph_audit.json,window_audit.tsv,checks.tsv,group_counts.tsv}`。

本地四窗全部通过，每窗14项共56项检查；双worker的真实全流程CPU冒烟也4/4完成，但使用小模型和单epoch，只用于流程验证。持久检查报告在 `artifacts/dbpedia_sliding_pilot_validation_2026_10_08/README.md`。2000窗按规则保留7,402个孤立节点（约45.1%），服务器试跑应检查近期模型效果和GraphMask稀疏性。

四窗试跑结束后入口直接退出，不自动启动101窗。先审阅正式试跑的准确率、Macro-F1、GraphMask保真度、门控启用状态与留存占比，再决定是否扩展；之后使用完整年度入口时，参数兼容的已完成阶段可以复用。

## 构图与训练

沿用原实验的日期规则：完整且有效的生卒区间只要与窗口相交即入图；仅知道出生年或死亡年时，该已知端点在窗口内才入图；两端均缺失或死亡早于出生的节点排除。不会把缺失死亡年自动补成 2022。保留两端节点均入窗的关系，包含原有反向消息；窗口中的孤立节点也保留。

这是人物生命区间相交的关系子图。数据没有关系发生时间，因此曲线横轴应写 **Window center year**，不能解释为“该年发生的关系”。相邻窗口会反复包含相同人物和边。

每窗单独重建特征编码、划分 70/10/20 训练/验证/测试节点、过滤局部少于 3 个监督样本的职业类别，然后沿用原 RGCN 超参数训练。类别总词表、7 个关系组、反向关系定义保持一致。默认只跑 multi_group，共 **101 个 RGCN + 101 个 GraphMask**，不是每个小组分别训练模型。可显式加 `--representations binary,multi_group`，任务数将翻倍。

GraphMask 使用原实验的 `checkpoint_selection=any-stage`：从所有训练阶段中，选择满足验证保真度要求且整体硬留存率最低的 checkpoint，不要求选择时 Layer 0 门控已经启用。Layer 1、Layer 0 仍按原顺序训练，每层 3 epoch，其他参数沿用原实验。若选中的 checkpoint 尚未启用 Layer 0，其 Layer 0 消息全部保留，组占比等于测试采样消息的组构成；报告保留门控启用状态以说明这一情况。没有满足验证保真度的候选时任务失败并保留历史。

模型、时期图、关系重编码图、GraphMask 都写入带 `sliding_1900_2000_pm20_step1_v1` / `life_windows_1900_2000_pm20_step1_v1` 的独立目录。参数兼容且完成的阶段可复用，失败阶段重试。已有八时期实验结果不用于替代任何滑窗模型。

## 本地规模预览（原始图统计，无模型训练）

运行：

```bash
python DBpedia/preview_sliding_windows.py
```

输出 `artifacts/dbpedia_sliding_window_preview_2026_10_07/`：

- `window_sizes.tsv`：101 个窗口的节点、边、孤立节点、日期完整程度、相邻窗口交并比及进出节点数。
- `window_relation_groups.tsv`：7 组的原始及重编码去重后图边数量和构成比例。这些是构图统计，不是 GraphMask 留存结果。
- `preview_metadata.json`：日期规则、输入表路径、完整窗口配置。

| 中心年 | 窗口 | 节点数 | 原始关系数 | multi_group 去重后关系数 |
| ---: | --- | ---: | ---: | ---: |
| 1900 | 1880–1920 | 18,506 | 17,309 | 17,191 |
| 1901 | 1881–1921 | 18,804 | 17,647 | 17,527 |
| 1902 | 1882–1922 | 19,116 | 18,003 | 17,881 |
| 1920 | 1900–1940 | 25,665 | 24,575 | 24,434 |
| 1950 | 1930–1970 | 35,797 | 35,674 | 35,542 |
| 1980 | 1960–2000 | 26,410 | 19,604 | 19,538 |
| 2000 | 1980–2020 | 16,411 | 8,443 | 8,417 |

表中关系数只计算原方向，不含生成的 `__rev`；送入 RGCN 的有向消息数为其两倍。1900/1901 的节点交并比为 97.55%，原始关系交并比为 96.81%。近期窗口人数下降，受源数据覆盖和上述缺失日期规则影响，不能视为真实社会网络人数下降。

## 服务器运行（tmux）

同步整个仓库代码后，在仓库根目录并使用已经装好 torch/PyG 的环境：

```bash
tmux new -s dbpedia-sliding
export DBPEDIA_PYTHON_BIN=/home/dingqiyang/anaconda3/envs/wywikidata/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export DBPEDIA_GROUP_DEVICE=cuda:0

# 只打印计划，不启动训练。
bash DBpedia/run_sliding_rgcn_graphmask.sh plan all

# 建议先检查早期、中期、近期三个窗口的模型效果和 Layer 0 删边表现。
bash DBpedia/run_sliding_rgcn_graphmask.sh run all --periods center_1900,center_1950,center_2000

# 同参数继续全部年度窗口；已完成的三个窗口会跳过。
bash DBpedia/run_sliding_rgcn_graphmask.sh run all
```

请按运行当时的空闲情况选择物理卡号。这里只运行一个任务，卡映射后进程内设备是 `cuda:0`。`Ctrl-b d` 离开 tmux；`tmux attach -t dbpedia-sliding` 回到会话。同一 model root 有进程锁，不能同时在两个 tmux 会话执行这份完整矩阵。

本入口支持旧入口的所有阶段及 `--periods`、`--representations`、`--config`、`--device`；默认使用新滑窗配置。会继承 `DBPEDIA_GROUP_*` 覆盖变量：如此前为其他实验设置过输出根目录、时期选择或超参数，应先用 `env | sort` 查看并清理这些覆盖，避免改写滑窗实验设定。

## 多 GPU 自动调度（推荐用于年度矩阵）

入口为 `DBpedia/run_sliding_multi_gpu.sh`，使用同一份滑窗配置和同一套结果目录，GraphMask 继续使用 `any-stage`。只需在一个 tmux 会话启动一次：

```bash
export DBPEDIA_PYTHON_BIN=/home/dingqiyang/anaconda3/envs/wywikidata/bin/python

# 先看计划：允许使用物理 GPU 4、5，选择其中两张。
CUDA_VISIBLE_DEVICES=4,5 bash DBpedia/run_sliding_multi_gpu.sh plan all --num-gpus 2

# 正式运行全部年度窗口。
CUDA_VISIBLE_DEVICES=4,5 bash DBpedia/run_sliding_multi_gpu.sh run all --num-gpus 2
```

也可直接列卡号，数量由列表自动确定：

```bash
bash DBpedia/run_sliding_multi_gpu.sh run all --gpus 4,5

# 例如之后可用三张卡：2、4、5。
bash DBpedia/run_sliding_multi_gpu.sh run all --gpus 2,4,5
```

两条 `run` 命令是不同可用卡情形下的选择，不应同时启动。`--gpus` 明确指定宿主卡号并覆盖继承的 `CUDA_VISIBLE_DEVICES`；使用系统分配的可见卡列表时应采用 `--num-gpus`。如果只传 `--num-gpus N` 而没有设置可见卡列表，会通过 `nvidia-smi` 列出本机卡号并选择前 N 张；不会判断哪些卡空闲。数量超过选定列表时直接报错，不擅自使用列表外的卡。

调度规则：

- 每张卡一个独立 worker，一次负责一个窗口；按该窗配置依次构图、重编码、RGCN 训练、GraphMask 训练与测试报告。若选了 binary 和 multi_group，该窗口两种表示由同一 worker 依次完成。
- 使用动态任务队列：worker 完成一个窗口便领取下一个，速度较快的卡可多做几个窗口。101 窗用两张卡时通常约各 50 个，实际分配随每窗运行时间变化。任务少于卡数时只启动需要的 worker。
- 每个 worker 都只看见自己的那张卡，进程内设备固定 `cuda:0`；外部 `DBPEDIA_GROUP_DEVICE` 不会把 worker 指到另一张逻辑卡。
- worker 只写自己窗口的产物；父进程在全部 worker 结束后统一写 `matrix_summary`、`relation_group_summary`、`pipeline_failures.json`。各 worker 的失败记录隔离，单个窗口失败后继续其他窗口，最终退出码非零。
- 父进程持有与单卡入口相同的 model-root 锁，防止同时启动两个调度器或同目录单卡任务。保持一套配置的四个输出根目录配套使用，不要让另一个 model root 指向这套共用产物。

可先用两个窗口试跑：

```bash
bash DBpedia/run_sliding_multi_gpu.sh run all --gpus 4,5 \
  --periods center_1900,center_1950
```

运行时终端显示每窗开始、完成、失败和累计进度。详细日志和分配记录写到 GraphMask 根目录的 `multi_gpu_runs/<运行ID>/`：

```text
dispatch_plan.json     本次配置、选定 GPU、时期列表
gpu_4.log              GPU 4 worker 的完整日志
gpu_5.log              GPU 5 worker 的完整日志
dispatch_status.json   每窗实际分配到哪张卡、成功/失败、未完成窗口与 worker 退出码
```

每个阶段自己的 `train.log`、`probe.log` 等仍保存在原窗口目录。离开 tmux 用 `Ctrl-b d`，训练继续；按 `Ctrl-C` 停止时调度器会清理 worker 及其训练子进程，并保留本次中断状态。之后可用不同数量或卡号重新执行：已完成且参数兼容的阶段自动跳过，失败/未完成阶段重新跑。GPU 数量和物理卡号不加入训练阶段合同，因此换卡不会要求重训已有完成窗口；其他训练参数与 Python 环境应保持一致。

只重新汇总报告不需要指定 GPU：

```bash
bash DBpedia/run_sliding_multi_gpu.sh run summarize
```

多卡验证包括真实 spawn 进程的任务分配、单窗失败隔离、worker 启动失败/异常退出不挂起、进程组清理及共享锁。另已使用真实 DBpedia 数据在临时目录通过两个 worker 并行准备并重编码 1900/1901 两窗，最终汇总为两行且计数与单卡一致。当前本地验证使用 CPU，CUDA 训练速度和显存需在服务器上确认。

## 更换中心年范围

例如覆盖到 2022：

```bash
python DBpedia/configure_sliding_windows.py --start-year 1900 --end-year 2022
bash DBpedia/run_sliding_rgcn_graphmask.sh plan all \
  --config config/dbpedia_grouped_sliding_1900_2022_pm20_step1_v1.json
```

生成器只写配置，不构图、不训练；已有同名且内容不同的配置会拒绝覆盖。中心年 2022 对应 2002–2042，后半段超出 DBpedia 2022 数据覆盖，不能与完整历史窗口直接比较。默认中心年 2000 的窗口结束于 2020，避开这个右端问题。改变范围会使用新的输出根目录；正式开跑前确定范围。

## 后续曲线和需要下载的结果

汇总目录为 `runs_graphmask/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1/`，沿用 `matrix_summary.tsv/json`、`relation_group_summary.tsv/json`、`pipeline_failures.json`。每个 `center_YYYY/multi_group/seed_42/` 下保留 `manifest.json`、`training_history.json`、`validation.json` 及 `test_report/`。

Layer 0 应区分两条统计口径：

1. **Retained message share**：某组硬保留消息 / 本层所有硬保留消息，七组加总约 100%，对应此前的热图口径。
2. **Within-group retention rate**：某组硬保留消息 / 该组全部消息，反映这个组删掉多少。

两者都来自各窗 test roots 的邻居采样报告，不是全图唯一边统计。无图边和采样未观测到的组保留为空，不能填 0。既有八时期绘图脚本固定旧时期，暂不能直接绘制年度结果；收到新报告后应使用中心年份作为横轴绘制七组曲线，并附 Layer 0 总留存率及门控启用状态。

独立训练会引入模型、测试节点及采样变化；固定 seed 并不意味着每窗测试的是同一批人。相邻窗口的细小波动应先当描述性结果，不能直接解释为真实年度突变。可先检查三个试跑窗口的准确率、Macro-F1、保真度和 Layer 0 留存，再决定是否增加多 seed 重复。

下载时至少保留两份汇总、完整配置和各窗上述小报告；图表不需要下载 RGCN 或 probe 的 `.pt` 权重。详细单人边解释另需 `root_top_edges.csv.gz`。运行记录和 `.pt` 权重仍保存在服务器。

## 验证

新增测试覆盖窗口边界、端点日期跨窗归属、非法滑窗配置拒绝、真实 induced-artifact 接口的重索引与复用、101 窗命令计划及 checkpoint 策略。旧 partition 配置继续要求有序、互不重叠且无空隙，不会因新增滑窗模式而放宽。

本次相关测试共 33 项通过（包括旧 DBpedia / Freebase Pipeline 回归）。另用真实 DBpedia 全图在 `/private/tmp` 独立目录跑通中心年 1900 的 prepare → multi_group collapse：18,506 节点、34,618 条原始有向消息，重编码后 34,382 条消息、14 个有向关系类型，配置 manifest 校验通过。未启动滑窗 RGCN 或 GraphMask 训练。
