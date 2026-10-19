# 统一 L1 输入预测 L1 的三库滑窗实验

状态：脚本与配置准备完成；尚未在服务器试跑或启动正式训练。

## 研究问题与对照

移除 Wikidata 的 L2/L3 输入、Freebase 的时间辅助特征后，1900–2000 的关系组占比与组内保留率趋势是否保持？DBpedia 在同一 BHHT 五类 L1 数据上恢复旧七组词表，将 Influence 与 Succession 合并训练，与现有八组结果比较。

三套均 **input L1 → predict L1**，`auxiliary_features=none`。源文件中的 L2/L3、日期和国家列可留在数据中，但不进入模型特征；日期仅供构图。预测目标根节点的职业屏蔽继续由现有训练代码执行，不修改其规则。

三套职业人群、原始关系覆盖和标签生成方式仍有差异；这次统一的是输入层级、辅助特征、日历、划分参数与训练预算，并非完全相同的抽样总体。

| 数据集 | 入口 | 关系组 | 配置标识 |
|---|---|---:|---|
| Wikidata | `Wikidata/run_sliding_l1_only.sh` | 7 | `wikidata_grouped_sliding_1900_2000_pm20_step1_l1_only_alive2026_v2` |
| Freebase | `Freebase/run_sliding_l1_only.sh` | 5 | `freebase_grouped_sliding_1900_2000_pm20_step1_l1_only_alive2026_v4` |
| DBpedia | `DBpedia/run_sliding_bhht_l1_merged7.sh` | 7 | `dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_merged7_alive2026_v2` |

配置位于 `config/<配置标识>.json`。DBpedia 使用已审核的 BHHT L1 数据，不切回旧细职业标签；复用的只是旧 `config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json` 关系分组。合并组包含 influenced、influencedBy、successor、predecessor；其余六组不变。basis 数按有向关系数分别为14/10/14。

## 固定协议

- 101个中心年：1900–2000，每年一步，窗口 `[year-20, year+20]`。
- **birth >1920** 且缺失死亡日期时，构图生命区间延伸到2026；1920年出生者不触发。仅影响入图，原始死亡日期不改写。
- 节点生命区间与窗口重叠才入图，边的两端均在窗口才保留；无双日期、单日期及无效生命区间规则沿用现有构图代码。
- 训练 seed=42；每窗重新分层划分70/10/20，split seed=20260814，局部最小类别数3。
- 2层 RGCN，最多50轮，val Macro-F1早停（patience6，min_delta0.001）；hidden128、branch64、dropout0.2、lr0.001、weight_decay0.0001；sampled fanouts15,10，batch512。
- GraphMask 每层3轮，batch32，beta0.03，lr0.0001，multiplier lr0.01；验证相对Macro-F1下降阈值0.05。
- **any-stage checkpoint**：在满足验证约束的候选中选择整体保留率最低者；不强制启用Layer0后才能选。
- 测试集评价 accuracy、Macro-F1、预测一致性、整体/Layer0保留率、各组自身保留率及保留消息份额；原图份额与采样消息份额用于解释趋势。

Freebase相对旧v3还统一了两项：birth≥1920改为birth>1920、split seed42改为20260814。这些变化会记录在配置的protocol_changes中，不能把结果差异全部归于删除时间特征。

## 服务器用法（连接后先核对 GPU、磁盘与环境）

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
git pull --ff-only
source /home/dingqiyang/anaconda3/etc/profile.d/conda.sh
conda activate wywikidata
hostname
pwd
command -v python
```

解释器应继续使用原环境 `/home/dingqiyang/anaconda3/envs/wywikidata/bin/python`。入口也接受 `L1_ONLY_PYTHON_BIN` 或各库的 `*_PYTHON_BIN`，但不会自动切换到其他解释器。源缓存的已有兼容性校验会检查实际命令及文件记录，不因本次任务而跳过。

只看计划，不训练、不创建结果目录：

```bash
bash Wikidata/run_sliding_l1_only.sh plan all --gpus 1,2 --periods center_1900,center_1950,center_2000
bash Freebase/run_sliding_l1_only.sh plan all --gpus 3,4 --periods center_1900,center_1950,center_2000
bash DBpedia/run_sliding_bhht_l1_merged7.sh plan all --gpus 5 --periods center_1900,center_1950,center_2000
```

GPU编号仅为语法例子；实际使用前读取nvidia-smi，按获准的卡分配，不自动挑空闲卡。`--num-gpus N` 只取显式池中的前N张，例如 `--gpus 2,3,4,5 --num-gpus 2` 使用2、3；也可先设置 `CUDA_VISIBLE_DEVICES=2,3,4,5` 再传 `--num-gpus 2`。不允许仅给数量就枚举整台共享服务器。

确认资源后，在已有 `wywikidata:codex-control` 中运行对应命令。训练例子（不是自动启动授权）：

```bash
bash Wikidata/run_sliding_l1_only.sh run all --gpus 1,2 --periods center_1900,center_1950,center_2000
```

试跑完成并核对配置、F1、fidelity与报告后，去掉 `--periods` 即为全部101窗。Freebase、DBpedia换成对应入口即可。每张指定卡一个worker，多个窗口并行调度；每个worker在一窗内依次完成RGCN→GraphMask→报告。

## 日志、版本与恢复

公共入口在 `artifacts/l1_only_sliding_launches/<dataset>/<run_id>/` 保存：

- `full.log`：调度输出；每卡的详细训练日志由原runner保存在该配置 `graphmask_root/multi_gpu_runs/<run_id>/` 下。
- `exit_code`、`experiment_manifest.json`：实际命令、主机、解释器、Git版本与工作区状态、有效配置、窗口、GPU池、时间和退出状态。
- `code_snapshot/`：本次执行的入口、配置、日历、分组和关键训练/构图代码快照。

源缓存沿用已有路径和标签，不重写旧实验；每套新的窗口、关系图、RGCN和GraphMask目录均使用新配置标识，互不覆盖。已有输出锁及兼容性检查保留。旧shell的实验路径/seed/预算覆盖若改变固定协议，会在启动前报错，应解除相应 `*_GROUP_*` 变量后重试。

在tmux运行可在SSH断开后继续；只保留 `log` 与 `codex-control` 两窗，不为每个GPU新增tmux窗口。将log窗切到本次full.log，必要时读取对应每卡worker日志。

重复相同命令会复用有效、兼容的已完成阶段；这是阶段级续跑，不是未完成epoch的optimizer状态恢复。窗口因OOM失败时保留已完成阶段，先查看GPU占用再按同参数补跑缺失窗口；不自动降低batch或改seed。
