# 主 Wikidata：20世纪逐年滑窗 RGCN → GraphMask

用户确认的研究范围是1900–2000共101个中心年、每年移动一次、闭区间
`[center-20, center+20]`。先在主数据上复现逐年关系组消息占比分析，不扩展
全部历史年份、不新增关系消融对照、不把这些高度重叠窗口视为独立重复。

## 协议与对照

- 使用主原始输入 `Q_R_Q_extended.txt`，目标为职业 L1；全图目标类别至少20人，
  窗口局部不足3人的类别保持节点但不参加监督。
- 保留主实验七组：inherited、intimate_partnership、education_mentorship、
  professional_collaboration、influence_succession、religious_ordination、other_acquired。
  新的显式 taxonomy 与原 `all_remaining` 定义等价，覆盖当前40种基础关系，
  正反向共14组关系类型；组级同端点重复消息边去重。没有拆分影响与继任。
- 完整生卒区间按相交纳入；出生 **>1920** 且缺失死亡日期者，仅为窗口归属
  假定存活至2026。1920不包含，原 `death_year` 和模型死亡/年龄缺失标志不改写。
  其他仅知一个日期的记录按端点规则，两个日期都未知/死亡早于出生者排除。
- 节点入选后保留两个端点都在窗中的边，每窗重编号，按 split seed=20260814
  独立分层70/10/20划分。原关系没有事件日期，不做关系发生年代的推断。
- 邻居职业特征为 L1/L2/L3，辅助特征 `none`，延续主实验的职业遮蔽：val/test
  三层职业隐藏，当前训练 seed 的三层职业每次 forward 再隐藏。
- 模型 seed=42，两层 RGCN，sampled train/eval，fanout=15,10，batch=512，
  hidden=128、branch=64、14 bases；最多50 epochs，validation macro-F1
  选 checkpoint，patience=6、min_delta=.001。每窗只训练一个 multi_group 模型。
- GraphMask 使用冻结的同窗 RGCN 最佳权重；fanout `auto` 继承15,10，batch=32，
  workers=0，每层3 epochs，beta=.03，验证集相对 macro-F1 差异阈值=.05。
  `any-stage` checkpoint 选择，按满足保真的最小整体验证保留率选择；不要求
  Layer0门控已启用。此采样 RGCN 实验与旧四时期 R-GAT/full-neighborhood 分析
  协议有区别，不将其结果直接混为同一个模型对照。
- 评价同一 test roots 上的原 RGCN 与删边后预测：Accuracy、macro-F1、预测一致性、
  整体/各层保留率、各组支持消息数。主图纵轴为各组硬保留的 Layer0采样消息
  除以全部 Layer0硬保留采样消息，正反向合并，合计100%，不是组内保留率。
  不 bootstrap，不平滑；Layer0未启用年份标为空心点，无采样支持留缺口。

## 启动条件

2026-10-13已完成代码及本地小型CPU验证，用户随后明确授权GPU 0、1、3：
先试跑1900/1950/2000，检查通过后扩展101窗。实验实际状态以服务器
`orchestration_runs/.../status.json` 为准，不能把授权或脚本准备视为已训练。
启动前重新检查GPU、磁盘、依赖、原始输入、既有进程/锁及结果。

在服务器自身工作区内执行，不修改系统、其他用户文件或共享conda环境：

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
conda activate wywikidata
git pull --ff-only
# 2,3,4仅是参数示例，实际卡由当时GPU状态和用户确认决定
bash Wikidata/run_sliding_multi_gpu.sh plan all --gpus 2,3,4
```

授权后的独立训练tmux窗口可一次执行自动试跑、验证和扩展入口：

```bash
python Wikidata/run_pilot_then_full.py --gpus 0,1,3
```

它记录每阶段退出码、日志及5秒一次的GPU显存/利用率。三个代表窗必须全部成功，
且通过来源/阶段合同、日期补全、split、采样、checkpoint选择、有限指标、
验证集F1保真及Layer0计数/分母检查才扩展。试跑采样中任一卡剩余显存低于1GiB，
或结束时显存不足，就停止扩展并保留现场。Layer0未启用本身不是失败。
两阶段在同一tmux任务中顺序执行；SSH断开仍运行；兼容试跑结果会被完整矩阵复用。

也可以手动分阶段。三个代表窗口测显存和耗时：

```bash
bash Wikidata/run_sliding_multi_gpu.sh run all --gpus 2,3,4 \
  --periods center_1900,center_1950,center_2000
```

小规模检查通过后，在相同协议和已授权GPU范围内扩展到101窗：

```bash
bash Wikidata/run_sliding_multi_gpu.sh run all --gpus 2,3,4
```

每张指定卡一个worker，共用队列，每次完成一个窗口后领取下一窗，工作进程内
`cuda:0` 对应该卡。指定GPU不会扫描并占用其他卡。CPU构图阶段不需要GPU池，
可单独用 `run prepare`；`plan` 不构图、不训练、不写结果。

原始主数据已有但默认基础artifact缺失时，入口只准备一次独立的
`artifacts/wikidata_level1_sliding_source_v1/`。输入大小/mtime、命令、准备参数和
产物记录必须兼容才复用；拒绝覆盖无来源记录的已有图。
设备配置支持显式CLI或WIKIDATA环境变量；本地CPU验证使用
`--device cpu --cpu-workers 2`，不需要GPU编号。DBPEDIA环境变量不会重定向主实验。

基础图锁及模型根锁阻止重复入口，同一基础图的入口有意串行，避免运行期间被
另一入口改写。Linux启动前会检查同输出根的既有进程，发现时拒绝重复运行，
不自行停止进程。兼容完成阶段可复用；阶段失败或中断会保留
状态并在重跑时重试未完成阶段，不混入不同参数的checkpoint。当前通用训练器
没有optimizer级精确续训接口，因此未完成的训练阶段重新运行，而不是声称从
最后一个epoch恢复。tmux保证正常SSH断开不会中断训练。不要同时重复启动入口。

## 输出与记录

窗口图：`artifacts/wikidata_life_windows_1900_2000_pm20_step1_alive2026_v1/`。
关系组图、RGCN、GraphMask分别位于 `artifacts/`、`runs/`、`runs_graphmask/`
下的 `wikidata_grouped_sliding_1900_2000_pm20_step1_alive2026_v1/`。

主日志与退出码：`artifacts/wikidata_sliding_run_logs/<UTC时间_PID>.log/.exit_code`。
GraphMask根的 `multi_gpu_runs/<UTC时间_PID>/` 保存每卡日志、dispatch plan/status
及 experiment_manifest：研究问题、原模型对照、完整配置、seed、划分规则、
预算、Git提交/未提交状态、实际代码快照、主机、Python、命令、选择规则、结果路径和退出状态。
另有各阶段合同、日志、RGCN权重、GraphMask probe/history/validation与test report。

总体汇总为 `matrix_summary.tsv`，分组汇总为 `relation_group_summary.tsv`。
GPU分配日志、模型/Mask权重及报告留在自己的实验根目录，不提交大型数据到Git。
日志查看和交互用独立tmux窗口，不向正在训练或用户输入的窗口发送命令。

完成后用 `Wikidata/package_sliding_results.py` 打包小型报告，也可在本地用
`scripts/download_sliding_reports.py` 只读流式下载，无需在服务器创建档案。
本地 `Wikidata/plot_sliding_relation_groups.py --received-root ... --output-dir ...`
生成逐组/总览PNG/SVG/PDF并核对消息分母、test支持、采样协议和选中checkpoint。

本地CPU fixture 验证使用两窗、60个模拟人物、简化训练参数，只验证流程、
存活规则和复用，不是正式实验指标。还修复了完全没有训练死亡记录时的数值
编码：无训练支持的死亡/年龄维度设为中性0、保留真实缺失标志，避免NaN，
且不借用val/test日期拟合，也不填入假定2026。
