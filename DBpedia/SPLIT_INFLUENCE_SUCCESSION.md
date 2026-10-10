# 八组年度滑窗重训：拆开 Influence 与 Succession

新的 taxonomy：`config/dbpedia_tie_taxonomy_acquired_subgroups_v2.json`。

| Group | Relations |
|---|---|
| Inherited ties | child, parent, father, mother, relative |
| Intimate partnership | spouse, partner |
| Education / mentorship | doctoralAdvisor, doctoralStudent, academicAdvisor, notableStudent, trainer, training, coach |
| Professional collaboration | associate, employer |
| Influence | influenced, influencedBy |
| Succession | successor, predecessor |
| Religious authority / recognition | beatifiedBy, canonizedBy |
| Other acquired ties | president, primeMinister, governor, governorGeneral, monarch, chancellor, lieutenant, vicePresident, deputy, appointer, opponent |

本次“继任”是职务接替的 Succession，不是亲缘组 Inherited ties。八组仍覆盖原33种谓词，每种谓词恰属一组。正向和反向消息分别编码，共16种有向模型关系；`num_bases.multi_group=16`，延续原实验每种关系一个basis的设置。

当前配置：`config/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3.json`。

范围仍为1900–2000共101个中心年份，每窗±20年。2026-10-10新增规则：出生年严格大于1920、死亡日期缺失的节点，推定生命区间为`[birth_year, 2026]`，以区间相交判断窗口归属。出生于1920及以前的缺失死亡节点、仅有死亡日期的节点仍按已知端点入窗；已有死亡日期保持原值，两端均缺失或有效死亡早于出生仍排除。2026是配置中固定的截止年，复跑时不会随电脑日期自动变化。

这是构图用的生命区间假设。原始`death_year`保留缺失，死亡/年龄特征的缺失标记保留；节点表另记录`life_period_effective_death_year`及`life_period_death_year_imputed`，manifest记录规则和人数。新窗口规则位于`config/dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2.json`。

新规则改变窗口节点集，必须重建原始谓词窗口图并在每窗按原来的seed及70/10/20规则重新划分监督节点；再做八组collapse、RGCN与GraphMask训练。训练超参数仍沿用前版。不能复用旧窗口图、七组collapse图、旧RGCN或probe。

新窗口图、关系图、RGCN、GraphMask输出到独立目录，旧七组及未补齐寿命的八组结果保持可访问：

```text
artifacts/dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2/
artifacts/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3/
runs/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3/
runs_graphmask/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3/
```

GraphMask仍用`any-stage`选择checkpoint，不强制Layer 0门控已启用。关系组拆开会改变RGCN的消息变换、分组去重及GraphMask输入，所以旧图不能简单拆分占比后冒充八组模型结果。

## 服务器运行

推送后，在服务器仓库根目录及原来可用的Python环境中，用tmux运行。若未补齐寿命的前版已开跑，先在其tmux窗口Ctrl+C停止，再同步并启动这版：

```bash
git pull
bash DBpedia/run_sliding_split_influence_succession.sh plan all --gpus 0,1
bash DBpedia/run_sliding_split_influence_succession.sh run all --gpus 0,1
```

也支持`CUDA_VISIBLE_DEVICES=0,1 bash DBpedia/run_sliding_split_influence_succession.sh run all --num-gpus 2`。仍为每GPU一个worker，按空闲worker动态分配完整窗口；兼容已完成阶段的断点继续。新入口固定实验路径、seed42和101窗multi_group范围，防止旧shell路径覆盖导致误用七组目录。GPU选择及既有batch/worker等训练环境选项仍可使用。

Python默认取当前环境的`python`。确需指定时，设置`DBPEDIA_PYTHON_BIN`为先前跑通实验的解释器路径。

## 完成后打包与绘图

```bash
python DBpedia/package_sliding_results.py \
  --config config/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3.json \
  --output artifacts/dbpedia_sliding_split_influence_succession_alive2026_v3_reports.tar.gz
```

如果训练时用环境变量调整了配置，打包时的`--config`可以改成新`model_root`下的`grouped_pipeline_resolved_config.json`，以导出实际运行参数。

本地解压到新目录后，`DBpedia/plot_sliding_relation_groups.py`自动识别七组v1或八组v2，按1901–2000年绘制全部关系组，无bootstrap。给八组结果选择新的`--output-dir`。八组总览为4×2面板，包含独立Influence与Succession曲线。旧的`export_ppt_summary.py`和八历史时期热图脚本仍针对原实验，年度曲线请使用上述年度绘图脚本。

构图审计的关系ID检查也改为按taxonomy组数动态确定，可运行：

```bash
python DBpedia/audit_sliding_window_artifacts.py \
  --config config/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3.json \
  --periods center_1900,center_1901,center_1950,center_2000
```

这项审计需要相应八组collapse图已生成。

## 生存假设的构图效果

`DBpedia/compare_alive_assumption_isolation.py`从原始表独立比较101窗的新旧归属和诱导边，区分旧孤立节点恢复连接、旧节点仍孤立以及新增孤立节点。结果保存在`artifacts/dbpedia_alive2026_isolation_2026_10_10/`；这是GraphMask前的图结构统计。

| Window center | Old isolated nodes | New isolated nodes | Old rate | New rate |
|---|---:|---:|---:|---:|
| 1950 | 3,945 | 3,627 | 11.02% | 9.98% |
| 1980 | 7,094 | 2,372 | 26.86% | 6.32% |
| 2000 | 7,402 | 2,884 | 45.10% | 8.43% |

2000窗节点从16,411增至34,202；原始方向关系从8,443增至32,818。旧孤立节点中5,151个恢复连接，2,251个仍孤立；新增节点中633个孤立，故新孤立总数为2,884。这些数值描述构图改善，预测效果需要新训练结果确认。

可运行`python DBpedia/compare_alive_assumption_isolation.py`复现；八组构图预览可用`python DBpedia/preview_sliding_windows.py --config config/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3.json --output-dir artifacts/dbpedia_alive2026_preview`。

此前不补齐寿命的八组v2配置仍保留；需要回放时通过通用`run_sliding_multi_gpu.sh --config`显式选它。专用`run_sliding_split_influence_succession.sh`现在固定选择带alive2026规则的v3配置。
