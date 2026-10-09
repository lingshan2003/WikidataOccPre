# Freebase 第二版数据：BHHT L1 与临时单标签

生成日期：2026-10-09。依据用户授权，将128种待审职业暂归Other；多L1人物暂选其一。原始数据、旧版映射与第一版实验文件保留。

## 分类与选择规则

输入为 `docs/freebase_bhht_semantic_review_2026-10-08/profession_l1_semantic_crosswalk.tsv`：541种作者精确唯一映射，1,148种语义复核映射，128种待补语境。新职业表 `profession_l1_crosswalk_v2.tsv` 共1,817行：保留原始 `level1` 和 `review_status`，新增 `training_level1`。待审词的 `training_level1=Other`、`temporary_other_fallback=1`；作者/审核已有的377种正式Other仍单独识别。原有语义审查状态未伪装成人工确认。

人物按原职业数组顺序，取第一个具有作者/语义审核L1的职业，正式Other也属于可选的有效L1。只有整组职业都未解决时，取第一个职业，人物标签暂用Other。因此临时Other不会优先覆盖已知职业。这与第一版“优先已有映射”的选择逻辑一致；数组顺序不代表职业的重要性。

每人原始职业、原语义L1序列、补Other后L1序列、全部候选大类、待审职业、所选职业、选择原因、置信度和是否临时Other均保存在 `label_selection_audit.tsv`。L2、L3仍缺失，不参与本次训练。

## 全图统计

总计 100,756 人，99,644 条规范基础关系三元组；添加生成反向边后 199,288 条消息边。原始关系事实为 145,876 条。

|最终实验L1|人物数|
|---|---:|
|Culture|72,897|
|Leadership|10,833|
|Discovery/Science|8,389|
|Sports/Games|7,226|
|Other|1,411|

|标签选择过程|人物数|
|---|---:|
|保留原审核唯一L1|83,499|
|补Other后仍有多个L1，按顺序暂选已审核职业|17,165|
|含待审词，补Other后仍为唯一Other|45|
|所有职业待审，人物标签暂用Other|47|

最终Other共1,411人，其中1,364人选中正式Other，47人因全部职业待审而使用临时Other。至少含一个待审职业的1,732人均保留相关标记。补Other后的多L1人数17,165含原已确认多L1的15,525人，也包含因待审词被临时设Other而形成多类的人；不是新增确认的跨域职业事实。

## 时期与分组数据

沿用第一版生存时期规则和5个关系组：inherited、intimate_partnership、education_mentorship、influence_succession、other_acquired，加各自生成反向共10种消息关系。8个时期源图和多组图均已生成。

|时期|节点|分组前消息边|分组后消息边|监督类别|Train/Val/Test|
|---|---:|---:|---:|---:|---|
|before_1500|502|1,342|1,342|4|351/50/100|
|1500_1900|15,879|28,674|28,664|5|11,115/1,588/3,176|
|1901_1920|23,079|33,110|33,104|5|16,155/2,309/4,615|
|1921_1940|37,871|54,832|54,824|5|26,510/3,787/7,574|
|1941_1960|48,876|65,648|65,628|5|34,213/4,888/9,775|
|1961_1980|48,285|52,136|51,970|5|33,800/4,828/9,657|
|1981_2000|30,409|22,394|22,330|5|21,286/3,042/6,081|
|2001_2020|10,582|3,442|3,442|5|7,407/1,058/2,117|

时期只保留两端均属于该期的关系。<1500仍有1名人物因所属局部类别不足3人不参与监督，节点保留。全图所有100,756人均有监督标签，70%/10%/20%划分为70,529/10,076/20,151人。各时期独立划分。

## 与第一版对照和验证

与第一版相比，923 人的最终标签改变，99,833 人相同。节点顺序、身份、生卒信息、原始职业和全图基础边张量均逐项核对一致；所有时期的基础图结构也与第一版一致。变化详情见 `label_changes_from_v1.tsv`，验证见 `v1_comparison.json` 和 `period_validation.json`。

按新标签以seed=42重新分层划分，因此虽然比例和seed沿用第一版，具体train/val/test成员并不相同。不能把两版分数差全部归因于标签修正。当前仅准备数据，尚未训练v2 RGCN或GraphMask。

专用标签/来源/选择规则和现有Freebase回归测试共48项通过；实图核查五类标签全覆盖、全图mask互斥且覆盖所有节点、8个时期训练/验证/测试均非空、分组图标签和mask与v2时期源图一致。

## 文件位置

- `artifacts/freebase_provisional_l1_v2/`：nodes.csv、edges.csv、graph_data.pt、class_stats.csv、完整职业映射及人物选择审计。
- `artifacts/freebase_life_periods_20y_v2/<时期>/`：8个时期源图与局部划分。
- `artifacts/freebase_grouped_20y_v2/<时期>/multi_group/`：关系分组后的训练图。
- `config/freebase_grouped_rgcn_graphmask_20y_v2.json`：v2独立配置；配置格式version仍为1，数据版本dataset_version为2。
- `Freebase/run_grouped_rgcn_graphmask_20y_v2.sh`：v2入口，默认固定使用v2配置，仍需手动指定设备；显式--config可覆盖。
- `artifacts/freebase_training_inputs_v2.tar.gz`：在已有可运行v1的服务器仓库中重建v2所需的输入、适配代码和配置。无需BHHT全量人物数据库，也不需要旧person_l1_audit.tsv。

## 在服务器运行

先将 freebase_training_inputs_v2.tar.gz 上传到原服务器项目目录，在已能运行v1的项目根目录解压。包内相对路径已对齐。它包含v2所需两个原始小表、BHHT语义职业表及相应代码和配置。包中不使用本地派生图的路径/mtime记录；在服务器重新生成，可避免跨机器来源记录不匹配。

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
tar -xzf freebase_training_inputs_v2.tar.gz
conda activate wywikidata
# 4仅为示例，请按当时的空闲设备手动选择。
export CUDA_VISIBLE_DEVICES=4
export FREEBASE_GROUP_DEVICE=cuda:0

# 如环境中曾覆盖过第一版输出目录，先清除这些覆盖，使用v2配置路径。
unset FREEBASE_GROUP_SOURCE_DATA FREEBASE_GROUP_PERIOD_ROOT FREEBASE_GROUP_RELATION_ROOT
unset FREEBASE_GROUP_MODEL_ROOT FREEBASE_GROUP_GRAPHMASK_ROOT
bash Freebase/run_grouped_rgcn_graphmask_20y_v2.sh plan
bash Freebase/run_grouped_rgcn_graphmask_20y_v2.sh run all
```

只准备数据：将最后一条的 `all` 改为 `prepare`；再运行 `run collapse` 可生成多组图。RGCN结果写入 `runs/freebase_grouped_20y_v2/`，GraphMask结果写入 `runs_graphmask/freebase_grouped_20y_v2/`。所有v2派生根目录均独立于v1。
