# Freebase 职业 L1 逐项语义复核（2026-10-08）

本轮按用户要求使用三个 GPT-6 Luna high 子智能体逐项审查非精确同词职业，并由主代理合并、复核高频词、分类变更、同义写法与独立审计建议。仅处理 L1，L2 暂不裁决。

这是智能体语义复核稿，未经过人工逐项批准，也未冻结为训练标签。保留作者分类与语义推断的来源区别，不把新增判断说成论文作者已验证的完整职业分类。

## 覆盖与结果

|处理范围|职业种数|
|---|---:|
|保留精确同词、L1 唯一的作者映射|541|
|本轮语义复核后给出 L1|1,148|
|本轮复核后仍需岗位或实体语境|128|
|总计|1,817|

复核范围为 1,274 种非精确同词职业，另加 Conservationist / Ecologist 两个精确跨 L1 冲突，共 1,276 种；541 种精确唯一职业未在本轮重新逐项语义审查。

主代理单独裁决/确认 79 项；补充具体职责依据的第二遍记录 612 项。与原有唯一查表候选相比，96 项改换 L1，33 项撤回为待补语境。

|L1|职业种数|
|---|---:|
|Culture|577|
|Discovery/Science|348|
|Leadership|266|
|Other|377|
|Sports/Games|121|

## 判断口径

- 阅读完整职业短语，区分职业职责与报道题材、工作场景、通用词片段；不将 jockey、player、criminal、engineer 等子串直接当成最终大类。
- 优先复用作者完整同词或明确同义职业的类别，语义新增映射保留职责依据和置信度。专门艺术教学、文化制作与传播等边界使用本轮明确写出的操作口径；这些扩展不等于作者原始算法。
- Other 是作者的正式类别，涵盖普通工种、服务及部分家庭/负面声名身份；代码、人名、机构污染、岗位不明或真正跨域的词留空，不能丢入 Other 兜底。
- 专业知识或技术工具本身不等于科学研发；维修、用户支持与实际工程研发分别按职责判断。
- 数字营销相关活动词可按作者 marketing 的明确功能口径映射 Other；宽泛的行业、学科或组织词若不能确定个人职责，仍保留待查。
- `(Profession)` / `(Job title)` 和编号只在语义审查中作为元数据理解；`(Film job)` 等有实际岗位含义的限定词保留。

|示例|本轮 L1|说明|
|---|---|---|
|Music executive|Culture|本轮按唱片、艺人及音乐内容业务的专门管理岗位归文化产业；这是完整短语的语义映射，不是作者对该完整词的精确分类。若人物另有CEO等职业应分别按其职责映射。|
|Disc jockey|Culture|选择、混音并播放录制音乐供广播或现场受众收听，是音乐传播岗位；不是赛马骑师。|
|Keyboard player|Culture|演奏键盘乐器的音乐人；player在完整短语中指乐器演奏者。|
|Criminal defense lawyer|Leadership|刑事辩护律师为刑事案件当事人提供法律代理与辩护；criminal修饰案件领域，不代表律师是犯罪者。|
|Sports commentator|Culture|通过广播或媒体解说体育赛事，职业职能是内容传播，体育是其报道题材。|
|Agriculturalist|Other|农业实务专家向农户或企业提供建议；本轮优先按作者同义agriculturist的Worker/Business(small)口径，而不因expert一词自动判为学术研究者。|
|Official|待补语境|“Official”只表示某种正式职务或身份，未说明是政府、体育、企业还是其他机构，缺少岗位语境。|
|Conservationist|待补语境|自然保护者可能从事生态科研，也可能是环境倡议者；没有具体职责语境时不能断言是科学家或政治倡议者。|

## 回到人物层面

对现有 100,756 名人节点的 320,525 条人物—职业对应关系重新统计：

|条件|人物数|
|---|---:|
|所有职业均有 L1，且汇总后只有一个 L1|83,499|
|所有职业均有 L1，但汇总后跨多个 L1|15,525|
|至少有一个职业仍未映射|1,732|

这是词表复核后的覆盖统计，不是新的训练人物标签。一个职业映射到单一 L1，不意味着一个多职业人物也只能有一个 L1；人物多域职业仍需另行确定实验标签策略。未改动训练标签、已完成的模型结果或原始数据。

## 文件与验证

- `profession_l1_semantic_crosswalk_compact.csv`：1817 行中文简表，适合查看职业、L1、状态、置信度及理由。
- `profession_l1_semantic_crosswalk.tsv`：完整来源、旧候选、语义判断、主审及 L2 deferred 标记。
- `semantic_review_1276.tsv`：本轮复核范围；`needs_context.tsv`：剩余待查词及可能类别。
- `changes_from_lookup.tsv`、`withdrawn_unique_lookup.tsv`：改换和撤回的原查表候选。
- `batch_*_review.tsv`：三个子智能体的原始逐项结果；`primary_adjudications.tsv`：主审裁决。
- `audit_findings.tsv`：独立审计原始建议；不是最终分类表。主审裁决及补写依据可覆盖这些原始建议。
- `rationale_*_review.tsv`：具体职责依据第二遍补写；最终类别仍由主审裁决优先。
- `summary.json`：可核对的完整计数。

校验：1817 个职业完整覆盖；1276 个复核输入输出一一对应且无重复；合法 L1/置信度/待查空值/候选数组/必填依据均检查；当前图人物职业频次与词表一致；元数据同词归一后的 L1 不一致记录 0 行。

复现：`python Freebase/merge_bhht_semantic_review.py`；验证测试：`python -m unittest discover -s tests -p test_freebase_bhht_semantic_merge.py`。

作者规则见 `external_data/notable_people/bhht_2022/derived/author_keyword_l123_rules.csv`，层级见同目录 `author_level1_level2_hierarchy.csv`；词义资料链接在每条 evidence 中。作者父数据及规则抽取说明见 `external_data/notable_people/bhht_2022/README.md`。
