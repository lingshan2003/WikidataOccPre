# Freebase 职业 L1 逐项语义复核口径

用户授权：使用子智能体逐项核查非精确同词职业，按真实完整职业语义优先输出正确 L1，L2 冲突暂不裁决。本轮不是运行关键词规则，也不是照抄旧 LLM 映射。

## 任务边界

- 审查 1,274 种非精确同词职业，另核查 Conservationist / Ecologist 两个精确跨 L1 冲突；合计 1,276 种。
- 仅选 Culture、Discovery/Science、Leadership、Sports/Games、Other 五个有效 L1，无法可靠判断时留空并标 `needs_context`。Other 是正式类别，不可用于兜底未知字符串。
- 基于完整职业短语及其限定词逐项判断。不得用泛词子串批量分类，不得把旧标签当答案，不能用“多数人/邻居”推断职业类别。
- 作者完整职业同义词和明确类别定义优先；词项存在不恰当宽泛命中时，以完整短语语义修正，并记录依据。新推断标为语义审查，不冒充作者逐词验证。
- L2 不填写新裁决；输入 L2 仅供理解作者分类含义。
- 每一输入行都须产生一条结果，独立填写简短中文语义依据。不得默认继承候选后只检查少数例外。
- 不更改训练标签、原始职业、旧 crosswalk 或其他批次文件。

## 类别与消歧原则

- **Culture**：艺术、文学、表演、音乐、影视、出版、新闻、广播、时尚、建筑/视觉设计、文化传播与文化制作。表演教练、配音/录音/影视声音岗位、专门艺术制作管理属于文化职能。`Sports commentator` 是媒体评论者，`Science writer` 是写作者，不能仅按题材改为体育或科学。
- **Discovery/Science**：科学研究、学术教育、医疗专业人员、工程与技术开发、探索发明。软件开发和通用工程不因 `designer`、`architect` 或 `mechanic` 子串归入艺术/普通工种。作者将医疗、教育置于 Academia，将发明/工程置于 Explorer/Inventor/Developer；本轮只裁决 L1。
- **Leadership**：政治、公共治理、法律/司法、军事、宗教权威、贵族身份，以及泛企业管理/金融/商业领导。律师不是犯罪者；飞行员须区分军用/民用；慈善/倡议等角色不得无依据等同政治领袖。
- **Sports/Games**：竞技体育、比赛、运动员、武术、运动训练/裁判/体育队伍管理。`Disc jockey`、`Keyboard player`、`Acting coach` 等不能因 jockey/player/coach 归体育。`Ice dancer` 在竞技滑冰语境属于体育。
- **Other**：作者正式定义的普通工种/日常服务、小规模经营、家庭关联、犯罪/负面声名及其他非以上四域的明确身份。未知代码、人名、字段污染、泛称 `Agent` / `Official` / `Staff` 等若缺少岗位语境，应留作 needs_context，而非滥填 Other。
- 专门文化生产经营岗位如 Music executive / Film executive 需按文化产业中的具体职业功能判断，不能仅因 `executive` 自动改为 Leadership；若确为公司 CEO/泛企业高管，按企业管理职能归 Leadership；不能确定时标 needs_context。用户提及 Music executive 偏 Culture，须结合完整角色给出解释，勿单凭词项冲突决定。
- 有一条作者精确职业能解释复合短语的语义核心时，优先复用该含义，例如房地产经纪可参考作者 estate_agent/realtor/broker，不能仅凭 real_estate 泛词判大企业领导。
- 对真实跨领域的复合职业（例如同时包含歌手和政治家），不得任意选择一个大类，应 needs_context 并列出候选。
- 排名、地区、年代、性别、`(Profession)` 和 `#编号` 等标记不是职业含义，可以在人工理解时忽略；这种同义化需在依据中写明。泛称或缩略词若有多个合理含义，不凭样本人名硬推全局词义。

## 证据

- 作者来源：`external_data/notable_people/bhht_2022/derived/author_keyword_l123_rules.csv` 及 `author_level1_level2_hierarchy.csv`。
- 原始规则源码：`external_data/notable_people/bhht_2022/raw/author_code/Data_Construction/wikidata/prog_occ.do`；主类汇总 `7_prog_occupations_B.do:1413–1437`。
- 人物主字段共现只供参考，不能充当全局权威规则。
- 对罕见/不熟悉词义应查权威词义或职业定义；查到的网页 URL 写入 evidence。无法取得可靠定义则保留 needs_context。
- 常见清晰职业可基于明确语义和作者相近职业类别作判断，不必逐条联网。

## 输出 TSV 字段

`rank, raw_value, semantic_level1, review_status, confidence, semantic_rationale, evidence, candidate_l1s_json, reviewer`

- `review_status`：resolved 或 needs_context。
- `confidence`：high / medium / low；needs_context 必须 low 且 semantic_level1 为空。
- `semantic_rationale`：职业实际职责和为何归这一 L1，不能只写“按语义判断”。
- `evidence`：`author_role_analogy:<词项/类别>`、`occupation_meaning:<明确职责>` 或可靠定义网页 URL 等。不得捏造已浏览的出处。
- `candidate_l1s_json`：resolved 时 `["所选L1"]`；needs_context 时列出合理候选，污染/无职业含义时可为 `[]`。
- `reviewer`：本批子智能体标识。

所有结果是智能体语义复核稿，非人工审核或绝对正确标签。
