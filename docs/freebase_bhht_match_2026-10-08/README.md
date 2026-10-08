# Freebase 1,817 种职业的 BHHT 作者规则查表结果

本轮仅生成独立查表、对照和复核材料；没有修改上一版职业 crosswalk、人物标签或训练结果，也没有执行新的 LLM 分类。

## 统计

| 查表结果 | 职业种数 | 人物—职业关联次数 |
|---|---:|---:|
| 精确同词，作者 L1/L2 唯一 | 541 | 219,093 |
| 精确同词，L1 唯一但 L2 冲突 | 0 | 0 |
| 精确同词，跨 L1 冲突 | 2 | 47 |
| 仅子串命中，单一 L1/L2 候选 | 748 | 81,753 |
| 仅子串命中，L1 唯一但 L2 冲突 | 39 | 5,383 |
| 仅子串命中，跨 L1 冲突 | 125 | 5,268 |
| 无作者规则命中 | 362 | 8,981 |

关联次数允许同一人物重复计入，不能当作不重复人数。图中 100,756 人，职业种数 1,817；两者均用原始节点表重新核验。

## 匹配口径

1. 标签仅小写、ASCII 空格转下划线。作者词项的首尾下划线只在完整标签相等测试中作为边界标记去除。保留括号、标点、连字符、词形和非 ASCII 字符，不做模糊匹配或新同义词推断。
2. 完整标签精确命中优先；无精确时，严格按作者的两端补下划线后字面子串规则查表。保留全部候选，跨类时不按第一条强制定类。
3. `candidate_level1` 仅在当前查表层级的 L1 唯一时填写，`candidate_level2` 仅在 L1/L2 组合唯一时填写。它们是职业词的候选分类，不是人物主职业或冻结训练标签。
4. 作者完整子串分类器的首命中另存 `author_first_match_*`。它可能与本轮“精确优先”的结果不同，不能把两套口径混称为作者原始分类器。
5. 人物主字段只做独立参考。主关键词对应多个分类或与规则不一致时标记复核，不从同人共现或多数频数强行补出作者规则。
6. `Other` 是作者的正式分类，与旧标签中未映射的临时兜底不同。旧 LLM 的一致/不一致只做对照，不参与本轮规则选择。

人物主字段参考与当前候选组合不一致：13 种（其中 L1 不一致 8 种）。这些标记不能自动推翻作者规则，但需要在冻结标签前复核。
在精确结果唯一的职业中，8 种的父类组合不同于作者完整子串首命中，其中 L1 不同 4 种。

## 文件

- `profession_bhht_matches_compact.csv`：便于查看的 11 列简表，1,817 行，中文表头，含候选 L1/L2/L3、旧标签和复核标记。
- `profession_bhht_matches.tsv`：全部 1,817 种职业的完整主表，保留 L1/L2/L3 候选、全部候选组、原始规则编号、旧 LLM 映射与参考证据。
- `exact_author_matches.tsv`：543 种精确同词命中，含 2 种跨 L1 冲突。精确唯一的行仍需留意参考一致性标记。
- `substring_candidates.tsv`：912 种仅子串命中的候选，全部需要转用复核。
- `conflicts_for_review.tsv`：多候选、来源参考不一致、精确与首命中类别不同、或与旧 LLM 的唯一 L1 不一致的行，按图内频次排序。
- `unmatched_for_llm.tsv`：362 种无作者规则命中的职业，供后续补词/LLM分类；部分有非权威人物主字段参考，单独标记。
- `author_llm_l1_disagreements.tsv`：本轮唯一 L1 候选与旧 `proposed` LLM 标签不同的行。
- `rule_evidence.tsv`：每个匹配词项一行，含作者原词、源码路径/行号、原始优先顺序，以及是否精确、是否本轮所选层级。
- `summary.json`：完整计数、输入路径/大小/修改时间与人物覆盖统计。

## 优先复核的跨大类子串冲突

| 职业 | 图内人数 | 候选 L1/L2 |
|---|---:|---|
| Criminal defense lawyer | 2,304 | [["Leadership","Administration/Law"],["Other","Other"]] |
| Music executive | 1,298 | [["Culture","Culture-core"],["Leadership","Corporate/Executive/Business (large)"]] |
| Child Actor | 206 | [["Culture","Culture-core"],["Other","Family"]] |
| Martial artist | 182 | [["Culture","Culture-core"],["Sports/Games","Sports/Games"]] |
| Multi-instrumentalist | 131 | [["Culture","Culture-core"],["Other","Other"]] |
| Sports commentator | 121 | [["Culture","Culture-periphery"],["Sports/Games","Sports/Games"]] |
| Real Estate Broker | 105 | [["Leadership","Corporate/Executive/Business (large)"],["Other","Worker/Business (small)"]] |
| Science writer | 88 | [["Culture","Culture-core"],["Discovery/Science","Academia"]] |
| Audio Engineer | 86 | [["Culture","Culture-periphery"],["Discovery/Science","Explorer/Inventor/Developer"]] |
| Baseball Manager | 50 | [["Leadership","Corporate/Executive/Business (large)"],["Sports/Games","Sports/Games"]] |
| Mixed Martial Artist | 44 | [["Culture","Culture-core"],["Sports/Games","Sports/Games"]] |
| Mechanical Engineer | 41 | [["Discovery/Science","Explorer/Inventor/Developer"],["Other","Worker/Business (small)"]] |
| Sports instructor | 41 | [["Discovery/Science","Academia"],["Sports/Games","Sports/Games"]] |
| Game Show Host | 32 | [["Culture","Culture-core"],["Sports/Games","Sports/Games"]] |
| Game designer | 29 | [["Culture","Culture-periphery"],["Sports/Games","Sports/Games"]] |

## 未命中的高频职业

| 职业 | 图内人数 | 旧状态 |
|---|---:|---|
| Health professional | 1,687 | proposed |
| Healthcare professional | 1,555 | proposed |
| Cameraman | 1,466 | proposed |
| Official | 913 | needs_review |
| Agent | 378 | needs_review |
| TV Personality (Profession) | 346 | proposed |
| Arranger | 265 | proposed |
| Agriculturalist | 252 | needs_review |
| Beautician | 198 | out_of_scope |
| Marketer | 166 | needs_review |
| Spokesperson | 149 | needs_review |
| Public speaker | 96 | needs_review |
| Supermodel | 53 | proposed |
| VJ | 53 | proposed |
| Master of Ceremonies | 53 | proposed |

## 单一候选与旧 LLM 提议不同的高频职业

单一子串候选不等于可靠分类。例如 Disc jockey 只命中 `jockey` 得到体育候选；本轮只保留规则证据和差异，不自动覆盖文化标签。旧 LLM 也不是裁决依据，须结合具体职业语义复核。

| 职业 | 图内人数 | 本轮候选 L1 | 旧 LLM L1 | 查表方式 |
|---|---:|---|---|---|
| Disc jockey | 269 | Sports/Games | Culture | author_substring |
| Keyboard player | 169 | Sports/Games | Culture | author_substring |
| Acting Coach | 42 | Sports/Games | Culture | author_substring |
| Magician | 36 | Other | Culture | author_substring |
| Stripper | 21 | Other | Culture | author_exact |
| Ice dancer | 19 | Culture | Sports/Games | author_substring |
| Vocal coach | 16 | Sports/Games | Culture | author_substring |
| TV chef | 13 | Other | Culture | author_substring |
| Recording Engineer (Film job) | 13 | Discovery/Science | Culture | author_substring |
| Impersonator | 11 | Other | Culture | author_exact |
| Sound Technician | 10 | Other | Culture | author_substring |
| Mixing engineer | 10 | Discovery/Science | Culture | author_substring |
| Dialect coach | 9 | Sports/Games | Culture | author_substring |
| Aircraft designer | 7 | Culture | Discovery/Science | author_substring |
| Magician (Profession) #81 | 5 | Other | Culture | author_substring |

## 来源与复现

作者数据/规则：BHHT，https://doi.org/10.21410/7E4/RDAG3O (v2.2)；https://doi.org/10.21410/7E4/YLG6YR (v2.3)。职业规则来自 `Data_Construction/wikidata/prog_occ.do`，L2→L1 来自 `7_prog_occupations_B.do:1413–1437`。许可与署名见 `external_data/notable_people/bhht_2022/README.md`。

```bash
python Freebase/match_bhht_professions.py
```

下一步先复核冲突与高影响子串候选，再对未命中项补词/LLM分类；本表不自动选择多职业人物的主职业。
