# CBDB 日期合格人物的具体官职与职业 v1

## 重建命令

在仓库根目录运行，只依赖 Python 标准库：

```bash
python3 scripts/cbdb_export_detailed_occupations.py
```

默认只读数据库 `external_data/cbdb/2026.09.14/cbdb_20260914.sqlite3`，写入新的 `external_data/cbdb/2026.09.14/detailed_occupations_v1/`。可用 `--database`、`--occupation-rules`、`--output-dir` 指定路径。为避免覆盖已使用的数据，目标目录存在时脚本停止；重建请指定新的输出目录。

本次从原始数据库重新应用旧日期规则。它不读取旧模型，不限旧实验图的 22,392 个关系端点，也不按时期过滤人物或官职。

## 提取规则

1. 日期规则复用 `scripts/cbdb_build_flat_triples.py::load_people`。生年或卒年至少一项非 0、非 NULL；保留负年份；排除超过 2026 的年份及生年晚于卒年的记录，共 93,688 人。
2. 读取这些人的 `POSTED_TO_OFFICE_DATA` 正人物、正官职码记录，按 `c_office_id` 关联 `OFFICE_CODES.c_office_chn` 与 `c_dy`。有名官职不再统一映射成“做官”。官职、散官、虚衔、爵位保留原貌，尚不推断官阶或现代职业类别。
3. 以 `(人物 ID, 官职 ID)` 去重，同职多次任职在职业列表中只出现一次。不同官职 ID 即使中文名称相同，也保留各自代码与朝代；原始任官行及支持行数另外保存。
4. 按官职 ID 升序写入 `occ1/occ1_code`、`occ2/occ2_code` 等槽位，直到个人全部官职写完，不截断。编号不表示主职、首次任职、官阶或重要程度。任官起止年份保存于原始证据表，尚未用于时期绑定。
5. 官名“未详”、空白及缺码不算具体官职。有其他有名官职时，只把有名部分写入职业槽位；原始未知码仍留在 `OfficeCodes`、码表及任官证据中。仅知做官但无法给出具体官名的人保留“做官”宽类，并单列、标记资格。
6. 非任官者的 STATUS 规则沿用 `config/cbdb_occupation_whitelist_v1.json`：含“做官”候选时优先，否则只保留唯一非官宽类；多个不同非官宽类的 226 人仍不分配标签。本次只放开官职粒度，没有扩大 STATUS 白名单或改变非官冲突规则。

## 结果

| 内容 | 数量 |
| --- | ---: |
| 日期合格人物 | 93,688 |
| 有职业的主表人物 | 38,789 |
| 有正官职码任官记录的人物 | 36,346 |
| 至少有一项具体官名的人物 | 35,139 |
| 有多个具体官职的人物 | 18,246 |
| 去重后的具体“人物—官职”组合 | 122,222 |
| 实际出现的有名官职代码 | 9,989 |
| 个人最多职业槽位 | 89 |
| 仅有未详官职记录、缺具体官名的人物 | 1,207 |
| 仅有 STATUS“做官”、缺具体任官记录的人物 | 789 |
| 保留的原始有效任官行 | 146,612 |

两个实际出现的未详官职码为 `OFFICE:19999` 与 `OFFICE:72802`，涉及 1,415 组人物—官职组合、1,417 条原始任官行。其中有些人物另有具体官职，因此仅缺具体官名的人数为 1,207，不能把两个人数混用。

## 输出与字段

- `people_with_occupations.csv`：主表，38,789 人，每人一行。包括原 STATUS 宽类及明确标记的“做官但缺具体官名”兜底。
- `people_with_specific_offices.csv`：主表中有具体官名的 35,139 人，保留相同字段。
- `date_filtered_people.csv`：全部 93,688 名日期合格人物与职业资格；不展开宽表槽位，便于回溯未获职业者。
- `person_occupations_long.tsv.gz`：125,872 行，每个人的每个职业一行；与宽表槽位一一对应，保留朝代、来源、支持记录数。
- `office_inventory.csv`：实际出现的 9,991 个正官职码，其中 9,989 个有名、2 个未详。包含原官名、朝代、人物数和原始行数。
- `office_postings_source.tsv.gz`：日期合格人物全部正官职码原始任官行。保留 posting ID、原起止年、官职性质及文献来源等所有原始字段。
- `officials_without_specific_office.csv`：单列 1,996 名仅知做官、缺具体官名的人。
- `summary.json`、`README.md`：统计摘要和口径说明。

宽表使用 UTF-8 BOM CSV；压缩长表、原始任官证据使用 UTF-8 TSV。缺失年份为空，不填 0。

主表的 `Node` 为 `CBDB:<personid>`，可与现有关系表对齐；`PersonID` 是原库整数 ID。`SpecificOfficeCount` 是具体官职数量，`OfficeCodeCount` 是所有有效正官职码数量，`UnknownOfficeCodeCount` 是其中未详/缺名码数量，`OccupationCount` 是实际写入的槽位数量。`LegacyBroadOccupation` 仅供与旧口径比较，不作为具体官职标签。

`OccupationResolution` 区分：

| 值 | 含义 |
| --- | --- |
| `specific_office` | 有具体官名；槽位只保存有名官职 |
| `office_unknown_broad` | 有任官记录但官名未详，只保留“做官”宽类 |
| `status_broad` | STATUS 提供的旧宽类职业，不能当作具体官职 |
| `ambiguous_nonofficial_status` | 全日期表中，多个不同非官宽类冲突而不分配职业 |
| `no_occupation` | 全日期表中，没有合格职业证据 |

槽位代码使用 `OFFICE:<id>`、`STATUS_CLASS:<宽类>` 或 `OFFICE_UNKNOWN:做官` 区分身份。后两者不应合并成某个具体官职。

## 验证

```bash
python3 -m unittest tests.test_cbdb_detailed_occupations
```

测试覆盖同名异码/异朝代保留、重复任职合并、未详及缺码处理、STATUS-only 兜底、非官冲突规则、日期资格、宽长表对齐及原始证据不截断。本次真实导出另与原始 SQLite 的全部人物—官职集合、支持行数和人物资格核对，未使用哈希。
