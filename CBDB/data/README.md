# Git 同步的 CBDB 清洗输入

`person_relation_triples.tsv.gz` 是原库 CBDB 2026-09-14 按现有职业白名单、明确生卒年规则生成的原始关系码三元组表：92,168条，22,392人，98,557条来源支持行。它复制自 `external_data/cbdb/2026.09.14/flat_triples_v1/person_relation_triples.tsv.gz`，内容与字段保持一致。

原来的 `external_data/` 被 Git 忽略；这份约1.8MB的输入放在 `CBDB/data/`，作为默认流水线输入随代码提交同步。原始SQLite不进入Git；需要重新清洗时使用 `--from-sqlite --database PATH`。职业标签仍为试验口径。
