"""Validate independent L1 reviews, retain provenance, and summarize person coverage."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'docs/freebase_bhht_semantic_review_2026-10-08'
LEVELS = {'Culture', 'Discovery/Science', 'Leadership', 'Sports/Games', 'Other'}


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if str(path).endswith('.tsv') else ','))


def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t' if str(path).endswith('.tsv') else ',', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def validate_decision(r):
    assert r['semantic_rationale'].strip() and r['evidence'].strip(), ('missing rationale/evidence', r)
    candidates = json.loads(r['candidate_l1s_json'])
    assert isinstance(candidates, list) and set(candidates) <= LEVELS, r
    assert len(candidates) == len(set(candidates)), r
    if r['review_status'] == 'resolved':
        assert r['semantic_level1'] in LEVELS and r['confidence'] in {'high', 'medium'}, r
        assert candidates == [r['semantic_level1']], r
    else:
        assert r['review_status'] == 'needs_context' and r['confidence'] == 'low' and not r['semantic_level1'], r


def validate_batch(inputs, reviews, reviewer):
    expected = {(r['rank'], r['raw_value']) for r in inputs}
    actual = [(r['rank'], r['raw_value']) for r in reviews]
    assert len(inputs) == len(expected) == len(actual) == len(set(actual)), 'duplicate or missing batch rows'
    assert expected == set(actual), 'batch input/output mismatch'
    for r in reviews:
        assert r['reviewer'] == reviewer, r
        validate_decision(r)


def write_report(directory, summary, merged):
    s = summary
    persons = s['person_resolution_not_training_labels']
    examples = ['Music executive', 'Disc jockey', 'Keyboard player', 'Criminal defense lawyer',
                'Sports commentator', 'Agriculturalist', 'Official', 'Conservationist']
    index = {r['raw_value']: r for r in merged}
    lines = [
        '# Freebase 职业 L1 逐项语义复核（2026-10-08）', '',
        '本轮按用户要求使用三个 GPT-6 Luna high 子智能体逐项审查非精确同词职业，并由主代理合并、复核高频词、分类变更、同义写法与独立审计建议。仅处理 L1，L2 暂不裁决。', '',
        '这是智能体语义复核稿，未经过人工逐项批准，也未冻结为训练标签。保留作者分类与语义推断的来源区别，不把新增判断说成论文作者已验证的完整职业分类。', '',
        '## 覆盖与结果', '',
        '|处理范围|职业种数|', '|---|---:|',
        f"|保留精确同词、L1 唯一的作者映射|{s['retained_exact_unique']:,}|",
        f"|本轮语义复核后给出 L1|{s['semantic_resolved']:,}|",
        f"|本轮复核后仍需岗位或实体语境|{s['needs_context']:,}|",
        f"|总计|{s['professions']:,}|", '',
        '复核范围为 1,274 种非精确同词职业，另加 Conservationist / Ecologist 两个精确跨 L1 冲突，共 1,276 种；541 种精确唯一职业未在本轮重新逐项语义审查。', '',
        f"主代理单独裁决/确认 {s['primary_adjudications']} 项；补充具体职责依据的第二遍记录 {s['specific_rationale_second_pass']} 项。与原有唯一查表候选相比，{s['changed_from_unique_lookup']} 项改换 L1，{s['withdrawn_unique_lookup_to_needs_context']} 项撤回为待补语境。", '',
        '|L1|职业种数|', '|---|---:|',
    ]
    lines += [f'|{label}|{s["level1_profession_counts"].get(label, 0):,}|' for label in sorted(LEVELS)]
    lines += ['', '## 判断口径', '',
              '- 阅读完整职业短语，区分职业职责与报道题材、工作场景、通用词片段；不将 jockey、player、criminal、engineer 等子串直接当成最终大类。',
              '- 优先复用作者完整同词或明确同义职业的类别，语义新增映射保留职责依据和置信度。专门艺术教学、文化制作与传播等边界使用本轮明确写出的操作口径；这些扩展不等于作者原始算法。',
              '- Other 是作者的正式类别，涵盖普通工种、服务及部分家庭/负面声名身份；代码、人名、机构污染、岗位不明或真正跨域的词留空，不能丢入 Other 兜底。',
              '- 专业知识或技术工具本身不等于科学研发；维修、用户支持与实际工程研发分别按职责判断。',
              '- 数字营销相关活动词可按作者 marketing 的明确功能口径映射 Other；宽泛的行业、学科或组织词若不能确定个人职责，仍保留待查。',
              '- `(Profession)` / `(Job title)` 和编号只在语义审查中作为元数据理解；`(Film job)` 等有实际岗位含义的限定词保留。', '',
              '|示例|本轮 L1|说明|', '|---|---|---|']
    for raw in examples:
        r = index[raw]
        lines.append(f"|{raw}|{r['level1'] or '待补语境'}|{r['semantic_rationale']}|")
    lines += ['', '## 回到人物层面', '',
              f"对现有 {s['graph_people']:,} 名人节点的 {s['person_profession_pairs']:,} 条人物—职业对应关系重新统计：", '',
              '|条件|人物数|', '|---|---:|',
              f"|所有职业均有 L1，且汇总后只有一个 L1|{persons.get('unique_l1', 0):,}|",
              f"|所有职业均有 L1，但汇总后跨多个 L1|{persons.get('multiple_l1', 0):,}|",
              f"|至少有一个职业仍未映射|{persons.get('incomplete_mapping', 0):,}|", '',
              '这是词表复核后的覆盖统计，不是新的训练人物标签。一个职业映射到单一 L1，不意味着一个多职业人物也只能有一个 L1；人物多域职业仍需另行确定实验标签策略。未改动训练标签、已完成的模型结果或原始数据。', '',
              '## 文件与验证', '',
              '- `profession_l1_semantic_crosswalk_compact.csv`：1817 行中文简表，适合查看职业、L1、状态、置信度及理由。',
              '- `profession_l1_semantic_crosswalk.tsv`：完整来源、旧候选、语义判断、主审及 L2 deferred 标记。',
              '- `semantic_review_1276.tsv`：本轮复核范围；`needs_context.tsv`：剩余待查词及可能类别。',
              '- `changes_from_lookup.tsv`、`withdrawn_unique_lookup.tsv`：改换和撤回的原查表候选。',
              '- `batch_*_review.tsv`：三个子智能体的原始逐项结果；`primary_adjudications.tsv`：主审裁决。',
              '- `audit_findings.tsv`：独立审计原始建议；不是最终分类表。主审裁决及补写依据可覆盖这些原始建议。',
              '- `rationale_*_review.tsv`：具体职责依据第二遍补写；最终类别仍由主审裁决优先。',
              '- `summary.json`：可核对的完整计数。', '',
              f"校验：1817 个职业完整覆盖；1276 个复核输入输出一一对应且无重复；合法 L1/置信度/待查空值/候选数组/必填依据均检查；当前图人物职业频次与词表一致；元数据同词归一后的 L1 不一致记录 {s['alias_l1_inconsistency_rows']} 行。", '',
              '复现：`python Freebase/merge_bhht_semantic_review.py`；验证测试：`python -m unittest discover -s tests -p test_freebase_bhht_semantic_merge.py`。', '',
              '作者规则见 `external_data/notable_people/bhht_2022/derived/author_keyword_l123_rules.csv`，层级见同目录 `author_level1_level2_hierarchy.csv`；词义资料链接在每条 evidence 中。作者父数据及规则抽取说明见 `external_data/notable_people/bhht_2022/README.md`。', '']
    (directory / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def merge(directory, source, nodes):
    original = read(source)
    selected = {r['raw_value'] for r in original if r['lookup_method'] != 'author_exact' or r['match_status'] != 'exact_unique'}
    reviews = {}
    for batch in range(1, 4):
        inputs = read(directory / f'batch_{batch}_input.tsv')
        outputs = read(directory / f'batch_{batch}_review.tsv')
        validate_batch(inputs, outputs, f'bhht_semantic_batch_{batch}')
        for r in outputs:
            assert r['raw_value'] not in reviews, 'overlapping batches'
            reviews[r['raw_value']] = r
    assert set(reviews) == selected and len(original) == 1817 and len(selected) == 1276
    rationales = {}
    for batch in range(1, 4):
        input_path = directory / f'rationale_{batch}_input.tsv'
        if not input_path.exists():
            continue
        inputs = read(input_path)
        outputs = read(directory / f'rationale_{batch}_review.tsv')
        expected = {(r['rank'], r['raw_value']) for r in inputs}
        actual = [(r['rank'], r['raw_value']) for r in outputs]
        assert len(inputs) == len(expected) == len(actual) == len(set(actual)) and expected == set(actual)
        for r in outputs:
            assert r['raw_value'] not in rationales and r['raw_value'] in selected
            assert r['reviewer'] == f'rationale_batch_{batch}'
            assert r['semantic_rationale'].strip() and r['evidence'].strip()
            rationales[r['raw_value']] = r
    adjudications = {}
    if (directory / 'primary_adjudications.tsv').exists():
        for r in read(directory / 'primary_adjudications.tsv'):
            validate_decision(r)
            assert r['raw_value'] in selected and r['raw_value'] not in adjudications, r
            assert r['rank'] == reviews[r['raw_value']]['rank'], r
            adjudications[r['raw_value']] = r
    merged = []
    for o in original:
        raw = o['raw_value']
        agent = reviews.get(raw)
        decision = adjudications.get(raw, agent)
        rationale = adjudications.get(raw, rationales.get(raw, decision))
        row = {k: o[k] for k in ['rank', 'raw_value', 'graph_people']}
        row.update({
            'level1': decision['semantic_level1'] if decision else o['candidate_level1'],
            'review_status': decision['review_status'] if decision else 'retained_exact_unique',
            'confidence': decision['confidence'] if decision else 'source_exact',
            'mapping_source': 'primary_semantic_adjudication' if raw in adjudications else 'agent_semantic_review' if agent else 'author_exact_unique',
            'semantic_rationale': rationale['semantic_rationale'] if rationale else '完整职业词与作者规则精确一致，且规则候选 L1 唯一；本轮保留原映射。',
            'evidence': rationale['evidence'] if rationale else 'author_keyword_l123_rules.csv:' + o['candidate_keywords_json'],
            'rationale_reviewer': rationale['reviewer'] if rationale else 'author_lookup_retained',
            'candidate_l1s_json': decision['candidate_l1s_json'] if decision else json.dumps([o['candidate_level1']], ensure_ascii=False),
            'reviewer': decision['reviewer'] if decision else 'author_lookup_retained',
            'primary_reviewed': str(int(raw in adjudications)),
            'agent_level1': agent['semantic_level1'] if agent else '',
            'agent_review_status': agent['review_status'] if agent else '',
            'original_lookup_method': o['lookup_method'],
            'original_match_status': o['match_status'],
            'original_candidate_level1': o['candidate_level1'],
            'original_parent_pairs_json': o['candidate_parent_pairs_json'],
            'previous_llm_level1': o['previous_llm_level1'],
            'previous_llm_status': o['previous_llm_status'],
            'level2_review_status': 'deferred',
            'human_review_status': 'not_reviewed',
            'label_status': 'semantic_review_draft_not_frozen',
        })
        merged.append(row)
    fields = list(merged[0])
    write(directory / 'profession_l1_semantic_crosswalk.tsv', merged)
    write(directory / 'semantic_review_1276.tsv', [r for r in merged if r['raw_value'] in selected], fields)
    unresolved = [r for r in merged if not r['level1']]
    write(directory / 'needs_context.tsv', unresolved, fields)
    changed = [r for r in merged if r['original_candidate_level1'] and r['level1'] and r['original_candidate_level1'] != r['level1']]
    write(directory / 'changes_from_lookup.tsv', changed, fields)
    withdrawn = [r for r in merged if r['original_candidate_level1'] and not r['level1']]
    write(directory / 'withdrawn_unique_lookup.tsv', withdrawn, fields)
    disagreements = [r for r in merged if r['previous_llm_status'] == 'proposed' and r['level1'] and r['previous_llm_level1'] != r['level1']]
    write(directory / 'changes_from_previous_llm.tsv', disagreements, fields)
    compact = [{'职业': r['raw_value'], '关联人物数': r['graph_people'], 'L1': r['level1'], '状态': r['review_status'], '置信度': r['confidence'], '来源': r['mapping_source'], '语义依据': r['semantic_rationale'], '候选L1': r['candidate_l1s_json']} for r in merged]
    write(directory / 'profession_l1_semantic_crosswalk_compact.csv', compact)
    counts = Counter()
    occurrences = Counter()
    index = {r['raw_value']: r for r in merged}
    for p in read(nodes):
        professions = json.loads(p['Professions'])
        assert professions and len(professions) == len(set(professions))
        occurrences.update(professions)
        labels = [index[v]['level1'] for v in professions]
        state = 'incomplete_mapping' if any(not v for v in labels) else 'unique_l1' if len(set(labels)) == 1 else 'multiple_l1'
        counts[state] += 1
    assert occurrences == Counter({r['raw_value']: int(r['graph_people']) for r in merged}), 'inventory differs from current graph'
    summary = {
        'version': 'freebase_bhht_semantic_l1_review_v1',
        'scope': 'L1 only; L2 deferred; draft not human-approved or frozen',
        'professions': len(merged), 'reviewed_professions': len(selected),
        'retained_exact_unique': len(merged) - len(selected),
        'semantic_resolved': len(selected) - len(unresolved), 'needs_context': len(unresolved),
        'primary_adjudications': len(adjudications), 'changed_from_unique_lookup': len(changed),
        'specific_rationale_second_pass': len(rationales),
        'withdrawn_unique_lookup_to_needs_context': len(withdrawn),
        'changed_from_previous_proposed_llm': len(disagreements),
        'level1_profession_counts': dict(Counter(r['level1'] or 'UNRESOLVED' for r in merged)),
        'semantic_review_confidence_counts': dict(Counter(r['confidence'] for r in merged if r['raw_value'] in selected)),
        'graph_people': sum(counts.values()), 'person_profession_pairs': sum(occurrences.values()),
        'person_resolution_not_training_labels': dict(counts),
        'unresolved_person_profession_pairs_not_unique_people': sum(int(r['graph_people']) for r in unresolved),
    }
    aliases = {}
    for r in merged:
        key = re.sub(r'\s*#\d+$', '', r['raw_value'].casefold())
        key = re.sub(r'\s*\((profession|job title)\)', '', key)
        aliases.setdefault(key, []).append(r)
    conflicts = [r for group in aliases.values() if len(group) > 1 and len({r['level1'] for r in group}) > 1 for r in group]
    write(directory / 'alias_l1_inconsistencies.tsv', conflicts, fields)
    summary['alias_l1_inconsistency_rows'] = len(conflicts)
    summary['review_by_original_match_status'] = {
        status: dict(Counter(r['review_status'] for r in merged if r['raw_value'] in selected and r['original_match_status'] == status))
        for status in sorted({r['original_match_status'] for r in merged if r['raw_value'] in selected})
    }
    audit_path = directory / 'audit_findings.tsv'
    if audit_path.exists():
        dispositions = []
        for finding in read(audit_path):
            actual = index[finding['raw_value']]
            accepted = actual['level1'] == finding['recommended_l1'] or (
                not actual['level1'] and finding['recommended_l1'] == 'needs_context')
            state = 'specific_rationale_supplied' if finding['finding_type'] == 'rationale_specificity' else (
                'accepted' if accepted else 'primary_alternative')
            if state == 'specific_rationale_supplied':
                assert finding['raw_value'] in rationales or finding['raw_value'] in adjudications
            dispositions.append({
                'rank': finding['rank'], 'raw_value': finding['raw_value'],
                'finding_type': finding['finding_type'], 'recommended_l1': finding['recommended_l1'],
                'final_level1': actual['level1'], 'disposition': state,
                'final_rationale': actual['semantic_rationale'],
            })
        write(directory / 'audit_dispositions.tsv', dispositions)
        summary['independent_audit_disposition_counts'] = dict(Counter(r['disposition'] for r in dispositions))
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(directory, summary, merged)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-dir', type=Path, default=DEFAULT)
    p.add_argument('--source', type=Path, default=ROOT / 'docs/freebase_bhht_match_2026-10-08/profession_bhht_matches.tsv')
    p.add_argument('--nodes', type=Path, default=ROOT / 'external_data/freebase/descriptive_v2_local/05_final/nodes.csv')
    a = p.parse_args()
    merge(a.review_dir, a.source, a.nodes)
