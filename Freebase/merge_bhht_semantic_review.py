"""Validate independent L1 reviews, retain provenance, and summarize person coverage."""
import argparse
import csv
import json
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
        row = {k: o[k] for k in ['rank', 'raw_value', 'graph_people']}
        row.update({
            'level1': decision['semantic_level1'] if decision else o['candidate_level1'],
            'review_status': decision['review_status'] if decision else 'retained_exact_unique',
            'confidence': decision['confidence'] if decision else 'source_exact',
            'mapping_source': 'primary_semantic_adjudication' if raw in adjudications else 'agent_semantic_review' if agent else 'author_exact_unique',
            'semantic_rationale': decision['semantic_rationale'] if decision else '完整职业词与作者规则精确一致，且规则候选 L1 唯一；本轮保留原映射。',
            'evidence': decision['evidence'] if decision else 'author_keyword_l123_rules.csv:' + o['candidate_keywords_json'],
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
        'changed_from_previous_proposed_llm': len(disagreements),
        'level1_profession_counts': dict(Counter(r['level1'] or 'UNRESOLVED' for r in merged)),
        'semantic_review_confidence_counts': dict(Counter(r['confidence'] for r in merged if r['raw_value'] in selected)),
        'graph_people': sum(counts.values()), 'person_profession_pairs': sum(occurrences.values()),
        'person_resolution_not_training_labels': dict(counts),
        'unresolved_person_profession_pairs_not_unique_people': sum(int(r['graph_people']) for r in unresolved),
    }
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-dir', type=Path, default=DEFAULT)
    p.add_argument('--source', type=Path, default=ROOT / 'docs/freebase_bhht_match_2026-10-08/profession_bhht_matches.tsv')
    p.add_argument('--nodes', type=Path, default=ROOT / 'external_data/freebase/descriptive_v2_local/05_final/nodes.csv')
    a = p.parse_args()
    merge(a.review_dir, a.source, a.nodes)
