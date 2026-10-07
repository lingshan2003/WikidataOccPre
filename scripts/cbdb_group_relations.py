"""Group CBDB v1 triples with explicit code mappings, preserving roles and evidence."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_data/cbdb/2026.09.14'
DIRECTIONS = {
    'to_parent', 'to_child', 'to_ancestor', 'to_descendant', 'to_teacher',
    'to_student', 'to_superior', 'to_subordinate', 'actor_to_recipient',
    'recipient_to_actor', 'peer', 'unspecified', 'to_examiner', 'to_examinee',
    'to_influencer', 'to_follower', 'to_writer', 'to_requester', 'to_patron', 'to_client',
}
INVERSE_DIRECTIONS = {
    'to_parent': 'to_child', 'to_child': 'to_parent',
    'to_ancestor': 'to_descendant', 'to_descendant': 'to_ancestor',
    'to_teacher': 'to_student', 'to_student': 'to_teacher',
    'to_superior': 'to_subordinate', 'to_subordinate': 'to_superior',
    'actor_to_recipient': 'recipient_to_actor',
    'recipient_to_actor': 'actor_to_recipient', 'peer': 'peer',
    'unspecified': 'unspecified', 'to_examiner': 'to_examinee',
    'to_examinee': 'to_examiner', 'to_influencer': 'to_follower',
    'to_follower': 'to_influencer', 'to_writer': 'to_requester',
    'to_requester': 'to_writer', 'to_patron': 'to_client', 'to_client': 'to_patron',
}
NODE_ATTRIBUTES = ('NameZh', 'Birth', 'Death', 'Occupation', 'OccupationSource')


def load_mapping(path: Path):
    config = json.loads(path.read_text(encoding='utf-8'))
    mapping = {}
    for entry in config['relations']:
        relation = entry['relation']
        if relation in mapping:
            raise ValueError(f'Duplicate mapping: {relation}')
        if entry['group'] not in config['groups']:
            raise ValueError(f'Unknown group: {entry}')
        if entry['direction'] not in DIRECTIONS:
            raise ValueError(f'Unknown direction: {entry}')
        if not entry['reason'].strip():
            raise ValueError(f'Missing reason: {relation}')
        mapping[relation] = entry
    return config, mapping


def write_tsv(path: Path, fields, rows):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows, mapping, groups):
    """Fail on unknown codes or conflicting node attributes; never swap endpoints."""
    nodes, aggregates, annotated = {}, {}, []
    original_keys = set()
    for row in rows:
        relation = row['Relation']
        if relation not in mapping:
            raise ValueError(f'Unmapped raw relation: {relation}')
        if row['Node1'] == row['Node2']:
            raise ValueError(f'Self loop in source: {row}')
        raw_key = (row['Node1'], relation, row['Node2'])
        if raw_key in original_keys:
            raise ValueError(f'Duplicate source triple: {raw_key}')
        original_keys.add(raw_key)
        support = int(row['SupportRows'])
        if support <= 0:
            raise ValueError(f'Nonpositive support: {raw_key}')
        entry = mapping[relation]
        model_relation = entry['group'] + '__' + entry['direction']
        new_row = dict(row)
        new_row.update(
            OriginalRelationGroup=row['RelationGroup'],
            RelationGroup=entry['group'],
            RelationGroupLabelZh=groups[entry['group']]['label_zh'],
            RelationFamily=groups[entry['group']]['family'],
            Direction=entry['direction'], ModelRelation=model_relation,
            RelationQualifier=entry.get('qualifier', ''),
        )
        annotated.append(new_row)
        for endpoint in ('Node1', 'Node2'):
            node = row[endpoint]
            attributes = {a: row[f'{endpoint}_{a}'] for a in NODE_ATTRIBUTES}
            if not attributes['Occupation'] or not (attributes['Birth'] or attributes['Death']):
                raise ValueError(f'Unqualified source node: {node}')
            if node in nodes and nodes[node] != attributes:
                raise ValueError(f'Conflicting source node attributes: {node}')
            nodes[node] = attributes
        key = (row['Node1'], model_relation, row['Node2'])
        if key not in aggregates:
            aggregate = {k: v for k, v in new_row.items() if k.startswith(('Node1', 'Node2'))}
            aggregate.update(
                Relation=model_relation, RelationGroup=entry['group'],
                RelationGroupLabelZh=groups[entry['group']]['label_zh'],
                RelationFamily=groups[entry['group']]['family'],
                Direction=entry['direction'], SupportRows=0, RawTripleCount=0,
                OriginalRelations=set(), RelationQualifiers=set(),
            )
            aggregates[key] = aggregate
        aggregate = aggregates[key]
        aggregate['SupportRows'] += support
        aggregate['RawTripleCount'] += 1
        aggregate['OriginalRelations'].add(relation)
        if entry.get('qualifier'):
            aggregate['RelationQualifiers'].add(entry['qualifier'])
    grouped = []
    for key in sorted(aggregates):
        row = aggregates[key]
        for field in ('OriginalRelations', 'RelationQualifiers'):
            row[field] = json.dumps(sorted(row[field]), ensure_ascii=False)
        grouped.append(row)
    return annotated, grouped, nodes


def code_inventory(conn):
    inventory = {}
    for code, label, path, inverse1, inverse2 in conn.execute(
        'SELECT c_kincode,c_kinrel_chn,c_kinrel_simplified,c_kin_pair1,c_kin_pair2 '
        'FROM KINSHIP_CODES WHERE c_kincode>0'
    ):
        inventory[f'KIN:{code}'] = dict(
            label_zh=label, official_subtype='', simplified_path=path,
            inverse_relations=[f'KIN:{i}' for i in (inverse1, inverse2) if i and i>0],
        )
    for code, label, inverse, role, subtype in conn.execute(
        'SELECT c.c_assoc_code,c.c_assoc_desc_chn,c.c_assoc_pair,c.c_assoc_role_type,'
        'r.c_assoc_type_code FROM ASSOC_CODES c LEFT JOIN ASSOC_CODE_TYPE_REL r '
        'ON c.c_assoc_code=r.c_assoc_code WHERE c.c_assoc_code>0'
    ):
        inventory[f'ASSOC:{code}'] = dict(
            label_zh=label, official_subtype=subtype or '', simplified_path='',
            inverse_relations=[f'ASSOC:{inverse}'] if inverse and inverse>0 else [],
            official_role=role or '',
        )
    return inventory


def inverse_audit(mapping, inventory):
    issues = []
    for relation, entry in sorted(mapping.items()):
        candidates = [i for i in inventory[relation]['inverse_relations'] if i in mapping]
        if not candidates:
            continue
        if any(mapping[i]['group'] == entry['group'] and
               mapping[i]['direction'] == INVERSE_DIRECTIONS[entry['direction']]
               for i in candidates):
            continue
        issues.append(dict(
            relation=relation, label_zh=inventory[relation]['label_zh'],
            group=entry['group'], direction=entry['direction'],
            candidates=json.dumps([
                {'relation': i, 'label_zh': inventory[i]['label_zh'],
                 'group': mapping[i]['group'], 'direction': mapping[i]['direction']}
                for i in candidates
            ], ensure_ascii=False),
        ))
    return issues


def export(source: Path, mapping_path: Path, database: Path, output: Path):
    config, mapping = load_mapping(mapping_path)
    conn = sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True)
    try:
        inventory = code_inventory(conn)
    finally:
        conn.close()
    if set(inventory) != set(mapping):
        raise ValueError(f'Mapping/code-table mismatch: missing={sorted(set(inventory)-set(mapping))}, '
                         f'extra={sorted(set(mapping)-set(inventory))}')
    with gzip.open(source, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        original_fields = list(reader.fieldnames)
        annotated, grouped, nodes = group_rows(reader, mapping, config['groups'])
    if not annotated:
        raise ValueError('Empty source graph')
    output.mkdir(parents=True, exist_ok=True)
    extra = ['OriginalRelationGroup', 'RelationGroupLabelZh', 'RelationFamily',
             'Direction', 'ModelRelation', 'RelationQualifier']
    write_tsv(output/'person_relation_triples_annotated.tsv.gz', original_fields+extra, annotated)
    grouped_fields = ['Node1', 'Relation', 'Node2', 'RelationGroup', 'RelationGroupLabelZh',
                      'RelationFamily', 'Direction']
    grouped_fields += [f'{endpoint}_{a}' for endpoint in ('Node1', 'Node2') for a in NODE_ATTRIBUTES]
    grouped_fields += ['SupportRows', 'RawTripleCount', 'OriginalRelations', 'RelationQualifiers']
    write_tsv(output/'person_relation_triples_grouped.tsv.gz', grouped_fields, grouped)
    write_tsv(output/'nodes.tsv.gz', ['Node', *NODE_ATTRIBUTES],
              [dict(Node=node, **nodes[node]) for node in sorted(nodes)])
    raw_counts = Counter(row['Relation'] for row in annotated)
    raw_support = Counter()
    for row in annotated:
        raw_support[row['Relation']] += int(row['SupportRows'])
    mapping_rows = []
    for relation, entry in sorted(mapping.items()):
        meta = inventory[relation]
        mapping_rows.append(dict(
            relation=relation, label_zh=meta['label_zh'],
            official_subtype=meta['official_subtype'], simplified_path=meta['simplified_path'],
            inverse_relations=json.dumps(meta['inverse_relations']),
            group=entry['group'], group_label_zh=config['groups'][entry['group']]['label_zh'],
            family=config['groups'][entry['group']]['family'], direction=entry['direction'],
            qualifier=entry.get('qualifier', ''), reason=entry['reason'],
            input_triples=raw_counts[relation], support_rows=raw_support[relation],
        ))
    write_tsv(output/'relation_mapping.tsv', list(mapping_rows[0]), mapping_rows)
    issues = inverse_audit(mapping, inventory)
    exceptions = config.get('inverse_audit_exceptions', {})
    unexpected = [r['relation'] for r in issues if r['relation'] not in exceptions]
    if unexpected:
        raise ValueError(f'Unreviewed inverse mapping conflicts: {unexpected}')
    for issue in issues:
        issue['input_triples'] = raw_counts[issue['relation']]
        issue['review_reason'] = exceptions[issue['relation']]
    write_tsv(output/'inverse_code_audit.tsv',
              ['relation', 'label_zh', 'group', 'direction', 'candidates', 'input_triples', 'review_reason'], issues)
    group_rows_stats, directional_stats = [], []
    for group, definition in sorted(config['groups'].items()):
        raw = [r for r in annotated if r['RelationGroup'] == group]
        collapsed = [r for r in grouped if r['RelationGroup'] == group]
        endpoints = {r[e] for r in raw for e in ('Node1', 'Node2')}
        group_rows_stats.append(dict(
            group=group, label_zh=definition['label_zh'], family=definition['family'],
            input_raw_codes=len({r['Relation'] for r in raw}), input_triples=len(raw),
            grouped_triples=len(collapsed), support_rows=sum(int(r['SupportRows']) for r in raw),
            people=len(endpoints), grouped_share_pct=round(100*len(collapsed)/len(grouped), 6),
            model_relation_count=len({r['Relation'] for r in collapsed}),
        ))
        for model_relation in sorted({r['Relation'] for r in collapsed}):
            selected = [r for r in collapsed if r['Relation'] == model_relation]
            directional_stats.append(dict(
                relation=model_relation, group=group, direction=selected[0]['Direction'],
                grouped_triples=len(selected), support_rows=sum(r['SupportRows'] for r in selected),
                raw_codes=len({code for r in selected for code in json.loads(r['OriginalRelations'])}),
            ))
    write_tsv(output/'group_support.tsv', list(group_rows_stats[0]), group_rows_stats)
    write_tsv(output/'directional_relation_support.tsv', list(directional_stats[0]), directional_stats)
    vocabulary = {r['relation']: i for i, r in enumerate(sorted(directional_stats, key=lambda r:r['relation']))}
    (output/'relation_vocabulary.json').write_text(
        json.dumps({'relation_to_id': vocabulary, 'generated_reverse_edges': False},
                   ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    taxonomy = {group: [r['relation'] for r in directional_stats if r['group']==group]
                for group in config['groups']}
    (output/'relation_group_taxonomy.json').write_text(
        json.dumps({'name': config['name'], 'version': config['version'],
                    'groups': {g:rs for g,rs in taxonomy.items() if rs}},
                   ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    summary = dict(
        name=config['name'], version=config['version'], source=str(source), mapping=str(mapping_path),
        database=str(database), mapped_positive_codes=len(mapping), input_triples=len(annotated),
        input_raw_relation_codes=len(raw_counts), output_grouped_triples=len(grouped),
        collapsed_parallel_triples=len(annotated)-len(grouped), output_people=len(nodes),
        input_support_rows=sum(int(r['SupportRows']) for r in annotated),
        output_support_rows=sum(r['SupportRows'] for r in grouped),
        configured_semantic_groups=len(config['groups']),
        observed_semantic_groups=len({r['RelationGroup'] for r in grouped}),
        observed_directional_model_relations=len(directional_stats),
        family_grouped_triples=dict(Counter(r['RelationFamily'] for r in grouped)),
        output_node_occupations=dict(Counter(n['Occupation'] for n in nodes.values())),
        inverse_code_audit_issues=len(issues), period_assignment='not performed',
        occupation_status='unchanged provisional v1; not validated by relation grouping',
        group_support=group_rows_stats,
    )
    if summary['input_support_rows'] != summary['output_support_rows']:
        raise AssertionError('Support conservation failed')
    if sum(r['RawTripleCount'] for r in grouped) != len(annotated):
        raise AssertionError('Raw triple conservation failed')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA/'flat_triples_v1/person_relation_triples.tsv.gz')
    parser.add_argument('--mapping', type=Path, default=ROOT/'config/cbdb_relation_groups_v1.json')
    parser.add_argument('--database', type=Path, default=DATA/'cbdb_20260914.sqlite3')
    parser.add_argument('--output-dir', type=Path, default=DATA/'grouped_relations_v1')
    args = parser.parse_args()
    source = args.input.resolve()
    if args.output_dir.resolve() == source.parent:
        parser.error('Output directory must differ from the source directory')
    summary = export(source, args.mapping, args.database, args.output_dir)
    print(json.dumps({k:v for k,v in summary.items() if k!='group_support'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
