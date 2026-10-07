"""Export main-experiment-style broad CBDB groups and binary controls."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cbdb_group_relations import NODE_ATTRIBUTES, load_mapping, write_tsv

DATA = ROOT / 'external_data/cbdb/2026.09.14'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def resolve_groups(profile, fine_config, fine_mapping):
    fine_to_broad = {}
    for group, definition in profile['groups'].items():
        for fine_group in definition['fine_groups']:
            if fine_group in fine_to_broad:
                raise ValueError(f'Overlapping fine group: {fine_group}')
            fine_to_broad[fine_group] = group
    if set(fine_to_broad) != set(fine_config['groups']):
        raise ValueError('Experiment profile must explicitly cover every fine group exactly once')
    overrides = profile['code_overrides']
    for code, entry in overrides.items():
        if code not in fine_mapping or entry['group'] not in profile['groups'] or not entry['reason']:
            raise ValueError(f'Invalid code override: {code}')
    resolved = {
        code: overrides[code]['group'] if code in overrides else fine_to_broad[entry['group']]
        for code, entry in fine_mapping.items()
    }
    for code, group in resolved.items():
        if code.startswith('KIN:'):
            expected = ('intimate_partnership' if fine_mapping[code]['group']=='intimate_partnership'
                        else 'inherited')
            if group != expected:
                raise ValueError(f'KIN family/partner boundary violated: {code}')
    return resolved


def collapse_rows(rows, resolved, fine_mapping, binary=False):
    nodes, aggregates, seen = {}, {}, set()
    for row in rows:
        code = row['Relation']
        if code not in resolved:
            raise ValueError(f'Unmapped source relation: {code}')
        key = (row['Node1'], code, row['Node2'])
        if key in seen or row['Node1']==row['Node2']:
            raise ValueError(f'Duplicate or self-loop source triple: {key}')
        seen.add(key)
        support = int(row['SupportRows'])
        if support <= 0:
            raise ValueError('Source support must be positive')
        for endpoint in ('Node1', 'Node2'):
            node = row[endpoint]
            attributes = {a:row[f'{endpoint}_{a}'] for a in NODE_ATTRIBUTES}
            if not attributes['Occupation'] or not (attributes['Birth'] or attributes['Death']):
                raise ValueError(f'Unqualified node: {node}')
            if node in nodes and nodes[node]!=attributes:
                raise ValueError(f'Conflicting node attributes: {node}')
            nodes[node] = attributes
        broad = resolved[code]
        tie = 'inherited' if broad=='inherited' else 'acquired'
        relation = tie if binary else broad
        collapsed_key = (row['Node1'], relation, row['Node2'])
        fine = fine_mapping[code]
        if collapsed_key not in aggregates:
            aggregate = {k:v for k,v in row.items() if k.startswith(('Node1','Node2'))}
            aggregate.update(Relation=relation, RelationGroup=relation, TieClass=tie,
                             SupportRows=0, RawTripleCount=0)
            for field in ('OriginalRelations','ExperimentGroups','FineGroups','FineDirections','RelationQualifiers'):
                aggregate[field] = set()
            aggregates[collapsed_key] = aggregate
        aggregate = aggregates[collapsed_key]
        aggregate['SupportRows'] += support
        aggregate['RawTripleCount'] += 1
        aggregate['OriginalRelations'].add(code)
        aggregate['ExperimentGroups'].add(broad)
        aggregate['FineGroups'].add(fine['group'])
        aggregate['FineDirections'].add(fine['direction'])
        if fine.get('qualifier'):
            aggregate['RelationQualifiers'].add(fine['qualifier'])
    output = []
    # The existing loader uses first appearance for node IDs. Order targets before
    # relation labels so binary and multi-group CSVs get identical node ordering.
    for key in sorted(aggregates, key=lambda k:(k[0],k[2],k[1])):
        row = aggregates[key]
        for field in ('OriginalRelations','ExperimentGroups','FineGroups','FineDirections','RelationQualifiers'):
            row[field] = json.dumps(sorted(row[field]), ensure_ascii=False)
        output.append(row)
    return output, nodes


def export_training_csv(path, rows):
    """Bridge single-layer occupation to level1 only; never invent L2/L3 or country."""
    fields = ['Node1','Relation','Node2']
    for endpoint in ('Node1','Node2'):
        fields.extend(f'{endpoint}_{suffix}' for suffix in
                      ('Birth','Death','occ_level1','occ_level2','occ_level3','Country'))
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            exported = {k:row[k] for k in ('Node1','Relation','Node2')}
            for endpoint in ('Node1','Node2'):
                exported.update({
                    f'{endpoint}_Birth':row[f'{endpoint}_Birth'],
                    f'{endpoint}_Death':row[f'{endpoint}_Death'],
                    f'{endpoint}_occ_level1':row[f'{endpoint}_Occupation'],
                    f'{endpoint}_occ_level2':'', f'{endpoint}_occ_level3':'',
                    f'{endpoint}_Country':'',
                })
            writer.writerow(exported)


def export(source, profile_path, output):
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    fine_path = ROOT/profile['fine_mapping']
    fine_config, fine_mapping = load_mapping(fine_path)
    resolved = resolve_groups(profile, fine_config, fine_mapping)
    with gzip.open(source, 'rt', encoding='utf-8', newline='') as stream:
        raw = list(csv.DictReader(stream, delimiter='\t'))
    if not raw:
        raise ValueError('Empty source graph')
    observed = Counter(row['Relation'] for row in raw)
    unknown = set(observed)-set(resolved)
    if unknown:
        raise ValueError(f'Unmapped input codes: {sorted(unknown)}')
    multi, nodes = collapse_rows(raw, resolved, fine_mapping)
    binary, binary_nodes = collapse_rows(raw, resolved, fine_mapping, binary=True)
    if nodes != binary_nodes:
        raise AssertionError('Representations must preserve the same node attributes')
    support = sum(int(r['SupportRows']) for r in raw)
    for rows in (multi,binary):
        if sum(r['SupportRows'] for r in rows)!=support or sum(r['RawTripleCount'] for r in rows)!=len(raw):
            raise AssertionError('Source support/triple conservation failed')
    output.mkdir(parents=True, exist_ok=True)
    raw_groups = {g:[] for g in profile['groups']}
    for code in sorted(observed):
        raw_groups[resolved[code]].append(code)
    if any(not members for members in raw_groups.values()):
        raise ValueError('Every configured broad group must have observed support in the full graph')
    mapping_rows = [dict(
        relation=code, fine_group=fine_mapping[code]['group'], fine_direction=fine_mapping[code]['direction'],
        experiment_group=resolved[code], tie_class='inherited' if resolved[code]=='inherited' else 'acquired',
        input_triples=observed[code],
        rule=profile['code_overrides'].get(code,{}).get('reason','Explicit fine-group-to-experiment-group mapping'),
    ) for code in sorted(resolved)]
    write_tsv(output/'relation_mapping.tsv', list(mapping_rows[0]), mapping_rows)
    annotated = [dict(row, FineGroup=fine_mapping[row['Relation']]['group'],
                      FineDirection=fine_mapping[row['Relation']]['direction'],
                      ExperimentGroup=resolved[row['Relation']],
                      TieClass='inherited' if resolved[row['Relation']]=='inherited' else 'acquired')
                 for row in raw]
    write_tsv(output/'person_relation_triples_annotated.tsv.gz', list(annotated[0]), annotated)
    write_tsv(output/'nodes.tsv.gz', ['Node',*NODE_ATTRIBUTES],
              [dict(Node=node,**attributes) for node,attributes in sorted(nodes.items())])
    # Raw-code taxonomies can be used on a raw-vocabulary artifact with existing collapse-relations.
    write_json(output/'raw_relation_taxonomy.json', dict(name=profile['name']+'_raw',version=1,groups=raw_groups))
    write_json(output/'raw_tie_taxonomy.json', dict(name=profile['name']+'_raw_ties',version=1,
               groups={'inherited':raw_groups['inherited'],
                       'acquired':sorted(code for g,codes in raw_groups.items() if g!='inherited' for code in codes)}))
    support_rows = []
    representations = {}
    for name, rows in [('multi_group',multi),('binary',binary)]:
        directory = output/name
        directory.mkdir(exist_ok=True)
        write_tsv(directory/'person_relation_triples.tsv.gz', list(rows[0]), rows)
        export_training_csv(directory/'Q_R_Q_extended.csv.gz', rows)
        names = list(profile['groups']) if name=='multi_group' else ['inherited','acquired']
        # Mirror the standard preparer's convention; semantic role suffixes are absent.
        vocabulary = {r:i for i,r in enumerate(sorted([v for g in names for v in (g,g+'__rev')]))}
        write_json(directory/'relation_vocabulary.json',dict(relation_to_id=vocabulary,
                   convention='Base groups plus preparer-generated __rev; no fine role suffixes'))
        write_json(directory/'relation_taxonomy.json',dict(name=profile['name']+'_'+name,version=1,
                   groups={g:[g] for g in names}))
        write_json(directory/'tie_taxonomy.json',dict(name=profile['name']+'_'+name+'_ties',version=1,
                   groups={'inherited':['inherited'],'acquired':[g for g in names if g!='inherited']}))
        counts = Counter(r['Relation'] for r in rows)
        raw_counts = Counter(('inherited' if resolved[r['Relation']]=='inherited' else 'acquired')
                             if name=='binary' else resolved[r['Relation']] for r in raw)
        for group in names:
            selected = [r for r in rows if r['Relation']==group]
            endpoints = {r[e] for r in selected for e in ('Node1','Node2')}
            support_rows.append(dict(representation=name,group=group,
                label_zh=profile['groups'][group]['label_zh'] if name=='multi_group' else
                         ('亲属与家庭归属' if group=='inherited' else '其余社会关系'),
                input_triples=raw_counts[group],grouped_triples=counts[group],people=len(endpoints),
                support_rows=sum(r['SupportRows'] for r in selected)))
        representations[name]=dict(base_relation_groups=len(names),model_relation_types_after_reverse=2*len(names),
               grouped_triples=len(rows),message_edges_after_reverse_and_deduplication=2*len(rows),
               output_people=len(nodes),support_rows=support,collapsed_parallel_triples=len(raw)-len(rows),
               relation_counts=dict(counts))
    write_tsv(output/'group_support.tsv',list(support_rows[0]),support_rows)
    summary=dict(name=profile['name'],version=1,source=str(source),profile=str(profile_path),
                 fine_mapping=str(fine_path),mapped_positive_codes=len(resolved),observed_raw_codes=len(observed),
                 input_triples=len(raw),input_support_rows=support,output_people=len(nodes),
                 output_node_occupations=dict(Counter(n['Occupation'] for n in nodes.values())),
                 representations=representations,period_assignment='not performed',
                 train_val_test_split='not performed',occupation_status='unchanged provisional single-label v1')
    write_json(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DATA/'flat_triples_v1/person_relation_triples.tsv.gz')
    parser.add_argument('--profile',type=Path,default=ROOT/'config/cbdb_experiment_relation_groups_v1.json')
    parser.add_argument('--output-dir',type=Path,default=DATA/'experiment_groups_v1')
    args = parser.parse_args()
    if args.output_dir.resolve() in (args.input.resolve().parent,DATA/'grouped_relations_v1'):
        parser.error('Use a separate experiment output directory')
    summary = export(args.input,args.profile,args.output_dir)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
