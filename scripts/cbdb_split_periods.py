#!/usr/bin/env python3
"""Export CBDB period-induced graphs without interpolating missing life dates."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cbdb_build_experiment_groups import DATA, export_training_csv, write_json
from scripts.cbdb_group_relations import NODE_ATTRIBUTES, write_tsv

REPRESENTATIONS = ('multi_group', 'binary')
ASSIGNMENT = 'any_known_birth_or_death'


def load_period_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(config, dict) or not isinstance(config.get('name'), str)
            or not config['name'].strip() or type(config.get('version')) is not int
            or config['version'] < 1 or config.get('assignment') != ASSIGNMENT):
        raise ValueError('Period config requires name, positive integer version, and '
                         f'assignment={ASSIGNMENT}')
    periods = config.get('periods')
    if not isinstance(periods, list) or not periods:
        raise ValueError('Period config must contain a nonempty periods list')
    seen = set()
    for index, period in enumerate(periods):
        if not isinstance(period, dict) or set(period) != {'id', 'label_zh', 'start', 'end'}:
            raise ValueError('Each period requires exactly id, label_zh, start, end')
        identifier = period['id']
        if (not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', identifier)
                or identifier in seen):
            raise ValueError(f'Invalid or duplicate period id: {identifier!r}')
        seen.add(identifier)
        if not isinstance(period['label_zh'], str) or not period['label_zh'].strip():
            raise ValueError(f'Missing label for {identifier}')
        for field in ('start', 'end'):
            if period[field] is not None and type(period[field]) is not int:
                raise ValueError(f'{identifier}.{field} must be an integer or null')
        start, end = period['start'], period['end']
        if (start is None) != (index == 0) or (end is None) != (index == len(periods)-1):
            raise ValueError('Only first start and last end must be null (unbounded coverage)')
        if start is not None and end is not None and start > end:
            raise ValueError(f'Reversed period bounds: {identifier}')
        if index and start != periods[index-1]['end']+1:
            raise ValueError('Period bounds must be consecutive: gaps and overlaps are forbidden')
    return config


def select_periods(config, selection='all'):
    if selection == 'all':
        return config['periods']
    requested = selection.split(',') if isinstance(selection, str) else list(selection)
    if not requested or any(not x or x != x.strip() for x in requested) or len(set(requested)) != len(requested):
        raise ValueError('Select all or a comma-separated list of distinct period IDs')
    known = {period['id'] for period in config['periods']}
    if set(requested)-known:
        raise ValueError(f'Unknown period IDs: {sorted(set(requested)-known)}')
    return [p for p in config['periods'] if p['id'] in requested]


def known_year(value):
    """Zero is the CBDB missing-year sentinel; negative years remain intact."""
    if value in ('', None):
        return None
    year = int(value)
    return None if year == 0 else year


def belongs_to_period(node, period):
    years = [known_year(node[field]) for field in ('Birth', 'Death')]
    return any(year is not None
               and (period['start'] is None or year >= period['start'])
               and (period['end'] is None or year <= period['end']) for year in years)


def read_tsv(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(fields) != len(set(fields)) or any(None in row or None in row.values() for row in rows):
        raise ValueError(f'Malformed TSV: {path}')
    return fields, rows


def loader_node_order(rows):
    return list(dict.fromkeys([row['Node1'] for row in rows]+[row['Node2'] for row in rows]))


def _load_source(input_dir):
    node_fields, node_rows = read_tsv(input_dir/'nodes.tsv.gz')
    if not {'Node', *NODE_ATTRIBUTES}.issubset(node_fields) or not node_rows:
        raise ValueError('Source nodes table is empty or missing required node attributes')
    nodes = {}
    for row in node_rows:
        identifier = row['Node']
        birth, death = known_year(row['Birth']), known_year(row['Death'])
        if (not identifier or identifier in nodes or not row['Occupation']
                or (birth is None and death is None) or (birth is not None and death is not None and birth > death)):
            raise ValueError(f'Duplicate or unqualified source node: {identifier}')
        nodes[identifier] = row
    graphs = {}
    for representation in REPRESENTATIONS:
        fields, rows = read_tsv(input_dir/representation/'person_relation_triples.tsv.gz')
        required = {'Node1', 'Node2', 'Relation', 'TieClass', 'SupportRows', 'RawTripleCount'}
        required.update(f'{endpoint}_{field}' for endpoint in ('Node1', 'Node2') for field in NODE_ATTRIBUTES)
        if not required.issubset(fields) or not rows:
            raise ValueError(f'Empty source graph or missing columns: {representation}')
        seen = set()
        for row in rows:
            key = row['Node1'], row['Relation'], row['Node2']
            if key in seen or key[0] == key[2] or not key[1] or key[1].endswith('__rev'):
                raise ValueError(f'Duplicate, self-loop, or invalid base triple: {key}')
            seen.add(key)
            if (int(row['SupportRows']) <= 0 or int(row['RawTripleCount']) <= 0
                    or row['TieClass'] not in ('inherited', 'acquired')):
                raise ValueError(f'Invalid support or tie class: {key}')
            for endpoint in ('Node1', 'Node2'):
                node = nodes.get(row[endpoint])
                if node is None or any(row[f'{endpoint}_{field}'] != node[field] for field in NODE_ATTRIBUTES):
                    raise ValueError(f'Missing node or conflicting endpoint attributes: {key}')
        rows.sort(key=lambda row:(row['Node1'], row['Node2'], row['Relation']))
        graphs[representation] = (fields, rows)
    if loader_node_order(graphs['multi_group'][1]) != loader_node_order(graphs['binary'][1]):
        raise ValueError('Binary and multi-group source graphs must have identical node ordering')
    if set(loader_node_order(graphs['multi_group'][1])) != set(nodes):
        raise ValueError('Source nodes table must exactly cover graph endpoints')
    return node_fields, nodes, graphs


def _validate_output(input_dir, output, overwrite):
    source, target = input_dir.resolve(), output.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError('Input and output directories must be separate, non-nested paths')
    if output.is_symlink():
        raise ValueError('Output directory must not be a symbolic link')
    if output.exists():
        if not output.is_dir():
            raise ValueError(f'Output is not a directory: {output}')
        if any(output.iterdir()):
            marker = output/'summary.json'
            if not overwrite:
                raise ValueError(f'Output is nonempty; use --overwrite to regenerate: {output}')
            if not marker.is_file() or json.loads(marker.read_text()).get('operation') != 'cbdb_split_periods':
                raise ValueError('Refusing to overwrite a directory not generated by cbdb_split_periods')


def export_periods(input_dir, period_config, output_dir, selection='all', overwrite=False):
    input_dir, period_config, output_dir = map(Path, (input_dir, period_config, output_dir))
    _validate_output(input_dir, output_dir, overwrite)
    if output_dir.resolve() in period_config.resolve().parents:
        raise ValueError('Period config must be outside the output directory')
    config = load_period_config(period_config)
    periods = select_periods(config, selection)
    node_fields, nodes, graphs = _load_source(input_dir)
    membership = {p['id']:{n for n, row in nodes.items() if belongs_to_period(row, p)} for p in periods}
    assigned_people = set().union(*membership.values())
    assignment_count = Counter(n for members in membership.values() for n in members)
    representations = {}
    summary_rows, period_summaries = [], {}
    # Validate all selected graphs before creating any output.
    selected_graphs = {}
    for representation, (fields, rows) in graphs.items():
        selected_graphs[representation] = {}
        multiplicity = Counter()
        for period in periods:
            identifier = period['id']
            selected = [row for row in rows if row['Node1'] in membership[identifier]
                        and row['Node2'] in membership[identifier]]
            if not selected:
                raise ValueError(f'Empty period graph: {identifier}/{representation}; choose supported periods')
            tie_names = {row['TieClass'] for row in selected}
            if tie_names != {'inherited', 'acquired'}:
                raise ValueError(f'{identifier}/{representation} lacks inherited or acquired support required by training')
            selected_graphs[representation][identifier] = selected
            multiplicity.update((r['Node1'], r['Relation'], r['Node2']) for r in selected)
        representations[representation] = dict(
            input_triples=len(rows), emitted_triples=sum(multiplicity.values()),
            unique_retained_triples=len(multiplicity), excluded_triples=len(rows)-len(multiplicity),
            triples_in_multiple_periods=sum(count > 1 for count in multiplicity.values()),
            repeated_triple_copies=sum(count-1 for count in multiplicity.values()),
            input_support_rows=sum(int(r['SupportRows']) for r in rows),
            retained_support_rows=sum(int(r['SupportRows']) for r in rows
                if (r['Node1'], r['Relation'], r['Node2']) in multiplicity),
            emitted_support_rows=sum(int(r['SupportRows'])*multiplicity[(r['Node1'], r['Relation'], r['Node2'])]
                                     for r in rows))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.stage-', dir=output_dir.parent))
    try:
        for period in periods:
            identifier = period['id']
            period_dir = staging/identifier
            period_dir.mkdir()
            eligible = [nodes[n] for n in sorted(membership[identifier])]
            write_tsv(period_dir/'eligible_nodes.tsv.gz', node_fields, eligible)
            period_summary = dict(period=period, eligible_nodes=len(eligible),
                eligible_occupations=dict(Counter(n['Occupation'] for n in eligible)), representations={})
            canonical_order = None
            for representation, (fields, rows) in graphs.items():
                selected = selected_graphs[representation][identifier]
                order = loader_node_order(selected)
                if canonical_order is not None and order != canonical_order:
                    raise AssertionError(f'Period graph node ordering differs: {identifier}')
                canonical_order = order
                directory = period_dir/representation
                directory.mkdir()
                active = [nodes[n] for n in order]
                write_tsv(directory/'eligible_nodes.tsv.gz', node_fields, eligible)
                write_tsv(directory/'active_nodes.tsv.gz', node_fields, active)
                write_tsv(directory/'person_relation_triples.tsv.gz', fields, selected)
                export_training_csv(directory/'Q_R_Q_extended.csv.gz', selected)
                relations = sorted({r['Relation'] for r in selected})
                ties = {tie:sorted({r['Relation'] for r in selected if r['TieClass'] == tie})
                        for tie in ('inherited', 'acquired')}
                if set(ties['inherited']) & set(ties['acquired']):
                    raise ValueError('A relation cannot have conflicting inherited/acquired classes')
                vocabulary = {relation:i for i, relation in enumerate(sorted(
                    r for base in relations for r in (base, base+'__rev')))}
                name = f"{config['name']}_{identifier}_{representation}"
                write_json(directory/'relation_taxonomy.json', dict(name=name, version=config['version'],
                           groups={r:[r] for r in relations}))
                write_json(directory/'tie_taxonomy.json', dict(name=name+'_ties', version=config['version'], groups=ties))
                write_json(directory/'relation_vocabulary.json', dict(relation_to_id=vocabulary,
                           convention='Base groups plus preparer-generated __rev; no fine role suffixes'))
                info = dict(eligible_nodes=len(eligible), active_nodes=len(active),
                    isolated_eligible_nodes=len(eligible)-len(active), grouped_triples=len(selected),
                    excluded_input_triples=len(rows)-len(selected),
                    base_relation_groups=len(relations), model_relation_types_after_reverse=2*len(relations),
                    message_edges_after_reverse_and_deduplication=2*len(selected),
                    support_rows=sum(int(r['SupportRows']) for r in selected),
                    raw_triple_count=sum(int(r['RawTripleCount']) for r in selected),
                    relation_counts=dict(Counter(r['Relation'] for r in selected)),
                    active_occupations=dict(Counter(n['Occupation'] for n in active)))
                period_summary['representations'][representation] = info
                write_json(directory/'summary.json', dict(period=period, representation=representation, **info))
                summary_rows.append(dict(period=identifier, label_zh=period['label_zh'], representation=representation,
                    **{field:info[field] for field in ('eligible_nodes','active_nodes','isolated_eligible_nodes',
                       'grouped_triples','base_relation_groups','model_relation_types_after_reverse','support_rows')},
                    eligible_occupations=json.dumps(period_summary['eligible_occupations'], ensure_ascii=False, sort_keys=True),
                    active_occupations=json.dumps(info['active_occupations'], ensure_ascii=False, sort_keys=True),
                    relation_counts=json.dumps(info['relation_counts'], ensure_ascii=False, sort_keys=True)))
            write_tsv(period_dir/'active_nodes.tsv.gz', node_fields, [nodes[n] for n in canonical_order])
            write_json(period_dir/'summary.json', period_summary)
            period_summaries[identifier] = period_summary
        summary = dict(operation='cbdb_split_periods', name=config['name'], version=config['version'],
            assignment=ASSIGNMENT, period_config=str(period_config.resolve()), input_dir=str(input_dir.resolve()),
            selected_periods=[p['id'] for p in periods], input_people=len(nodes),
            unique_assigned_people=len(assigned_people), excluded_people=len(nodes)-len(assigned_people),
            people_in_multiple_periods=sum(c > 1 for c in assignment_count.values()),
            repeated_person_copies=sum(c-1 for c in assignment_count.values()),
            emitted_eligible_people=sum(len(s) for s in membership.values()),
            representations=representations, periods=period_summaries,
            edge_policy='Both endpoints must belong to the period; source triples and attributes retained exactly',
            train_val_test_split='not performed',
            overlap_policy='Cross-period people/edges can repeat; periods are not independent cohorts')
        write_json(staging/'summary.json', summary)
        write_tsv(staging/'period_summary.tsv', list(summary_rows[0]), summary_rows)
        write_json(staging/'period_config.json', config)
        # Stage first so malformed input never leaves a partially exported graph.
        backup = None
        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.old-', dir=output_dir.parent))
            backup.rmdir()
            output_dir.rename(backup)
        try:
            staging.rename(output_dir)
        except BaseException:
            if backup is not None:
                backup.rename(output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DATA/'experiment_groups_v1')
    parser.add_argument('--period-config', type=Path, default=ROOT/'config/cbdb_periods_v1.json')
    parser.add_argument('--output-dir', type=Path, default=DATA/'periods_v1')
    parser.add_argument('--periods', default='all', help='all or comma-separated period IDs')
    parser.add_argument('--overwrite', action='store_true', help='Replace only a previously generated period directory')
    args = parser.parse_args()
    try:
        summary = export_periods(args.input_dir, args.period_config, args.output_dir, args.periods, args.overwrite)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({k:summary[k] for k in ('operation','selected_periods','input_people','representations')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
