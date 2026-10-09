"""Apply reviewed BHHT L1 mappings with explicitly temporary v2 selections."""
from collections import Counter
import csv
import json
from pathlib import Path

POLICY = 'bhht_first_reviewed_other_fallback'
VERSION = 'freebase_bhht_provisional_single_l1_v2'
LEVELS = {'Culture', 'Discovery/Science', 'Leadership', 'Sports/Games', 'Other'}


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t' if str(path).endswith('.tsv') else ',')


def load_crosswalk(path):
    mapping = {}
    for source in rows(path):
        raw, level, state = source['raw_value'], source['level1'], source['review_status']
        if not raw or raw in mapping:
            raise ValueError(f'Empty/duplicate BHHT occupation: {raw!r}')
        if state not in {'resolved', 'retained_exact_unique', 'needs_context'}:
            raise ValueError(f'Unexpected BHHT review status: {raw!r} {state!r}')
        if (state == 'needs_context' and level) or (state != 'needs_context' and level not in LEVELS):
            raise ValueError(f'Inconsistent BHHT L1/status: {raw!r}')
        record = dict(source)
        record.update(training_level1=level or 'Other',
                      temporary_other_fallback=str(int(not level)),
                      mapping_application='reviewed_or_author_l1' if level else 'temporary_unresolved_to_other',
                      dataset_version=VERSION,
                      training_label_status='user_authorized_provisional_not_ground_truth')
        mapping[raw] = record
    if not mapping:
        raise ValueError('Empty BHHT occupation crosswalk')
    return mapping


def select(raw, mapping):
    if not raw or len(raw) != len(set(raw)):
        raise ValueError('Expected a nonempty, unique recorded occupation array')
    missing = set(raw) - mapping.keys()
    if missing:
        raise ValueError(f'Occupations missing from BHHT crosswalk: {sorted(missing)}')
    known = [value for value in raw if mapping[value]['level1']]
    unresolved = [value for value in raw if not mapping[value]['level1']]
    reviewed_l1 = sorted({mapping[v]['level1'] for v in known})
    effective_l1 = sorted({mapping[v]['training_level1'] for v in raw})
    original = 'review_incomplete_mapping' if unresolved else (
        'review_unique_l1' if len(reviewed_l1) == 1 else 'review_multiple_l1')
    chosen = known[0] if known else raw[0]
    if not known:
        reason = 'temporary_other_all_professions_unresolved'
    elif len(effective_l1) > 1:
        reason = 'forced_first_reviewed_profession'
    elif unresolved:
        reason = 'unique_after_other_fallback'
    else:
        reason = 'preserved_unique_reviewed_l1'
    return {
        'selected_raw_profession': chosen,
        'temporary_target_label': mapping[chosen]['training_level1'],
        'selection_reason': reason,
        'original_resolution_status': original,
        'provisional_resolution_status': 'provisional_unique_l1' if len(effective_l1) == 1 else 'provisional_multiple_l1',
        'reviewed_l1_set_json': json.dumps(reviewed_l1, ensure_ascii=False),
        'provisional_l1_set_json': json.dumps(effective_l1, ensure_ascii=False),
        'unresolved_professions_json': json.dumps(unresolved, ensure_ascii=False),
        'raw_semantic_l1_sequence_json': json.dumps([mapping[v]['level1'] for v in raw], ensure_ascii=False),
        'raw_provisional_l1_sequence_json': json.dumps([mapping[v]['training_level1'] for v in raw], ensure_ascii=False),
        'selected_mapping_status': mapping[chosen]['review_status'],
        'selected_mapping_confidence': mapping[chosen]['confidence'],
        'selected_mapping_source': mapping[chosen]['mapping_source'],
        'target_label_is_temporary_fallback': int(not known),
        'has_any_temporary_other_profession': int(bool(unresolved)),
        'label_policy': POLICY,
        'label_status': 'user_authorized_provisional_not_ground_truth',
    }


def load_node_tables(nodes_path, crosswalk_path, *, maximum_year, date_parser):
    mapping = load_crosswalk(crosswalk_path)
    nodes, audit, dates = [], [], []
    seen, frequencies = set(), Counter()
    for r in rows(nodes_path):
        name = r['Name']
        if not name or name in seen:
            raise ValueError(f'Empty/duplicate Freebase person: {name!r}')
        seen.add(name)
        raw = json.loads(r['Professions'])
        decision = select(raw, mapping)
        frequencies.update(raw)
        birth, birth_state = date_parser(r['BirthYears'], maximum_year)
        death, death_state = date_parser(r['DeathYears'], maximum_year)
        nodes.append({
            'node_id': name, 'birth_year': birth, 'death_year': death,
            'occupation_level1': decision['temporary_target_label'],
            'occupation_level2': '', 'occupation_level3': '', 'country': '',
            'freebase_id': r['FreebaseID'], 'freebase_id_status': r['IDStatus'],
            'raw_professions_json': r['Professions'],
            'selected_raw_profession': decision['selected_raw_profession'],
            'label_selection_reason': decision['selection_reason'],
            'original_resolution_status': decision['original_resolution_status'],
            'provisional_l1_set_json': decision['provisional_l1_set_json'],
            'target_label_is_temporary_fallback': decision['target_label_is_temporary_fallback'],
        })
        audit.append({'person_name': name, 'freebase_id': r['FreebaseID'],
                      'raw_professions_json': r['Professions'], **decision})
        dates.append({'person_name': name, 'birth_years_json': r['BirthYears'],
                      'death_years_json': r['DeathYears'], 'birth_status': birth_state,
                      'death_status': death_state,
                      'death_before_birth': bool(birth != '' and death != '' and death < birth)})
    expected = Counter({v: int(r['graph_people']) for v, r in mapping.items()})
    if frequencies != expected:
        raise ValueError('BHHT crosswalk population/frequencies differ from the supplied Freebase graph')
    crosswalk = list(mapping.values())
    details = {
        'dataset_version': VERSION,
        'occupation_crosswalk_summary': {
            'professions': len(mapping),
            'reviewed_or_author_professions': sum(bool(r['level1']) for r in crosswalk),
            'formal_other_professions': sum(r['level1'] == 'Other' for r in crosswalk),
            'temporary_other_professions': sum(not r['level1'] for r in crosswalk),
        },
        'person_resolution_after_other_fallback': dict(Counter(r['provisional_resolution_status'] for r in audit)),
        'people_with_any_unresolved_profession_before_fallback': sum(r['has_any_temporary_other_profession'] for r in audit),
        'people_assigned_temporary_other_without_any_reviewed_l1': sum(r['target_label_is_temporary_fallback'] for r in audit),
        'person_profession_pairs': sum(frequencies.values()),
        'selection_order_note': 'First original-array occupation with a reviewed/author L1, including formal Other; if none, first raw occupation with temporary Other. Array order is not occupational importance.',
        'level2_policy': 'deferred_and_missing_in_training',
    }
    return nodes, audit, dates, crosswalk, details
