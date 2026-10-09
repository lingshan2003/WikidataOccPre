"""Protect the reviewed-vs-temporary Other distinction and recorded selection order."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from Freebase.bhht_labels import load_crosswalk, load_node_tables, select
from Freebase.prepare_graph import single_year

ROOT = Path(__file__).resolve().parents[1]


def source(raw, label, count=1):
    return dict(raw_value=raw, level1=label, review_status='resolved' if label else 'needs_context',
                confidence='medium' if label else 'low', mapping_source='agent_semantic_review', graph_people=count)


def write(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), delimiter='\t' if path.suffix == '.tsv' else ',')
        w.writeheader(); w.writerows(records)


class BhhtV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'crosswalk.tsv'
        write(self.path, [source('Writer', 'Culture'), source('Physicist', 'Discovery/Science'),
                          source('Mechanic', 'Other'), source('Unknown', '')])
        self.mapping = load_crosswalk(self.path)

    def test_unknown_other_cannot_override_reviewed_occupation(self):
        r = select(['Unknown', 'Writer'], self.mapping)
        self.assertEqual((r['temporary_target_label'], r['selected_raw_profession']), ('Culture', 'Writer'))
        self.assertEqual(json.loads(r['provisional_l1_set_json']), ['Culture', 'Other'])
        self.assertEqual(r['target_label_is_temporary_fallback'], 0)
        self.assertEqual(r['has_any_temporary_other_profession'], 1)

    def test_formal_other_is_valid_and_preserves_original_array_order(self):
        r = select(['Mechanic', 'Writer'], self.mapping)
        self.assertEqual(r['temporary_target_label'], 'Other')
        self.assertEqual(r['target_label_is_temporary_fallback'], 0)
        self.assertEqual(select(['Writer', 'Mechanic'], self.mapping)['temporary_target_label'], 'Culture')

    def test_first_reviewed_role_is_deterministic_without_majority_voting(self):
        self.assertEqual(select(['Physicist', 'Writer'], self.mapping)['temporary_target_label'], 'Discovery/Science')
        self.assertEqual(select(['Writer', 'Physicist'], self.mapping)['temporary_target_label'], 'Culture')

    def test_all_unknown_maps_to_explicitly_temporary_other(self):
        r = select(['Unknown'], self.mapping)
        self.assertEqual(r['temporary_target_label'], 'Other')
        self.assertEqual(r['target_label_is_temporary_fallback'], 1)
        self.assertEqual(self.mapping['Unknown']['level1'], '')
        self.assertEqual(self.mapping['Unknown']['training_level1'], 'Other')
        self.assertEqual(self.mapping['Mechanic']['temporary_other_fallback'], '0')

    def test_missing_and_duplicate_professions_are_rejected(self):
        for raw in ([], ['Writer', 'Writer'], ['Unlisted']):
            with self.assertRaises(ValueError):
                select(raw, self.mapping)

    def test_inconsistent_source_review_cannot_be_reinterpreted_silently(self):
        write(self.path, [{**source('Unknown', ''), 'level1': 'Other'}])
        with self.assertRaises(ValueError):
            load_crosswalk(self.path)

    def test_actual_population_must_match_crosswalk_frequencies(self):
        nodes = Path(self.tmp.name) / 'nodes.csv'
        write(self.path, [source('Writer', 'Culture'), source('Unknown', '')])
        write(nodes, [dict(Name='A', FreebaseID='/m/a', IDStatus='unique', Professions='["Unknown", "Writer"]', BirthYears='[1900]', DeathYears='[]')])
        records, audit, dates, crosswalk, stats = load_node_tables(nodes, self.path, maximum_year=2026, date_parser=single_year)
        self.assertEqual(records[0]['occupation_level1'], 'Culture')
        self.assertEqual(stats['occupation_crosswalk_summary']['temporary_other_professions'], 1)
        self.assertEqual(audit[0]['raw_semantic_l1_sequence_json'], '["", "Culture"]')
        write(self.path, [source('Writer', 'Culture', 2), source('Unknown', '')])
        with self.assertRaisesRegex(ValueError, 'frequencies'):
            load_node_tables(nodes, self.path, maximum_year=2026, date_parser=single_year)

    def test_v2_configuration_has_separate_derived_roots_and_manual_device(self):
        c1 = json.loads((ROOT / 'config/freebase_grouped_rgcn_graphmask_20y_v1.json').read_text())
        c2 = json.loads((ROOT / 'config/freebase_grouped_rgcn_graphmask_20y_v2.json').read_text())
        for key in ['source_data', 'period_root', 'relation_root', 'model_root', 'graphmask_root']:
            self.assertNotEqual(c1[key], c2[key])
        self.assertEqual(c2['dataset_version'], 2)
        self.assertIsNone(c2['device'])
        self.assertNotIn('audit_file', c2['source_prepare'])


if __name__ == '__main__':
    unittest.main()
