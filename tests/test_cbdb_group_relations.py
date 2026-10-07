"""Checks for role-preserving CBDB grouping and source-evidence conservation."""
import json
import unittest

from pathlib import Path

from scripts.cbdb_group_relations import group_rows, load_mapping


class CBDBGroupingTest(unittest.TestCase):
    def setUp(self):
        self.groups = {
            'kin_parent_child': {'label_zh': '亲子', 'family': 'kinship'},
        }
        self.mapping = {
            'KIN:75': dict(group='kin_parent_child', direction='to_parent', qualifier=''),
            'KIN:76': dict(group='kin_parent_child', direction='to_parent', qualifier='specific'),
            'KIN:180': dict(group='kin_parent_child', direction='to_child', qualifier=''),
        }

    def row(self, relation, source='CBDB:1', target='CBDB:2', support=1):
        row = dict(Node1=source, Node2=target, Relation=relation,
                   RelationGroup='kinship', RelationLabelZh='原始名称', SupportRows=str(support))
        for endpoint, node in [('Node1', source), ('Node2', target)]:
            row.update({
                f'{endpoint}_NameZh': node, f'{endpoint}_Birth': '1000',
                f'{endpoint}_Death': '', f'{endpoint}_Occupation': '做官',
                f'{endpoint}_OccupationSource': 'POSTED_TO_OFFICE_DATA',
            })
        return row

    def test_merge_parallel_codes_conserves_support_and_keeps_roles(self):
        rows = [self.row('KIN:75', support=2), self.row('KIN:76', support=3),
                self.row('KIN:180', source='CBDB:2', target='CBDB:1', support=4)]
        annotated, grouped, nodes = group_rows(rows, self.mapping, self.groups)
        self.assertEqual(len(annotated), 3)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(nodes), 2)
        parent = next(r for r in grouped if r['Direction'] == 'to_parent')
        self.assertEqual((parent['Node1'], parent['Node2']), ('CBDB:1', 'CBDB:2'))
        self.assertEqual(parent['Relation'], 'kin_parent_child__to_parent')
        self.assertEqual(parent['SupportRows'], 5)
        self.assertEqual(parent['RawTripleCount'], 2)
        self.assertEqual(json.loads(parent['OriginalRelations']), ['KIN:75', 'KIN:76'])
        self.assertEqual(json.loads(parent['RelationQualifiers']), ['specific'])
        self.assertEqual(sum(r['SupportRows'] for r in grouped), 9)
        self.assertEqual(rows[0]['RelationGroup'], 'kinship')
        self.assertEqual(annotated[0]['OriginalRelationGroup'], 'kinship')

    def test_unknown_code_fails_instead_of_silently_dropping(self):
        with self.assertRaisesRegex(ValueError, 'Unmapped raw relation'):
            group_rows([self.row('KIN:99999')], self.mapping, self.groups)

    def test_conflicting_person_attributes_fail(self):
        rows = [self.row('KIN:75'), self.row('KIN:76')]
        rows[1]['Node1_Birth'] = '1001'
        with self.assertRaisesRegex(ValueError, 'Conflicting source node attributes'):
            group_rows(rows, self.mapping, self.groups)

    def test_duplicate_raw_triples_fail(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate source triple'):
            group_rows([self.row('KIN:75'), self.row('KIN:75')], self.mapping, self.groups)


class CBDBSemanticMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.mapping = load_mapping(
            Path(__file__).resolve().parents[1] / 'config/cbdb_relation_groups_v1.json')

    def test_roles_match_parent_teacher_recruitment_and_examiner_names(self):
        expected = {
            'KIN:75': ('kin_parent_child', 'to_parent'),
            'KIN:180': ('kin_parent_child', 'to_child'),
            'KIN:107': ('kin_social_parent_child', 'to_parent'),
            'ASSOC:22': ('education_mentorship', 'to_teacher'),
            'ASSOC:23': ('education_mentorship', 'to_student'),
            'ASSOC:24': ('official_hierarchy', 'to_subordinate'),
            'ASSOC:25': ('official_hierarchy', 'to_superior'),
            'ASSOC:558': ('education_assessment', 'to_examinee'),
            'ASSOC:559': ('education_assessment', 'to_examiner'),
            'ASSOC:646': ('religious_relation', 'to_student'),
            'ASSOC:647': ('religious_relation', 'to_teacher'),
        }
        for relation, role in expected.items():
            with self.subTest(relation=relation):
                entry = self.mapping[relation]
                self.assertEqual((entry['group'], entry['direction']), role)

    def test_influence_and_commissioning_are_not_direct_instruction_or_authorship(self):
        self.assertEqual(self.mapping['ASSOC:258']['group'], 'influence_succession')
        self.assertEqual(self.mapping['ASSOC:309']['group'], 'family_social')
        self.assertEqual(self.mapping['ASSOC:154']['direction'], 'to_writer')
        self.assertEqual(self.mapping['ASSOC:155']['direction'], 'to_requester')
        self.assertEqual(self.mapping['ASSOC:601']['group'], self.mapping['ASSOC:602']['group'])
        self.assertEqual(self.mapping['ASSOC:603']['group'], self.mapping['ASSOC:604']['group'])

    def test_conflicting_descriptions_abstain_on_roles_and_keep_reason(self):
        for code in [68, 69, 72, 73, 140, 141, 223, 224, 456, 457, 648, 649]:
            entry = self.mapping[f'ASSOC:{code}']
            self.assertEqual(entry['direction'], 'unspecified')
            self.assertIn('不一致', entry['qualifier'])


if __name__ == '__main__':
    unittest.main()
