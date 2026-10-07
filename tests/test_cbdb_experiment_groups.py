"""Contract checks for the broad relation vocabulary requested for CBDB experiments."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.cbdb_build_experiment_groups import collapse_rows, export_training_csv, resolve_groups
from scripts.cbdb_group_relations import load_mapping

ROOT = Path(__file__).resolve().parents[1]


class CBDBExperimentGroupsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads((ROOT/'config/cbdb_experiment_relation_groups_v1.json').read_text())
        cls.fine, cls.mapping = load_mapping(ROOT/'config/cbdb_relation_groups_v1.json')
        cls.resolved = resolve_groups(cls.profile, cls.fine, cls.mapping)

    def row(self, code, source='CBDB:1', target='CBDB:2', support=1):
        row = dict(Node1=source, Node2=target, Relation=code, SupportRows=str(support))
        for endpoint, node in [('Node1',source),('Node2',target)]:
            row.update({f'{endpoint}_NameZh':node, f'{endpoint}_Birth':'1000',
                        f'{endpoint}_Death':'', f'{endpoint}_Occupation':'做官',
                        f'{endpoint}_OccupationSource':'POSTED_TO_OFFICE_DATA'})
        return row

    def test_kin_table_is_inherited_except_direct_partnership(self):
        for code, group in self.resolved.items():
            if code.startswith('KIN:'):
                expected = ('intimate_partnership' if self.mapping[code]['group']=='intimate_partnership'
                            else 'inherited')
                self.assertEqual(group, expected)
        for code in ['KIN:134','KIN:135','KIN:168']:
            self.assertEqual(self.resolved[code], 'intimate_partnership')
        self.assertEqual(self.resolved['KIN:117'], 'inherited')  # 姻亲 is not one's partner.

    def test_collaboration_adoption_and_teacher_boundaries(self):
        for code in [549,550,568,569]:
            self.assertEqual(self.resolved[f'ASSOC:{code}'], 'inherited')
        for code in [251,341,342,456,457,541,542,572,573,213,214,499,500]:
            self.assertEqual(self.resolved[f'ASSOC:{code}'], 'professional_collaboration')
        self.assertEqual(self.resolved['ASSOC:22'], 'education_mentorship')
        self.assertEqual(self.resolved['ASSOC:558'], 'education_assessment')
        self.assertEqual(self.resolved['ASSOC:44'], 'textual_relations')

    def test_coarse_model_types_have_no_role_suffix_and_preserve_endpoint_order(self):
        rows = [self.row('KIN:75',support=2),self.row('KIN:107',support=3),
                self.row('KIN:180',source='CBDB:2',target='CBDB:1',support=4)]
        grouped, nodes = collapse_rows(rows,self.resolved,self.mapping)
        self.assertEqual(len(grouped),2)
        self.assertEqual({r['Relation'] for r in grouped},{'inherited'})
        forward = next(r for r in grouped if r['Node1']=='CBDB:1')
        self.assertEqual(forward['Node2'],'CBDB:2')
        self.assertEqual(forward['SupportRows'],5)
        self.assertEqual(sum(r['RawTripleCount'] for r in grouped),3)
        self.assertEqual(len(nodes),2)

    def test_binary_merges_distinct_acquired_themes_with_conserved_support(self):
        rows = [self.row('ASSOC:22',support=2),self.row('ASSOC:44',support=3)]
        multi,_ = collapse_rows(rows,self.resolved,self.mapping)
        binary,_ = collapse_rows(rows,self.resolved,self.mapping,binary=True)
        self.assertEqual(len(multi),2)
        self.assertEqual(len(binary),1)
        self.assertEqual(binary[0]['Relation'],'acquired')
        self.assertEqual(binary[0]['SupportRows'],5)

    def test_binary_and_multi_group_keep_loader_node_order_identical(self):
        rows=[self.row('ASSOC:44',target='CBDB:2'),self.row('ASSOC:22',target='CBDB:3')]
        multi,_=collapse_rows(rows,self.resolved,self.mapping)
        binary,_=collapse_rows(rows,self.resolved,self.mapping,binary=True)
        def loader_order(rs):
            return list(dict.fromkeys([r['Node1'] for r in rs]+[r['Node2'] for r in rs]))
        self.assertEqual(loader_order(multi),loader_order(binary))

    def test_training_adapter_preserves_single_label_without_fabricating_hierarchy(self):
        import csv,gzip
        rows,_ = collapse_rows([self.row('KIN:75')],self.resolved,self.mapping)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'input.csv.gz'
            export_training_csv(path,rows)
            with gzip.open(path,'rt') as stream:
                row=next(csv.DictReader(stream))
            self.assertEqual(row['Node1_occ_level1'],'做官')
            self.assertEqual(row['Node1_occ_level2'],'')
            self.assertEqual(row['Node1_occ_level3'],'')
            self.assertEqual(row['Node1_Country'],'')


if __name__=='__main__':
    unittest.main()
