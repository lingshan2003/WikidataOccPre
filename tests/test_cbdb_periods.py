"""Contracts for CBDB endpoint-year membership and period graph export."""
import copy
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.cbdb_group_relations import NODE_ATTRIBUTES, write_tsv
from scripts.cbdb_split_periods import (
    ROOT, belongs_to_period, export_periods, load_period_config, loader_node_order,
    read_tsv, select_periods,
)


class CBDBPeriodsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads((ROOT/'config/cbdb_periods_v1.json').read_text())

    def config_path(self, config=None):
        path = self.root/'periods.json'
        path.write_text(json.dumps(config or self.config))
        return path

    def test_any_known_endpoint_no_lifetime_interpolation(self):
        periods = load_period_config(self.config_path())['periods']
        node = dict(Birth='600', Death='1600')
        self.assertEqual([p['id'] for p in periods if belongs_to_period(node,p)], ['to_699','1500_1699'])
        self.assertTrue(belongs_to_period(dict(Birth='-148', Death=''), periods[0]))
        self.assertFalse(belongs_to_period(dict(Birth='0', Death=''), periods[0]))
        self.assertTrue(belongs_to_period(dict(Birth='', Death='700'), periods[1]))
        self.assertFalse(belongs_to_period(dict(Birth='699', Death=''), periods[1]))

    def test_configuration_rejects_gaps_overlaps_bad_types_and_unsafe_ids(self):
        mutations = [
            lambda c:c['periods'][1].update(start=701),
            lambda c:c['periods'][1].update(start=699),
            lambda c:c['periods'][0].update(start=1),
            lambda c:c['periods'][2].update(start=None),
            lambda c:c['periods'][1].update(start=True),
            lambda c:c['periods'][1].update(id='../bad'),
            lambda c:c.update(assignment='lifetime_overlap'),
        ]
        for mutation in mutations:
            bad = copy.deepcopy(self.config)
            mutation(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_period_config(self.config_path(bad))

    def test_period_selection_is_config_order_and_rejects_unknown_duplicates(self):
        self.assertEqual([p['id'] for p in select_periods(self.config,'1700_plus,to_699')],
                         ['to_699','1700_plus'])
        for selection in ('to_699,to_699', 'missing', '', 'all,to_699'):
            with self.assertRaises(ValueError):
                select_periods(self.config, selection)

    def fixture(self):
        source = self.root/'source'
        source.mkdir()
        nodes = {
            'A':dict(Node='A', NameZh='甲', Birth='650', Death='750', Occupation='做官', OccupationSource='office'),
            'B':dict(Node='B', NameZh='乙', Birth='700', Death='800', Occupation='教育', OccupationSource='status'),
            'C':dict(Node='C', NameZh='丙', Birth='600', Death='699', Occupation='做官', OccupationSource='office'),
            'E':dict(Node='E', NameZh='丁', Birth='', Death='650', Occupation='宗教', OccupationSource='status'),
        }
        write_tsv(source/'nodes.tsv.gz',['Node',*NODE_ATTRIBUTES],list(nodes.values()))
        multi = []
        for first, second, relation in [('A','B','inherited'),('A','B','textual_relations'),
                                        ('A','C','inherited'),('A','C','textual_relations'),
                                        ('B','C','textual_relations'),('E','B','textual_relations')]:
            row = dict(Node1=first, Relation=relation, Node2=second, RelationGroup=relation,
                TieClass='inherited' if relation=='inherited' else 'acquired',
                SupportRows='2', RawTripleCount='1', OriginalRelations='["ASSOC:44"]',
                Evidence='preserve me')
            row.update({f'{endpoint}_{field}':nodes[node][field] for endpoint,node in [('Node1',first),('Node2',second)]
                        for field in NODE_ATTRIBUTES})
            multi.append(row)
        binary = [dict(row,Relation=row['TieClass'], RelationGroup=row['TieClass']) for row in multi]
        for representation, rows in [('multi_group',multi),('binary',binary)]:
            directory = source/representation
            directory.mkdir()
            rows.sort(key=lambda row:(row['Node1'],row['Node2'],row['Relation']))
            write_tsv(directory/'person_relation_triples.tsv.gz',list(rows[0]),rows)
        return source, multi

    def test_export_preserves_rows_taxonomies_order_and_excludes_cross_period_edges(self):
        source, multi = self.fixture()
        output = self.root/'output'
        summary = export_periods(source,self.config_path(),output,'to_699,700_899')
        stats = summary['representations']['multi_group']
        self.assertEqual(stats['input_triples'],6)
        self.assertEqual(stats['emitted_triples'],4)
        self.assertEqual(stats['excluded_triples'],2)
        self.assertEqual(summary['people_in_multiple_periods'],1)
        self.assertEqual(summary['emitted_eligible_people'],5)
        self.assertEqual(summary['periods']['to_699']['eligible_nodes'],3)
        self.assertEqual(summary['periods']['to_699']['representations']['multi_group']['active_nodes'],2)
        for period, expected_nodes in [('to_699',{'A','C'}),('700_899',{'A','B'})]:
            orders = []
            for representation in ('multi_group','binary'):
                directory = output/period/representation
                _,rows = read_tsv(directory/'person_relation_triples.tsv.gz')
                if representation == 'multi_group':
                    self.assertEqual(rows,[r for r in multi if r['Node1'] in expected_nodes and r['Node2'] in expected_nodes])
                orders.append(loader_node_order(rows))
                _,active = read_tsv(directory/'active_nodes.tsv.gz')
                self.assertEqual([row['Node'] for row in active],orders[-1])
                self.assertTrue(all(row['Evidence']=='preserve me' and row['SupportRows']=='2' for row in rows))
                taxonomy = json.loads((directory/'relation_taxonomy.json').read_text())['groups']
                self.assertEqual(set(taxonomy),{row['Relation'] for row in rows})
                ties = json.loads((directory/'tie_taxonomy.json').read_text())['groups']
                self.assertTrue(ties['inherited'] and ties['acquired'])
                self.assertEqual(set(ties['inherited'])|set(ties['acquired']),set(taxonomy))
                self.assertFalse(set(ties['inherited'])&set(ties['acquired']))
                with gzip.open(directory/'Q_R_Q_extended.csv.gz','rt') as stream:
                    csv_rows = list(csv.DictReader(stream))
                self.assertEqual(loader_node_order(csv_rows),orders[-1])
                self.assertTrue(all(r['Node1_occ_level2']=='' and r['Node2_Country']=='' for r in csv_rows))
            self.assertEqual(*orders)

    def test_repeated_triples_counted_without_changing_support(self):
        source,_ = self.fixture()
        # Make both A and C belong to both selected periods.
        _,nodes = read_tsv(source/'nodes.tsv.gz')
        next(row for row in nodes if row['Node']=='C')['Death']='750'
        write_tsv(source/'nodes.tsv.gz',['Node',*NODE_ATTRIBUTES],nodes)
        for representation in ('multi_group','binary'):
            path = source/representation/'person_relation_triples.tsv.gz'
            fields,rows = read_tsv(path)
            for row in rows:
                for endpoint in ('Node1','Node2'):
                    if row[endpoint]=='C':
                        row[f'{endpoint}_Death']='750'
            write_tsv(path,fields,rows)
        summary = export_periods(source,self.config_path(),self.root/'output','to_699,700_899')
        stats = summary['representations']['multi_group']
        self.assertEqual(stats['triples_in_multiple_periods'],2)
        self.assertEqual(stats['repeated_triple_copies'],2)
        self.assertEqual(stats['emitted_support_rows']-stats['retained_support_rows'],4)

    def test_output_protection_and_atomic_failure(self):
        source,_ = self.fixture()
        config = self.config_path()
        with self.assertRaisesRegex(ValueError,'separate'):
            export_periods(source,config,source/'periods','to_699')
        output = self.root/'output'
        with self.assertRaisesRegex(ValueError,'Empty period graph'):
            export_periods(source,config,output)
        self.assertFalse(output.exists())
        export_periods(source,config,output,'to_699')
        with self.assertRaisesRegex(ValueError,'nonempty'):
            export_periods(source,config,output,'to_699')
        export_periods(source,config,output,'700_899',overwrite=True)
        self.assertFalse((output/'to_699').exists())
        output2 = self.root/'unrelated'
        output2.mkdir()
        (output2/'notes.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError,'Refusing'):
            export_periods(source,config,output2,'to_699',overwrite=True)
        self.assertEqual((output2/'notes.txt').read_text(),'keep')


if __name__ == '__main__':
    unittest.main()
