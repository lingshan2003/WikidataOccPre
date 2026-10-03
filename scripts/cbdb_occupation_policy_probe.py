"""Read-only CBDB occupation coverage and policy sensitivity probe; v1 remains unchanged."""
import csv,json,sqlite3,sys
from collections import defaultdict,Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from scripts.cbdb_build_flat_triples import load_people,load_occupations,collect_triples
conn=sqlite3.connect(f'file:{root}/external_data/cbdb/2026.09.14/cbdb_20260914.sqlite3?mode=ro',uri=True)
out=root/'docs/cbdb_occupation_probe_2026-10-03'
out.mkdir(exist_ok=True)
people,_=load_people(conn)
names=dict(conn.execute('select c_personid,c_name_chn from BIOG_MAIN'))
rules=json.loads((root/'config/cbdb_occupation_whitelist_v1.json').read_text())
code_labels={code:label for label,codes in rules.items() for code in codes}
official=set(x[0] for x in conn.execute('select distinct c_personid from POSTED_TO_OFFICE_DATA where c_personid>0 and c_office_id>0'))
statuses=defaultdict(set); code_people=defaultdict(set); code_rows=Counter(); evidence=defaultdict(list)
for pid,code,source,pages,notes in conn.execute('select c_personid,c_status_code,c_source,c_pages,c_notes from STATUS_DATA'):
 statuses[pid].add(code);code_people[code].add(pid);code_rows[code]+=1
 if len(evidence[code])<3:
  evidence[code].append(dict(person_id=pid,name=names.get(pid),source_id=source,pages=pages,notes=notes))
labels={pid:{code_labels[c] for c in codes if c in code_labels} for pid,codes in statuses.items()}
baseline,base_counts=load_occupations(conn,people,root/'config/cbdb_occupation_whitelist_v1.json')
base_triples,_=collect_triples(conn,set(baseline))
base_nodes={p for (_,_,s,t) in base_triples for p in (s,t)}

def distribution(occupations):
 triples,_=collect_triples(conn,set(occupations))
 nodes={p for (_,_,s,t) in triples for p in (s,t)}
 return dict(eligible_people=len(occupations),eligible_occupations=dict(Counter(v[0] for v in occupations.values())),triples=len(triples),output_people=len(nodes),output_occupations=dict(Counter(occupations[p][0] for p in nodes)),official_official_triples=sum(occupations[s][0]==occupations[t][0]=='做官' for _,_,s,t in triples))

# Sensitivity analysis only: no frozen mapping or production outputs are changed.
priority={};strict={}
for pid in people:
 candidate=labels.get(pid,set()); nonofficial=candidate-{'做官'}
 has_official=pid in official or '做官' in candidate
 if len(nonofficial)==1: priority[pid]=(next(iter(nonofficial)),'STATUS_DATA')
 elif not nonofficial and has_official: priority[pid]=('做官','office_evidence')
 domains=candidate|({'做官'} if pid in official else set())
 if len(domains)==1:strict[pid]=(next(iter(domains)),'unique_domain')
unknown=set(people)-set(baseline)
date_triples,_=collect_triples(conn,set(people))
date_nodes={p for (_,_,s,t) in date_triples for p in (s,t)}
notes_ids={p for p,n in conn.execute('select c_personid,c_notes from BIOG_MAIN') if n and str(n).strip()}
source_ids={p for (p,) in conn.execute('select distinct c_personid from BIOG_SOURCE_DATA')}
text_ids={p for (p,) in conn.execute('select distinct c_personid from BIOG_TEXT_DATA')}
summary=dict(database='2026-09-14 original SQLite, read only',whitelist='v1 unreviewed; all alternative results are sensitivity probes',date_people=len(people),baseline=distribution(baseline),unique_nonofficial_priority=distribution(priority),unique_all_domains_only=distribution(strict),date_people_with_nonofficial_candidates=sum(bool(labels.get(p,set())-{'做官'}) for p in people),date_people_with_official_and_nonofficial=sum((p in official or '做官' in labels.get(p,set())) and bool(labels.get(p,set())-{'做官'}) for p in people),baseline_official_people_with_nonofficial=sum(baseline[p][0]=='做官' and bool(labels.get(p,set())-{'做官'}) for p in baseline),baseline_output_official_people_with_nonofficial=sum(baseline[p][0]=='做官' and bool(labels.get(p,set())-{'做官'}) for p in base_nodes),date_people_with_multiple_nonofficial=sum(len(labels.get(p,set())-{'做官'})>1 for p in people),missing_labels=dict(people=len(unknown),with_any_status=len(unknown&set(statuses)),without_any_status=len(unknown-set(statuses)),with_any_biog_notes=len(unknown&notes_ids),with_source_reference=len(unknown&source_ids),with_text_role=len(unknown&text_ids)),date_only_graph=dict(people=len(date_nodes),triples=len(date_triples),baseline_labeled_people=len(date_nodes&set(baseline)),unlabeled_people=len(date_nodes-set(baseline))),nonofficial_candidate_distribution=dict(Counter(label for p in people for label in labels.get(p,set())-{'做官'})),text_role_codes=list(conn.execute('select * from TEXT_ROLE_CODES')),assume_office_codes=list(conn.execute('select * from ASSUME_OFFICE_CODES')))
meta={c:(zh,en,t,cat) for c,zh,en,t,cat in conn.execute('select c.c_status_code,c.c_status_desc_chn,c.c_status_desc,r.c_status_type_code,t.c_status_type_chn from STATUS_CODES c left join STATUS_CODE_TYPE_REL r on c.c_status_code=r.c_status_code left join STATUS_TYPES t on r.c_status_type_code=t.c_status_type_code')}
fields=['status_code','name_zh','name_en','official_type_code','official_type_name','rows','people','date_people','date_people_with_posting','baseline_output_people','v1_label','review_decision','review_reason','examples_json']
with (out/'status_inventory.tsv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
 for code in sorted(code_rows,key=lambda c:(-code_rows[c],c)):
  zh,en,t,cat=meta.get(code,('','','',''));ps=code_people[code]
  w.writerow(dict(status_code=code,name_zh=zh,name_en=en,official_type_code=t,official_type_name=cat,rows=code_rows[code],people=len(ps),date_people=len(ps&set(people)),date_people_with_posting=len(ps&set(people)&official),baseline_output_people=len(ps&base_nodes),v1_label=code_labels.get(code,''),review_decision='待审核',review_reason='',examples_json=json.dumps(evidence[code],ensure_ascii=False)))
assumptions=defaultdict(set)
for pid,value in conn.execute('select c_personid,coalesce(c_assume_office_code,0) from POSTED_TO_OFFICE_DATA where c_personid>0 and c_office_id>0'):
 assumptions[pid].add(value)
summary['office_assumption_probe']=dict(date_posting_people=sum(p in people for p in assumptions),date_people_any_explicit_assumed=sum(p in people and 1 in a for p,a in assumptions.items()),date_people_unknown_only=sum(p in people and a=={0} for p,a in assumptions.items()),date_people_all_explicit_not_assumed=sum(p in people and a<=set(range(2,6)) for p,a in assumptions.items()))
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['text_role_codes','assume_office_codes']},ensure_ascii=False,indent=2))
print('text roles',summary['text_role_codes'])
print('assume office codes',summary['assume_office_codes'])
