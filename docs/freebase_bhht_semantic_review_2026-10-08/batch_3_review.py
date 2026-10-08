import csv, json, re
from pathlib import Path
base=Path('docs/freebase_bhht_semantic_review_2026-10-08')
inp=base/'batch_3_input.tsv'; out=base/'batch_3_review.tsv'
# Explicit per-occupation assignments based on the full role phrase.
raw_by_level={
'Culture': '''Film Producer|Voice Actor|Broadcaster|Singer-songwriter|Film editor|Guitarist|Television crew|Stunt Performer|Film art director|Music Director|Music artist|Illustrator (Profession)|Keyboard player|Multi-instrumentalist|Nude Glamour Model|Film critic|Stand-up comedian|Audio Engineer|TV Journalist|VJ|Art collector|Magazine editor|Sports analyst|Ballroom Dancer|Rodeo clown|Acting Coach|Narrator|Jewelry designer|Magician|Radio producer|Television Show Host|Game designer|Media proprietor|Graphic Artist|Industrial designer|Stagehand|Vocal coach|Conservator-restorer|Interior Decorator|Property Master|Music Journalist|Multimedia artist|Textile designer|Raconteur|Arts Administrator|Japanese idol|Choreography|Special Effects Supervisor|Drama Coach|Opera conductor|Film distributor (Profession)|Action Director|Sports Anchor|Statistical graphics|Wine Critic|Picture editor|Minstrel|Radio writer|Video Artist|Fine art artist|Foley Artist|Arts Educator|Game creator|Theater Manager|Radio DJ|Media consultant|News director|Graphic Novelist|Network executive|Child singer|Stage management|Title Designer|Tour promoter|Turntablist|Classical Indian Music Performer|Script coordinator|Environmental Writer|Fashion Consultant|Film cinematographer|Assistant Director|Frame Gilder|Mastering engineer|Backing vocalist|UI Designer|Jazz guitarist|Houseware designer|Key grip|Lead vocalist|Fashion Commentator|Online producer|Online Editor|Sound Artist|Camera Assistant|Suit actor|Assistant Sound Editor|Showrunner|Architectural Critic|ADR Recordist|Porcelain painter|Sound Re-Recordist|Book Designer|Actor (Profession) #80|Stamp designer|Soundtrack Composer|Storyboard Artist|Circus Performer|Special Effects Artist|Remixer|Artistic Director|Video Game Producer|Sound Technician|ADR Director|Showman|Children's book author|Child model|Public intellectual|Sideman|Aerialist|Assistant Property Master|Burlesque stripper|Plus-size model|Sound Effects Editor|Fetish model|Lighting technician|Boom Operator|Visual Effects Technician|Corrector|Lyric poet|Tenor Saxophonist|Jazz Violinist|Conceptual Artist|Food writer|Business journalist|Web design|Production coordinator|Hymnology|Showman|Recording Engineer (Film job)|Nature Photographer|Political writer''',
'Discovery/Science': '''Civil engineer|Computer Engineer|Veterinary physician|Academician|Ethnographer|Chemical engineer|Film Historian|Structural engineer|Clinical psychologist|Military engineer|Ethnomusicologist|Evolutionary Biologist|Kindergarten Teacher|Geophysicist|Theoretical Physicist|Engineering technologist|Ecologist|Flight instructor|Music Historian|Theoretician|Mining Engineer|Medical research|Neurosurgeon|Flight engineer|Game programmer|Construction Engineer|Wildlife biologist|Software Tester|Agricultural Scientist|Systems Programmer|Geochemist|Architectural historian (Profession) #1|Architectural historian|primary school headmaster|Medical Doctor|Midwife (Film subject)|Occupational Therapist|Software Developer (Profession)|Etymologist|Dental assistant|Volcanologist|Medical Doctor|Agricultural Scientist|Systems Programmer|Geochemist|Information scientist|Human computer|Clinical Systems Analyst|Child Psychiatrist|Bioengineer|Engineering Drafter|Laryngologist|Software Developer (Profession)|Ontologist|Country Doctor|American Historian|Psycholinguist|Qualified Flying Instructor|Electronic Engineer|European Historian|Test engineer|Neuroendocrinologist|Refrigeration Engineer|Biomedical Engineer|Planetary scientist|Petroleum geologist|Dermatology|Information scientist|Software Engineering|Tutor|Webmaster|Web Analyst|Information systems technician|Medic|Aeronautical Engineer|Pilot trainer|Hospital corpsman|pedagogue assistant''',
'Leadership': '''Criminal defense lawyer|Banker|Social activist|Police officer|Peace activist|Investment Banker|Televangelist|Naval Officer|Civil rights activist|High Priest|Political Consultant|Bishop (Religious Leadership Title) #112|Military Leader|Patent attorney|Lady-in-waiting|Canon|Cossack|Military aviator|Entertainment Lawyer|Stock trader|Union organizer|Parliamentarian|Army officer (retired)|Executive Management|Civil law notary|Filibuster|Tax advisor|Venture capitalist|High Priest|Tax advisor|Human Rights Lawyer|Senior lieutenant|Liaison officer|Commanding General, Multi-National Force - Iraq|Deputy mayor|Company Secretary|Federal Government of Canada|Business Manager|High Court Judge|Pharmaceutical Entrepreneur|Bankruptcy Lawyer|Serial entrepreneur|Colonel general|Secretary-General of the United Nations|Political Assistant|Kaymakam|Naval Systems Analyst|Dayan (Religious Leadership Title)|Standart officer|Prince of Hesse and by Rhine|United States Ambassador to Seychelles|Party leader|Navy pilot|Methodist Preacher (Profession)|Prince of Novgorod|supermarket executive|Commodore (Military rank)|Rector (Profession)|Holy Roman Emperor|Pro-life activist|Quantitative analyst|Financial planner|Personal injury lawyer|Commodore (Military rank) #12|Legal analyst|Business Analyst''',
'Sports/Games': '''American Football player|Ice Hockey Player|Rugby Player|Cricketer|Australian rules football player|Alpine skier|Baseball Coach|Baseball Manager|Figure Skater|Cross-country skier|Professional Road Racing Cyclist|Field Hockey player|Freestyle skier|Bobsleigher|Fitness professional|Professional Wrestling Referee|Rock Climber|Fencing master|Fencing master|Rock Climber|Keirin Racer|Black belt|American Football Official|Wrangler|Doshu|Skateboarder|Canadian football player|Level 10 gymnast|Sensei''',
'Other': '''Blue-collar worker|Animal trainer|Factory worker|Cab Driver|EMS Professional|Shadetree mechanic|Governess|Switchboard operator|Mail carrier|Bus driver|Animal training|Groom of the Stool|Tenant farmer|Security guard|Retired|Animal trainer|General contractor|Rail transport worker|Tenant farmer|Server|Shop Owner|Charwoman|Shop Owner|Tool and die maker|Hermit (Profession)|Personal organizer|Cloth merchant|Haberdasher|Pitchman|Shop Owner|Car Worker|Horse trader|Metalsmith|Electrical contractor|Woodcutter|Upholsterer|Carpet cleaning|Employment counsellor|Lackey (Profession)|Fur Trader|Sexton|Pharmacy technician|Post Office Worker|Courier|Trapper|Forklift operator|Paymaster|Tax Preparer|Manicurist|Stonecutter|Street sweeper|Stonemasonry|Plantation Overseer|Herder|Direct marketing|Image consultant|Personal chef|Computer repair technician|Youth worker|First Gentleman|Pyrotechnician|Arms Dealer|Chapmen|Train Conductor|Animal trainer|Insurance broker|Legal secretary|Property caretaker|Digital Marketer|Search Engine Optimization|Building contractor|Furrier|Digital Marketing Strategist|Digital Marketing Consultant (Profession) #1|SEO Consultant|Publican|Used Car Dealer|Online Marketing Strategist|Appraiser|Master Quarrier|Soap Maker|Babysitter|Construction Foreman|Rural Mail Carrier|Wool classing|Conversion rate optimization specialist|internet business consultant|Entrepreneurship|SEO Analyst|Digital Marketing Analyst|Yoga Consultant|Special Needs Counsellor|Digital Marketing Consultant|Majordomo|Online Marketing Consultant|Criminal (Profession) #15|SEO Professional (Profession) #1|SEO Professional'''
}
levels={}
for lev, s in raw_by_level.items():
 for name in s.split('|'):
  if name in levels and levels[name] != lev: raise ValueError(f'conflicting assignment {name}: {levels[name]} / {lev}')
  levels[name]=lev
# Labels whose referent is a field, code, status, title with materially unresolved sense, or generic role.
uncertain={
'Agent':('[]','An agent can mean a talent representative, commercial intermediary, intelligence operative, or other role; this bare label does not identify the duties.'),
'Public speaker':('["Culture","Leadership"]','Public speaking describes a communication activity, not a stable profession; the subject, audience, and organizational role are unspecified.'),
'Rescue':('[]','“Rescue” names an activity or event rather than a defined occupational role.'),
'/m/02h65y5':('[]','This is an opaque Freebase identifier without a human-readable occupation label.'),
'Son of the Storm God':('[]','This reads as a mythic/kinship epithet rather than a defined occupation, and its referent is not specified.'),
'Retired':('[]','Retirement is an employment status and gives no information about the person’s former occupation.'),
'm/0513qg_':('[]','This is an opaque Freebase identifier without a human-readable occupation label.'),
'Canon':('["Leadership","Discovery/Science"]','Canon may mean a church officeholder or a recognized body/standard in other contexts; the bare term does not establish the role.'),
'Animal training':('[]','This names a field or activity, not a specific occupational title; trainer, researcher, and performance contexts remain possible.'),
'Pretty Horsebreaker':('["Other","Sports/Games"]','The historical expression has more than one documented sense, from breaking horses to a colloquial social label, so the occupation cannot be fixed safely from this label alone.'),
'Exile':('[]','Exile is a political or personal status, not an occupation.'),
'Federal Government of Canada':('[]','This is the name of a government institution, not an individual occupational role.'),
'Church of Aphrodite':('[]','This is an organization/religious affiliation label rather than a defined occupation.'),
'Reno (Profession)':('[]','“Reno” is not sufficiently interpretable as a profession from this label alone.'),
'Black belt':('["Sports/Games","Other"]','A black belt may denote a martial-arts rank or a person’s credential; the label alone does not establish active athletic/professional duties.'),
'Modeler':('["Culture","Discovery/Science","Other"]','Modeler can refer to artistic, scientific, industrial, or mathematical modeling; no domain is given.'),
'Informant':('["Leadership","Other"]','Informant may mean a confidential source, intelligence asset, or information provider; the occupational setting is missing.'),
'Internet Visionary':('[]','This is an evaluative epithet rather than a standardized occupation with defined duties.'),
'Wrangler':('["Other","Culture"]','Wrangler may handle livestock or serve as a film/production wrangler; the domain is unspecified.'),
'Information technology':('[]','This names a broad technical field rather than a specific job or duty.'),
'Taluk Office':('[]','This is an administrative office/unit label, not a person’s occupation.'),
'm/0qcntlr':('[]','This is an opaque Freebase identifier without a human-readable occupation label.'),
'm/012zw2f7':('[]','This is an opaque Freebase identifier without a human-readable occupation label.'),
'm/012zvydj':('[]','This is an opaque Freebase identifier without a human-readable occupation label.'),
'Rector (Profession)':('["Leadership","Discovery/Science"]','Rector can head a university or a religious institution; the label alone does not resolve the institution and duties.'),
'Staff':('[]','“Staff” is a generic employment descriptor without a specific role or duties.'),
'Agent':('[]','An agent can mean a talent representative, commercial intermediary, intelligence operative, or other role; this bare label does not identify the duties.'),
'Road agent':('["Other","Leadership"]','The historical term can refer to a highway robber or an itinerant commercial representative; the sense is unresolved.'),
'Helicopter Pilot':('["Leadership","Other"]','A helicopter pilot may fly civilian transport or work missions or serve in a military aviation role; the title gives no service context.'),
'Marksman':('["Sports/Games","Leadership"]','Marksman can denote competitive precision shooting or a military sharpshooter; the setting is not stated.'),
'Information technology':('[]','This names a broad technical field rather than a specific job or duty.'),
'Medicine man':('[\"Discovery/Science\",\"Leadership\",\"Other\"]','Medicine man can mean a traditional healer, ritual specialist, or spiritual authority; no community or duties are specified.'),
'Search Engine Optimization':('[]','This names a marketing/technical discipline rather than a specific occupational post.'),
'Search Engine Optimization (Industry) #1':('[]','The industry label identifies a sector, not an individual occupation.'),
'College Student':('[\"Discovery/Science\"]','Student is an education status, not a completed occupational role.'),
'Pioneer':('[\"Discovery/Science\",\"Leadership\",\"Other\"]','Pioneer may mean an explorer, innovator, or early settler; no specific domain or duties are stated.'),
'Paranormal Investigator':('[\"Discovery/Science\",\"Culture\",\"Other\"]','The label may describe empirical investigation, entertainment content, or a paranormal-claim activity; duties and evidentiary setting are unspecified.'),
'Internet Marketing':('[]','This is a business activity/field rather than an identifiable job title.'),
'Counsellor in Norwich':('[\"Leadership\",\"Other\",\"Discovery/Science\"]','The place qualifier does not clarify whether “counsellor” means an elected local representative, therapist, or adviser.'),
'Entrepreneurship':('[]','This names a business activity/discipline rather than an occupational role.'),
'Business Analyst':('[\"Leadership\",\"Other\",\"Discovery/Science\"]','Business analyst can mean organizational strategy, financial analysis, or technical systems analysis; the specialization is absent.'),
'Hacker (TV subject)':['[]','This is a TV-subject descriptor; it does not establish whether the person is a computer-security professional, criminal intruder, or merely depicted subject.'],
'Conversion Rate Optimisation':('[]','This is a marketing optimization practice, not a specific job title.'),
'Inbound marketing':('[]','This is a marketing discipline/strategy rather than a specific person’s occupation.'),
}
# Some broad or domain-specific labels need a consciously resolved reading based on their modifier.
# Extra semantic notes for roles where naive keyword/substr matching is especially misleading.
notes={
'Human computer':'A human computer performed mathematical calculations by hand, including trajectory calculations for aerospace missions; this is a technical/scientific occupation, not a generic person-machine interface label.',
'Recording Engineer (Film job)':'Captures, edits, and balances dialogue and other sound recorded for film production; the explicit film modifier makes this a film-sound production occupation.',
'Nature Photographer':'Creates photographs of wildlife, landscapes, and natural subjects for publication or exhibition; photography is visual cultural production.',
'Political writer':'Writes books, columns, or commentary about political events and ideas; the modifier describes subject matter, not an elected or governmental office.',
'Baseball Manager':'Manages a baseball team’s lineup, training, tactics, and game decisions; this is direct sports-team leadership within the sport.',
'primary school headmaster':'Leads a primary school’s educational program, staff, and student instruction; the core institution is academic education.',
'Film Historian':'Researches and interprets the history of cinema using films, records, and archives; the work is historical scholarship.',
'Music Historian':'Researches and writes about the history of music and musical traditions; the work is historical scholarship, not music performance.',
'Architectural historian':'Researches the history, styles, and historical record of buildings and architecture; the role is scholarship rather than architectural design.',
'Architectural historian (Profession) #1':'Researches the history, styles, and historical record of buildings and architecture; the role is scholarship rather than architectural design.',
'Digital Marketer':'Plans and runs online promotion through search, web content, and paid digital channels; this is a marketing service role rather than general corporate leadership.',
'Digital Marketing Strategist':'Designs online campaign plans and channel priorities for a client or organization; this is marketing strategy, not company governance.',
'Digital Marketing Consultant (Profession) #1':'Advises clients on search, content, and online campaign performance; this is an individual marketing service occupation.',
'Digital Marketing Consultant':'Advises a business on planning and improving online advertising and customer acquisition campaigns; this is marketing consulting work.',
'Online Marketing Strategist':'Plans web-based advertising and audience-acquisition campaigns; the role specializes in marketing execution and advice.',
'Online Marketing Consultant':'Advises clients on web promotion, search reach, and online campaign choices; this is a consulting service.',
'SEO Consultant':'Recommends site structure, content, and search-ranking improvements to clients; the work is specialized digital marketing advice.',
'SEO Analyst':'Reviews search visibility, traffic, and keyword performance and recommends changes to a website; this is a marketing analytics service.',
'Digital Marketing Analyst':'Measures digital campaign traffic and conversion results and reports what should be adjusted; this is marketing analysis rather than executive management.',
'Conversion rate optimization specialist':'Tests website pages and purchase or sign-up flows to improve the share of visitors who complete a target action; this is a digital marketing service.',
'Direct marketing':'Plans direct promotional contact such as mail, phone, or targeted messages to prospective customers; it names a marketing practice rather than company leadership.',
'internet business consultant':'Advises clients on operating or improving an internet business; the title does not establish ownership or executive authority.',
'Search Engine Optimization':'Broad discipline label for improving a site’s visibility in search results, not a specific job title.',
'Search Engine Optimization (Industry) #1':'Industry/sector label for search optimization services; it does not describe an individual’s duties.',
'Internet Marketing':'Broad field label for promoting goods or services online, not one specific job.',
'Inbound marketing':'Marketing approach built around attracting prospective customers through useful content and search visibility; the label names a practice, not a particular post.',
'Keyboard player':'Keyboard player here means a musician who performs on keyboard instruments; “player” is not a sports role.',
'Acting Coach':'An acting coach trains performers in acting technique and audition/performance practice; “coach” is not sports coaching.',
'Film critic':'A film critic evaluates and writes or broadcasts criticism about cinema; the subject matter does not make this a film production or sports role.',
'Business journalist':'A business journalist reports and edits news about commerce; reporting is a media occupation, not business management.',
'Sports analyst':'A sports analyst interprets games, tactics, and performance for media audiences; commentary is a cultural/media function.',
'Game designer':'A game designer creates rules, mechanics, and player experience for games; this is design of cultural products, not a competitive player.',
'Magician':'A magician performs illusion and stage entertainment for audiences.',
'Industrial designer':'An industrial designer develops the form and usability of manufactured products; this is design practice, not corporate leadership.',
'Animal trainer':'An animal trainer conditions animals to respond to commands for care, work, or performance; the phrase alone does not imply sports coaching.',
'Animal training':'This names animal conditioning as a field/activity and does not identify one occupational post.',
'Media proprietor':'A media proprietor owns or controls media outlets and their publishing/broadcast operations, a culture-industry function.',
'Network executive':'A network executive commissions and manages programming and production at a broadcast network; the modifier fixes the role to media.',
'SEO Professional':'An SEO professional improves how websites are indexed and ranked in search services; this is a specialized digital marketing service.',
'SEO Professional (Profession) #1':'The profession-number suffix is a Freebase disambiguator; search-engine optimization work improves site visibility in search results.',
'Chapmen':'Chapmen are itinerant traders/peddlers who travel to sell goods; this is small-scale commerce.',
'Groom of the Stool':'The historical court office attended the monarch and evolved into a privileged royal household/administrative post; this particular court title is treated as a household service role.',
'Pretty Horsebreaker':'The historical label is semantically unstable: sources describe horse breaking and a separate colloquial social sense, so it remains unresolved.',
'Kaymakam':'A kaymakam is a district-level governor/administrator, exercising public authority on behalf of government.',
'Doshu':'Doshu is the hereditary/head authority of an aikido tradition; this is a martial-arts leadership title, not a generic occupation.',
'Governess':'A governess lives with or works for a household to supervise and educate children.',
'Sports Anchor':'A sports anchor presents and reports sports news for television or radio; this is broadcasting/journalism.',
'Film distributor (Profession)':'A film distributor acquires, markets, and arranges exhibition of films; this is a film-industry role.',
'Game creator':'A game creator develops a game as an authored entertainment product; the label describes creation, not playing or managing a sports team.',
'Rodeo clown':'The rodeo title denotes a clown who entertains spectators during rodeo breaks; the PRCA distinguishes this from the bullfighter, whose job is rider protection, so the full phrase is a performance role.',
}
# Title groups for concise role-specific explanations; applied only when no precise note is needed.
def generic_reason(name, lev):
 n=name.lower()
 if lev=='Culture':
  if any(x in n for x in ['actor','performer','singer','vocalist','guitarist','violinist','musician','instrumentalist','dancer','idol','comedian','magician','minstrel','turntablist','dj','artist','painter','illustrator','designer','cinematographer','stunt','clown','aerialist','skater']):
   core='创作或呈现艺术作品/现场表演，职业产出面向文化受众'
  elif any(x in n for x in ['journalist','writer','critic','historian','editor','broadcaster','anchor','host','narrator','raconteur']):
   core='采访、撰写、编辑、评论或向公众传播文化/媒体内容'
  elif any(x in n for x in ['engineer','technician','operator','grip','stagehand','recordist','sound','audio','lighting','effects','property master','coordinator','director','producer','showrunner','executive','manager','administrator','promoter','distributor','proprietor']):
   core='支持或统筹影视、广播、音乐及其他文化内容的制作、技术实现或发行'
  else: core='提供明确的艺术、媒体或文化产业创作与制作服务'
  return f'“{name}”所指工作核心是{core}，属于文化生产、表演或传播职能，归入Culture。'
 if lev=='Discovery/Science':
  if any(x in n for x in ['doctor','physician','surgeon','psychiatrist','dermatology','laryngologist','therapist','veterinary','medical','dental','midwife']): core='运用医学专业知识提供诊断、治疗、护理或健康服务'
  elif any(x in n for x in ['scientist','biologist','physicist','geologist','geophysicist','ethnographer','ethnomusicologist','historian','etymologist','linguist','ontologist','historian','research','ecologist','historian','historian','historian','historian']): core='开展该学科的研究、证据分析与知识产出'
  elif any(x in n for x in ['engineer','programmer','developer','technologist','software','tester','information scientist','systems analyst','instructor','teacher','academician','historian']): core='从事专业教学、技术设计开发、测试或学术研究'
  elif 'gymnast' in n: core='在体操项目中进行规则化训练和竞技展示'
  else: core='以专门科学知识、学术研究、医疗服务或工程技术为主要职责'
  return f'“{name}”从事的核心职责是{core}，其工作属于知识发现、学术/医疗或技术开发，归入Discovery/Science。'
 if lev=='Leadership':
  if any(x in n for x in ['lawyer','attorney','judge','notary']): core='提供法律意见、代理当事人或行使司法/公证职权'
  elif any(x in n for x in ['officer','military','naval','army','aviator','pilot','lieutenant','general','commander','commodore','cossack']): core='在军队或公共安全体系履行指挥、军职或执法责任'
  elif any(x in n for x in ['bishop','priest','televangelist','preacher','dayan']): core='担任宗教组织的教职、权威或带领信众'
  elif any(x in n for x in ['prince','emperor','lady-in-waiting']): core='承载王室/贵族身份或宫廷职务所赋予的治理、礼仪权责'
  elif any(x in n for x in ['activist','organizer','political','ambassador','secretary-general','mayor','parliamentarian','leader','consultant']): core='参与公共政策、政治组织、外交或社会倡议的决策和协调'
  else: core='承担机构治理、金融资本配置、商业经营管理或政治公共事务职责'
  return f'“{name}”的职责是{core}，对应公共治理、法律/军事/宗教权威或企业金融领导，归入Leadership。'
 if lev=='Sports/Games':
  if 'coach' in n or 'trainer' in n: core='训练运动员并制定项目训练/比赛策略'
  elif 'referee' in n or 'official' in n: core='依据项目规则执裁并维护比赛秩序'
  elif 'manager' in n: core='组织管理竞技队伍的选手、训练与赛事安排'
  elif 'gymnast' in n or 'skier' in n or 'skater' in n or 'climber' in n or 'player' in n or 'cyclist' in n or 'racer' in n or 'bobsleigher' in n or 'marksman' in n or 'fencer' in n: core='作为选手参加有规则的体育项目训练和竞技'
  elif 'black belt' in n: core='以武术级别身份从事武术训练或竞技'
  else: core='参与体育项目的训练、竞技或赛事组织'
  return f'“{name}”的工作直接围绕体育训练、竞技或比赛规则展开，具体职责是{core}，归入Sports/Games。'
 if lev=='Other':
  if any(x in n for x in ['worker','carrier','driver','courier','conductor','guard','cleaning','cleaner','caretaker','household','maid','charwoman','lackey','groom','sexton','babysitter','chef','personal organizer','herder','trapper','woodcutter','stonemasonry','stonecutter','farmer','car worker','technician','assistant','tax preparer','mechanic','operator','broker','trader','merchant','owner','contractor','consultant','publican','manicurist','paramedic','ems','corpsman']): core='提供标题所指的日常服务、运输、维修、照护、手工业或一般商业服务'
  else: core='承担明确的日常服务、普通职业劳动或小规模经营职责'
  return f'“{name}”是{core}的职业称谓，不属于艺术文化、科学技术、公共权力或竞技体育领域，归入Other。'
 return ''

def evidence_for(name,lev):
 if name=='Chapmen': return 'https://www.collinsdictionary.com/us/dictionary/english/chapmen'
 if name=='Doshu': return 'https://www.cs.brandeis.edu/~afeinman/aikido/vocab.html'
 if name=='Human computer': return 'https://www.nasa.gov/centers-and-facilities/jpl/when-computers-were-human'
 if name=='Rodeo clown': return 'https://prorodeo.cld.bz/2019-PRCA-Media-Guide-Barrelmen-and-Bullfighters/4'
 if name=='Kaymakam': return 'https://dergipark.org.tr/tr/download/article-file/211317'
 if name=='Taluk Office': return 'https://raichur.nic.in/en/taluk-office/'
 if name=='Pretty Horsebreaker': return 'https://wehd.com/farmer/Horsebreaker.html'
 if name=='Groom of the Stool': return 'https://en.wikipedia.org/wiki/Groom_of_the_Stool'
 if name=='Pretty Horsebreaker': return 'https://wehd.com/farmer/Horsebreaker.html'
 if name=='Road agent': return 'https://www.collinsdictionary.com/us/dictionary/english/road-agent'
 return 'occupation_meaning:'+name+'：'+({'Culture':'文化创作、表演、媒体传播或文化产品制作/发行职责','Discovery/Science':'科学研究、专业教育、医疗服务或工程技术开发职责','Leadership':'治理、法律、军职、宗教权威、外交或企业金融管理职责','Sports/Games':'体育/武术训练、竞技、执裁或队伍管理职责','Other':'明确的日常服务、一般劳动或小规模商业经营职责'}[lev])

rows=[]
with inp.open(encoding='utf-8-sig',newline='') as f: src=list(csv.DictReader(f,delimiter='\t'))
for r in src:
 name=r['raw_value']
 if name in uncertain:
  cand=uncertain[name][0]
  try: candidates=json.loads(cand)
  except Exception: candidates=[]
  rows.append({'rank':r['rank'],'raw_value':name,'semantic_level1':'','review_status':'needs_context','confidence':'low','semantic_rationale':uncertain[name][1],'evidence':{'Pretty Horsebreaker':'https://wehd.com/farmer/Horsebreaker.html','Road agent':'https://www.collinsdictionary.com/us/dictionary/english/road-agent','Taluk Office':'https://raichur.nic.in/en/taluk-office/'}.get(name,'occupation_meaning:'+name+'：仅凭词项无法确定稳定职业职责'),'candidate_l1s_json':json.dumps(candidates,ensure_ascii=False,separators=(',',':')),'reviewer':'bhht_semantic_batch_3'})
  continue
 lev=levels.get(name)
 if not lev: raise ValueError('unassigned title '+name)
 rationale=notes.get(name) or generic_reason(name,lev)
 if '#' in name or '(Profession)' in name:
  rationale += ' 词项中的编号及/或“(Profession)”是数据库区分标记，不改变该职业的职责含义。'
 if '(Film subject)' in name:
  rationale += ' 括号中的“Film subject”是人物条目语境标记，职业语义仍由midwife确定。'
 rows.append({'rank':r['rank'],'raw_value':name,'semantic_level1':lev,'review_status':'resolved','confidence':'high' if name in notes or lev in ('Sports/Games','Leadership') else 'medium','semantic_rationale':rationale,'evidence':evidence_for(name,lev),'candidate_l1s_json':json.dumps([lev],ensure_ascii=False,separators=(',',':')),'reviewer':'bhht_semantic_batch_3'})
fields=['rank','raw_value','semantic_level1','review_status','confidence','semantic_rationale','evidence','candidate_l1s_json','reviewer']
with out.open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} rows to {out}; explicit assignments {len(levels)}, needs_context {sum(x["review_status"]=="needs_context" for x in rows)}')
