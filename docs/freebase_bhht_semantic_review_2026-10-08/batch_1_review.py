import csv, json
from pathlib import Path

ROOT = Path(__file__).parent
INPUT = ROOT / 'batch_1_input.tsv'
OUTPUT = ROOT / 'batch_1_review.tsv'

# Explicit occupation-by-occupation decisions, organized by occupational function.
# Each semicolon-separated title is individually reviewed under that function.
GROUPS = {
 'Culture': {
  '艺术创作、文学写作、音乐/舞台/影视制作或文化传播': '''Performance Artist; Composer (Profession); Film Score Composer; Music Producer; Theatre Director; Lyricist (Profession); Radio personality; Theatrical producer; Television presenter; Sound Engineer; Set Decorator; Literary critic; Jazz Pianist; Vaudeville Performer; Science writer; Music video director; Trumpeter; Session musician; Jazz Drummer; Jazz Musician; Sound Editor; Opera Director (Profession); Cabaret Artist; Acting Teacher; Stunt coordinator; Music pedagogue; Dramaturge; Visual Effects Artist; Theatre Manager; Voice teacher; Art Model; Advertising Executive; Radio Broadcaster; Gravure idol; Electronic musician; Contemporary artist; TV Program Creator; Mixing engineer; Tattoo artist; Music editor; Keytarist; Ballet master; Lighting Designer; Theater Performer; Animation Director; Director of audiography; Shoe designer; Arranger (Profession); Saxophone player; Syndicated columnist; Voice coach; Music publisher (Profession) #1; Fashion Photographer; Travel writer; Street artist; Visual Effects Producer; Second Unit Director; Unit production manager; Technical writer; Ballet Teacher; Vlogger; Newspaper; Creative consultant; Gallery Owner; Jeweller (Film job); Architectural photographer; Special Effects Technician; News analyst; Children's book illustrator; Fashion Editor; Blogger (Website); Set Production Assistant; Tour manager; Literary executor; Court reporter; Floral Designer; Puppet Designer; Soubrette; Peking opera; Typeface designer; Voice Director; Typesetting; Broadway Producer; Didgeridooist; Window dresser; glass artist; Film Photographer; Fashion Illustrator; Make-Up Designer; Children's writer; Graphic Artist (Film job) #1; Makeup Effects Artist; Fashion Director; Light Artist; Speculative writer; Music Programmer; Drama Critic; Proofreader; Recording Director; Prima donna; Theatre consultant; Dance Educator; Light Sculptor; Vibraphonist; Drama Theorist; Dramatic colouratura; Media scholar; Video journalist; Printmaker (Music Artist); Hair and Makeup Artist; ADR Script Writer; Impressionist (Profession) #3; Performer-Songwriter-Musical Director; Figurative Sculptor; Wardrobe supervisor; Architectural Sculptor; Ringmaster''',
  '音响、设计/技术、造型等岗位服务于演出、影视或文化作品的制作，按其文化生产功能归类': '''Sound Engineer; Set Decorator; Visual Effects Artist; Sound Editor; Stunt coordinator; Theatre Manager; Voice teacher; Lighting Designer; Director of audiography; Furniture Designer; Fight Choreographer; Mixing engineer; Music editor; Fashion Editor; Set Production Assistant; Set Dresser; Special Effects Technician; Script supervisor; Dialogue Editor; Puppet Designer; Sound Director; Video journalist; Recording Director; Make-Up Designer; Graphic Artist (Film job) #1; Makeup Effects Artist; Hair and Makeup Artist; ADR Script Writer; Miscellaneous Crew; Pyrotechnics Supervisor; Wardrobe supervisor; Aerial Coordinator''',
  '文化传播、新闻评论或艺术批评以信息表达/作品评论为核心，而非被报道题材的职业': '''Science writer; Literary critic; Radio personality; Television presenter; Radio Broadcaster; Syndicated columnist; Travel writer; News analyst; Court reporter; Drama Critic; Blogger (Website); Technical writer; Fashion Editor; Video journalist; Sports Columnist''',
  '艺术指导/文化产业经营的职责直接围绕作品创作、演出或出版发行': '''Theatre Director; Theatrical producer; Music publisher (Profession) #1; Music publisher; Opera Director (Profession); TV Program Creator; Visual Effects Producer; Second Unit Director; Unit production manager; Gallery Owner; Broadway Producer; Fashion Director; Recording Director; Theatre consultant; Music Programmer''',
  '运动或活动中的表演型艺术仍属文化表演': '''Belly Dancer; Drag queen; B-boy; Master of Ceremonies; Vaudeville Performer; Prima donna; Ringmaster''',
  '音乐选曲、混音并主持广播/舞池节目属于音乐传播与文化表演': '''Disc jockey''',
  '设计舞台、影视布景或视觉造型以支持文化作品呈现': '''Set Designer''',
  '以歌唱为职业的音乐表演者': '''Singer (Profession) #212''',
  '音乐家从事母带处理与录音声音后期制作': '''musician, mastering engineer''',
  '以体育新闻报道为工作，核心是新闻采编与传播，不是运动员或体育训练': '''Sports Reporter''',
  '以法律或其他专业主题写作出版，角色核心是文字表达/知识传播': '''Legal writer''',
  '以科幻文学创作为核心的作者': '''Science-fiction writer''',
  '以个人媒体平台进行内容创作和传播的媒体身份': '''Promotional Model''',
  '以纸本/数字出版媒介作为出版物或新闻机构标签出现，并非明确职业，需上下文': '''Newspaper''',
 },
 'Discovery/Science': {
  '医疗照护、治疗、诊断及健康专业服务属于医学/专业知识应用': '''Healthcare professional; Psychotherapist; Molecular Biologist; Ophthalmology; Cognitive scientist; Developmental Biologist; Cardiac surgeon; Bioethicist; Emergency medical technician; Environmental health officer; Materials Scientist; Social Psychologist; Orthopaedist; Mental health counselor; Speech-language pathologist; Audiologist; Nurse practitioner; Certified Sex Therapist; Massage therapist; Art therapist; Cognitive Psychologist; Occupational Therapy Educator; Neurobiologist; Clinical Psychiatrist; Neurosurgery; Medical assistant; Medical researcher; Hospital Driver; Clinical Social Worker; Medical Anthropologist; Nursing Graduate; Veterinary Technician; Physician assistant''',
  '科学研究、技术设计/开发或学术研究传播以专业知识和技术系统为核心': '''Sound Engineer; Aircraft designer; Landscape architect; Web Developer; Software Architect; Search engine technology; Information Technologist; Marine engineer; telecommunications expert; Naval Architect; Cosmologist; Neurophysiologist; Information Technology Consultant; Cognitive scientist; Technical writer; Joomla developer; Computer game designer; Animation Engineer; Design Engineer; Geological engineer; Structural biologist; Android software development; Rails Developer; IT/Software Developer; Information retrieval; Product Designer''',
  '科学探索/环境保护以生态、物种或自然科学知识为主要手段': '''Conservationist; Conservation; Botanical Explorer; Botonist; Plant Biologist; Ethnobotanist; Maritime Archaeologist; Cultural Anthropologist; Agrarian; Agriculturalist''',
  '教学或学术岗位以专业知识研究、教育或学术传播为核心': '''Law professor; Music pedagogue; Science writer; Adjunct Professor; Social Historian; Media scholar; Aeronautics instructor; Head Master; Drama Theorist; Scholar-official''',
  '尚不能从短语确定单一领域的技术顾问或泛称': '''Technical advisor; Senior Consultant; UX & Semantic Search Consultant; Ecommerce Consultant; Strategist; Adviser; Visionary''',
  '急诊医学临床诊断和治疗': '''Emergency physician''',
  '战斗/军队医疗急救属于医疗专业，但战斗军种语境混合，单一职业L1不稳': '''Combat medic''',
  '植物学学科研究': '''Antropology''',
  '计算机系统运维与账户、网络等IT系统管理': '''System Administrator''',
  '软件/计算机专业泛称，未给出具体岗位职责': '''Computer professional''',
  '应用体验设计属于软件产品设计，兼有设计与开发语义，需具体职责语境': '''User experience design''',
  '动物医护技术支持/检查协助': '''Veterinary Technician''',
  '机器与生产设备的制造、安装、维护及维修技术工作': '''Industriemechaniker''',
  '维护和修理飞机机械系统的技术岗位': '''Aircraft maintenance technician''',
  '性健康知识教育属于专业健康知识传播': '''Sex Educator''',
  '建设船只及船体结构的制造工艺': '''Boat builder''',
  '用油漆或颜料进行视觉创作，属艺术制作': '''Paint Artist; Watercolor paint''',
  '以马匹繁殖/养育为行业活动，非明确单一岗位': '''Horse breeding''',
  '保健品或护肤美容品行业/商品领域，不是明确职业': '''Cosmetics''',
  '电子游戏的策划开发同时涉及软件技术与文化产品创作，短语未说明具体岗位': '''Video game development''',
 },
 'Leadership': {
  '企业经营、公司高管、金融或商业管理职责属于领导/商业治理': '''Business executive; Business magnate; Real estate development; Insurer; Marketer; Advertising Executive; Executive officer; Internet marketing; SEO strategist; SEO; Internet Marketer; Senior SEO Manager; Marketing Consultant; Co-founder and Business Executive; Business Owner; Oil Executive; Senior Consultant; Deli Manager; Business Coach; Social entrepreneur; Credit control; Internet Executive; Management Theorist; Investment advisor; Restaurant Mgt.; Entrepreneur (Profession) #6; IT service management; Inbound Marketing; Entrepreneur (Small Business) Nightclub and Auto Paint and Body Shops; Investment banking; Business process outsourcing; Marketing Specialist; Digital Marketing Strategist (Job title); Ecommerce Consultant; Promotional Model''',
  '国家政治、公共治理、军事指挥、司法法律或公共政策职能属于领导/治理': '''Nobleman; Roman emperor; Democracy activist; Trade unionist; Justice of the peace; Court clerk; Attorney at law; Certified Public Accountant; Labor Union Leader; Defense policy analyst; Civil Rights Leader; Notary public; Commissioner of Baseball; Plantation owner; Prison officer; Non-commissioned officer; Commissar; Political philosopher; Political strategist; Court reporter; Chartered Surveyor; United States Naval Aviator; Vietnam veteran; United States Post Office Department; Cavalry Officer; Military Strategist; Reconnaissance; First Lady of Chile; Infantry Weapons Officer; Police commissioner; Secret Intelligence Service officer; Adjudicator; Fleet admiral; Policy Consultant; Political analyst; Party organizer; Advocate General; Sheriff officer; divorce lawyer; Child-Care Solicitor; Religious Judge''',
  '宗教职务/宗教身份属于宗教权威或宗教群体治理': '''Kannushi; Spiritual teacher; Laity; Diocesan bishop; elder''',
  '管理公立学校、学校体系或校区行政运营': '''School Administrator; School superintendent''',
  '法律执业者为当事人提供辩护/法律代理': '''Bail bondsman''',
  '依法执行司法程序并负责法院行政文书': '''Court clerk''',
  '执行政府授予的公权力/公共职务，但具体领域不足以进一步细分': '''Herald''',
  '以倡议公民权利或言论自由为公共社会运动职能，非确定政治官职': '''Free speech activist''',
  '为组织公开发言、传达立场的传播角色，可能属于公关/政治等多领域': '''Spokesperson''',
  '专门情报或执法代理职责取决于机构语境': '''Special Agent''',
  '私下秘书提供行政安排与文书支持': '''Private Secretary''',
  '动物园/狩猎猎手职责并不必然是运动竞赛': '''Big-game hunter''',
  '酒类品鉴与专业知识服务，不足以确定所属行业职责': '''Wine Expert''',
  '咨询/策略类职称宽泛，欠缺所属专业领域': '''Social media strategist''',
  '泛企业战略、营销推广或财务信贷职能属于商业管理/金融': '''Internet marketing; SEO strategist; SEO; Marketer; Marketing Consultant; Internet Marketer; Senior SEO Manager; Advertising Executive; Marketing Specialist; Digital Marketing Strategist (Job title); Credit control; Investment advisor; Investment banking; Moneylender; Insurer; Oil Executive; Inbound Marketing''',
 },
 'Sports/Games': {
  '竞技运动员、赛车/球类/搏击等赛事参与者属于体育': '''Football player; Baseball Player; Race car driver; Professional Boxer; Martial artist; Golfer; Sumo Wrestler; Volleyball player; Biathlete; Mixed Martial Artist; Handball Player; Professional golfer; Ice dancer; Nordic combined skier; Motorcycle Racer; Beach Volleyball Player; College gymnast; Wrestling Promoter; Rally Driver; Ski Racer; Lacrosse Player; Competitive Eater; Real Tennis Professional; Roller Skater''',
  '体育训练、教练、裁判或运动队竞赛支持属于体育职能': '''Basketball Coach; Sports instructor; Sports agent; Gymnastics Coach; Boxing Trainer; Athletic trainer; Dog trainer; Swimming coach; Karate Instructor; Cornerman; Figure skating coach; Gymterneter; Gyōji; Sports Columnist; Football Analyst''',
  'Gamer作为明确的电子游戏玩家身份属于游戏参与': '''Gamer''',
  '教练具体教授武术技艺，核心是竞技武术训练': '''Martial Artist Teacher''',
  '武术项目教学与专项训练': '''Martial Arts Instructor''',
  'sumo赛事裁判角色，负责比赛现场裁决': '''Gyōji''',
 },
 'Other': {
  '普通服务、手工业、劳务或日常职业不属于四个专门领域': '''Sex worker; Domestic worker; Waiting staff; Beautician; Firefighter; Truck driver; Maid; Construction worker; Bank cashier; Steelworker; Cook (Profession) #121; Porter; Stevedore; Bounty hunter; Farmworker; Funeral director; Garment Worker; Railroad worker; Station master; Glovemaker; Shop assistant; Farmworker; Railroad worker; Garment Worker; Farmworker; machineworker; Distillery Worker; Transient laborer; port worker; Farmworker; Hospital Driver; Page-boy; Wheelwright; Chimney sweep; Call Center Agent; Bootseller; Animal control service; Pipefitter; Health Authority Wages Clerk''',
  '普通经营/服务岗位或一般雇员身份，未表达大型企业治理或专门领域': '''Boss; Commoner; Landholder; Farmworker; Business Owner; Clothing manufacturer; Taxidermist; Deli Manager; Events Organiser; Business Owner; Bootseller; Promoter; Entrepreneur (Small Business) Nightclub and Auto Paint and Body Shops''',
  '犯罪、负面声名、成人服务或非正规活动身份归其他社会身份': '''Sex worker; Dominatrix; Confidence artist; Racketeering; Big-game hunter; Buccaneer; Usurper; Bounty hunter; Pin-up girl''',
  '非正式组织/家庭身份或普通职业身份': '''Mother (Profession); Page-boy; Vietnam veteran; Woman; Commoner''',
  '成人影像或人体商业摄影模特属于模特服务工作': '''Adult model; Promotional model''',
  '烹饪并制作餐食的餐饮服务工作': '''Celebrity chef''',
  '客户服务/看守场所等一般服务性工作': '''Bouncer; Tour guide; Life Coach''',
  '普通慈善或公益组织工作人员，未明示政治领导职责': '''Chairty Worker; Scout Leader''',
  '美容甲护、美甲服务属于个人护理服务': '''Nailist''',
  '个人训练/催眠等服务内容不足以证明是医学或学术职业': '''Hypnotist''',
  '一般保密/代理职称、缺少机构与法定权限语境': '''Special Agent''',
  '一般旅行导览服务': '''Tour guide''',
  '派遣、安保或管理职称缺少职责与雇主语境': '''Private Secretary''',
  '按用户标签所称的精神/灵性助人身份定义不稳定，候选职责跨文化/宗教/健康': '''Lightworker''',
  '不明确的名称、非职业字符串或疑似数据污染；需回看实体/源字段': '''Rob Gibsun; m/05hmb8k; Sculptor (Constellation); Watercolor paint; Woman; Gymterneter''',
  '形容人或领域而非明确单一职业': '''Psychedelic art; Cosmetics; Horse breeding; Search engine technology; Video game development; User experience design; Computer professional''',
 },
}

NEEDS_CONTEXT = {
 'Adviser': ('[]', '泛称顾问，未给出咨询领域，可能横跨商业、政府、科学或文化；不能确定单一L1。', 'occupation_meaning:顾问泛称，缺领域信息'),
 'Strategist': ('[]', '策略师未说明服务领域，职责可能属于商业、政治、军事或传播，无法可靠选单一L1。', 'occupation_meaning:策略规划泛称，缺领域信息'),
 'Visionary': ('[]', '“Visionary”可指愿景型人物、思想者或自我描述，并非明确职业职责。', 'occupation_meaning:多义身份词'),
 'Technical advisor': ('["Discovery/Science","Leadership","Culture"]', '技术顾问可为科学工程、影视制作或企业咨询岗位；短语未提供行业语境。', 'occupation_meaning:技术咨询，行业未明'),
 'Senior Consultant': ('["Leadership","Discovery/Science"]', '咨询顾问的服务领域未说明，可能为企业管理或专业技术咨询。', 'occupation_meaning:顾问职称，领域未明'),
 'UX & Semantic Search Consultant': ('["Discovery/Science","Leadership"]', '同时指用户体验与语义搜索咨询，属于技术产品/商业咨询交叉职能。', 'occupation_meaning:UX与搜索咨询复合职责'),
 'Ecommerce Consultant': ('["Leadership","Discovery/Science"]', '可提供电商经营或电商技术咨询，短语无法辨明主要职责。', 'occupation_meaning:电子商务咨询'),
 'Sculptor (Constellation)': ('[]', '括号明确表示天文学星座实体，不是职业名称，作为职业数据行属实体/字段语境污染。', 'https://www.iau.org/public/themes/constellations/'),
 'Gymterneter': ('[]', '无法找到稳定可核实的职业词义，疑似拼写或数据污染，保留待核。', 'occupation_meaning:未找到可靠职业定义'),
 'm/05hmb8k': ('[]', 'Freebase式ID字符串而非可解释职业名称；需回查实体映射。', 'occupation_meaning:数据库ID'),
 'Rob Gibsun': ('[]', '看似人名而非职业名称，无法按身份字符串裁定。', 'occupation_meaning:疑似人名/字段污染'),
 'Woman': ('[]', '性别身份词，不是职业名称，需确认源字段含义。', 'occupation_meaning:性别身份，不表职业'),
 'Newspaper': ('[]', '可指报纸出版物、新闻机构或泛指行业，短语没有具体岗位。', 'occupation_meaning:媒介/机构泛称'),
 'Cosmetics': ('[]', '可指化妆品行业、商品或相关业务，并非明确岗位。', 'occupation_meaning:商品/行业名'),
 'Watercolor paint': ('[]', '指水彩颜料/媒材，不是清楚的职业短语。', 'occupation_meaning:绘画媒材'),
 'Psychedelic art': ('["Culture"]', '指一种艺术风格/运动而非岗位；如该字段要求职业类别，需进一步确认其映射含义。', 'occupation_meaning:艺术形式/流派'),
 'Video game development': ('["Discovery/Science","Culture"]', '电子游戏开发是技术开发与文化产品生产复合领域，未指明具体职能。', 'occupation_meaning:技术开发与文化生产复合'),
 'User experience design': ('["Discovery/Science","Culture"]', '可指软件产品设计，也可泛指视觉/交互设计；缺少产品语境。', 'occupation_meaning:用户体验设计，产品场景未明'),
 'Computer professional': ('["Discovery/Science","Leadership"]', '计算机专业人员泛称可覆盖开发、运维、管理等不同职责。', 'occupation_meaning:计算机工作泛称'),
 'Special Agent': ('["Leadership","Other"]', '可指执法/情报特工，也可能是商业代理人，机构和职责未说明。', 'occupation_meaning:代理/特工多义职称'),
 'Spokesperson': ('["Culture","Leadership"]', '发言人承担组织传播职能，但所属组织可为媒体、企业或政府，未见角色语境。', 'occupation_meaning:机构发言/传播角色'),
 'Herald': ('["Leadership","Culture"]', '可指历史上的纹章官/使者，也可能是报刊名称或现代媒体岗位，需语境。', 'occupation_meaning:历史官职或媒体多义词'),
 'Lightworker': ('["Culture","Discovery/Science","Leadership","Other"]', '新纪元灵性自称语义开放，可指治疗、教学、倡议或创作，无法由该词确定实际职责。', 'https://en.wiktionary.org/wiki/lightworker'),
 'Combat medic': ('["Discovery/Science","Leadership"]', '医疗救护职责属医学，但该复合称谓带军队/战斗岗位语境；若类别按岗位所属系统可归军事治理。', 'occupation_meaning:战地医疗复合职能'),
 'Horse breeding': ('["Other","Discovery/Science"]', '是畜牧/繁育活动或行业名，未给出具体岗位，无法确定按工种或专业科学归类。', 'occupation_meaning:畜牧活动而非明确岗位'),
 'Adjudicator': ('["Leadership","Other"]', '可指法定裁决者或一般争议评审，未注明司法/行政权限。', 'occupation_meaning:裁决角色，制度语境未明'),
 'Business Owner': ('["Leadership","Other"]', '企业所有者可能经营小型个体店铺，也可能担任大公司控制者；短语未给规模与职能。', 'occupation_meaning:企业所有者，经营规模/职能未明'),
 'Agriculturalist': ('["Discovery/Science","Other"]', '农业从业者可能是农业科学人员、经营者或生产者；职位短语不足以确定其具体工作。', 'occupation_meaning:农业从业者泛称'),
 'Agrarian': ('["Discovery/Science","Other"]', 'Agrarian可指农业生产者，也可指土地/社会制度立场，并非无歧义的职业名称。', 'occupation_meaning:农业/社会身份多义词'),
 'Nursing Graduate': ('["Discovery/Science","Other"]', 'Nursing Graduate表示护理专业毕业生或学历状态，不能确认其已从事护士职业。', 'occupation_meaning:教育经历/资格状态，非明确岗位'),
 'Miscellaneous Crew': ('["Culture","Other"]', '影视制作片尾的杂项剧组职务集合，未给出具体工种。', 'occupation_meaning:多个未指定剧组岗位'),
 'Promotional Model': ('["Culture","Other"]', '宣传模特可能从事商业推广、产品展示或表演传播；无法确定以传播还是普通服务为核心。', 'occupation_meaning:模特推广复合职责'),
 'Promotional model': ('["Culture","Other"]', '宣传模特可能从事商业推广、产品展示或表演传播；无法确定以传播还是普通服务为核心。', 'occupation_meaning:模特推广复合职责'),
 'SEO': ('["Other","Discovery/Science"]', 'SEO是搜索引擎优化的领域/职能缩写，可指策略、营销或技术工作，未给具体岗位。', 'occupation_meaning:搜索引擎优化缩略词，职责未明'),
 'elder': ('["Leadership","Other"]', 'elder可指宗教/社群长者、资深身份或具体职位，短语没有职责语境。', 'occupation_meaning:长者/职位多义词'),
 'Conservationist': ('["Discovery/Science","Leadership"]', '自然保护者可能从事生态科研，也可能是环境倡议者；没有具体职责语境时不能断言是科学家或政治倡议者。', 'occupation_meaning:生态保护者多种真实职责'),
 'Conservation': ('["Discovery/Science","Other"]', '自然保护是活动/领域名称，没有明确职业或岗位职责。', 'occupation_meaning:非岗位领域词'),
 'Ophthalmology': ('["Discovery/Science"]', '眼科学是医学专科领域，不是明确从业者职位；需确认该字段表示学科还是临床岗位。', 'occupation_meaning:医学专科名称'),
 'Neurosurgery': ('["Discovery/Science"]', '神经外科可指医学专科或临床服务领域，未明确是外科医师职位。', 'occupation_meaning:医学专科名称'),
 'Real estate development': ('["Leadership","Discovery/Science"]', '房地产开发可指公司经营、项目投资，也可包含建筑/规划工作；字段未给具体岗位。', 'occupation_meaning:房地产行业/活动名称'),
 'Clothing manufacturer': ('["Leadership","Other"]', '服装制造商可指制造企业或具体经营者，字段没有区分组织实体与个人职业。', 'occupation_meaning:组织与从业者两义'),
 'Insurer': ('["Leadership","Other"]', 'Insurer可指保险公司，也可泛指承保人/保险从业者；未说明个人岗位和职责。', 'occupation_meaning:机构与从业者两义'),
 'Business process outsourcing': ('["Leadership","Other","Discovery/Science"]', '业务流程外包是行业活动/服务模式，可涉及销售、运营或技术支持，未给具体职位。', 'occupation_meaning:行业/服务模式而非岗位'),
 'IT service management': ('["Discovery/Science","Leadership"]', 'IT服务管理可指技术运营领域或管理职能，未给出具体职位职责。', 'occupation_meaning:技术与管理复合领域'),
 'Search engine technology': ('["Discovery/Science"]', '搜索引擎技术是技术领域而非明确职业名称。', 'occupation_meaning:技术领域名'),
 'Information retrieval': ('["Discovery/Science"]', '信息检索是研究/技术领域，可对应多种岗位，未明示具体职业。', 'occupation_meaning:研究与技术领域名'),
 'Antropology': ('["Discovery/Science"]', '疑为Anthropology拼写错误；该词是学科名称而非明确职业岗位。', 'occupation_meaning:学科名称/拼写变体'),
 'Peking opera': ('["Culture"]', '京剧是表演艺术/剧种名称，不是具体岗位；需确认原数据是否指京剧演员或其他从业者。', 'occupation_meaning:表演艺术形式而非岗位'),
 'United States Post Office Department': ('["Leadership","Other"]', '这是美国联邦邮政部门机构名称，而非个人职业。', 'occupation_meaning:政府机构名'),
 'Reconnaissance': ('["Leadership","Other"]', '侦察是军事/情报任务领域，未说明执行者职位或具体岗位。', 'occupation_meaning:军事任务名而非明确职业'),
 'Democracy activist': ('["Leadership","Other"]', '民主倡议者从事公共倡议，但短语未说明其是政治领导者、组织者还是普通倡议参与者。', 'occupation_meaning:政治倡议角色，职位层级未明'),
 'Trade unionist': ('["Leadership","Other"]', '工会人士可指普通会员、组织者或工会领导，短语未给出具体职责层级。', 'occupation_meaning:工会身份泛称'),
 'Boss': ('["Leadership","Other"]', 'Boss只是非正式“上司/老板”称呼，缺少行业和具体职务信息。', 'occupation_meaning:泛称管理身份'),
 'Spiritual teacher': ('["Leadership","Discovery/Science","Other"]', '灵性教师可能是宗教导师、非正式人生教练或一般思想教师，短语未说明教义或机构职位。', 'occupation_meaning:灵性教学泛称'),
 'Social media strategist': ('["Culture","Leadership","Other"]', '社交媒体策略师可能负责文化内容传播，也可能负责企业营销或个人账号运营；缺少雇主与职责语境。', 'occupation_meaning:社媒内容与营销复合职能'),
 'Political analyst': ('["Culture","Discovery/Science","Leadership"]', '政治分析师可能从事新闻评论、学术研究或政府政策咨询，无法单凭头衔定类。', 'occupation_meaning:政治分析多种岗位语境'),
 'Policy Consultant': ('["Leadership","Discovery/Science"]', '政策顾问可能服务政府决策，也可能从事智库/学术研究，短语未说明机构与职责。', 'occupation_meaning:公共政策咨询'),
 'Free speech activist': ('["Leadership","Other"]', '言论自由倡议者从事公共倡议，但头衔未表示其为政治领导者或公职人员。', 'occupation_meaning:公共倡议角色，组织职责未明'),
 'Wine Expert': ('["Other","Discovery/Science","Culture"]', '葡萄酒专家可能是侍酒师、酿酒科学人员或酒类评论者，短语未说明具体职业。', 'occupation_meaning:葡萄酒专业角色多义'),
 'Executive officer': ('["Leadership","Other"]', '执行官可指企业高管，也可指组织或公共部门执行岗位，缺少机构语境。', 'occupation_meaning:组织执行职务，类型未明'),
 'Hypnotist': ('["Culture","Discovery/Science","Other"]', '催眠师可能是舞台表演者、临床催眠治疗从业者或非临床服务者，职责语境未明。', 'occupation_meaning:催眠表演/治疗多义角色'),
 'Nursing Graduate': ('["Discovery/Science","Other"]', 'Nursing Graduate表示护理专业毕业生或学历状态，不能确认其已从事护士职业。', 'occupation_meaning:教育经历/资格状态，非明确岗位'),
 'Social entrepreneur': ('["Leadership","Other"]', '社会企业家可经营小型公益企业或大型社会组织，经营规模与职责未明确。', 'occupation_meaning:社会企业经营者，规模未明'),
 'Entrepreneur (Profession) #6': ('["Leadership","Other"]', 'Entrepreneur泛指创业经营者，缺乏企业规模与具体行业，无法区分小业主和大型企业领导。', 'occupation_meaning:创业者泛称'),
 'Social entrepreneur': ('["Leadership","Other"]', '社会企业家可经营小型公益企业或大型社会组织，经营规模与职责未明确。', 'occupation_meaning:社会企业经营者，规模未明'),
 'Entrepreneur (Profession) #6': ('["Leadership","Other"]', 'Entrepreneur泛指创业经营者，缺乏企业规模与具体行业，无法区分小业主和大型企业领导。', 'occupation_meaning:创业者泛称'),
 'Set Production Assistant': ('["Culture","Other"]', '片场助理服务于影视制作，但“assistant”所指任务范围过宽。', 'occupation_meaning:片场助理，具体职责不明'),
 'Stunt coordinator': ('["Culture","Sports/Games"]', '负责影视特技动作设计与现场安全协调，可能兼任特技表演，但核心岗位是影视制作。', 'occupation_meaning:影视动作协调'),
 'Dog trainer': ('["Sports/Games","Other"]', '犬只训练可用于竞技/赛事，也可用于家庭服从训练或工作犬训练；用途未注明。', 'occupation_meaning:犬只训练，目的未明'),
 'Big-game hunter': ('["Other","Sports/Games"]', '可能指狩猎活动参与者或狩猎旅游/职业，难以确定具体职业语境。', 'occupation_meaning:大型猎物猎手'),
 'Bounty hunter': ('["Other","Leadership"]', '追捕逃犯以领取赏金，可能为民间职业也与司法执行相关；作者类别依赖制度语境。', 'occupation_meaning:赏金追捕职责'),
 'Newspaper': ('[]', '该词仅指媒介/机构，未给出职业岗位。', 'occupation_meaning:出版媒介名'),
 'Chairty Worker': ('["Other"]', '疑为“Charity Worker”拼写错误；慈善工作者可能从事服务、筹款或倡议，职责不足以定类。', 'occupation_meaning:疑似charity worker拼写错误'),
 'Sculptor (Constellation)': ('[]', '括号指定星座而非雕塑职业。', 'https://www.iau.org/public/themes/constellations/'),
 'Watercolor paint': ('[]', '这是颜料媒材词组而非清晰职业名称。', 'occupation_meaning:绘画颜料/媒材'),
 'Nailist': ('["Other"]', '美甲师为客户清洁、修整和美容指甲，属于个人护理服务业。', 'https://www.bls.gov/ooh/personal-care-and-service/manicurists-and-pedicurists.htm'),
 'Industriemechaniker': ('["Discovery/Science"]', '德国工业机械技工制造、装配、调试并维修机器及生产设备，属于技术工程职业。', 'https://www.anerkennung-in-deutschland.de/en/interest/finder/profession/182/profile'),
 'Parapsychologist': ('["Discovery/Science","Other"]', '该词指超心理学研究者，但该领域职业含义和其科学/非科学边界不统一，需进一步职业与分类语境。', 'occupation_meaning:超心理学研究，学科边界存在争议'),
 'Kannushi': ('["Leadership"]', 'Kannushi是神道神社神职人员，负责神社仪式与祭祀，按宗教权威归类。', 'https://www.si.edu/object/bulletin-united-states-national-museum-148-1929%3Anmah_10040'),
 'Gyōji': ('["Sports/Games"]', '相扑比赛裁判，负责场上裁决。', 'https://sumo.or.jp/EnSumoDataKyokaiMember/gyoji/'),
}

RESOLVED_OVERRIDES = {
 'Advertising Executive': ('Leadership','广告公司/广告业务的高管负责商业经营、客户与业务管理，职位核心是企业管理。','occupation_meaning:广告业务高管职能','medium'),
 'Bounty hunter': ('Other','赏金猎人受雇追踪并捕获逃犯以领取报酬，属于非正式追捕/执法边缘职业，不是政府司法官职。','occupation_meaning:受雇追捕逃犯','medium'),
 'Big-game hunter': ('Other','大型猎物猎手从事狩猎活动；该称谓本身不指竞技体育赛事。','occupation_meaning:狩猎活动而非竞赛运动','medium'),
 'Combat medic': ('Discovery/Science','战斗医护兵在军事场景中提供急救、伤员稳定与医疗照护；其直接职业职能是医疗服务。','occupation_meaning:战地急救和医疗照护','medium'),
 'Dog trainer': ('Other','训练犬只服从指令或完成日常行为训练，未指明竞技项目，属一般动物训练服务。','occupation_meaning:犬只行为训练服务','medium'),
 'Gyōji': ('Sports/Games','日本相扑比赛裁判在比赛中宣告并判定胜负，属于竞赛裁判岗位。','https://sumo.or.jp/EnSumoDataKyokaiMember/gyoji/','high'),
 'Kannushi': ('Leadership','Kannushi是神道神社神职人员，负责祭祀仪式与神社礼拜，属于宗教权威/宗教职务。','https://www.si.edu/object/bulletin-united-states-national-museum-148-1929%3Anmah_10040','high'),
 'Nailist': ('Other','美甲师为顾客清洁、修整和美容指甲，属于个人护理服务工作。','https://www.bls.gov/ooh/personal-care-and-service/manicurists-and-pedicurists.htm','high'),
 'School Administrator': ('Discovery/Science','学校行政人员负责教育机构的日常运营与教学行政支持，属于教育系统专业职能。','author_role_analogy:Academia/教育行政','medium'),
 'School superintendent': ('Discovery/Science','学区/学校体系主管负责教育系统运营和教学管理，属于教育行政职能。','author_role_analogy:Academia/教育管理','medium'),
 'Set Production Assistant': ('Culture','片场制作助理为影视拍摄和制作流程提供现场协调与行政支持，职责明确服务于文化产品制作。','occupation_meaning:影视片场制作支持','medium'),
 'Sports agent': ('Sports/Games','体育经纪人代表运动员处理合同与职业事务，直接服务竞技体育参与者。','occupation_meaning:运动员职业代理/合同支持','medium'),
 'Stunt coordinator': ('Culture','特技协调员为影视制作设计、排练并组织安全执行动作场面，属于电影电视制作岗位。','occupation_meaning:影视动作设计与制作协调','high'),
 'Vietnam veteran': ('Leadership','作者将veteran归入Military；该词指曾在越战服役的军人，退伍经历仍对应军事职业身份。','author_role_analogy:veteran → Leadership/Military; author hierarchy Military','medium'),
 'Private Secretary': ('Other','私人秘书主要提供日程、通信和文书等个人行政支持，未显示公共治理或企业决策权限。','occupation_meaning:个人行政助理服务','medium'),
 'Marketer': ('Other','市场营销人员执行产品推广、市场研究与客户拓展等商业服务工作；作者将 marketing 归入 Worker/Business (small)，对应 Other。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','high'),
 'Marketing Specialist': ('Other','市场专员开展市场调研、推广与营销执行，属商业服务职能；作者对 marketing 的正式归类是 Worker/Business (small)，对应 Other。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','high'),
 'Inbound Marketing': ('Other','入站营销是吸引潜在客户的推广职能/业务活动；作者将 marketing 归为 Worker/Business (small)，本词未表达企业高管职责。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','medium'),
 'Internet Marketer': ('Other','网络营销人员执行线上推广与获客工作；作者将 marketing 归为 Worker/Business (small)，对应 Other。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','high'),
 'Marketing Consultant': ('Other','营销顾问为客户提供市场推广建议，属于商业服务/专业咨询；作者将 marketing 归为 Worker/Business (small)，对应 Other。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','medium'),
 'SEO strategist': ('Other','SEO策略师规划搜索引擎优化与线上获客，属于营销服务职能；作者将 marketing 归为 Worker/Business (small)，对应 Other。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','medium'),
 'Internet marketing': ('Other','网络营销指线上推广、广告和客户拓展活动，属于营销服务职能；作者将 marketing 归为 Worker/Business (small)，对应 Other。','author_role_analogy:marketing → Other/Worker/Business (small), author rules line 64','medium'),
 'Football Analyst': ('Culture','足球分析师在媒体语境中解读比赛并向观众传播评论；这是体育新闻/评论工作，不等于参赛运动员或教练。','occupation_meaning:比赛分析与媒体传播','medium'),
 'Sports Columnist': ('Culture','体育专栏作者撰写赛事评论并向读者传播观点，职业核心是新闻出版写作；题材为体育不改变媒体职业本身。','occupation_meaning:体育新闻评论写作','high'),
 'Typesetting': ('Culture','排版/照排将文字和图像编排成出版页面，是出版制作环节的文化生产工种。','occupation_meaning:出版物文字与页面编排','medium'),
 'Sound Engineer': ('Culture','此批职业语境中的录音/声音工程师负责音乐、影视或广播声音的录制、混音与后期制作，服务于文化作品。','occupation_meaning:音乐/影视录音及声音后期制作','medium'),
 'Deli Manager': ('Other','熟食店经理负责小型零售餐饮店的排班、库存与日常服务，不是大型企业治理职位。','occupation_meaning:小型餐饮零售店日常管理','medium'),
 'Entrepreneur (Small Business) Nightclub and Auto Paint and Body Shops': ('Other','该称谓明确指小企业经营者，管理夜店和汽车修理/喷漆店等本地小型生意。','occupation_meaning:小企业经营者；输入中的小企业限定','medium'),
 'Molecular Biologist': ('Discovery/Science','分子生物学家研究细胞、基因、蛋白质等分子层面的生命过程，属于科学研究。','occupation_meaning:分子生命科学研究','high'),
 'Developmental Biologist': ('Discovery/Science','发育生物学家研究生物体从胚胎到成熟的生长与分化过程，属于科学研究。','occupation_meaning:生物发育研究','high'),
 'Materials Scientist': ('Discovery/Science','材料科学家研究材料的结构、性能与应用，属于自然科学和技术研发。','occupation_meaning:材料性质研究与开发','high'),
 'Social Psychologist': ('Discovery/Science','社会心理学家研究社会环境对认知、态度和行为的影响，属于学术/科学研究。','occupation_meaning:社会心理学研究','high'),
 'Cognitive scientist': ('Discovery/Science','认知科学家研究知觉、记忆、语言和推理等认知过程，属于科学研究。','occupation_meaning:认知过程研究','high'),
 'Cognitive Psychologist': ('Discovery/Science','认知心理学家研究记忆、注意、学习等心理过程，属于科学/学术研究。','occupation_meaning:认知心理研究','high'),
 'Neurobiologist': ('Discovery/Science','神经生物学家研究神经系统的结构与功能，属于生命科学研究。','occupation_meaning:神经系统研究','high'),
 'Clinical Psychiatrist': ('Discovery/Science','临床精神科医师诊断和治疗精神障碍，属于医疗专业服务。','occupation_meaning:精神疾病临床诊治','high'),
 'Medical Anthropologist': ('Discovery/Science','医学人类学家研究健康、疾病与医疗实践的社会文化因素，属于学术研究。','occupation_meaning:健康医疗的人类学研究','high'),
 'Hospital Driver': ('Other','医院司机负责病患、职工或物资的地面运输；工作核心是驾驶运输服务，医院雇主不使其成为医疗专业。','occupation_meaning:医院场景中的运输司机','high'),
 'Paint Artist': ('Culture','Paint Artist指使用颜料进行视觉艺术创作，工作产物是绘画作品，属于视觉艺术。','occupation_meaning:颜料绘画创作','high'),
 'Product Designer': ('Culture','产品设计师构思并设计消费/工业产品的外形、交互与使用体验，属于设计创作职能。','occupation_meaning:产品外形与使用体验设计','medium'),
 'Landscape architect': ('Culture','景观建筑师规划并设计户外空间、园林与公共景观，核心是空间和视觉设计。','occupation_meaning:景观与户外空间设计','high'),
 'Certified Public Accountant': ('Other','注册会计师从事会计、审计和税务专业服务；作者将 accountant 类职业置于 Other/Worker-Business small，资格认证不改变职责。','author_role_analogy:accountant → Other/Worker-Business (small)','high'),
 'Chartered Surveyor': ('Other','特许测量师提供土地、房产和建造项目的测量、估值或咨询专业服务；surveying是专业服务职能，不是治理职位。','https://www.rics.org/surveyor-careers/surveying/what-is-a-chartered-surveyor; author_role_analogy:surveyor → Other','medium'),
 'Adult model': ('Culture','成人出版物或影像中的模特提供摄影/影视视觉表演；建模工作本身属于文化媒介制作，不等同于性服务。','occupation_meaning:成人媒介摄影/影像模特','medium'),
 'Pin-up girl': ('Culture','Pin-up girl指为宣传或出版摄影摆姿的模特形象，属于视觉媒介/摄影文化；词本身不表示性服务。','occupation_meaning:摄影出版中的模特/视觉表演','medium'),
 'Laity': ('Other','Laity指宗教团体中的普通信众/非神职成员，是宗教身份类别，不表示神职或宗教权威职位。','occupation_meaning:非神职普通信众','high'),
 'First Lady of Chile': ('Other','该词通常指智利国家领导人配偶的礼仪/家庭身份；未表示本人担任政府官职或政策职务。','occupation_meaning:国家领导人配偶身份','medium'),
 'Scholar-official': ('Leadership','Scholar-official指兼具儒学学者身份的政府官僚/文官，实际公共职能是行政治理。','occupation_meaning:政府文官/官僚身份','medium'),
 'Commissioner of Baseball': ('Sports/Games','棒球专员负责职业棒球联盟的赛事规则与联赛管理，属于体育竞赛治理职务。','occupation_meaning:棒球联盟治理','high'),
 'Music pedagogue': ('Discovery/Science','音乐教育者设计并教授音乐课程与训练方法，核心是专业教学，按作者Academia类别归入Discovery/Science。','author_role_analogy:Academia/专业教育','medium'),
 'Drama Theorist': ('Discovery/Science','戏剧理论研究者分析戏剧文本、形式与历史，是学术研究而非舞台表演岗位。','occupation_meaning:戏剧理论研究','high'),
 'Media scholar': ('Discovery/Science','媒体学者研究传播媒介、内容与社会影响，属于学术研究。','occupation_meaning:媒介研究','high'),
 'Political philosopher': ('Discovery/Science','政治哲学家研究政治观念与制度的思想基础，不是从事政府治理的政治官员。','occupation_meaning:政治思想学术研究','high'),
 'Business Coach': ('Other','商业教练为企业主或团队提供经营辅导服务，不等同于公司高管或决策职位。','occupation_meaning:商业辅导服务','medium'),
 'Botonist': ('Discovery/Science','Botonist应为Botanist的拼写变体，指研究植物分类、结构或生理的植物学者。','occupation_meaning:Botanist植物学研究者；输入疑似拼写错误','high'),
 'Bail bondsman': ('Other','保释担保人/保释债券经纪人安排保释金担保并提供相关商业服务，不是司法裁判或政府执法职位。','occupation_meaning:保释担保商业服务','medium'),
 'Structural biologist': ('Discovery/Science','结构生物学家研究生物分子与细胞结构，属于生命科学研究。','occupation_meaning:生物结构研究','high'),
 'Plant Biologist': ('Discovery/Science','植物生物学家研究植物的生理、遗传、生态或分类，属于生命科学研究。','occupation_meaning:植物生命科学研究','high'),
 'Neurophysiologist': ('Discovery/Science','神经生理学家研究神经系统及神经细胞的功能活动，属于生物医学研究。','occupation_meaning:神经系统功能研究','high'),
 'Medical researcher': ('Discovery/Science','医学研究者开展疾病机制、诊断或治疗相关研究，核心是科研工作。','occupation_meaning:医学研究','high'),
 'Digital Marketing Strategist (Job title)': ('Other','数字营销策略师规划线上推广和客户获取，属于营销服务职能；作者将marketing映射到Worker/Business (small)，即Other。','author_role_analogy:marketing → Other/Worker-Business (small), author rules line 64','medium'),
 'Boat builder': ('Other','船匠按设计和工艺制造、装配船体与船只，属于手工业/制造工种；标题不表示船舶设计工程师。','occupation_meaning:船只制造手工业','medium'),
 'Chairty Worker': ('Other','“Chairty”明显是“Charity”的拼写错误；慈善机构工作人员提供公益项目与社区服务，词项没有组织领导职务信息。','occupation_meaning:慈善服务工作人员；raw_value疑为Charity Worker拼写错误','medium'),
 'Management Theorist': ('Discovery/Science','管理理论研究者提出或分析组织与管理理论，核心工作是学术研究，不是企业管理者。','occupation_meaning:管理学理论研究','high'),
 'Agriculturalist': ('Discovery/Science','农业专家运用农业知识向农户或农业企业提供建议，属于农业专业知识与科学应用。','https://www.oxfordlearnersdictionaries.com/us/definition/american_english/agriculturalist','high'),
 'Industriemechaniker': ('Discovery/Science','德国工业机械技工制造、安装、调试、维护并修理机器和生产设备，属于工程技术工作。','https://www.anerkennung-in-deutschland.de/en/interest/finder/profession/182/profile','high'),
 'Soubrette': ('Culture','Soubrette指歌剧/轻歌剧中的轻型女高音声部或由其饰演的女仆型角色，属于歌唱表演。','https://www.treccani.it/vocabolario/soubrette/','high'),
 'Dramatic colouratura': ('Culture','戏剧花腔女高音是歌剧演唱声部，结合花腔技巧与戏剧表现，属于表演艺术。','https://www.uc.edu/content/dam/refresh/cont-ed-62/olli/olli_docs/6-dramatic-soprano.pdf','high'),
 'Hospital Driver': ('Other','医院司机负责病患、职工或物资的地面运输；工作核心是驾驶运输服务，医院雇主不使其成为医疗专业。','occupation_meaning:医院场景中的运输司机','high'),
 'Paint Artist': ('Culture','Paint Artist指使用颜料进行视觉艺术创作，工作产物是绘画作品，属于视觉艺术。','occupation_meaning:颜料绘画创作','high'),
 'Product Designer': ('Culture','产品设计师构思并设计消费/工业产品的外形、交互与使用体验，属于设计创作职能。','occupation_meaning:产品外观与体验设计','medium'),
 'Landscape architect': ('Culture','景观建筑师规划并设计户外空间、园林与公共景观，核心是空间和视觉设计。','occupation_meaning:景观与户外空间设计','high'),
 'Certified Public Accountant': ('Other','注册会计师从事会计、审计和税务专业服务；作者把 accountant 类职业置于 Other 的 Worker/Business (small)，资格认证不改变该职责。','author_role_analogy:accountant → Other/Worker/Business (small)','high'),
 'Chartered Surveyor': ('Other','特许测量师提供土地、房产和建造项目的测量、估值或咨询专业服务；作者相近 surveyor 职业归 Other，特许资格不等于治理职位。','https://www.rics.org/surveyor-careers/surveying/what-is-a-chartered-surveyor; author_role_analogy:surveyor → Other','medium'),
 'Adult model': ('Culture','成人出版物或影像中的模特提供摄影/影视视觉表演；建模工作本身属于文化媒介制作，不等同于性服务。','occupation_meaning:成人媒介摄影/影像模特','medium'),
 'Pin-up girl': ('Culture','Pin-up girl指为宣传或出版摄影摆姿的模特形象，属于视觉媒介/摄影文化；词本身不表示性服务。','occupation_meaning:摄影出版中的模特/视觉表演','medium'),
 'Laity': ('Other','Laity指宗教团体中的普通信众/非神职成员，是宗教身份类别，不表示神职或宗教权威职位。','occupation_meaning:非神职普通信众','high'),
 'First Lady of Chile': ('Other','该词通常指智利国家领导人配偶的礼仪/家庭身份；未表示本人担任政府官职或政策职务。','occupation_meaning:国家领导人配偶身份','medium'),
 'Scholar-official': ('Leadership','Scholar-official指兼具儒学学者身份的政府官僚/文官，实际公共职能是行政治理。','occupation_meaning:政府文官/官僚身份','medium'),
 'Commissioner of Baseball': ('Sports/Games','棒球专员负责职业棒球联盟的赛事规则与联赛管理，属于体育竞赛治理职务。','occupation_meaning:棒球联盟治理','high'),
}

def main():
    with INPUT.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    decided = {}
    for level, groups in GROUPS.items():
        for rationale, titles in groups.items():
            for title in titles.split(';'):
                title = title.strip()
                if title and title not in decided:
                    decided[title] = (level, rationale)
    missing = sorted({r['raw_value'] for r in rows} - decided.keys() - NEEDS_CONTEXT.keys())
    print('rows',len(rows),'unique',len({r['raw_value'] for r in rows}),'missing',len(missing))
    print('\n'.join(missing))
    if missing:
        raise SystemExit(1)

    out=[]
    for row in rows:
        title = row['raw_value']
        if title in RESOLVED_OVERRIDES:
            level, rationale, evidence, confidence = RESOLVED_OVERRIDES[title]
            out.append({'rank':row['rank'],'raw_value':title,'semantic_level1':level,
                'review_status':'resolved','confidence':confidence,'semantic_rationale':rationale,
                'evidence':evidence,'candidate_l1s_json':json.dumps([level],ensure_ascii=False),
                'reviewer':'bhht_semantic_batch_1'})
            continue
        if title in NEEDS_CONTEXT:
            candidates, rationale, evidence = NEEDS_CONTEXT[title]
            out.append({'rank':row['rank'],'raw_value':title,'semantic_level1':'',
                'review_status':'needs_context','confidence':'low','semantic_rationale':rationale,
                'evidence':evidence,'candidate_l1s_json':candidates,'reviewer':'bhht_semantic_batch_1'})
            continue
        level, function = decided[title]
        rationale = f'“{title}”指{function}，据此其主要职责属于{level}。'
        evidence = f'occupation_meaning:{title}—{function}'
        # Direct, unambiguous role labels receive high confidence; broad compounds stay medium.
        confidence = 'high' if title in {
            'Football player','Baseball Player','Professional Boxer','Ice dancer','Sumo Wrestler',
            'Composer (Profession)','Film Score Composer','Singer (Profession) #212','Cardiac surgeon',
            'Emergency physician','Nobleman','Roman emperor','Attorney at law','Diocesan bishop',
            'Molecular Biologist','Plant Biologist','Structural biologist','Gyōji','Nailist',
            'Industriemechaniker','Kannushi','Disc jockey','Ballet Teacher','Basketball Coach',
            'Figure skating coach','Gamer','Sex worker','Domestic worker'
        } else 'medium'
        out.append({'rank':row['rank'],'raw_value':title,'semantic_level1':level,
            'review_status':'resolved','confidence':confidence,'semantic_rationale':rationale,
            'evidence':evidence,'candidate_l1s_json':json.dumps([level],ensure_ascii=False),
            'reviewer':'bhht_semantic_batch_1'})
    fields=['rank','raw_value','semantic_level1','review_status','confidence','semantic_rationale','evidence','candidate_l1s_json','reviewer']
    with OUTPUT.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader(); writer.writerows(out)

if __name__ == '__main__': main()
