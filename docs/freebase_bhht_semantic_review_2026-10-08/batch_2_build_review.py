import csv, json
from pathlib import Path
base = Path(__file__).parent
src = base / 'batch_2_input.tsv'
out = base / 'batch_2_review.tsv'
with src.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

# Individual row-number decisions based on the complete occupation phrase.
sets = {
'Culture': '''1 3 4 6 7 9 10 13 14 15 16 18 19 20 22 24 28 29 30 37 38 41 43 47 48 51 53 54 55 59 64 66 72 73 74 77 79 80 81 84 85 86 89 92 93 98 99 100 102 103 105 108 109 111 114 116 119 125 127 130 133 134 135 136 138 139 142 145 149 151 152 153 155 158 160 161 163 167 170 174 176 179 180 184 185 187 188 189 194 195 198 201 204 205 207 210 212 218 219 222 226 227 229 231 234 237 243 247 253 256 259 265 266 267 268 278 279 282 284 293 298 299 302 305 308 311 318 325 327 329 332 339 343 344 355 358 362 363 365 366 370 371 379 380 389 397 402 403 404 415 420 422''',
'Discovery/Science': '''5 23 35 44 49 60 76 86 88 93 102 105 113 120 121 129 132 139 140 142 145 146 148 149 150 151 156 159 162 164 165 171 175 181 183 186 193 194 195 196 199 207 210 211 220 221 223 227 229 235 238 242 251 256 259 260 261 262 264 271 274 278 285 287 291 292 297 300 301 305 313 322 326 333 335 337 338 341 346 348 353 356 357 358 359 361 367 368 374 375 381 382 383 385 388 393 395 396 410 412 413 414 416 423 424''',
'Leadership': '''2 8 12 17 21 26 32 39 42 50 61 62 68 78 83 91 96 99 106 110 115 118 120 126 128 130 141 143 147 153 157 159 162 172 173 177 182 190 192 199 202 208 215 216 217 225 228 232 239 241 249 250 252 255 263 269 270 272 277 280 281 283 289 290 296 303 307 309 313 315 317 320 321 323 330 331 336 340 342 345 347 350 354 359 364 373 377 390 391 398 400 401 405 409 417 421''',
'Sports/Games': '''11 17 27 33 36 40 56 63 65 69 71 82 87 88 95 117 122 131 137 190 208 214 219 236 240 252 273 275 276 286 290 295 304 316 317 330 369 372 378 399 408 415 418''',
'Other': '''25 31 34 39 45 56 57 58 67 75 78 90 91 95 97 101 104 107 112 124 128 144 154 157 166 168 169 172 175 178 182 191 197 200 206 209 212 214 220 221 230 232 233 240 244 245 248 249 254 255 257 258 263 270 273 275 280 281 283 285 287 288 289 294 295 296 297 300 301 306 307 310 312 314 315 319 324 326 328 334 335 340 341 342 345 347 349 350 351 352 354 356 357 360 364 369 373 375 376 378 381 383 384 386 387 390 392 393 395 398 401 406 407 408 411 412 417 418 419 421 423 424'''
}
labels = {}
for lab, s in sets.items():
    for n in map(int, s.split()):
        labels.setdefault(n, []).append(lab)

# Resolve the deliberately overlapping candidate group lists per occupation.
overrides = {
86:'Discovery/Science',93:'Culture',99:'Culture',102:'Culture',105:'Culture',130:'Culture',
139:'Discovery/Science',142:'Culture',145:'Culture',149:'Discovery/Science',151:'Culture',
153:'Culture',194:'Culture',195:'Discovery/Science',207:'Culture',210:'Discovery/Science',
212:'Other',219:'Culture',227:'Culture',229:'Culture',256:'Discovery/Science',259:'Culture',
278:'Culture',305:'Discovery/Science',358:'Discovery/Science',415:'Sports/Games',
88:'Sports/Games',120:'Discovery/Science',159:'Discovery/Science',162:'Discovery/Science',
175:'Discovery/Science',199:'Other',220:'Discovery/Science',221:'Discovery/Science',
285:'Other',287:'Other',297:'Discovery/Science',300:'Other',301:'Other',313:'Leadership',
326:'Other',335:'Discovery/Science',341:'Discovery/Science',356:'Discovery/Science',
357:'Discovery/Science',359:'Discovery/Science',375:'Discovery/Science',381:'Discovery/Science',
383:'Discovery/Science',393:'Discovery/Science',395:'Discovery/Science',412:'Discovery/Science',
423:'Discovery/Science',424:'Culture',17:'Sports/Games',39:'Other',78:'Other',91:'Other',
128:'Other',157:'Leadership',172:'Leadership',182:'Discovery/Science',190:'Culture',
208:'Leadership',232:'Other',249:'Leadership',252:'Culture',255:'Other',263:'Other',
270:'Leadership',280:'Leadership',281:'Other',283:'Other',289:'Other',290:'Sports/Games',
296:'Leadership',307:'Leadership',315:'Leadership',317:'Sports/Games',330:'Sports/Games',
340:'Leadership',342:'Leadership',345:'Leadership',347:'Leadership',350:'Leadership',
354:'Leadership',364:'Leadership',373:'Leadership',390:'Other',398:'',401:'Leadership',
417:'Other',421:'Other',56:'Sports/Games',95:'Other',214:'Other',240:'Discovery/Science',
273:'Other',275:'',295:'Sports/Games',369:'Sports/Games',378:'Other',408:'Sports/Games',
418:'Other',52:'Leadership',70:'Other',94:'Culture',394:'Culture',
8:'',9:'Leadership',25:'Leadership',26:'Other',62:'Other',67:'',68:'Other',
70:'Other',83:'Other',107:'Leadership',112:'Culture',114:'Other',144:'Leadership',
147:'Other',156:'Other',168:'Culture',173:'Other',178:'Sports/Games',197:'Culture',
200:'Culture',202:'',209:'',215:'Culture',230:'Leadership',233:'Leadership',237:'Other',
244:'',248:'',250:'Discovery/Science',258:'',269:'',272:'Culture',276:'Other',277:'Leadership',
282:'Discovery/Science',286:'',287:'',288:'Culture',294:'Culture',306:'Culture',312:'Leadership',
320:'',323:'',328:'Leadership',334:'Culture',343:'',351:'Discovery/Science',352:'Culture',
372:'',386:'Culture',388:'',389:'Other',390:'Culture',391:'Other',392:'Discovery/Science',
399:'Sports/Games',406:'Discovery/Science',411:'Culture',424:'Other',
57:'Culture',58:'Culture',107:'Leadership',178:'Culture',201:'Culture',214:'Other',
258:'',286:'',309:'Leadership',343:'',366:'Culture',359:'Leadership',384:'Leadership',396:'Leadership'
}
labels = {n: labs[0] for n, labs in labels.items() if len(labs) == 1}
for n, lab in overrides.items():
    if lab:
        labels[n] = lab

# Deliberately unresolved short labels, broad statuses, domain names, and corrupted/unclear strings.
needs = {
8: ('Official', '“Official”只表示某种正式职务或身份，未说明是政府、体育、企业还是其他机构，缺少岗位语境。', '["Leadership","Sports/Games","Other"]'),
67: ('Aircrew', 'Aircrew 可指军用或民用航空机组成员，职责背景可能分别归军事或普通服务，缺少具体岗位语境。', '["Leadership","Other"]'),
225: ('Government agent', '未说明代理机构和工作职责；政府代理人可能承担执法、情报、行政或其他职能，无法仅由“agent”确定 L1。', '["Leadership","Other"]'),
246: ('Pilgrim Fathers', '这是历史群体/身份称谓，而非明确职业名称；文本没有提供可归类的工作职责。', '[]'),
199: ('Registrar', 'Registrar 可指学校、法院、医疗机构或其他组织中的登记/行政岗位，缺少机构语境，无法确定 L1。', '["Discovery/Science","Leadership","Other"]'),
46: ('Promoter', '“Promoter”可指赛事、艺人、产品或企业推广者；没有领域限定，无法确定其主要职责所属大类。', '["Culture","Sports/Games","Leadership"]'),
75: ('Student', '这是学习身份而非职业职责，所学领域也未提供，不能据此确定 L1。', '["Discovery/Science"]'),
123: ('Avatar of Vashanka', '该短语可能是虚构神祇的化身/角色称号，也可能涉及宗教身份；无法确认它是现实职业还是作品设定。', '[]'),
167: ('Production Supervisor', '未说明所监督的制作属于影视/演出、工业制造或其他行业，职责领域不足以确定。', '["Culture","Other"]'),
203: ('Public figure', '这是公众知名度/身份描述，不说明其职业职能，可能覆盖多个真实 L1。', '["Culture","Leadership","Sports/Games","Other"]'),
202: ('Partisan', '可指党派支持者，也可指武装抵抗组织成员，两种含义的实际职责和类别不同。', '["Leadership","Other"]'),
209: ('Lineman', 'Lineman 可指美式足球位置，也可指电力/通信线路工，语境缺失会导致不同 L1。', '["Sports/Games","Other"]'),
244: ('Expert', '“Expert”只表示具备专长，没有指出专业领域或工作职责。', '[]'),
258: ('Handmade soap', '这是手工皂产品/制作领域的短语，没有明确说明它表示制作者这一职业。', '[]'),
286: ('Seashore Hotel', '该值看似酒店名称或住宿业务实体，而非个人职业名称，不能据此归入 L1。', '[]'),
320: ('Most Eligible', '这是含义不明的描述性短语，无法确认是身份、荣誉称号还是职业。', '[]'),
323: ('Underwriting', '这是保险/金融职能或业务名称，没有说明具体岗位职责或是否指从业者。', '["Leadership","Other"]'),
372: ('Kidnapping', '这是犯罪行为名称，不是明确职业名称；无法确认其作为职业字段的真实语义。', '[]'),
213: ('Semantic Web', '这是技术领域名称而非明确岗位；缺少从事研究、开发或倡议等职责信息。', '[]'),
224: ('Guest speaker', '只说明临时受邀发言的身份，未给出演讲领域或主要职业，不能据此定 L1。', '[]'),
225: ('Government agent', '未说明代理机构和工作职责；政府雇员、情报人员、执法者或其他代理角色的 L1 会不同。', '["Leadership","Other"]'),
248: ('Independent contractor', '这是雇佣/合同关系形式，不表明承包工作的职业领域。', '["Culture","Discovery/Science","Leadership","Sports/Games","Other"]'),
269: ('FLMI', 'FLMI 是保险业教育/专业资格称号，而非足以判定具体岗位的职业名称；持证者可能从事不同职责。', '["Leadership","Other"]'),
287: ('Agricultural Agent', '可能指农业推广顾问、商业代理或政府农业职员，职责语境不足以在学术、治理或普通服务之间判定。', '["Discovery/Science","Leadership","Other"]'),
343: ('CNR (Composition)', '“CNR”缩写和括号中的 Composition 未能可靠解释为具体岗位，不能推定为作曲或其他类别。', '[]'),
388: ('Information professional', '该泛称可覆盖图书馆、档案、信息技术和信息管理等不同岗位，未提供具体职责。', '["Culture","Discovery/Science","Other"]'),
398: ('Financial Services', '这是金融服务行业/领域名称，没有指出具体金融职业或职责。', '["Leadership","Other"]'),
275: ('Canine Professional', '该泛称可能指犬类训练、护理、饲养或其他犬业工作，不能由此确定具体职责类别。', '["Sports/Games","Other"]'),
425: ('Affiliate', '该词可表示商业关联者、营销推广者或其他从属关系，缺少实际岗位语境。', '["Culture","Leadership","Other"]'),
}

# Final row-by-row corrections from the full raw-value audit.
overrides.update({
 25:'Leadership', 57:'Culture', 58:'Culture', 70:'Other', 83:'Other', 107:'Leadership',
 112:'Culture', 114:'Other', 144:'Leadership', 147:'Other', 156:'Other', 168:'Culture',
 173:'Other', 178:'Sports/Games', 197:'Culture', 200:'Culture', 215:'Culture',
 230:'Leadership', 233:'Leadership', 237:'Other', 250:'Discovery/Science', 272:'Culture',
 276:'Other', 282:'Discovery/Science', 288:'Culture', 294:'Culture', 306:'Culture',
 328:'Leadership', 334:'Culture', 351:'Discovery/Science', 352:'Culture', 359:'Leadership',
 366:'Culture', 384:'Leadership', 386:'Culture', 389:'Other', 390:'Culture', 391:'Other',
 392:'Discovery/Science', 396:'Leadership', 399:'Sports/Games', 406:'Discovery/Science',
 411:'Culture', 424:'Other'
})
needs.update({
 67: ('Aircrew', 'Aircrew 可指军用或民用航空机组成员，职责背景可能分别归军事或普通服务，缺少具体岗位语境。', '["Leadership","Other"]'),
 202: ('Partisan', '可指党派支持者，也可指武装抵抗组织成员，两种含义的实际职责和类别不同。', '["Leadership","Other"]'),
 209: ('Lineman', 'Lineman 可指美式足球位置，也可指电力/通信线路工，语境缺失会导致不同 L1。', '["Sports/Games","Other"]'),
 225: ('Government agent', '未说明代理机构和工作职责；政府代理人可能承担执法、情报、行政或其他职能，无法仅由“agent”确定 L1。', '["Leadership","Other"]'),
 244: ('Expert', '“Expert”只表示具备专长，没有指出专业领域或工作职责。', '[]'),
 258: ('Handmade soap', '这是手工皂产品/制作领域的短语，没有明确说明它表示制作者这一职业。', '[]'),
 286: ('Seashore Hotel', '该值看似酒店名称或住宿业务实体，而非个人职业名称，不能据此归入 L1。', '[]'),
 320: ('Most Eligible', '这是含义不明的描述性短语，无法确认是身份、荣誉称号还是职业。', '[]'),
 323: ('Underwriting', '这是保险/金融职能或业务名称，没有说明具体岗位职责或是否指从业者。', '["Leadership","Other"]'),
 372: ('Kidnapping', '这是犯罪行为名称，不是明确职业名称；无法确认其作为职业字段的真实语义。', '[]')
})
overrides.update({67:'',202:'',209:'',225:'',244:'',258:'',286:'',320:'',323:'',372:''})
for n, lab in overrides.items():
    if lab:
        labels[n] = lab

# Rows outside a resolved list must be expressly accounted for; catch coverage mistakes.
unreviewed = [(i,r['raw_value']) for i,r in enumerate(rows,1) if i not in labels and i not in needs]
if unreviewed:
    print('UNREVIEWED', unreviewed)
    raise SystemExit(1)
for i,r in enumerate(rows,1):
    if i in needs and r['raw_value'] != needs[i][0]:
        raise ValueError(f'needs row mismatch {i}: {r["raw_value"]} != {needs[i][0]}')

role_basis = {
'Culture': '该岗位通过创作、演出、编辑、制作或传播文化内容完成工作，归入 Culture。',
'Discovery/Science': '该岗位以专业研究、教学、医疗服务或技术开发为核心，属于知识与科学专业职能。',
'Leadership': '该岗位履行治理、司法、军事指挥、宗教权威或组织/商业管理职责，归入 Leadership。',
'Sports/Games': '该岗位直接参与竞技项目、竞赛组织、体育训练/裁判或游戏活动，归入 Sports/Games。',
'Other': '该岗位属于一般服务、日常工种、小型经营、家庭身份或其他非四类核心职能，归入 Other。',
}
extra = {
26: ('房地产经纪人撮合买卖/租赁双方、评估并营销房产；作者同义职业 estate_agent / realtor / broker 属普通职业，归 Other。', None),
7: ('Music executive 管理唱片、艺人或音乐制作业务，职责处于音乐文化产业内部，按专门文化产业职能归 Culture。', None),
24: ('Sports commentator 通过播报和评论赛事向观众提供媒体内容，是体育媒体工作者而非运动员。', None),
168: ('Radio jockey 在广播中主持节目、介绍音乐并与听众沟通，属于广播媒体传播。', None),
252: ('Professional wrestling promoter 组织并推广职业摔角比赛，直接服务于摔角竞赛。', None),
270: ('Muezzin 在清真寺宣诵每日礼拜的宣礼词，承担宗教礼仪职务，归入 Leadership。', 'https://en.wikisource.org/wiki/The_New_International_Encyclop%C3%A6dia/Muezzin'),
273: ('Plantsman 指熟悉园艺植物与栽培的园艺专家；按园艺栽培这一普通职业职能归 Other。', 'https://www.oxfordlearnersdictionaries.com/us/definition/english/plantsman'),
9: ('宗教领袖承担宗教共同体的权威与指导职责，按作者宗教领导类别归入 Leadership。', None),
25: ('销售管理负责规划销售目标、带领销售团队并监督业绩，属于组织管理职能。', None),
50: ('房地产企业家经营地产项目或相关业务，核心是商业经营管理，归入 Leadership。', None),
62: ('人权倡议者推动权利保护与公众倡议，但词项本身没有公共职务或治理权；按非领导型倡议身份归 Other。', None),
68: ('Web analytics 指收集并分析网站访问数据以支持网站运营/营销决策，是普通分析岗位。', None),
83: ('酒店经理管理住宿接待和日常服务运营，属于酒店服务业管理，不据此推为大型企业领导。', None),
107: ('Crown Princess 是王室继承序列中的贵族身份称谓，归入 Leadership 的 Nobility 语义。', None),
114: ('Online Marketing Professional 从事线上营销活动与推广，是常规营销服务岗位，归 Other。', None),
144: ('法院官员在法院体系履行司法行政或法庭程序职责，属于法律/司法职能。', None),
147: ('民权倡议者从事权利倡导与社会动员；没有明确治理职位，不等同政治领导者。', None),
156: ('Chartered accountant 提供审计、会计和财务报告专业服务；按作者 accountant 近似职业归入 Other。', None),
173: ('Native American activist 表示社群权益倡议工作，未说明其担任政治公职或治理职位。', None),
197: ('Cinematography 是电影摄影工作，负责以摄影机、镜头和光线实现影片画面表达，属于影视制作。', None),
200: ('Music criticism 通过评论和分析音乐作品向读者/听众传播文化评价，属于文化写作。', None),
215: ('Fashion entrepreneur 创办或经营时尚产业业务，工作直接围绕服饰与时尚文化产品。', None),
230: ('船舶所有者拥有并经营船舶资产，属于商业经营者身份。', None),
233: ('Chief mate 是船长以下的高级甲板官员，负责航行值班并协助指挥船员，属组织/航海管理。', None),
237: ('Digital marketing 从事线上推广和营销执行，是常规商业服务岗位，归 Other。', None),
250: ('Software development 编写、维护和交付软件程序，属于技术开发。', None),
272: ('影视片场 Grip 搭建和操作摄影机支撑、轨道及照明设备，属于影视制作技术工种。', None),
277: ('原值“Poltician”是 politician 的明显拼写误差；按政治人物的公共治理职责归 Leadership。', None),
282: ('摄影史研究者研究摄影媒介、作品和历史脉络，属于学术研究而非摄影创作岗位。', None),
288: ('特效总监负责规划并监督影视作品中的视觉/物理特效制作，属于影视制作职能。', None),
294: ('Stage combat 是舞台/影视表演中的编排式打斗设计和排演，属于表演制作。', None),
306: ('Escapology 指表演者从束缚或机关中脱身的魔术/杂技表演形式，属于娱乐表演。', 'https://www.dictionary.com/browse/escapology'),
334: ('Gambist 演奏 viola da gamba（维奥尔琴），是音乐表演者。', 'https://vitrinelinguistique.oqlf.gouv.qc.ca/fiche-gdt/fiche/8382969/gambiste'),
351: ('Web builder 构建和维护网站页面/功能，属于网站技术开发。', None),
352: ('Garden designer 规划庭园空间、植物配置和景观视觉，是设计创作工作。', None),
359: ('Arms control analyst 研究武器管制政策与国际安全议题，属于公共政策分析。', None),
366: ('Production stage manager 协调演出排练、演出流程和舞台部门执行，属于现场文化制作。', None),
384: ('Master mariner 是具备商船驾驶资格的高级船舶指挥人员，承担航行与船员管理职责。', None),
386: ('Media entrepreneur 创办或经营媒体业务，职责直接围绕媒体内容制作/传播。', None),
390: ('Salonnières 主持文学/艺术沙龙并组织文化交流，属于文化传播活动。', 'https://www.encyclopedia.com/women/encyclopedias-almanacs-transcripts-and-maps/salonnieres-fl-17th-and-18th-c'),
391: ('Illegal drug dealer 指非法交易毒品者，是犯罪身份，按 Other 处理。', None),
392: ('Help desk coordinator 安排技术支持请求、分派工单并协调用户问题处理，属于 IT 服务支持。', None),
396: ('Arms policy analyst 研究武器相关公共政策及其影响，属于政策分析而非科学研究。', None),
399: ('Wrestling booker 负责安排职业摔角比赛对阵和赛事内容，属于摔角竞赛组织。', None),
406: ('Ruby developer 使用 Ruby 编写和维护软件，属于软件技术开发。', None),
411: ('Communications consultant 为客户制定传播信息与沟通方案，属于媒体/传播服务。', None),
424: ('该职位专门为网站提供 SEO 与线上推广咨询，属于常规营销服务岗位，归 Other。', None),
61: ('历史上的 condottiere 是受雇佣兵团首领，属于军事指挥。', 'https://en.wikisource.org/wiki/1911_Encyclop%C3%A6dia_Britannica/Condottiere'),
86: ('音乐理论研究和教学属于音乐学术职能。', None),
123: (None, None),
127: ('魔术师以现场幻术表演娱乐观众，属于表演艺术。', None),
112: ('Acrobat 以身体技巧完成杂技表演，职责是现场娱乐演出，归入 Culture。', None),
160: ('vedette 在表演语境指歌舞/音乐厅的主要艺人。', 'https://www.larousse.fr/dictionnaires/francais/vedette/81241'),
179: ('Dubber 为影视作品录制配音/对白，属于影视声音制作。', None),
178: ('Professional wrestling booker 负责编排摔角赛事对阵与故事线，直接服务于摔角比赛。', None),
252: ('Professional wrestling promoter 组织并推广摔角赛事，属于摔角竞赛运营。', None),
276: ('Tokoyama 是相扑协会雇用、为相扑力士整理发髻的传统理发师；其职责是理发服务，故归 Other。', 'https://www.japantimes.co.jp/sports/2018/11/14/sumo/sumo-hairdressers-require-many-years-training-specialized-tools/'),
389: ('Stage mother 指陪伴并支持子女从事舞台/娱乐活动的家庭身份，不表示本人从事表演或制作工作，归 Other。', None),
288: ('特效总监负责规划并监督影视作品中的视觉/物理特效制作，属于影视制作职能。', None),
294: ('Stage combat 是舞台/影视表演中的编排式打斗设计和排演，属于表演制作。', None),
306: ('Escapology 指表演者从束缚或机关中脱身的魔术/杂技表演形式，属于娱乐表演。', 'https://www.dictionary.com/browse/escapology'),
208: ('Faqīh 是伊斯兰法学专家/法学家，作为宗教法律权威归入 Leadership。', 'https://www.sloughislamictrust.org.uk/dictionary/meaning/jurist-expert/'),
308: ('Benshi 是日本无声电影现场讲述/解释影片的表演者，属于电影文化传播。', 'https://aboutjapan.japansociety.org/content.cfm/a_brief_history_of_benshi'),
312: ('Faqīh 是伊斯兰法学专家/法学家，履行宗教法律解释职能。', 'https://www.sloughislamictrust.org.uk/dictionary/meaning/jurist-expert/'),
334: ('Gambist 演奏 viola da gamba（维奥尔琴），是音乐表演者。', 'https://vitrinelinguistique.oqlf.gouv.qc.ca/fiche-gdt/fiche/8382969/gambiste'),
390: ('Salonnières 主持文学/艺术沙龙并组织文化交流，属于文化传播活动。', 'https://www.encyclopedia.com/women/encyclopedias-almanacs-transcripts-and-maps/salonnieres-fl-17th-and-18th-c'),
}

out_rows=[]
for i, r in enumerate(rows,1):
    label='' if i in needs else labels.get(i, '')
    if not label:
        raw, rationale, candidates=needs[i]
        evidence = {
            269:'https://www.loma.org/en/professional-development/talent-mobility-suite/flmi/',
        }.get(i, 'occupation_meaning:原值只给出泛称、资格、身份或无法确认的文本，需补充具体职责/语境。')
        out_rows.append({'rank':r['rank'],'raw_value':r['raw_value'],'semantic_level1':'','review_status':'needs_context','confidence':'low','semantic_rationale':rationale,'evidence':evidence,'candidate_l1s_json':candidates,'reviewer':'bhht_semantic_batch_2'})
        continue
    rationale=role_basis[label].replace('该职业', f'“{r["raw_value"]}”')
    if i in extra and extra[i][0]: rationale=extra[i][0]
    url=extra.get(i,(None,None))[1]
    evidence=f'occupation_meaning:{rationale}' if url is None else url
    if url is None:
        evidence += f'; author_role_analogy:author L1 {label}'
    confidence='high'
    if i in {7,26,50,59,62,99,100,116,134,141,167,172,188,192,215,216,218,241,249,252,303,328,336,350,354,359,373,383,386,387,397,405,407,408,411,415,418,422,424}:
        confidence='medium'
    out_rows.append({'rank':r['rank'],'raw_value':r['raw_value'],'semantic_level1':label,'review_status':'resolved','confidence':confidence,'semantic_rationale':rationale,'evidence':evidence,'candidate_l1s_json':json.dumps([label],ensure_ascii=False,separators=(',',':')),'reviewer':'bhht_semantic_batch_2'})

fields=['rank','raw_value','semantic_level1','review_status','confidence','semantic_rationale','evidence','candidate_l1s_json','reviewer']
with out.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n')
    w.writeheader(); w.writerows(out_rows)
print(f'wrote {len(out_rows)} rows to {out}')
