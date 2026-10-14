import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = process.env.DBPEDIA_WORKBOOK_RUNTIME || path.join(repo, 'artifacts/dbpedia_bhht_review_workbook_20261014');
const require = createRequire(path.join(runtime, 'package.json'));
const { Workbook, SpreadsheetFile } = await import(require.resolve('@oai/artifact-tool'));
const out = process.argv[2] || path.join(repo, 'docs/dbpedia_bhht_match_2026-10-14');
const { records, summary } = JSON.parse(await fs.readFile(`${out}/workbook_data.json`, 'utf8'));
const finalPath = `${out}/dbpedia_bhht_mapping_review.xlsx`;
try {
  await fs.access(finalPath);
  throw new Error('Review workbook already exists; use a new path to preserve review entries.');
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const wb = Workbook.create();
const overview = wb.worksheets.add('Summary');
const mapping = wb.worksheets.add('Mapping');
const statusNames = {
  exact_l1_unique: '精确匹配，L1唯一', substring_l1_unique: '子串匹配，L1唯一',
  cross_l1_conflict: '跨L1冲突', unmatched: '未命中作者规则',
};
const reasons = {
  no_author_rule: '作者规则未命中', substring_requires_l1_transfer_review: '子串命中，需核对完整职业语义',
  multiple_author_l1_candidates: '多个L1候选', observed_reference_l1_disagrees: '作者观察参考出现不同L1',
  exact_lookup_differs_from_author_first_l1: '精确词与作者首次命中L1不同',
};
const data = records.map(r => {
  const parents = JSON.parse(r.candidate_parent_pairs_json);
  const l1 = [...new Set(parents.map(p => p[0]))].join(' / ');
  const flags = JSON.parse(r.l1_review_flags_json).map(f => reasons[f] || f);
  return [r.occupation_label, r.graph_nodes, statusNames[r.l1_lookup_status], l1,
    flags.join('\n') || 'L1精确候选，无L1冲突', null, null, null,
    JSON.parse(r.candidate_keywords_json).join(' / ')];
});
const end = records.length + 1;
mapping.getRange(`A1:I${end}`).values = [
  ['DBpedia职业原词', '图内人数', 'L1查表结果', '候选L1', 'L1复核说明', '审核L1', '审核结论', '审核理由', '命中作者关键词（参考）'],
  ...data,
];
mapping.showGridLines = false;
mapping.getRange(`A1:I${end}`).format.font = { name: 'Arial', size: 10, color: '#1F2937' };
mapping.getRange(`A1:I${end}`).format.verticalAlignment = 'center';
mapping.getRange(`A2:I${end}`).format.wrapText = true;
const widths = [380, 95, 210, 245, 420, 190, 185, 450, 400];
for (let c = 0; c < 9; c++) mapping.getRangeByIndexes(0, c, end, 1).format.columnWidthPx = widths[c];
mapping.getRange(`B2:B${end}`).setNumberFormat('#,##0');
mapping.getRange(`B2:B${end}`).format.horizontalAlignment = 'right';
mapping.tables.add(`A1:I${end}`, true, 'DBpediaBHHTMapping');
mapping.getRange('A1:I1').format = { fill: '#234563', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
  horizontalAlignment: 'center', verticalAlignment: 'center', rowHeightPx: 34, wrapText: true };
mapping.getRange(`F2:H${end}`).format.fill = '#FFF6D8';
mapping.getRange(`F2:F${end}`).dataValidation = { rule: { type: 'list', values: ['Culture', 'Discovery/Science', 'Leadership', 'Sports/Games', 'Other'] } };
mapping.getRange(`G2:G${end}`).dataValidation = { rule: { type: 'list', values: ['mapped', 'ambiguous', 'out_of_scope', 'pending_context'] } };
mapping.freezePanes.freezeRows(1);
mapping.freezePanes.freezeColumns(1);
mapping.getRange(`E2:E${end}`).conditionalFormats.add('containsText', { text: '多个L1候选', format: { fill: '#FCE8E6' } });
const visualLength = text => [...String(text || '')].reduce((n, c) => n + (c.charCodeAt(0) > 255 ? 2 : 1), 0);
for (let i = 0; i < data.length; i++) {
  const lines = Math.max(...data[i].map((v, c) => String(v || '').split('\n').reduce((n, s) => n + Math.max(1, Math.ceil(visualLength(s) / ((widths[c] - 16) / 6.5))), 0)));
  mapping.getRange(`A${i + 2}:I${i + 2}`).format.rowHeightPx = Math.max(32, lines * 17 + 10);
}
overview.showGridLines = false;
overview.getRange('A1:D25').format.font = { name: 'Arial', size: 10, color: '#1F2937' };
overview.getRange('A1:D25').format.verticalAlignment = 'center';
overview.getRange('A1:A25').format.columnWidthPx = 380;
overview.getRange('B1:C25').format.columnWidthPx = 135;
overview.getRange('D1:D25').format.columnWidthPx = 140;
overview.getRange('A2').values = [['DBpedia职业映射：优先审核L1']];
overview.getRange('A2').format.font = { name: 'Arial', size: 14, bold: true };
overview.getRange('A2:D2').format.rowHeightPx = 36;
overview.getRange('A3:D3').format.borders = { bottom: { style: 'thin', color: '#CBD5E1' } };
overview.getRange('A5:D5').values = [['查表结果', '职业种数', '对应人物数', '人物占比']];
overview.getRange('A5:D5').format = { fill: '#234563', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
  horizontalAlignment: 'center', rowHeightPx: 30 };
const statuses = ['exact_l1_unique', 'substring_l1_unique', 'cross_l1_conflict', 'unmatched'];
for (let i = 0; i < statuses.length; i++) {
  const row = i + 6, label = statusNames[statuses[i]];
  overview.getRange(`A${row}`).values = [[label]];
  overview.getRange(`B${row}`).formulas = [[`=COUNTIF(Mapping!$C$2:$C$${end},A${row})`]];
  overview.getRange(`C${row}`).formulas = [[`=SUMIF(Mapping!$C$2:$C$${end},A${row},Mapping!$B$2:$B$${end})`]];
  overview.getRange(`D${row}`).formulas = [[`=C${row}/$C$10`]];
}
overview.getRange('A10').values = [['合计']];
overview.getRange('B10:C10').formulas = [['=SUM(B6:B9)', '=SUM(C6:C9)']];
overview.getRange('D10').formulas = [['=SUM(D6:D9)']];
overview.getRange('A10:D10').format.font.bold = true;
overview.getRange('A10:D10').format.borders = { top: { style: 'thin', color: '#CBD5E1' } };
overview.getRange('B6:C10').setNumberFormat('#,##0');
overview.getRange('D6:D10').setNumberFormat('0.0%');
overview.getRange('A6:D10').format.rowHeightPx = 28;
overview.getRange('A12:C13').values = [
  ['精确L1候选且无L1复核标记', summary.strict_exact_candidate_occupations, summary.strict_exact_candidate_nodes],
  ['待L1语义复核', summary.pending_review_occupations, summary.pending_review_nodes],
];
overview.getRange('B12:C13').setNumberFormat('#,##0');
overview.getRange('A12:C13').format.rowHeightPx = 28;
overview.getRange('A15').values = [['已填审核结论 mapped']];
overview.getRange('B15').formulas = [[`=COUNTIF(Mapping!G2:G${end},"mapped")`]];
overview.getRange('C15').formulas = [[`=SUMIF(Mapping!G2:G${end},"mapped",Mapping!B2:B${end})`]];
overview.getRange('B15:C15').setNumberFormat('#,##0');
const notes = [
  'L1一致但L2/L3冲突，不阻塞L1候选。子串命中仍需核对完整职业语义。',
  '原职业选择保持不变。已有规则候选不等于完成LLM或人工审核。',
  'Mapping黄色列用于填写审核L1、结论和理由；空白表示尚未审核。',
  '参考关键词保留作者原词；不是另建的细职业本体。',
  'Source: DBpedia occupation assignments / current graph-node inventory.',
  'Source: BHHT author_keyword_l123_rules.csv; source_file/source_line保留于证据表。',
];
for (let i = 0; i < notes.length; i++) {
  const row = i + 18; overview.getRange(`A${row}`).values = [[notes[i]]];
  overview.getRange(`A${row}:D${row}`).format.rowHeightPx = 24;
}
wb.recalculate();
const countValues = overview.getRange('B6:C10').values;
for (let i = 0; i < statuses.length; i++) {
  if (countValues[i][0] !== summary.l1_lookup_status_counts[statuses[i]] || countValues[i][1] !== summary.graph_nodes_by_l1_lookup_status[statuses[i]]) throw new Error('Workbook totals disagree with audited Python counts');
}
if (countValues[4][0] !== summary.occupations || countValues[4][1] !== summary.graph_nodes) throw new Error('Wrong totals');
// Test the editable review summary in a temporary edit, then restore the blank input.
mapping.getRange('G2').values = [['mapped']]; wb.recalculate();
if (overview.getRange('B15').values[0][0] !== 1 || overview.getRange('C15').values[0][0] !== records[0].graph_nodes) throw new Error('Review summary does not respond to edits');
mapping.getRange('G2').values = [[null]]; wb.recalculate();
console.log((await wb.inspect({ kind: 'table', range: 'Summary!A5:D15', include: 'values,formulas', tableMaxRows: 11, tableMaxCols: 4, maxChars: 2300 })).ndjson);
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 20 }, maxChars: 700 });
console.log(errors.ndjson);
for (const [sheetName, range, file] of [['Summary', 'A1:D24', 'summary_preview.png'], ['Mapping', 'A1:I12', 'mapping_preview.png']]) {
  const image = await wb.render({ sheetName, range, scale: 1.5, format: 'png' });
  await fs.writeFile(path.join(runtime, file), new Uint8Array(await image.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(finalPath);
console.log(`SAVED ${finalPath}`);
