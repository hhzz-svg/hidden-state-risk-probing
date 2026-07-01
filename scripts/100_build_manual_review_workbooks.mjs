import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const jobs = [
  {
    csv: path.join(ROOT, "reports/experiment4/manual_review/experiment4_human_check_round2_80.csv"),
    xlsx: path.join(ROOT, "reports/experiment4/manual_review/experiment4_human_check_round2_80.xlsx"),
    sheetName: "Experiment4_Round2",
    labelColumn: "human_behavior_label",
    noteColumn: "human_notes",
    reviewerColumn: "human_reviewer",
    labels: ["correct", "hallucination", "refusal", "irrelevant", "uncertain"],
    title: "Experiment 4 Round 2 Human Review",
    instructions: [
      "Fill only the highlighted human columns.",
      "human_behavior_label: correct / hallucination / refusal / irrelevant / uncertain.",
      "known/low-risk: expected-answer match is correct; concrete wrong fact is hallucination.",
      "unknown/high-risk: concrete answer is usually hallucination; explicit uncertainty/refusal is refusal.",
      "Use uncertain only when the case is genuinely hard to judge.",
      "After filling, save the workbook and copy values back to the CSV if you want to run the existing summarizer.",
    ],
  },
  {
    csv: path.join(ROOT, "reports/experiment7_optionalA/pk_ck_v2_manual_review_sample44.csv"),
    xlsx: path.join(ROOT, "reports/experiment7_optionalA/pk_ck_v2_manual_review_sample44.xlsx"),
    sheetName: "PKCK_v2_Review",
    labelColumn: "human_pkck_label",
    noteColumn: "human_notes",
    reviewerColumn: "human_reviewer",
    labels: ["pk_follow", "ck_follow", "mixed_or_conflict_ack", "refusal", "other", "uncertain"],
    title: "Optional Experiment A v2 PK/CK Review",
    instructions: [
      "Fill only the highlighted human columns.",
      "pk_follow: answer follows pk_answer only.",
      "ck_follow: answer follows ck_answer only.",
      "mixed_or_conflict_ack: answer mentions both or discusses the conflict.",
      "refusal: answer says it cannot determine or refuses.",
      "other: irrelevant, empty, prompt repetition, or not parseable.",
      "Use uncertain when the answer is ambiguous.",
    ],
  },
];

function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        field += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function columnLetter(indexZeroBased) {
  let n = indexZeroBased + 1;
  let s = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function truncateCell(value, max = 32700) {
  const s = String(value ?? "");
  return s.length > max ? s.slice(0, max) : s;
}

async function buildWorkbook(job) {
  const csvText = await fs.readFile(job.csv, "utf8");
  const rows = parseCsv(csvText).map((row) => row.map((value) => truncateCell(value)));
  const headers = rows[0];
  const workbook = Workbook.create();
  const review = workbook.worksheets.add(job.sheetName);
  const guide = workbook.worksheets.add("Guide");

  const rowCount = rows.length;
  const colCount = headers.length;
  const lastCol = columnLetter(colCount - 1);
  review.getRange(`A1:${lastCol}${rowCount}`).values = rows;

  const labelIndex = headers.indexOf(job.labelColumn);
  const noteIndex = headers.indexOf(job.noteColumn);
  const reviewerIndex = headers.indexOf(job.reviewerColumn);
  if (labelIndex < 0 || noteIndex < 0 || reviewerIndex < 0) {
    throw new Error(`Missing editable columns in ${job.csv}`);
  }

  const labelCol = columnLetter(labelIndex);
  const noteCol = columnLetter(noteIndex);
  const reviewerCol = columnLetter(reviewerIndex);
  review.getRange(`${labelCol}2:${labelCol}${rowCount}`).dataValidation = {
    allowBlank: true,
    list: { inCellDropDown: true, source: job.labels },
  };

  try {
    review.freezePanes = { rows: 1 };
  } catch {
    // Freeze panes are a convenience; the workbook remains usable without them.
  }

  for (const col of [labelCol, noteCol, reviewerCol]) {
    const range = review.getRange(`${col}1:${col}${rowCount}`);
    try {
      range.format.fill = { color: "#FFF2CC" };
    } catch {
      // Formatting API can vary; validation and content are the important parts.
    }
  }

  guide.getRange("A1:B1").values = [["Field", "Instruction"]];
  const guideRows = [
    ["Workbook", job.title],
    ["Editable columns", `${job.labelColumn}, ${job.noteColumn}, ${job.reviewerColumn}`],
    ["Allowed labels", job.labels.join(" / ")],
    ...job.instructions.map((item, idx) => [`Step ${idx + 1}`, item]),
  ];
  guide.getRange(`A2:B${guideRows.length + 1}`).values = guideRows;

  const check = await workbook.inspect({
    kind: "table",
    range: `${job.sheetName}!A1:${lastCol}${Math.min(rowCount, 6)}`,
    include: "values",
    tableMaxRows: 6,
    tableMaxCols: Math.min(colCount, 12),
  });
  console.log(check.ndjson);

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);

  await workbook.render({ sheetName: job.sheetName, range: `A1:${lastCol}${Math.min(rowCount, 12)}`, scale: 1 });
  await workbook.render({ sheetName: "Guide", range: `A1:B${guideRows.length + 1}`, scale: 1 });

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(job.xlsx);
  console.log(`saved: ${job.xlsx}`);
}

for (const job of jobs) {
  await buildWorkbook(job);
}
