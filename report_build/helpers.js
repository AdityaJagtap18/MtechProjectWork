"use strict";
/* Shared helpers for assembling QGNN_V4_Final_Research_Report.docx.
 * Keeps one consistent visual language (fonts, headings, figure/table
 * captions, numbering) across every content_part*.js module. */

const fs = require("fs");
const path = require("path");
const {
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  ImageRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  ShadingType,
  PageBreak,
  VerticalAlign,
  convertInchesToTwip,
  LineRuleType,
} = require("docx");

const REPO_ROOT = path.resolve(__dirname, "..");
const DIAGRAMS = path.join(REPO_ROOT, "report_assets", "diagrams");
const CHARTS = path.join(REPO_ROOT, "report_assets", "charts");
const FINAL_FIGS = path.join(REPO_ROOT, "final_figures");

const FONT = "Calibri";
const COLOR_HEADING = "1F3864";
const COLOR_ACCENT = "2E5395";
const COLOR_MUTED = "595959";
const COLOR_RULE = "BFBFBF";

let figureCounter = 0;
let tableCounter = 0;
function resetCounters() {
  figureCounter = 0;
  tableCounter = 0;
}
function nextFigureNumber() {
  figureCounter += 1;
  return figureCounter;
}
function nextTableNumber() {
  tableCounter += 1;
  return tableCounter;
}

// ---------------------------------------------------------------------
// Headings & body text
// ---------------------------------------------------------------------

function h1(text) {
  // No paragraph border on headings: a bottom border on Heading1
  // (whether in the named style or inline) corrupts the rendering of later
  // images in LibreOffice's headless PDF export (verified by bisection).
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 480, after: 240 },
    children: [new TextRun({ text, bold: true })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text, bold: true })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 260, after: 140 },
    children: [new TextRun({ text, bold: true })],
  });
}

// runs: string | array of {text, bold, italic, color, superScript}
function toRuns(runs) {
  if (typeof runs === "string") {
    return [new TextRun({ text: runs })];
  }
  return runs.map((r) => {
    if (typeof r === "string") return new TextRun({ text: r });
    return new TextRun(r);
  });
}

function p(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 200, line: 276 },
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    children: toRuns(runs),
    ...opts.paraOpts,
  });
}

function pNoJustify(runs, opts = {}) {
  return p(runs, { alignment: AlignmentType.LEFT, ...opts });
}

function labelPara(label, runs, opts = {}) {
  const children = [
    new TextRun({ text: `${label} `, bold: true, italics: true }),
    ...toRuns(runs),
  ];
  return new Paragraph({
    spacing: { after: 200, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children,
    ...opts,
  });
}

// Bulleted list using a real numbering definition (never literal "-"/"*").
function bullets(items, opts = {}) {
  return items.map(
    (item) =>
      new Paragraph({
        numbering: { reference: "bullet-list", level: 0 },
        spacing: { after: 120, line: 264 },
        children: toRuns(item),
      })
  );
}

let numberedInstance = 0;
function numbered(items) {
  numberedInstance += 1;
  const instance = numberedInstance;
  return items.map(
    (item) =>
      new Paragraph({
        numbering: { reference: "numbered-list", level: 0, instance },
        spacing: { after: 120, line: 264 },
        children: toRuns(item),
      })
  );
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function rule() {
  return new Paragraph({
    spacing: { before: 120, after: 240 },
    border: {
      bottom: { color: COLOR_RULE, space: 1, style: BorderStyle.SINGLE, size: 6 },
    },
    children: [],
  });
}

// ---------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------

// key: filename stem (without .png), dir: DIAGRAMS|CHARTS, caption: string
// discussion: array of paragraphs (Paragraph[]) to render after the figure.
function figure(key, dir, captionText, opts = {}) {
  const filePath = path.join(dir, `${key}.png`);
  const data = fs.readFileSync(filePath);
  const pxW = data.readUInt32BE(16);
  const pxH = data.readUInt32BE(20);
  const aspect = pxW / pxH;
  const maxWidthIn = opts.maxWidthIn || 6.1;
  const maxHeightIn = opts.maxHeightIn || 7.7;
  let widthIn = maxWidthIn;
  let heightIn = widthIn / aspect;
  if (heightIn > maxHeightIn) {
    heightIn = maxHeightIn;
    widthIn = heightIn * aspect;
  }
  const num = nextFigureNumber();

  const imgPara = new Paragraph({
    keepNext: true,
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120, line: 240, lineRule: LineRuleType.AUTO },
    children: [
      new ImageRun({
        type: "png",
        data,
        transformation: {
          width: Math.round(widthIn * 96),
          height: Math.round(heightIn * 96),
        },
      }),
    ],
  });

  const captionPara = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({ text: `Figure ${num}. `, bold: true, size: 20 }),
      new TextRun({ text: captionText, size: 20, italics: false }),
    ],
  });

  return { paragraphs: [imgPara, captionPara], number: num };
}

// ---------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------

const CELL_MARGIN = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, widthDXA, fontSize = 19) {
  return new TableCell({
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: COLOR_ACCENT, color: "auto" },
    margins: CELL_MARGIN,
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text, bold: true, color: "FFFFFF", size: fontSize })],
      }),
    ],
  });
}

function bodyCell(text, widthDXA, opts = {}) {
  const align = opts.align || AlignmentType.LEFT;
  const shadeFill = opts.shade ? "EEF2F8" : undefined;
  return new TableCell({
    width: { size: widthDXA, type: WidthType.DXA },
    shading: shadeFill
      ? { type: ShadingType.CLEAR, fill: shadeFill, color: "auto" }
      : undefined,
    margins: CELL_MARGIN,
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        alignment: align,
        children: [
          new TextRun({ text: String(text), bold: !!opts.bold, size: opts.fontSize || 19 }),
        ],
      }),
    ],
  });
}

// columns: [{header, width(fraction of total), align}], rows: array of arrays
// widthTotalDXA default ~ 6.1in content width
function dataTable(columns, rows, opts = {}) {
  const totalDXA = opts.totalDXA || convertInchesToTwip(6.1);
  const colWidths = columns.map((c) =>
    Math.round(totalDXA * (c.width !== undefined ? c.width : 1 / columns.length))
  );
  const headerRow = new TableRow({
    tableHeader: true,
    children: columns.map((c, i) => headerCell(c.header, colWidths[i], opts.fontSize || 19)),
  });
  const bodyRows = rows.map((row, ri) => {
    const shade = opts.zebra && ri % 2 === 1;
    return new TableRow({
      children: row.map((cell, ci) =>
        bodyCell(cell, colWidths[ci], {
          align: columns[ci].align || AlignmentType.LEFT,
          bold: opts.boldRows && opts.boldRows.includes(ri),
          shade,
          fontSize: opts.fontSize,
        })
      ),
    });
  });
  return new Table({
    width: { size: totalDXA, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...bodyRows],
  });
}

function tableCaption(captionText) {
  const num = nextTableNumber();
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 180 },
    children: [
      new TextRun({ text: `Table ${num}. `, bold: true, size: 20 }),
      new TextRun({ text: captionText, size: 20 }),
    ],
  });
}

module.exports = {
  REPO_ROOT,
  DIAGRAMS,
  CHARTS,
  FINAL_FIGS,
  FONT,
  COLOR_HEADING,
  COLOR_ACCENT,
  COLOR_MUTED,
  COLOR_RULE,
  resetCounters,
  nextTableNumber,
  h1,
  h2,
  h3,
  p,
  pNoJustify,
  labelPara,
  bullets,
  numbered,
  pageBreak,
  rule,
  figure,
  dataTable,
  tableCaption,
  toRuns,
};
