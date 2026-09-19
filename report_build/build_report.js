"use strict";
const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  PageBreak,
  Header,
  Footer,
  PageNumber,
  LevelFormat,
  convertInchesToTwip,
  BorderStyle,
  TabStopType,
  LeaderType,
} = require("docx");

const H = require("./helpers");
const part1 = require("./content_part1");
const part2 = require("./content_part2");
const part3 = require("./content_part3");
const part4 = require("./content_part4");
const part5 = require("./content_part5");

const OUT_PATH = path.join(H.REPO_ROOT, "QGNN_V4_Final_Research_Report.docx");

// ---------------------------------------------------------------------
// Title page
// ---------------------------------------------------------------------
function titlePage() {
  const spacer = (h) => new Paragraph({ spacing: { after: h }, children: [] });
  return [
    spacer(1600),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({
          text: "QGNN-v4",
          bold: true,
          size: 64,
          color: H.COLOR_HEADING,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 480 },
      children: [
        new TextRun({
          text: "Final Research Report",
          bold: true,
          size: 40,
          color: H.COLOR_ACCENT,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 800 },
      children: [
        new TextRun({
          text:
            "A Hybrid Quantum-Classical Head on Frozen GraphSAGE Embeddings for Supply-Chain Risk Prediction",
          italics: true,
          size: 26,
          color: H.COLOR_MUTED,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      border: { top: { color: H.COLOR_RULE, space: 8, style: BorderStyle.SINGLE, size: 6 } },
      children: [],
    }),
    spacer(1200),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [new TextRun({ text: "M.Tech Research Project", size: 22 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [new TextRun({ text: "Author: Aditya Jagtap", size: 22 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [new TextRun({ text: "September 2026", size: 22 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [
        new TextRun({
          text: "Repository branch: qgnn-package-restructure — commit 755e076",
          size: 20,
          color: H.COLOR_MUTED,
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({
          text: "Every quantum-circuit result in this report is an ideal, noise-free classical simulation (PennyLane default.qubit, backprop). No physical quantum hardware was used at any point in this project.",
          size: 18,
          italics: true,
          color: H.COLOR_MUTED,
        }),
      ],
    }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ---------------------------------------------------------------------
// Table of contents (manual — built from the rendered PDF's own page
// numbers, since headless LibreOffice conversion does not resolve a
// docx TOC field's page numbers without an interactive field-update pass)
// ---------------------------------------------------------------------
const TOC_ENTRIES = [
  ["Abstract", 4, 1],
  ["1. Introduction", 5, 1],
  ["1.1 Research Objective", 5, 2],
  ["1.2 Research Questions", 5, 2],
  ["1.3 How This Report Is Organized", 5, 2],
  ["2. System Architecture Overview", 7, 1],
  ["3. Data and Information Flow", 9, 1],
  ["4. Graph Construction", 12, 1],
  ["5. Classical GraphSAGE Processing", 14, 1],
  ["6. QGNN-v4 Detailed Architecture", 16, 1],
  ["7. Quantum Circuit Diagram", 18, 1],
  ["8. What Is Trainable? Frozen vs. Trainable Components", 19, 1],
  ["9. Training and Evaluation Workflow", 21, 1],
  ["10. Research Experiment Roadmap", 23, 1],
  ["11. Experimental Setup", 25, 1],
  ["11.1 Technology Stack", 25, 2],
  ["11.2 Dataset Characteristics", 25, 2],
  ["11.3 Graph Characteristics", 26, 2],
  ["11.4 Training and Evaluation Protocol", 26, 2],
  ["12. Experiment Control: What Changed, What Stayed Fixed", 27, 1],
  ["12.1 What Changed in Each Experiment?", 27, 2],
  ["12.2 Master Experiment Inventory", 27, 2],
  ["13. Results", 29, 1],
  ["13.1 Classical Baseline", 29, 2],
  ["13.2 QGNN-v4 Baseline", 29, 2],
  ["13.3 Classical vs. QGNN-v4: Extended Metrics", 30, 2],
  ["13.4 Primary (Temporal) Evaluation", 32, 2],
  ["13.5 Severity (Out-of-Distribution) Evaluation", 32, 2],
  ["13.6 Ablation Studies", 33, 2],
  ["13.7 Representation Analysis", 37, 2],
  ["13.8 Quantum Resource Analysis", 38, 2],
  ["13.9 Seed Sensitivity", 39, 2],
  ["13.10 Fresh-Onset Analysis", 41, 2],
  ["13.11 Calibration Analysis", 42, 2],
  ["13.12 Quantum Representation Analysis", 44, 2],
  ["13.13 Primary vs. Severity Comparison", 46, 2],
  ["14. Discussion", 47, 1],
  ["14.1 Why Primary and Severity Behave Differently", 47, 2],
  ["14.2 Classical vs. Quantum Head Behavior", 47, 2],
  ["14.3 Representation Findings", 48, 2],
  ["14.4 Seed Sensitivity", 48, 2],
  ["14.5 Quantum-Specific Observations", 49, 2],
  ["14.6 Implications", 49, 2],
  ["15. Limitations", 50, 1],
  ["15.1 Severity-5 Event Limitation (the single most important caveat)", 50, 2],
  ["15.2 Fresh-Onset Limitation", 50, 2],
  ["15.3 Synthetic Dataset", 50, 2],
  ["15.4 Seed and Sample-Size Limitations", 50, 2],
  ["15.5 Parameter-Count Confounds", 50, 2],
  ["15.6 Ideal Simulator, No Hardware", 51, 2],
  ["15.7 Frozen Encoder, Not Co-Trained", 51, 2],
  ["15.8 Threshold Policy Held Fixed Throughout", 51, 2],
  ["15.9 Computational / Runtime Limitations", 51, 2],
  ["15.10 Scope of the “Ruled Out” Claims", 51, 2],
  ["16. Conclusion", 52, 1],
  ["16.1 Explicitly Demonstrated", 52, 2],
  ["16.2 Explicitly Not Demonstrated", 52, 2],
  ["17. Future Work", 54, 1],
  ["18. Reproducibility", 55, 1],
  ["18.1 How to Reproduce the Standing References", 56, 2],
  ["Appendix A — Final Decision Matrix", 57, 1],
];

const TOC_RIGHT_TAB = convertInchesToTwip(6.27);

function tocEntry(text, pageNum, level) {
  const isTop = level === 1;
  return new Paragraph({
    spacing: { after: isTop ? 100 : 60 },
    indent: { left: isTop ? 0 : convertInchesToTwip(0.3) },
    tabStops: [{ type: TabStopType.RIGHT, position: TOC_RIGHT_TAB, leader: LeaderType.DOT }],
    children: [
      new TextRun({ text, bold: isTop, size: isTop ? 22 : 20 }),
      new TextRun({ text: `\t${pageNum}`, bold: isTop, size: isTop ? 22 : 20 }),
    ],
  });
}

function tocPage() {
  return [
    H.h1("Table of Contents"),
    ...TOC_ENTRIES.map(([text, pageNum, level]) => tocEntry(text, pageNum, level)),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ---------------------------------------------------------------------
// Header / Footer
// ---------------------------------------------------------------------
function makeFooter() {
  return new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        border: { top: { color: H.COLOR_RULE, space: 4, style: BorderStyle.SINGLE, size: 4 } },
        children: [
          new TextRun({ text: "QGNN-v4 Final Research Report", size: 16, color: H.COLOR_MUTED }),
          new TextRun({ text: "   •   Page ", size: 16, color: H.COLOR_MUTED }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: H.COLOR_MUTED }),
          new TextRun({ text: " of ", size: 16, color: H.COLOR_MUTED }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: H.COLOR_MUTED }),
        ],
      }),
    ],
  });
}

function makeHeader() {
  return new Header({
    children: [
      new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [
          new TextRun({
            text: "QGNN-v4 — Hybrid Quantum-Classical Supply-Chain Risk Prediction",
            size: 16,
            italics: true,
            color: H.COLOR_MUTED,
          }),
        ],
      }),
    ],
  });
}

// ---------------------------------------------------------------------
// Assemble
// ---------------------------------------------------------------------
async function main() {
  H.resetCounters();

  const body = [
    ...titlePage(),
    ...tocPage(),
    ...part1.build(),
    ...part2.build(),
    ...part3.build(),
    ...part4.build(),
    ...part5.build(),
  ];

  const doc = new Document({
    creator: "Aditya Jagtap",
    title: "QGNN-v4 Final Research Report",
    description:
      "Hybrid quantum-classical head on frozen GraphSAGE embeddings for supply-chain risk prediction — final consolidated research report.",
    styles: {
      default: {
        document: {
          run: { font: H.FONT, size: 22 },
          paragraph: { spacing: { line: 276 } },
        },
      },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: H.FONT, size: 32, bold: true, color: H.COLOR_HEADING },
          paragraph: {
            spacing: { before: 480, after: 240 },
          },
        },
        {
          id: "Heading2",
          name: "Heading 2",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: H.FONT, size: 26, bold: true, color: H.COLOR_ACCENT },
          paragraph: { spacing: { before: 360, after: 180 } },
        },
        {
          id: "Heading3",
          name: "Heading 3",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: H.FONT, size: 23, bold: true, color: H.COLOR_MUTED },
          paragraph: { spacing: { before: 260, after: 140 } },
        },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullet-list",
          levels: [
            {
              level: 0,
              format: LevelFormat.BULLET,
              text: "•",
              alignment: AlignmentType.LEFT,
              style: {
                paragraph: {
                  indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) },
                },
              },
            },
          ],
        },
        {
          reference: "numbered-list",
          levels: [
            {
              level: 0,
              format: LevelFormat.DECIMAL,
              text: "%1.",
              alignment: AlignmentType.LEFT,
              style: {
                paragraph: {
                  indent: { left: convertInchesToTwip(0.4), hanging: convertInchesToTwip(0.25) },
                },
              },
            },
          ],
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: convertInchesToTwip(8.27), height: convertInchesToTwip(11.69) },
            margin: {
              top: convertInchesToTwip(0.9),
              bottom: convertInchesToTwip(0.9),
              left: convertInchesToTwip(1.0),
              right: convertInchesToTwip(1.0),
            },
          },
        },
        headers: { default: makeHeader() },
        footers: { default: makeFooter() },
        children: body,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT_PATH, buffer);
  console.log(`wrote ${OUT_PATH} (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
