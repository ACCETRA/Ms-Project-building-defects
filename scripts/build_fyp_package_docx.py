#!/usr/bin/env python3
"""Build a self-contained DOCX acceptance package from project evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "fyp.docx"

TEXT_FILES = [
    ROOT / "docs/V1_COMPARISON_REPORT.md",
    ROOT / "docs/DECISION_LOG.md",
    ROOT / "docs/CONTRIBUTOR_GPU_SETUP.md",
    ROOT / "docs/SAM_DECODER_TRAINING.md",
    ROOT / "docs/V1_TRAINING_RUNBOOK.md",
    ROOT / "docs/OUTPUT_CONTRACT.md",
    ROOT / "docs/HUMAN_ERROR_REVIEW.md",
]
JSON_FILES = [
    ROOT / "runs/evaluation/cubit_locked_detect.json",
    ROOT / "runs/evaluation/cubit_locked_segment.json",
    ROOT / "runs/evaluation/resnet_metrics.json",
    ROOT / "runs/evaluation/cross_domain_analysis.json",
    ROOT / "runs/evaluation/error_review_queue.json",
    ROOT / "runs/evaluation/source_specific/cif_segment.json",
    ROOT / "runs/evaluation/source_specific/dacl_segment.json",
    ROOT / "runs/evaluation/source_specific/s2ds_segment.json",
    ROOT / "runs/evaluation/source_specific/uav75_segment.json",
    ROOT / "runs/comparison/yolo_sam/v1-sample-10/summary.json",
    ROOT / "runs/florence/v1-validation-sample-10/summary.json",
]


def paragraph(text: str, style: str = "Normal") -> str:
    text = text.replace("\t", "    ")
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def heading(text: str, level: int = 1) -> str:
    return paragraph(text, f"Heading{min(level, 3)}")


def markdown_blocks(text: str) -> list[str]:
    blocks = []
    in_code = False
    code_lines = []
    for line in text.splitlines():
        if line.startswith("```"):
            if in_code:
                blocks.append(paragraph("\n".join(code_lines), "Code"))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("# "):
            blocks.append(heading(line[2:], 1))
        elif line.startswith("## "):
            blocks.append(heading(line[3:], 2))
        elif line.startswith("### "):
            blocks.append(heading(line[4:], 3))
        elif line.startswith("- ") or re.match(r"^\d+\. ", line):
            blocks.append(paragraph("• " + re.sub(r"^\d+\. ", "", line[2:] if line.startswith("- ") else line)))
        elif line.strip():
            clean = re.sub(r"[`*_]", "", line)
            blocks.append(paragraph(clean))
    if code_lines:
        blocks.append(paragraph("\n".join(code_lines), "Code"))
    return blocks


def document_xml() -> str:
    body = []
    body.append(heading("Project Alpha Building Defect Inspection", 1))
    body.append(paragraph("Consolidated FYP acceptance and verification package", "Subtitle"))
    body.append(paragraph(f"Generated {datetime.now(timezone.utc).isoformat()}"))
    body.append(heading("Executive Status", 1))
    body.append(paragraph("Acceptance status: demo_only. The project has a functional local browser prototype, trained YOLO/ResNet v1 baselines, locked/source-specific evaluations, and documented provenance. It is not a production-ready or structural-safety system."))
    body.append(paragraph("Human image-level error review is complete by project-owner confirmation and documented with its evidence boundary. Unresolved deployment or extended-model gates are target-site validation, S2DS six-class class identity, full Florence fine-tuning, and a fully trained SAM decoder checkpoint."))
    body.append(heading("Evidence Files", 1))
    for path in TEXT_FILES + JSON_FILES:
        if path.exists():
            body.append(paragraph(str(path.relative_to(ROOT))))
    for path in TEXT_FILES:
        if path.exists():
            body.append(heading(path.name, 1))
            body.extend(markdown_blocks(path.read_text(encoding="utf-8", errors="replace")))
    for path in JSON_FILES:
        if path.exists():
            body.append(heading(path.name, 2))
            body.append(paragraph(path.read_text(encoding="utf-8", errors="replace"), "Code"))
    body.append(heading("Harness Findings", 1))
    finding_dir = ROOT / "runs/harness/findings"
    if finding_dir.exists():
        for path in sorted(finding_dir.glob("*.json")):
            body.append(heading(path.name, 2))
            body.append(paragraph(path.read_text(encoding="utf-8", errors="replace"), "Code"))
    else:
        body.append(paragraph("No persisted harness findings were present when this package was generated."))
    sect = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>'
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + "".join(body) + sect + "</w:body></w:document>"


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content_types = '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:color w:val="2D7771"/><w:sz w:val="24"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:rPr><w:rFonts w:ascii="Consolas"/><w:sz w:val="16"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style></w:styles>'''
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/document.xml", document_xml())
    print(OUTPUT)


if __name__ == "__main__":
    build()
