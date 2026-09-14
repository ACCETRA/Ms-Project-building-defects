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

EVIDENCE_TEXT_FILES = [
    ROOT / "docs/FYP_COMPLETION_CHECKLIST.md",
    ROOT / "docs/V1_COMPARISON_REPORT.md",
    ROOT / "docs/RECOMMENDED_IMPROVEMENTS.md",
    ROOT / "docs/DATASET_REPRODUCTION_GUIDE.md",
    ROOT / "docs/OUTPUT_CONTRACT.md",
    ROOT / "docs/HUMAN_ERROR_REVIEW.md",
]
EMBEDDED_TEXT_FILES = [
    path for path in EVIDENCE_TEXT_FILES
    if path.name != "V1_COMPARISON_REPORT.md"
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
    body.append(paragraph("Project Alpha Building Defect Inspection", "Title"))
    body.append(paragraph("Consolidated FYP acceptance and verification package", "Subtitle"))
    body.append(paragraph(f"Generated {datetime.now(timezone.utc).isoformat()}"))
    body.append(heading("Executive Status", 1))
    body.append(paragraph("Acceptance status: complete for the approved demo_only FYP scope. The package contains a functional local browser application, real model inference, trained YOLO and ResNet checkpoints, bounded Florence and SAM routes, locked and source-specific evaluations, human-review documentation, and reproducibility evidence."))
    body.append(paragraph("The system is an academic demonstration. It does not make structural-safety determinations, and physical measurements remain pixel-only unless valid calibration metadata is supplied."))
    body.append(heading("Key Measured Results", 1))
    body.append(paragraph("YOLO11n detection, CUBIT-InSeg locked test, 701 images: precision 89.997%, recall 64.198%, mAP@50 63.165%, and mAP@50:95 54.107%."))
    body.append(paragraph("YOLO11n segmentation, CUBIT-InSeg locked test, 701 images: box mAP@50 63.169%, mask precision 82.735%, mask recall 60.749%, mask mAP@50 58.341%, and mask mAP@50:95 40.496%."))
    body.append(paragraph("Final training-run validation: detection box precision 93.166%, recall 25.744%, mAP@50 28.038%, and mAP@50:95 22.334%; segmentation box mAP@50 28.699% and mask precision 90.131%, mask recall 24.466%, mask mAP@50 25.581%, and mask mAP@50:95 16.803%. These are validation results, separate from the locked CUBIT test above."))
    body.append(paragraph("ResNet-50 group-safe validation, 1,104 images: micro precision 99.374%, recall 99.286%, F1 99.330%, and PR-AUC 99.374%. CODEBRIM official test, 632 images: micro precision 26.488%, recall 59.413%, F1 36.641%, and PR-AUC 36.856%."))
    body.append(paragraph("External mask mAP@50: CiF 2.462% on 2,500 records, S2DS 0.698% on 93 images, DACL10K 0.339% on 975 images, and UAV75 0.0047% on 15 images. These source-specific results are not combined into one accuracy number."))
    body.append(paragraph("The Florence fine-tuning smoke run completed one step and the SAM decoder smoke run completed one FP32 epoch. They verify executable routes and are not presented as accuracy benchmarks."))
    body.append(heading("Evidence Files", 1))
    for path in EVIDENCE_TEXT_FILES + JSON_FILES:
        if path.exists():
            body.append(paragraph(str(path.relative_to(ROOT))))
    for path in EMBEDDED_TEXT_FILES:
        if path.exists():
            body.append(heading(path.stem.replace("_", " ").title(), 1))
            body.extend(markdown_blocks(path.read_text(encoding="utf-8", errors="replace")))
    body.append(heading("Machine Readable Evidence", 1))
    body.append(paragraph("The evaluation JSON files listed in Evidence Files are included separately in the handoff. They preserve full precision, checkpoint paths, sample counts, thresholds, and source-specific metrics without duplicating raw coordinate arrays in this reader report."))
    sect = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>'
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + "".join(body) + sect + "</w:body></w:document>"


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content_types = '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:color w:val="000000"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:color w:val="000000"/><w:b/><w:sz w:val="40"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="000000"/><w:sz w:val="24"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas"/><w:sz w:val="16"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/><w:rPr><w:color w:val="000000"/><w:b/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/><w:rPr><w:color w:val="000000"/><w:b/><w:sz w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/><w:rPr><w:color w:val="000000"/><w:b/><w:sz w:val="22"/></w:rPr></w:style></w:styles>'''
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/document.xml", document_xml())
    print(OUTPUT)


if __name__ == "__main__":
    build()
