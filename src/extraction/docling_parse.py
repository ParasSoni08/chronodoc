"""
Run a single PDF through Docling and dump its raw output for inspection.

This does NOT map Docling's output onto the ChronoDoc entity/clause/graph
schema yet -- that mapping is designed after we've seen what Docling
actually returns on real sample documents.

Usage:
    python src/extraction/docling_parse.py path/to/document.pdf
"""

import argparse
import json
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_json"


def parse_pdf(pdf_path: Path) -> dict:
    # This machine is memory-constrained, so we trade throughput for a
    # smaller peak footprint: OCR off (not needed for native text-layer
    # PDFs), batch sizes down to 1 page at a time, table mode set to FAST.
    # Re-tune once run on a machine with more headroom.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.accelerator_options.num_threads = 1

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump raw Docling output for one PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF")
    args = parser.parse_args()

    pdf_path: Path = args.pdf_path
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.docling_raw.json"

    print(f"Parsing {pdf_path} with Docling...")
    doc_dict = parse_pdf(pdf_path)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)

    print(f"Raw Docling output written to {out_path}")


if __name__ == "__main__":
    main()
