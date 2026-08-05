"""
Map raw Docling output onto the ChronoDoc per-document extraction schema
(see chronodoc_brief.md) using a local Ollama model.

A document that Docling choked on in one pass (see docling_parse.py's note
on this machine's memory ceiling around page 13) can be split into several
raw JSON files; pass them in page order and this script stitches them back
into one logical document before structuring.

Design choice: `visual_elements` is populated directly from Docling's
`pictures` list, not by the LLM -- Docling already has real bboxes for
those, so asking a 3B model to re-detect them would just add noise.
Everything else (vendor, sections, clauses, entities, cross_references) is
extracted by the LLM from the page text, since that's the actual
structuring task.

Usage:
    python -m src.extraction.structure_entities \\
        data/extracted_json/foo_part1.docling_raw.json \\
        data/extracted_json/foo_part2.docling_raw.json \\
        --doc-id vendor-foo-contract-2014 \\
        --source-file foo.pdf \\
        --doc-type contract
"""

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional

import ollama
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_json"

MODEL = "llama3.1:8b"  # llama3.2:3b reliably hallucinated redaction markers over real values in testing
NUM_CTX = 8192
MAX_INPUT_CHARS = 12000

# Docling labels that are page furniture, not contract content.
SKIP_LABELS = {"page_header", "page_footer", "footnote"}


# --- Target schema (per chronodoc_brief.md) -------------------------------


class Vendor(BaseModel):
    name: str
    normalized_name: str


class LLMSection(BaseModel):
    heading: str
    start_source_ref: str = Field(description="id of the source element where this section begins, e.g. 'e17'")
    end_source_ref: str = Field(description="id of the source element where this section ends, e.g. 'e23'")


class LLMClause(BaseModel):
    clause_type: str
    text_summary: str
    source_ref: str = Field(description="id of the source element this was extracted from, e.g. 'e17'")
    confidence: float


class LLMEntity(BaseModel):
    # Deliberately restricted to party_name: asking the single-shot pass to
    # also find date/price/quantity/SLA_term entities produced duplicate,
    # inconsistently-grounded entities alongside the per-clause second pass
    # (extract_entity_for_clause) below, which does that job more reliably.
    entity_type: Literal["party_name"]
    value: str
    unit: Optional[str] = None
    source_ref: str = Field(description="id of the source element this was extracted from, e.g. 'e17'")
    confidence: float
    linked_clause_index: Optional[int] = Field(
        default=None, description="index into the clauses array this entity belongs to, if any"
    )


class CrossReference(BaseModel):
    from_clause_index: int = Field(description="index into the clauses array this reference originates from")
    to: str
    text: str


class LLMExtraction(BaseModel):
    """What we ask the LLM to produce -- everything except the parts we
    already know (doc_id/source_file/ingestion_date) or get more reliably
    from Docling directly (visual_elements). page_ref/bbox are deliberately
    NOT asked of the LLM: a small local model reproducing exact page
    numbers and bbox coordinates from memory is unreliable. Instead it
    cites a source_ref id, and we resolve that to page_ref/bbox ourselves
    from Docling's own provenance data."""

    doc_type: Literal["contract", "amendment", "renewal", "spec_sheet", "other"]
    effective_date: Optional[str] = None
    vendor: Vendor
    sections: list[LLMSection]
    clauses: list[LLMClause]
    entities: list[LLMEntity]
    cross_references: list[CrossReference]


# --- Merge raw Docling parts into one flat element list -------------------


def load_and_merge(raw_json_paths: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (texts, tables, pictures), each with page_no offset so a
    multi-part document reads as one continuous page range."""
    all_texts, all_tables, all_pictures = [], [], []
    page_offset = 0

    for path in raw_json_paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        part_page_count = len(doc["pages"])

        for t in doc["texts"]:
            for p in t.get("prov", []):
                p["page_no"] += page_offset
            all_texts.append(t)
        for t in doc["tables"]:
            for p in t.get("prov", []):
                p["page_no"] += page_offset
            all_tables.append(t)
        for p in doc["pictures"]:
            for prov in p.get("prov", []):
                prov["page_no"] += page_offset
            all_pictures.append(p)

        page_offset += part_page_count

    return all_texts, all_tables, all_pictures


def table_to_text(table: dict) -> str:
    cells = table["data"]["table_cells"]
    n_rows = max((c["end_row_offset_idx"] for c in cells), default=0)
    grid = [["" for _ in range(20)] for _ in range(n_rows)]
    max_col = 0
    for c in cells:
        r, col = c["start_row_offset_idx"], c["start_col_offset_idx"]
        if r < len(grid) and col < len(grid[r]):
            grid[r][col] = c["text"]
            max_col = max(max_col, col)
    lines = [" | ".join(row[: max_col + 1]) for row in grid]
    return "\n".join(lines)


def build_source_text(texts: list[dict], tables: list[dict]) -> tuple[str, dict[str, dict]]:
    """Returns (prompt_text, id_map) where id_map resolves each element id
    used in the prompt back to its real page_no/bbox for provenance."""
    elements = []
    id_map: dict[str, dict] = {}
    counter = 0

    for t in texts:
        if t["label"] in SKIP_LABELS or not t.get("text", "").strip():
            continue
        prov = t["prov"][0] if t.get("prov") else None
        eid = f"e{counter}"
        counter += 1
        id_map[eid] = {
            "page_ref": prov["page_no"] if prov else None,
            "bbox": _bbox_list(prov["bbox"]) if prov else None,
            "text": t["text"].strip(),
        }
        page_no = prov["page_no"] if prov else "?"
        elements.append((page_no, f"[{eid} p{page_no} {t['label']}] {t['text'].strip()}"))

    for tb in tables:
        prov = tb["prov"][0] if tb.get("prov") else None
        eid = f"e{counter}"
        counter += 1
        table_text = table_to_text(tb)
        id_map[eid] = {
            "page_ref": prov["page_no"] if prov else None,
            "bbox": _bbox_list(prov["bbox"]) if prov else None,
            "text": table_text,
        }
        page_no = prov["page_no"] if prov else "?"
        elements.append((page_no, f"[{eid} p{page_no} table]\n{table_text}"))

    elements.sort(key=lambda e: e[0] if isinstance(e[0], int) else 0)
    joined = "\n\n".join(e[1] for e in elements)

    if len(joined) > MAX_INPUT_CHARS:
        print(
            f"WARNING: source text is {len(joined)} chars, truncating to "
            f"{MAX_INPUT_CHARS} to fit the model's context window. Later "
            f"content will be missing from this extraction.",
            file=sys.stderr,
        )
        joined = joined[:MAX_INPUT_CHARS]

    return joined, id_map


def _bbox_list(bbox: dict) -> list[float]:
    return [bbox["l"], bbox["t"], bbox["r"], bbox["b"]]


_REF_RE = re.compile(r"e\d+")


def resolve_ref(source_ref: str, id_map: dict[str, dict]) -> dict:
    """The model sometimes echoes the full '[eN pM ...]' tag instead of
    just the id -- pull the eN token out rather than requiring an exact
    match."""
    match = _REF_RE.search(source_ref)
    if match:
        return id_map.get(match.group(0), {})
    return {}


def pictures_to_visual_elements(pictures: list[dict]) -> list[dict]:
    visual_elements = []
    for pic in pictures:
        for p in pic.get("prov", []):
            bbox = p["bbox"]
            visual_elements.append(
                {
                    "type": "picture",
                    "page_ref": p["page_no"],
                    "bbox": [bbox["l"], bbox["t"], bbox["r"], bbox["b"]],
                    "detected_only": True,
                    "note": "layout-detected by Docling, not semantically classified",
                }
            )
    return visual_elements


# --- LLM structuring call --------------------------------------------------

SYSTEM_PROMPT = """You are extracting structured data from a real commercial \
contract for a legal document intelligence system. You will be given the \
contract's text as a sequence of elements, each tagged with a unique id and \
page number, e.g. "[e17 p3 text] ...".

Rules:
- Only extract facts that are explicitly present in the given text. Do not \
invent vendor names, dates, or values.
- Every clause and entity you extract MUST include a source_ref set to the \
element id (the "eN" token, not the page number) of the element it was \
extracted from. Never invent an id that wasn't shown to you.
- Some values are redacted in the source with markers like [***] or [****]. \
Never invent a value to fill a redaction -- if a clause's key value is \
redacted, still record the clause (e.g. clause_type "pricing") but omit an \
Entity for the redacted value, or set its value to the literal redaction \
marker if you must reference it.
- confidence is your own estimate (0.0-1.0) of how clearly the source text \
supports this extraction.
- clause_type should be a short lowercase snake_case label describing what \
kind of clause it is (e.g. "pricing", "term", "exclusivity", "termination", \
"notices").
- entity_type must be exactly "party_name" -- this pass only extracts the \
contracting parties. Concrete values (dates, prices, quantities) inside \
each clause are extracted separately; do not produce them here.
- The contracting parties are named in the opening sentence, e.g. "is \
entered into by and between X ... and Y ...". Use those legal entity names \
for party_name entities and for vendor.name. Do NOT use a person's job \
title or a brand name mentioned only in a signature block (e.g. "SVP, \
Wolfspeed") as a party name -- that is not the contracting entity.
- vendor is the counterparty supplying goods/services (not the company that \
is clearly the buyer/customer, if that distinction is evident from the \
text).
- cross_references are places where the text explicitly points to another \
clause, section, or an external document/exhibit (e.g. "as specified in \
Exhibit B", "Paragraph 2(b)"). from_clause_index is the 0-based index of \
the clause (in the clauses array you are producing) that contains the \
reference. "to" must be the human-readable name of the target (e.g. \
"Exhibit B", "Paragraph 2(b)", "Section 5") -- never an element id.
- effective_date: look near the start of the document for a phrase like \
"dated as of ... (the 'Effective Date')" or "entered into as of ...". \
Extract it even if it's also embedded in a WHEREAS clause or a defined-term \
parenthetical. Use the format shown in the source (e.g. "June 30, 2020").
- vendor.normalized_name should be lowercase, spaces replaced with \
underscores, no punctuation.
- For sections, start_source_ref and end_source_ref must be element ids \
("eN") from the material -- the first and last element that belong to that \
section. Never invent a page range from memory; it is derived from these \
ids afterward.

Respond with JSON only, matching the provided schema."""


def structure_document(source_text: str, model: str = MODEL) -> LLMExtraction:
    client = ollama.Client()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source_text},
        ],
        format=LLMExtraction.model_json_schema(),
        options={"num_ctx": NUM_CTX, "temperature": 0.0},
    )
    return LLMExtraction.model_validate_json(response.message.content)


# --- Second pass: mine one entity per clause -------------------------------
#
# Asking a 3B model to produce sections+clauses+entities+cross_references in
# one shot reliably drops the entity extraction (it keeps finding party
# names and stopping there, even when explicitly told to look for more).
# A single narrow question per clause -- "does this specific clause have a
# concrete value, and what is it" -- is a much easier task for a small model
# and comes out far more reliable in practice.

CLAUSE_ENTITY_PROMPT_TEMPLATE = """Extract the key date, price, or quantity from this contract clause. \
Respond with JSON only: {{"has_value": bool, "entity_type": "price"|"date"|"SLA_term"|"quantity"|null, "value": string|null, "unit": string|null, "confidence": number}}.
If the value is redacted (marked [***] or [****]) or the clause is pure boilerplate with no concrete value, has_value is false.

Example:
Clause type: term
Clause text: The term of this Agreement shall expire on March 1, 2020.
Answer: {{"has_value": true, "entity_type": "date", "value": "March 1, 2020", "unit": null, "confidence": 0.95}}

Now do this one:
Clause type: {clause_type}
Clause text: {clause_text}
Answer:"""


class ClauseEntityExtraction(BaseModel):
    has_value: bool
    entity_type: Optional[Literal["price", "date", "SLA_term", "quantity"]] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    confidence: float = 0.0


def extract_entity_for_clause(clause_type: str, clause_text: str, model: str = MODEL) -> Optional[ClauseEntityExtraction]:
    # Deliberately NOT using format=<json schema> here: with this small model,
    # strict schema-constrained decoding biased it toward the trivially-valid
    # "has_value": false / all-nulls completion almost every time,  even on
    # clauses with an obvious date in them. Free-form JSON with a worked
    # example in the prompt was reliably correct in testing, so we parse
    # leniently instead of constraining generation.
    client = ollama.Client()
    prompt = CLAUSE_ENTITY_PROMPT_TEMPLATE.format(clause_type=clause_type, clause_text=clause_text)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    content = response.message.content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        print(f"WARNING: clause-entity response wasn't JSON: {content!r}", file=sys.stderr)
        return None
    try:
        result = ClauseEntityExtraction.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"WARNING: couldn't parse clause-entity response ({exc}): {content!r}", file=sys.stderr)
        return None
    return result if result.has_value else None


# --- CLI --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure raw Docling output into the ChronoDoc entity schema via Ollama."
    )
    parser.add_argument(
        "raw_json_paths",
        type=Path,
        nargs="+",
        help="One or more docling_raw.json files, in page order, for one logical document",
    )
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    for p in args.raw_json_paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    texts, tables, pictures = load_and_merge(args.raw_json_paths)
    source_text, id_map = build_source_text(texts, tables)
    print(f"Sending {len(source_text)} chars of source text to {args.model}...")

    extraction = structure_document(source_text, model=args.model)

    sections = []
    for i, s in enumerate(extraction.sections):
        start_prov = resolve_ref(s.start_source_ref, id_map)
        end_prov = resolve_ref(s.end_source_ref, id_map)
        start_page = start_prov.get("page_ref")
        end_page = end_prov.get("page_ref")
        if start_page is None or end_page is None:
            print(f"WARNING: section {s.heading!r} has an unresolved source_ref, dropping page_range", file=sys.stderr)
        sections.append(
            {
                "section_id": f"s{i + 1}",
                "heading": s.heading,
                "page_range": [start_page, end_page] if start_page is not None and end_page is not None else None,
            }
        )

    clauses = []
    clause_ids = []
    mined_entities = []  # from the per-clause second pass, below
    for i, c in enumerate(extraction.clauses):
        clause_id = f"c{i + 1}"
        clause_ids.append(clause_id)
        prov = resolve_ref(c.source_ref, id_map)
        if not prov:
            print(f"WARNING: clause references unknown source_ref {c.source_ref!r}, dropping provenance", file=sys.stderr)
        clauses.append(
            {
                "clause_id": clause_id,
                "clause_type": c.clause_type,
                "text_summary": c.text_summary,
                "page_ref": prov.get("page_ref"),
                "bbox": prov.get("bbox"),
                "confidence": c.confidence,
            }
        )

        # Deliberately using text_summary, not the raw cited source element:
        # the first pass's source_ref citation and its own text_summary can
        # disagree (seen in testing -- summary correctly said "June 29,
        # 2025" while source_ref pointed to an unrelated element mentioning
        # "June 30, 2020"). text_summary is what the model actually read
        # and paraphrased, so it's the safer ground truth for this pass.
        mined = extract_entity_for_clause(c.clause_type, c.text_summary, model=args.model)
        if mined is not None:
            mined_entities.append(
                {
                    "entity_type": mined.entity_type,
                    "value": mined.value,
                    "unit": mined.unit,
                    "page_ref": prov.get("page_ref"),
                    "bbox": prov.get("bbox"),
                    "confidence": mined.confidence,
                    "linked_clause": clause_id,
                }
            )

    entities = []
    for i, e in enumerate(extraction.entities):
        prov = resolve_ref(e.source_ref, id_map)
        if not prov:
            print(f"WARNING: entity references unknown source_ref {e.source_ref!r}, dropping provenance", file=sys.stderr)
        linked_clause = (
            clause_ids[e.linked_clause_index]
            if e.linked_clause_index is not None and 0 <= e.linked_clause_index < len(clause_ids)
            else None
        )
        entities.append(
            {
                "entity_id": f"e{i + 1}",
                "entity_type": e.entity_type,
                "value": e.value,
                "unit": e.unit,
                "page_ref": prov.get("page_ref"),
                "bbox": prov.get("bbox"),
                "confidence": e.confidence,
                "linked_clause": linked_clause,
            }
        )

    for m in mined_entities:
        m["entity_id"] = f"e{len(entities) + 1}"
        entities.append(m)

    effective_date = extraction.effective_date
    if effective_date is None:
        # The top-level effective_date field is asked for directly in the
        # single-shot pass and came back null often enough in testing to
        # need a fallback: the per-clause mining pass reliably finds a date
        # entity for whichever clause it classified as "effective_date"
        # (the model does this itself when the opening sentence reads like
        # "...(the 'Effective Date'), is entered into...").
        for clause_dict in clauses:
            if "effective" in clause_dict["clause_type"].lower():
                match = next((m for m in entities if m.get("linked_clause") == clause_dict["clause_id"]), None)
                if match:
                    effective_date = match["value"]
                    break

    cross_references = []
    for cr in extraction.cross_references:
        from_clause = (
            clause_ids[cr.from_clause_index]
            if 0 <= cr.from_clause_index < len(clause_ids)
            else None
        )
        cross_references.append({"from_clause": from_clause, "to": cr.to, "text": cr.text})

    doc = {
        "doc_id": args.doc_id,
        "source_file": args.source_file,
        "doc_type": extraction.doc_type,
        "ingestion_date": dt.date.today().isoformat(),
        "effective_date": effective_date,
        "vendor": extraction.vendor.model_dump(),
        "sections": sections,
        "clauses": clauses,
        "entities": entities,
        "visual_elements": pictures_to_visual_elements(pictures),
        "cross_references": cross_references,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.doc_id}.structured.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Structured document written to {out_path}")


if __name__ == "__main__":
    main()
