# ChronoDoc — Project Brief

## What this is
A portfolio project: a Vision-Language Contract Intelligence system with persistent
graph memory. It demonstrates enterprise-grade multimodal document AI for
legal/compliance/vendor-management use cases — specifically, answering
cross-document, cross-time questions that standard chunk-based RAG cannot
(e.g. "How does this vendor's new contract compare to the spec sheet they
submitted six months ago?").

**Constraint: the entire stack must be open source / local. No paid APIs
(no Claude/GPT-4V vision calls, no hosted vector DB, no hosted graph DB).**

## The problem, precisely
Standard RAG treats every PDF chunk in isolation. On this document class it
fails in four specific ways:
1. **Visual blindness** — tables, stamps, signature blocks, diagrams get
   flattened into unstructured text or dropped entirely if OCR is skipped;
   table row/column relationships are lost.
2. **Document isolation** — no structural memory that "March contract" and
   "September renewal" are the same vendor relationship at two points in
   time. Vector similarity has no concept of supersession.
3. **No provenance/version chain** — can't answer "is this still in effect"
   or "what changed" because there's no `SUPERSEDES` relationship, only
   cosine similarity.
4. **Silent staleness** — if an amendment isn't retrieved, RAG confidently
   answers with the outdated term and gives no signal anything changed.
   This is the dangerous failure mode: not "no answer," but "wrong answer
   stated with full confidence."

The demo must make this failure mode **visible**: run the same question
through (a) plain chunk-RAG and (b) ChronoDoc side by side, on a question
engineered to trigger the isolation failure (e.g. a price/SLA term that
changed between two versions of the same vendor relationship).

## Architecture — 3 layers
1. **Vision-Language Extraction Layer**: PDF → structured JSON (entities,
   clauses, tables, visual elements, page/bbox provenance). Docling does the
   visual/layout parsing; a local LLM (via Ollama) structures Docling's
   output into the typed JSON schema below.
2. **Persistent Graph Memory Layer**: entities become nodes in a graph
   database; relationships link documents across time (same vendor,
   supersession, cross-references). New documents link to *existing*
   entities/vendors rather than creating disconnected islands.
3. **Reasoning + Retrieval Layer**: NL query → graph traversal to find
   relevant entity clusters across time → pull grounding text for those
   entities → local LLM synthesizes a cited answer. Explicit temporal
   queries supported ("compare March vs September version").

## Tech stack (all open source / local)
| Layer | Tool |
|---|---|
| PDF layout/table parsing | Docling (IBM) |
| OCR fallback (scanned pages) | Tesseract (via Docling) |
| Entity/clause structuring | Ollama + local model (e.g. Llama 3.1 8B or Qwen2.5) |
| Chunk embeddings (fallback grounding layer) | sentence-transformers |
| Vector store | Chroma |
| Graph DB | Neo4j Community, local via Docker |
| Query/synthesis LLM | Ollama, local model |
| Orchestration | hand-rolled Python (no LangChain/LlamaIndex — the point of the project is showing the graph-traversal-then-retrieve pattern built explicitly, not abstracted away) |
| UI | Streamlit (built last, after the pipeline works end-to-end via script/notebook) |

## Graph schema

**Nodes**
- `:Vendor` — name, normalized_name, first_seen_date
- `:Document` — doc_id, filename, doc_type, ingestion_date, effective_date, page_count, source_hash
- `:Clause` — clause_id, clause_type, text_summary, page_ref, bbox, confidence
- `:Entity` — entity_id, entity_type (price/date/SLA_term/quantity/party_name), value, unit, page_ref, bbox, confidence
- `:Signature` / `:Stamp` — page_ref, bbox, associated_party, detected_only (bool)
- `:Section` — section_id, heading, page_range

**Edges**
- `(:Vendor)-[:SUBMITTED {date}]->(:Document)`
- `(:Document)-[:SUPERSEDES {date}]->(:Document)` — backbone of temporal queries
- `(:Document)-[:CONTAINS]->(:Clause)` / `(:Document)-[:CONTAINS]->(:Entity)`
- `(:Clause)-[:REFERENCES]->(:Clause | :Section)`
- `(:Entity)-[:SAME_AS {confidence}]->(:Entity)` — links the *same kind* of
  value across document versions (e.g. price in March vs price in
  September) WITHOUT merging them into one node, so a changed value is
  visible as two linked entities, not one — this is what powers the
  before/after demo
- `(:Document)-[:RELATES_TO]->(:Document)` — looser link than SUPERSEDES
  (e.g. spec sheet → contract, same vendor family, not strict versioning)
- `(:Vendor)-[:PARTY_TO]->(:Document)`

`SUPERSEDES`/`RELATES_TO` links may need to be **manually annotated at
ingestion** (a config mapping doc → predecessor) rather than purely
auto-inferred, if the sample documents don't form a naturally detectable
version chain.

## Per-document extraction JSON schema
```json
{
  "doc_id": "vendor-acme-contract-2026-03",
  "source_file": "acme_contract_march2026.pdf",
  "doc_type": "contract",
  "ingestion_date": "2026-08-05",
  "effective_date": "2026-03-12",
  "vendor": {"name": "Acme Fabrication Ltd", "normalized_name": "acme_fabrication"},
  "sections": [
    {"section_id": "s1", "heading": "Pricing", "page_range": [3, 4]}
  ],
  "clauses": [
    {
      "clause_id": "c1",
      "clause_type": "pricing",
      "text_summary": "Unit price for Part A set at $50,000, effective March 2026",
      "page_ref": 3,
      "bbox": [72, 140, 480, 210],
      "confidence": 0.91
    }
  ],
  "entities": [
    {
      "entity_id": "e1",
      "entity_type": "price",
      "value": 50000,
      "unit": "USD",
      "page_ref": 3,
      "bbox": [200, 150, 300, 165],
      "confidence": 0.95,
      "linked_clause": "c1"
    }
  ],
  "visual_elements": [
    {"type": "signature", "page_ref": 8, "bbox": [90, 700, 250, 740], "detected_only": true},
    {"type": "stamp", "page_ref": 1, "bbox": [400, 50, 500, 100], "detected_only": true,
     "note": "layout-detected, not semantically parsed"}
  ],
  "cross_references": [
    {"from_clause": "c1", "to": "external", "text": "as specified in Exhibit B"}
  ]
}
```

## Build order
1. **Single-document extraction pipeline first.** PDF → Docling raw output →
   local-LLM-structured JSON matching the schema above. Validate on 2-3
   sample documents before touching the graph at all.
2. Design/stand up the graph schema; write ingestion code that pushes
   entities in with proper linking logic (including `SAME_AS` resolution).
3. Build the query layer: NL question → graph traversal → context assembly
   → local LLM answer with citations back to source page/bbox.
4. Add temporal/cross-document comparison as the headline capability
   (before/after demo vs. plain chunk-RAG).
5. Build the Streamlit UI last, once the pipeline works end-to-end.

## Deliverables
- Ingestion pipeline (PDF → structured entities → graph nodes)
- Documented graph schema with an ER-style diagram
- Query interface (Streamlit) with citations to source pages
- 3-5 real/public sample documents (no fabricated synthetic ones)
- A before/after demo: standard chunk-RAG vs ChronoDoc on the same
  question, showing the specific failure ChronoDoc catches
