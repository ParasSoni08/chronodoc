# ChronoDoc

Vision-Language Contract Intelligence with persistent graph memory — a fully
local, open-source pipeline that answers cross-document, cross-time
questions about vendor contracts that standard chunk-based RAG gets
confidently wrong. See [`chronodoc_brief.md`](chronodoc_brief.md) for the
full project brief.

## The problem, in plain English

Imagine a messy filing cabinet full of years of vendor contracts and
amendments. Ask a standard RAG chatbot "What's our current pricing with
this vendor?" and it acts like a lazy intern: it reaches into the cabinet,
grabs the first page with matching keywords, and reads it back to you.

That fails in three specific ways:
- **It's blind to structure.** A pricing table gets flattened into word
  soup, and row/column relationships (which price goes with which tier)
  get lost.
- **It ignores time.** It might grab the 2022 contract and confidently
  quote a price, missing entirely that a 2024 amendment changed it.
- **It doesn't know "why."** It has no concept that Document B *replaced*
  Document A — every PDF is just an isolated bag of text chunks.

The dangerous part isn't "no answer" — it's a wrong answer delivered with
total confidence.

## The solution

ChronoDoc acts like an archivist with a detective's string board instead of
a filing cabinet:

1. **It sees the layout.** [Docling](https://github.com/docling-project/docling)
   parses real page layout — tables stay tables, with page/bbox provenance
   for every fact — instead of flattening everything to plain text.
2. **It builds a map.** Every document becomes a node in a graph; a
   `SUPERSEDES` edge explicitly says "this amendment replaces that
   contract." A `SAME_AS` edge links the *same kind of fact* (e.g. a term
   expiration date) across versions **without merging them** — so a
   changed value stays visible as two linked facts, not one silently
   overwritten value.
3. **It traces the timeline.** Ask a question and ChronoDoc walks the
   `SUPERSEDES` chain to the current document for the answer, then checks
   whether that fact's `SAME_AS` cluster shows it changed — and if so,
   says so explicitly, with citations to every document involved.

All of this runs **fully offline** — no OpenAI/Claude/Gemini API calls, no
hosted vector DB, no hosted graph DB. Every model and datastore is local.

| | Standard chunk-RAG | ChronoDoc |
|---|---|---|
| Reading tables | Flattened into disconnected text fragments | Rows/columns/headers preserved with bbox provenance |
| Document history | Every PDF is an isolated island | Explicit `SUPERSEDES` chain across versions |
| Answering questions | Confidently returns whichever chunk matched, even if outdated | Resolves to the current document, and explicitly flags when a fact changed |
| Privacy & cost | Often sends data to a cloud API, per-query cost | 100% local, zero marginal cost per query |

## Status

Four of the five architecture layers are built and validated end-to-end on
a real document pair (see below), including a working Streamlit UI. Not
yet done: the explicit side-by-side chunk-RAG-vs-ChronoDoc demo.

| Layer | Status |
|---|---|
| PDF → Docling raw extraction | ✅ `src/extraction/docling_parse.py` |
| Docling output → structured entities (local LLM) | ✅ `src/extraction/structure_entities.py` |
| Graph schema + ingestion (`SAME_AS`, `SUPERSEDES`) | ✅ `src/graph/schema.py`, `src/graph/ingest.py` |
| Query layer (NL → graph traversal → cited answer) | ✅ `src/query/traverse.py`, `src/query/synthesize.py` |
| Streamlit UI | ✅ `src/ui/app.py` |
| Before/after chunk-RAG vs. ChronoDoc demo | ⬜ not started |

**Note on the graph DB:** the brief specifies Neo4j Community via Docker.
This machine repeatedly ran out of memory running Docker Desktop's WSL2
backend, so the graph layer runs on
[Kuzu](https://github.com/kuzudb/kuzu) instead — an embedded graph
database with the same Cypher query model, no server/JVM/Docker required.
`docker-compose.yml` is still in the repo if a real Neo4j instance becomes
available later; the schema translates directly. One consequence: Kuzu
only allows one open connection to the database file per process, so the
CLI (`src/query/synthesize.py`) and the UI (`src/ui/app.py`) can't run
against the same `data/graph_db` at the same time — stop one before
starting the other.

**Note on `requirements.txt`:** `starlette` is pinned below Streamlit's
tested range. The latest `starlette` (1.x) changed its GZip middleware's
constructor signature in a way that breaks Streamlit's bundled copy —
the app still serves its HTML shell but every subsequent request 500s,
so content silently never renders. Caught by actually driving the UI in
a browser rather than just checking the process started.

## Sample data

Two real documents from SEC EDGAR (no fabricated samples), forming a
genuine version chain — the same vendor relationship at two points in time:

- **Original**: Exclusive Supply Agreement between Charles & Colvard, Ltd.
  and Cree, Inc., dated December 12, 2014
- **Amendment**: Second Amendment to that agreement, dated June 30, 2020

Both are real SEC filings; note that their pricing tables are redacted
under confidential-treatment provisions (`[***]`/`[****]`) — structure
extracts correctly, but the demo's numeric before/after uses the
contract's **term-expiration date** instead, which is not redacted:

| | 2014 Original | 2020 Amendment |
|---|---|---|
| Vendor identified | Cree, Inc. | Cree, Inc. |
| Term expiration | June 24, 2018 | June 29, 2025 |

Asking ChronoDoc *"What is Cree's current contract term end date?"*
correctly answers **June 29, 2025** while explicitly flagging that it
changed from **June 24, 2018** in the 2014 original — citing both
documents by page. That's the exact staleness failure plain RAG can't
catch.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with `llama3.1:8b`
pulled (`ollama pull llama3.1:8b`) — a smaller model (`llama3.2:3b`) was
tried first but proved unreliable at copying exact dates/prices; see the
brief and commit history for details.

## Usage

**1. Parse a PDF with Docling** (dumps raw layout/text/table output for
inspection — not yet mapped to the ChronoDoc schema):

```
python -m src.extraction.docling_parse data/raw_pdfs/your_document.pdf
```

If Docling runs out of memory on a long document (this project hit that
around page 13 on a memory-constrained machine), split the PDF into
smaller chunks and pass each through separately — see
`src/extraction/structure_entities.py`, which accepts multiple raw JSON
parts for one logical document.

**2. Structure the raw output into entities/clauses via a local LLM:**

```
python -m src.extraction.structure_entities \
    data/extracted_json/your_document.docling_raw.json \
    --doc-id vendor-name-contract-2024 \
    --source-file your_document.pdf
```

**3. Ingest into the graph**, annotating any known supersession
relationships:

```
python -m src.graph.ingest \
    data/extracted_json/doc-a.structured.json \
    data/extracted_json/doc-b.structured.json \
    --supersedes doc-b:doc-a
```

**4. Ask a question**, via the CLI:

```
python -m src.query.synthesize "What is Cree's current contract term end date?"
```

or the UI:

```
streamlit run src/ui/app.py
```

## Repo structure

```
chronodoc/
  chronodoc_brief.md       # full project brief/spec
  data/
    raw_pdfs/               # source PDFs
    extracted_json/         # Docling raw output + structured entity JSON
    graph_db/               # Kuzu database (gitignored, rebuild via src/graph/ingest.py)
  docs/
    chronodoc_overview.pptx # project overview slides
  src/
    extraction/
      docling_parse.py      # PDF -> raw Docling JSON
      structure_entities.py # raw JSON -> schema JSON via Ollama
    graph/
      schema.py             # Kuzu node/edge table definitions
      ingest.py             # structured JSON -> graph, SAME_AS/SUPERSEDES resolution
    query/
      traverse.py           # graph traversal: vendor/topic -> clauses+entities across versions
      synthesize.py         # NL question -> cited answer
    ui/
      app.py                # Streamlit chat UI over the query layer
  docker-compose.yml        # Neo4j Community, if swapping back from Kuzu
  requirements.txt
```
