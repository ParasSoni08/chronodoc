# ChronoDoc

Vision-Language Contract Intelligence with persistent graph memory. See
`chronodoc_brief.md` for the full project brief.

## Status

Step 1 (project skeleton + Docling raw-output inspection) in progress.
Graph schema, ingestion, and query layers are not yet built — they depend
on inspecting Docling's raw output against real sample documents first.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Neo4j (once we reach the graph layer):

```
docker compose up -d
```

## Usage so far

Dump raw Docling output for one PDF to `data/extracted_json/`:

```
python src/extraction/docling_parse.py path/to/document.pdf
```
