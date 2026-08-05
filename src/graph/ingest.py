"""
Load one or more ChronoDoc structured JSON documents (from
src/extraction/structure_entities.py) into the Kuzu graph.

SUPERSEDES/RELATES_TO links are manually annotated at ingestion, per the
brief -- these two sample documents don't give Docling/the LLM any
reliable signal to auto-infer "this document replaces that one" from
content alone (the 2020 amendment's text does reference the original by
name, but treating that as a general auto-inference rule is exactly the
kind of thing the brief says to defer until more documents are in hand).

Usage:
    python -m src.graph.ingest data/extracted_json/*.structured.json \\
        --supersedes charles-colvard-cree-esa-amendment2-2020:charles-colvard-cree-esa-2014
"""

import argparse
import json
import sys
from pathlib import Path

import kuzu

from src.graph.schema import DB_PATH, create_schema, get_connection

# Entities across documents, for the same vendor and entity_type, are
# considered SAME_AS candidates -- e.g. the "date" entity for a contract's
# term-expiration in doc A and doc B. This is deliberately coarse (matches
# on vendor + entity_type only, not on which clause_type the entity is
# linked to) since with only two sample documents there isn't yet a richer
# signal to key off; revisit once more documents are ingested and same-type
# entities start meaning different things within one vendor relationship.
SAME_AS_CONFIDENCE = 0.6


def load_structured_doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_document(conn: kuzu.Connection, doc: dict) -> None:
    doc_id = doc["doc_id"]

    conn.execute(
        """MERGE (v:Vendor {normalized_name: $normalized_name})
           ON CREATE SET v.name = $name, v.first_seen_date = $effective_date""",
        {
            "normalized_name": doc["vendor"]["normalized_name"],
            "name": doc["vendor"]["name"],
            "effective_date": doc.get("effective_date"),
        },
    )

    conn.execute(
        """MERGE (d:Document {doc_id: $doc_id})
           ON CREATE SET d.filename = $filename, d.doc_type = $doc_type,
                         d.ingestion_date = $ingestion_date,
                         d.effective_date = $effective_date""",
        {
            "doc_id": doc_id,
            "filename": doc["source_file"],
            "doc_type": doc["doc_type"],
            "ingestion_date": doc["ingestion_date"],
            "effective_date": doc.get("effective_date"),
        },
    )

    conn.execute(
        """MATCH (v:Vendor {normalized_name: $normalized_name}), (d:Document {doc_id: $doc_id})
           MERGE (v)-[:SUBMITTED {date: $effective_date}]->(d)
           MERGE (v)-[:PARTY_TO]->(d)""",
        {
            "normalized_name": doc["vendor"]["normalized_name"],
            "doc_id": doc_id,
            "effective_date": doc.get("effective_date"),
        },
    )

    for s in doc["sections"]:
        global_id = f"{doc_id}:{s['section_id']}"
        page_range = s.get("page_range") or [None, None]
        conn.execute(
            """MERGE (s:Section {section_id: $section_id})
               ON CREATE SET s.heading = $heading, s.page_start = $page_start, s.page_end = $page_end
               WITH s
               MATCH (d:Document {doc_id: $doc_id})
               MERGE (d)-[:CONTAINS]->(s)""",
            {
                "section_id": global_id,
                "heading": s["heading"],
                "page_start": page_range[0],
                "page_end": page_range[1],
                "doc_id": doc_id,
            },
        )

    for c in doc["clauses"]:
        global_id = f"{doc_id}:{c['clause_id']}"
        conn.execute(
            """MERGE (c:Clause {clause_id: $clause_id})
               ON CREATE SET c.clause_type = $clause_type, c.text_summary = $text_summary,
                             c.page_ref = $page_ref, c.bbox = $bbox, c.confidence = $confidence
               WITH c
               MATCH (d:Document {doc_id: $doc_id})
               MERGE (d)-[:CONTAINS]->(c)""",
            {
                "clause_id": global_id,
                "clause_type": c["clause_type"],
                "text_summary": c["text_summary"],
                "page_ref": c.get("page_ref"),
                "bbox": c.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                "confidence": c["confidence"],
                "doc_id": doc_id,
            },
        )

    for e in doc["entities"]:
        global_id = f"{doc_id}:{e['entity_id']}"
        conn.execute(
            """MERGE (e:Entity {entity_id: $entity_id})
               ON CREATE SET e.entity_type = $entity_type, e.value = $value, e.unit = $unit,
                             e.page_ref = $page_ref, e.bbox = $bbox, e.confidence = $confidence
               WITH e
               MATCH (d:Document {doc_id: $doc_id})
               MERGE (d)-[:CONTAINS]->(e)""",
            {
                "entity_id": global_id,
                "entity_type": e["entity_type"],
                "value": e["value"],
                "unit": e.get("unit"),
                "page_ref": e.get("page_ref"),
                "bbox": e.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                "confidence": e["confidence"],
                "doc_id": doc_id,
            },
        )
        if e.get("linked_clause"):
            clause_global_id = f"{doc_id}:{e['linked_clause']}"
            conn.execute(
                """MATCH (c:Clause {clause_id: $clause_id}), (e:Entity {entity_id: $entity_id})
                   MERGE (c)-[:HAS_ENTITY]->(e)""",
                {"clause_id": clause_global_id, "entity_id": global_id},
            )

    for vis in doc["visual_elements"]:
        global_id = f"{doc_id}:visual:{vis['page_ref']}:{vis['bbox']}"
        conn.execute(
            """MERGE (vi:VisualElement {visual_id: $visual_id})
               ON CREATE SET vi.type = $type, vi.page_ref = $page_ref, vi.bbox = $bbox,
                             vi.detected_only = $detected_only, vi.note = $note
               WITH vi
               MATCH (d:Document {doc_id: $doc_id})
               MERGE (d)-[:CONTAINS]->(vi)""",
            {
                "visual_id": global_id,
                "type": vis["type"],
                "page_ref": vis["page_ref"],
                "bbox": vis["bbox"],
                "detected_only": vis["detected_only"],
                "note": vis.get("note"),
                "doc_id": doc_id,
            },
        )

    # cross_references: best-effort match against this same document's own
    # sections/clauses by heading/type substring. If nothing matches, skip
    # rather than invent a target node -- we don't want REFERENCES edges
    # pointing at things that don't exist in the graph.
    for cr in doc.get("cross_references", []):
        if not cr.get("from_clause"):
            continue
        from_clause_global = f"{doc_id}:{cr['from_clause']}"
        target = cr["to"]
        conn.execute(
            """MATCH (c1:Clause {clause_id: $from_clause})
               MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(s:Section)
               WHERE s.heading CONTAINS $target OR $target CONTAINS s.heading
               MERGE (c1)-[:REFERENCES {text: $text}]->(s)""",
            {
                "from_clause": from_clause_global,
                "doc_id": doc_id,
                "target": target,
                "text": cr["text"],
            },
        )


def resolve_supersedes(conn: kuzu.Connection, doc_id: str, predecessor_id: str) -> None:
    result = conn.execute(
        """MATCH (successor:Document {doc_id: $doc_id}), (predecessor:Document {doc_id: $predecessor_id})
           RETURN successor.effective_date""",
        {"doc_id": doc_id, "predecessor_id": predecessor_id},
    )
    if not result.has_next():
        print(f"WARNING: --supersedes {doc_id}:{predecessor_id} -- one or both doc_ids not found in graph", file=sys.stderr)
        return
    effective_date = result.get_next()[0]

    conn.execute(
        """MATCH (successor:Document {doc_id: $doc_id}), (predecessor:Document {doc_id: $predecessor_id})
           MERGE (successor)-[:SUPERSEDES {date: $date}]->(predecessor)""",
        {"doc_id": doc_id, "predecessor_id": predecessor_id, "date": effective_date},
    )
    print(f"Linked {doc_id} -[:SUPERSEDES]-> {predecessor_id}")


def resolve_same_as(conn: kuzu.Connection) -> int:
    """Link entities across a vendor's documents as SAME_AS, without
    merging them. Matching on entity_type alone is too coarse -- e.g. a
    document can have several unrelated "date" entities (a signing date,
    a term-expiration date), and entity_type-only matching would link all
    of them across documents indiscriminately. Requiring the entities'
    linked clause_type to also match keeps the pairing to "the same kind
    of fact" (e.g. term-expiration date v1 <-> term-expiration date v2).
    Entities with no linked clause fall back to entity_type-only matching,
    since that's the best signal available for them.

    Computed in Python rather than one Cypher query: an OPTIONAL MATCH on
    a variable already bound by an earlier MATCH through a REL TABLE GROUP
    (CONTAINS) didn't correlate correctly on this Kuzu version -- it
    produced a full cartesian join instead of a per-row left-join. Doing
    the join here sidesteps that rather than fighting the query planner."""
    entities_result = conn.execute(
        """MATCH (v:Vendor)-[:PARTY_TO]->(d:Document)-[:CONTAINS]->(e:Entity)
           WHERE e.entity_type <> 'party_name'
           RETURN v.normalized_name, d.doc_id, e.entity_id, e.entity_type"""
    )
    entities = []
    while entities_result.has_next():
        vendor, doc_id, entity_id, entity_type = entities_result.get_next()
        entities.append({"vendor": vendor, "doc_id": doc_id, "entity_id": entity_id, "entity_type": entity_type})

    clause_type_result = conn.execute(
        "MATCH (c:Clause)-[:HAS_ENTITY]->(e:Entity) RETURN e.entity_id, c.clause_type"
    )
    clause_type_by_entity = {}
    while clause_type_result.has_next():
        entity_id, clause_type = clause_type_result.get_next()
        clause_type_by_entity[entity_id] = clause_type

    count = 0
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 :]:
            if e1["vendor"] != e2["vendor"] or e1["doc_id"] == e2["doc_id"]:
                continue
            if e1["entity_type"] != e2["entity_type"]:
                continue
            ct1 = clause_type_by_entity.get(e1["entity_id"])
            ct2 = clause_type_by_entity.get(e2["entity_id"])
            if ct1 != ct2:
                continue
            conn.execute(
                """MATCH (a:Entity {entity_id: $e1_id}), (b:Entity {entity_id: $e2_id})
                   MERGE (a)-[:SAME_AS {confidence: $confidence}]->(b)""",
                {"e1_id": e1["entity_id"], "e2_id": e2["entity_id"], "confidence": SAME_AS_CONFIDENCE},
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ChronoDoc structured JSON documents into the Kuzu graph.")
    parser.add_argument("structured_json_paths", type=Path, nargs="+")
    parser.add_argument(
        "--supersedes",
        action="append",
        default=[],
        metavar="DOC_ID:PREDECESSOR_DOC_ID",
        help="Manually annotate that DOC_ID supersedes PREDECESSOR_DOC_ID (repeatable)",
    )
    args = parser.parse_args()

    conn = get_connection()
    create_schema(conn)

    for path in args.structured_json_paths:
        doc = load_structured_doc(path)
        ingest_document(conn, doc)
        print(f"Ingested {doc['doc_id']} from {path.name}")

    for pair in args.supersedes:
        doc_id, predecessor_id = pair.split(":", 1)
        resolve_supersedes(conn, doc_id, predecessor_id)

    same_as_count = resolve_same_as(conn)
    print(f"Created {same_as_count} SAME_AS link(s)")

    print(f"Graph is ready at {DB_PATH}")


if __name__ == "__main__":
    main()
