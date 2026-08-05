"""
Graph traversal for ChronoDoc's query layer: given a vendor and a topic
keyword, walk the SUPERSEDES chain to find that vendor's documents in
temporal order, then pull the clauses/entities relevant to the topic from
each one -- including SAME_AS clusters that reveal a value changed between
versions.

This is deliberately keyword-based, not NL->Cypher translation: with two
sample documents there isn't enough variety yet to justify a learned/LLM
query planner. synthesize.py extracts the vendor name and topic keyword
from the user's question with a small LLM call, then hands them to the
functions here.
"""

import re
from dataclasses import dataclass, field

import kuzu

from src.graph.schema import get_connection


@dataclass
class ClauseHit:
    doc_id: str
    doc_type: str
    effective_date: str | None
    is_current: bool  # True if no other ingested document SUPERSEDES this one
    clause_id: str
    clause_type: str
    text_summary: str
    page_ref: int | None
    bbox: list[float] | None
    entities: list[dict] = field(default_factory=list)


def find_vendor(conn: kuzu.Connection, name_query: str) -> str | None:
    """Case-insensitive substring match against Vendor.name/normalized_name.
    Returns the vendor's normalized_name, or None if nothing matches."""
    result = conn.execute(
        """MATCH (v:Vendor)
           WHERE lower(v.name) CONTAINS lower($q) OR lower(v.normalized_name) CONTAINS lower($q)
           RETURN v.normalized_name""",
        {"q": name_query},
    )
    if result.has_next():
        return result.get_next()[0]
    return None


def get_document_chain(conn: kuzu.Connection, vendor_normalized_name: str) -> list[dict]:
    """All documents for this vendor, each flagged with whether a later
    document (also for this vendor) SUPERSEDES it -- the one with no
    successor is 'current'."""
    result = conn.execute(
        """MATCH (v:Vendor {normalized_name: $vendor})-[:PARTY_TO]->(d:Document)
           RETURN d.doc_id, d.doc_type, d.effective_date, d.ingestion_date""",
        {"vendor": vendor_normalized_name},
    )
    docs = []
    while result.has_next():
        doc_id, doc_type, effective_date, ingestion_date = result.get_next()
        docs.append({"doc_id": doc_id, "doc_type": doc_type, "effective_date": effective_date, "ingestion_date": ingestion_date})

    superseded_ids = set()
    for d in docs:
        r = conn.execute(
            "MATCH (a:Document {doc_id: $doc_id})-[:SUPERSEDES]->(b:Document) RETURN b.doc_id",
            {"doc_id": d["doc_id"]},
        )
        while r.has_next():
            superseded_ids.add(r.get_next()[0])

    for d in docs:
        d["is_current"] = d["doc_id"] not in superseded_ids

    docs.sort(key=lambda d: (d["effective_date"] or "", d["ingestion_date"] or ""))
    return docs


def _shares_prefix(a: str, b: str, min_len: int = 4) -> bool:
    n = min(len(a), len(b), min_len)
    return n >= min_len and a[:n] == b[:n]


def _topic_matches(keyword: str, clause_type: str, text_summary: str) -> bool:
    """Fuzzy match: real clause_type labels are free-text from an LLM
    ("price_payment_terms", "pricing", "price") and a plain substring check
    misses obvious matches ("pricing" is not a substring of
    "price_payment_terms" even though they're clearly the same topic).
    A shared word-prefix of >=4 chars catches these without needing a
    stemming library for a two-document demo."""
    keyword = keyword.lower()
    haystack_words = re.split(r"[^a-z]+", f"{clause_type} {text_summary}".lower())
    return any(_shares_prefix(keyword, w) or keyword in w or w in keyword for w in haystack_words if w)


def get_relevant_clauses(conn: kuzu.Connection, vendor_normalized_name: str, keyword: str) -> list[ClauseHit]:
    """Clauses across all of this vendor's documents whose clause_type or
    text_summary matches the keyword (see _topic_matches), each with its
    linked entities and its document's position in the SUPERSEDES chain."""
    docs = get_document_chain(conn, vendor_normalized_name)
    doc_by_id = {d["doc_id"]: d for d in docs}

    result = conn.execute(
        """MATCH (v:Vendor {normalized_name: $vendor})-[:PARTY_TO]->(d:Document)-[:CONTAINS]->(c:Clause)
           RETURN d.doc_id, c.clause_id, c.clause_type, c.text_summary, c.page_ref, c.bbox""",
        {"vendor": vendor_normalized_name},
    )

    hits = []
    while result.has_next():
        doc_id, clause_id, clause_type, text_summary, page_ref, bbox = result.get_next()
        if not _topic_matches(keyword, clause_type, text_summary):
            continue
        doc = doc_by_id.get(doc_id, {})
        hit = ClauseHit(
            doc_id=doc_id,
            doc_type=doc.get("doc_type", "unknown"),
            effective_date=doc.get("effective_date"),
            is_current=doc.get("is_current", True),
            clause_id=clause_id,
            clause_type=clause_type,
            text_summary=text_summary,
            page_ref=page_ref,
            bbox=list(bbox) if bbox is not None else None,
        )
        hit.entities = get_entities_for_clause(conn, clause_id)
        hits.append(hit)

    hits.sort(key=lambda h: (h.effective_date or "", h.doc_id))
    return hits


def get_entities_for_clause(conn: kuzu.Connection, clause_id: str) -> list[dict]:
    result = conn.execute(
        """MATCH (c:Clause {clause_id: $clause_id})-[:HAS_ENTITY]->(e:Entity)
           RETURN e.entity_id, e.entity_type, e.value, e.unit, e.page_ref""",
        {"clause_id": clause_id},
    )
    entities = []
    while result.has_next():
        entity_id, entity_type, value, unit, page_ref = result.get_next()
        entities.append({"entity_id": entity_id, "entity_type": entity_type, "value": value, "unit": unit, "page_ref": page_ref})
    return entities


def get_same_as_cluster(conn: kuzu.Connection, entity_id: str) -> list[dict]:
    """All entities SAME_AS-linked to this one (in either direction),
    including itself -- i.e. this fact's value across every document
    version it appears in."""
    result = conn.execute(
        """MATCH (e:Entity {entity_id: $entity_id})
           OPTIONAL MATCH (e)-[:SAME_AS]-(other:Entity)
           RETURN other.entity_id, other.value, other.entity_type""",
        {"entity_id": entity_id},
    )
    cluster = []
    while result.has_next():
        other_id, other_value, other_type = result.get_next()
        if other_id is not None:
            cluster.append({"entity_id": other_id, "value": other_value, "entity_type": other_type})
    return cluster
