"""
ChronoDoc graph schema, per chronodoc_brief.md, on Kuzu (an embedded graph
database -- no server process, no JVM, no Docker).

Swapped in for Neo4j-in-Docker: this machine repeatedly ran out of memory
running Docker Desktop's WSL2 backend alongside everything else this
session. Kuzu speaks openCypher and models the same node/edge shapes the
brief specifies, just as an in-process library instead of a server. If a
real Neo4j Community instance becomes available later, this schema
translates directly (docker-compose.yml is already in the repo for that).

Usage:
    python -m src.graph.schema  # creates/resets the DB at data/graph_db/
"""

from pathlib import Path

import kuzu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "graph_db"


def get_connection(db_path: Path = DB_PATH) -> kuzu.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(db_path))
    return kuzu.Connection(db)


NODE_TABLES = [
    """CREATE NODE TABLE IF NOT EXISTS Vendor(
        normalized_name STRING PRIMARY KEY,
        name STRING,
        first_seen_date STRING
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Document(
        doc_id STRING PRIMARY KEY,
        filename STRING,
        doc_type STRING,
        ingestion_date STRING,
        effective_date STRING,
        page_count INT64,
        source_hash STRING
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Section(
        section_id STRING PRIMARY KEY,
        heading STRING,
        page_start INT64,
        page_end INT64
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Clause(
        clause_id STRING PRIMARY KEY,
        clause_type STRING,
        text_summary STRING,
        page_ref INT64,
        bbox DOUBLE[4],
        confidence DOUBLE
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Entity(
        entity_id STRING PRIMARY KEY,
        entity_type STRING,
        value STRING,
        unit STRING,
        page_ref INT64,
        bbox DOUBLE[4],
        confidence DOUBLE
    )""",
    """CREATE NODE TABLE IF NOT EXISTS VisualElement(
        visual_id STRING PRIMARY KEY,
        type STRING,
        page_ref INT64,
        bbox DOUBLE[4],
        detected_only BOOLEAN,
        note STRING
    )""",
]

REL_TABLES = [
    "CREATE REL TABLE IF NOT EXISTS SUBMITTED(FROM Vendor TO Document, date STRING)",
    "CREATE REL TABLE IF NOT EXISTS PARTY_TO(FROM Vendor TO Document)",
    "CREATE REL TABLE IF NOT EXISTS SUPERSEDES(FROM Document TO Document, date STRING)",
    "CREATE REL TABLE IF NOT EXISTS RELATES_TO(FROM Document TO Document)",
    """CREATE REL TABLE GROUP IF NOT EXISTS CONTAINS(
        FROM Document TO Section,
        FROM Document TO Clause,
        FROM Document TO Entity,
        FROM Document TO VisualElement
    )""",
    "CREATE REL TABLE IF NOT EXISTS HAS_ENTITY(FROM Clause TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS SAME_AS(FROM Entity TO Entity, confidence DOUBLE)",
    """CREATE REL TABLE GROUP IF NOT EXISTS REFERENCES(
        FROM Clause TO Clause,
        FROM Clause TO Section,
        text STRING
    )""",
]


def create_schema(conn: kuzu.Connection) -> None:
    for stmt in NODE_TABLES + REL_TABLES:
        conn.execute(stmt)


def main() -> None:
    conn = get_connection()
    create_schema(conn)
    print(f"Schema created at {DB_PATH}")


if __name__ == "__main__":
    main()
