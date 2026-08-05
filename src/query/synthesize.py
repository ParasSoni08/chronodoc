"""
ChronoDoc's query layer: NL question -> graph traversal -> cited answer.

This is the headline capability from the brief: the answer must surface
supersession explicitly. If a fact (e.g. a term-expiration date) changed
between document versions, ChronoDoc must say so and cite both values --
never silently answer with a stale value the way plain chunk-RAG would.

Usage:
    python -m src.query.synthesize "What is Cree's current contract term end date?"
"""

import argparse
import json
import re
import sys

import ollama
from pydantic import BaseModel

from src.graph.schema import get_connection
from src.query.traverse import ClauseHit, find_vendor, get_relevant_clauses, get_same_as_cluster

MODEL = "llama3.1:8b"


class QueryIntent(BaseModel):
    vendor_query: str
    topic_keyword: str


INTENT_PROMPT_TEMPLATE = """Extract the vendor/company name and the single topic keyword (e.g. \
"term", "price", "pricing", "exclusivity", "quantity", "payment") this question is asking \
about. Respond with JSON only: {{"vendor_query": string, "topic_keyword": string}}.

Example:
Question: What is Acme Corp's current contract term end date?
Answer: {{"vendor_query": "Acme Corp", "topic_keyword": "term"}}

Now do this one:
Question: {question}
Answer:"""


def extract_intent(question: str, model: str = MODEL) -> QueryIntent | None:
    client = ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": INTENT_PROMPT_TEMPLATE.format(question=question)}],
        options={"temperature": 0.0},
    )
    match = re.search(r"\{.*\}", response.message.content, re.DOTALL)
    if not match:
        return None
    try:
        return QueryIntent.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValueError):
        return None


def build_context(hits: list[ClauseHit], conn) -> str:
    if not hits:
        return "(no matching clauses found in the graph)"

    blocks = []
    for h in hits:
        status = "CURRENT" if h.is_current else "SUPERSEDED"
        lines = [
            f"[{h.doc_id} | {h.doc_type} | effective {h.effective_date} | {status} | page {h.page_ref}]",
            f"Clause ({h.clause_type}): {h.text_summary}",
        ]
        for e in h.entities:
            lines.append(f"  Entity ({e['entity_type']}): {e['value']}" + (f" {e['unit']}" if e.get("unit") else ""))
            cluster = get_same_as_cluster(conn, e["entity_id"])
            other_values = [c["value"] for c in cluster if c["value"] != e["value"]]
            if other_values:
                lines.append(f"    Same fact in other document version(s): {', '.join(other_values)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


SYNTHESIS_SYSTEM_PROMPT = """You are answering a question about a vendor contract relationship using \
ONLY the context provided below -- context assembled from a graph that tracks documents across \
time, including which document is CURRENT and which are SUPERSEDED by a later version.

Rules:
- Answer using only the given context. Do not use outside knowledge.
- If the question asks about a current value (e.g. "current term", "current price"), answer using \
the value from the document marked CURRENT, not a SUPERSEDED one.
- If a fact shows "Same fact in other document version(s)" with a DIFFERENT value than the current \
one, you MUST explicitly say the term changed, state both the old and new values, and note which \
document each came from. This is the most important rule: never silently give just the current \
value when history shows it changed -- always surface that a change happened.
- Cite the source document id and page number for every fact you state, e.g. "(charles-colvard-cree-esa-amendment2-2020, p1)".
- If the context doesn't contain enough information to answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}
Answer:"""


def answer_question(question: str, model: str = MODEL) -> str:
    conn = get_connection()

    intent = extract_intent(question, model=model)
    if intent is None:
        return "Couldn't parse the question into a vendor + topic -- try rephrasing (e.g. \"What is <vendor>'s current <topic>?\")."

    vendor = find_vendor(conn, intent.vendor_query)
    if vendor is None:
        return f"No vendor matching {intent.vendor_query!r} found in the graph."

    hits = get_relevant_clauses(conn, vendor, intent.topic_keyword)
    context = build_context(hits, conn)

    client = ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": SYNTHESIS_SYSTEM_PROMPT.format(context=context, question=question)}],
        options={"temperature": 0.0},
    )
    return response.message.content


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask ChronoDoc a question about an ingested vendor relationship.")
    parser.add_argument("question", type=str)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    answer = answer_question(args.question, model=args.model)
    print(answer)


if __name__ == "__main__":
    main()
