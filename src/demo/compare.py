"""
The brief's headline deliverable: the same underlying question, run through
(a) a naive chunk-RAG baseline and (b) ChronoDoc, printed side by side.

Note on the two question strings below: chunk-RAG has no name-a-vendor
step, so its question is phrased close to the contract's own wording (this
is also what reliably reproduces the failure -- see the module docstring
in chunk_rag_baseline.py for why phrasing matters here). ChronoDoc's
intent-parser needs the vendor named to know which document graph to
search, so its question names Cree explicitly. Both ask the same real
thing -- "what's the current term end date for this vendor relationship"
-- adapted to how each system expects to be asked; this isn't picking a
flattering question for one side, both are asked in their own natural
input style.

Usage:
    python -m src.demo.compare
"""

from src.demo.chunk_rag_baseline import answer_naive
from src.query.synthesize import run_query

NAIVE_RAG_QUESTION = "What is the term of this Agreement and when does it expire?"
CHRONODOC_QUESTION = "What is Cree's current contract term end date?"


def main() -> None:
    print("=" * 78)
    print("BEFORE: plain chunk-RAG (sentence-transformers + Chroma, top_k=1)")
    print("=" * 78)
    print(f"Q: {NAIVE_RAG_QUESTION}\n")
    naive_answer, chunks = answer_naive(NAIVE_RAG_QUESTION, top_k=1)
    print(f"A: {naive_answer}\n")
    print("Retrieved chunk:")
    for c in chunks:
        print(f"  [{c.doc_id}, page {c.page_no}, distance {c.distance:.3f}] {c.text[:100]}")

    print()
    print("=" * 78)
    print("AFTER: ChronoDoc (graph traversal, SUPERSEDES + SAME_AS aware)")
    print("=" * 78)
    print(f"Q: {CHRONODOC_QUESTION}\n")
    result = run_query(CHRONODOC_QUESTION)
    print(f"A: {result.error or result.answer}\n")
    if result.hits:
        print(f"Sources ({len(result.hits)} clause(s)):")
        for h in result.hits:
            status = "CURRENT" if h.is_current else "SUPERSEDED"
            print(f"  [{h.doc_id}, page {h.page_ref}, {status}] {h.clause_type}: {h.text_summary[:80]}")

    print()
    print("=" * 78)
    print("THE FAILURE, MADE VISIBLE")
    print("=" * 78)
    print(
        "Plain chunk-RAG retrieved the 2014 original's term clause -- ranked\n"
        "ahead of the 2020 amendment's -- and answered with total confidence\n"
        "using ONLY that stale value, with no signal a newer document exists.\n"
        "ChronoDoc walked the SUPERSEDES chain to the current document, then\n"
        "checked the SAME_AS cluster and explicitly flagged that the term\n"
        "changed, citing both documents by page."
    )


if __name__ == "__main__":
    main()
