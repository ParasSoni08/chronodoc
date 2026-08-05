"""
A deliberately naive chunk-RAG baseline, for the brief's before/after demo.

This treats every Docling text/table element from the raw parse as one
chunk (a reasonable stand-in for how a typical PDF-chunking RAG pipeline
would split these documents), embeds them with sentence-transformers, and
answers a question using only the top-k nearest chunks by cosine
similarity -- with NO awareness of which document is current, no
supersession chain, no flagging that a fact might have changed. This is
intentionally the failure mode ChronoDoc's graph layer is built to catch:
a stale value can retrieve just as well as (or better than) the current
one, and nothing here would ever tell you that happened.

Usage:
    python -m src.demo.chunk_rag_baseline "What is Cree's current contract term end date?"
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from src.extraction.structure_entities import SKIP_LABELS, load_and_merge, table_to_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_JSON_DIR = PROJECT_ROOT / "data" / "extracted_json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL = "llama3.1:8b"

# Each entry: (doc_id, [raw docling JSON parts, in page order]).
# Mirrors the same documents ChronoDoc's graph has ingested, so the
# before/after comparison is apples-to-apples.
DOCUMENT_PARTS = {
    "charles-colvard-cree-esa-2014": [
        EXTRACTED_JSON_DIR / "charles_colvard_cree_esa_2014_part1.docling_raw.json",
        EXTRACTED_JSON_DIR / "charles_colvard_cree_esa_2014_part2.docling_raw.json",
    ],
    "charles-colvard-cree-esa-amendment1-2018": [
        EXTRACTED_JSON_DIR / "charles_colvard_cree_esa_1st_amendment_2018.docling_raw.json",
    ],
    "charles-colvard-cree-esa-amendment2-2020": [
        EXTRACTED_JSON_DIR / "charles_colvard_cree_esa_2nd_amendment_2020.docling_raw.json",
    ],
}


@dataclass
class Chunk:
    doc_id: str
    page_no: int
    text: str
    distance: float = 0.0


def build_chunks(doc_id: str, raw_json_paths: list[Path]) -> list[Chunk]:
    texts, tables, _pictures = load_and_merge(raw_json_paths)
    chunks = []
    for t in texts:
        if t["label"] in SKIP_LABELS or not t.get("text", "").strip():
            continue
        page_no = t["prov"][0]["page_no"] if t.get("prov") else 0
        chunks.append(Chunk(doc_id=doc_id, page_no=page_no, text=t["text"].strip()))
    for tb in tables:
        page_no = tb["prov"][0]["page_no"] if tb.get("prov") else 0
        chunks.append(Chunk(doc_id=doc_id, page_no=page_no, text=table_to_text(tb)))
    return chunks


_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def build_index(reset: bool = True) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection("chunks")
        except Exception:
            pass
    collection = client.get_or_create_collection("chunks")

    embedder = get_embedder()
    all_chunks: list[Chunk] = []
    for doc_id, parts in DOCUMENT_PARTS.items():
        all_chunks.extend(build_chunks(doc_id, parts))

    embeddings = embedder.encode([c.text for c in all_chunks]).tolist()
    collection.add(
        ids=[f"{c.doc_id}:{i}" for i, c in enumerate(all_chunks)],
        embeddings=embeddings,
        documents=[c.text for c in all_chunks],
        metadatas=[{"doc_id": c.doc_id, "page_no": c.page_no} for c in all_chunks],
    )
    return collection


def get_index() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection("chunks")
    except Exception:
        return build_index()


def retrieve(question: str, top_k: int = 3) -> list[Chunk]:
    collection = get_index()
    embedder = get_embedder()
    query_embedding = embedder.encode([question]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        chunks.append(Chunk(doc_id=metadata["doc_id"], page_no=metadata["page_no"], text=text, distance=distance))
    return chunks


NAIVE_RAG_PROMPT = """Answer the question using only the following context. If the context \
doesn't contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""


def answer_naive(question: str, top_k: int = 3, model: str = MODEL) -> tuple[str, list[Chunk]]:
    chunks = retrieve(question, top_k=top_k)
    context = "\n\n".join(f"[{c.doc_id}, page {c.page_no}]\n{c.text}" for c in chunks)

    client = ollama.Client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": NAIVE_RAG_PROMPT.format(context=context, question=question)}],
        options={"temperature": 0.0},
    )
    return response.message.content, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive chunk-RAG baseline over the same sample documents.")
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.rebuild_index:
        build_index()

    answer, chunks = answer_naive(args.question, top_k=args.top_k)
    print(answer)
    print("\n--- retrieved chunks ---")
    for c in chunks:
        print(f"[{c.doc_id}, page {c.page_no}, distance {c.distance:.3f}] {c.text[:100]}")


if __name__ == "__main__":
    main()
