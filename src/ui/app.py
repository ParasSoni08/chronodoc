"""
ChronoDoc's UI: ask a question about an ingested vendor relationship and
get a cited answer, with the underlying source clauses shown alongside it
so the graph's provenance is visible, not just the LLM's prose.

Run with:
    streamlit run src/ui/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.demo.chunk_rag_baseline import answer_naive
from src.demo.compare import CHRONODOC_QUESTION, NAIVE_RAG_QUESTION
from src.graph.schema import get_connection
from src.query.synthesize import MODEL, run_query
from src.query.traverse import get_document_chain

st.set_page_config(page_title="ChronoDoc", page_icon=":material/history_edu:", layout="wide")


@st.cache_resource
def get_graph_connection():
    return get_connection()


@st.cache_data(ttl=60)
def list_vendors() -> list[dict]:
    conn = get_graph_connection()
    result = conn.execute("MATCH (v:Vendor) RETURN v.name, v.normalized_name")
    vendors = []
    while result.has_next():
        name, normalized_name = result.get_next()
        vendors.append({"name": name, "normalized_name": normalized_name})
    return vendors


@st.cache_data(ttl=60)
def vendor_document_chain(normalized_name: str) -> list[dict]:
    conn = get_graph_connection()
    return get_document_chain(conn, normalized_name)


SUGGESTIONS = {
    ":material/history: Current contract term": "What is Cree's current contract term end date?",
    ":material/payments: Pricing structure": "What is Cree's pricing structure for the supply agreement?",
    ":material/handshake: Exclusivity commitment": "What is Cree's exclusivity commitment?",
}

with st.sidebar:
    st.subheader("Vendors in the graph")
    vendors = list_vendors()
    if not vendors:
        st.caption("No vendors ingested yet -- run `python -m src.graph.ingest` first.")
    for v in vendors:
        with st.container(border=True):
            st.markdown(f"**{v['name']}**")
            for doc in vendor_document_chain(v["normalized_name"]):
                status = ":green[current]" if doc["is_current"] else ":gray[superseded]"
                st.caption(f"{doc['doc_id']} — {status}  \neffective {doc['effective_date']}")
    st.divider()
    st.caption(f"Model: `{MODEL}`")

st.title("ChronoDoc")

chat_tab, demo_tab = st.tabs([":material/chat: Ask ChronoDoc", ":material/compare_arrows: Before / after"])

with chat_tab:
    st.caption("Ask about a vendor relationship. Answers are grounded in the graph and cite their source document + page.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("hits"):
                with st.expander(f"Sources ({len(msg['hits'])} clause(s))"):
                    for h in msg["hits"]:
                        status = ":green[CURRENT]" if h.is_current else ":gray[SUPERSEDED]"
                        st.markdown(f"**{h.doc_id}** — {status} — page {h.page_ref}")
                        st.caption(f"_{h.clause_type}_: {h.text_summary}")
                        for e in h.entities:
                            unit = f" {e['unit']}" if e.get("unit") else ""
                            st.caption(f"↳ {e['entity_type']}: {e['value']}{unit}")

    if not st.session_state.messages:
        selected = st.pills("Try asking:", list(SUGGESTIONS.keys()), label_visibility="collapsed")
    else:
        selected = None

    prompt = st.chat_input("Ask a question, e.g. \"What is Cree's current contract term end date?\"")
    if selected and not prompt:
        prompt = SUGGESTIONS[selected]

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Walking the graph..."):
                result = run_query(prompt, conn=get_graph_connection())
            if result.error:
                st.write(result.error)
                st.session_state.messages.append({"role": "assistant", "content": result.error, "hits": []})
            else:
                st.write(result.answer)
                if result.hits:
                    with st.expander(f"Sources ({len(result.hits)} clause(s))"):
                        for h in result.hits:
                            status = ":green[CURRENT]" if h.is_current else ":gray[SUPERSEDED]"
                            st.markdown(f"**{h.doc_id}** — {status} — page {h.page_ref}")
                            st.caption(f"_{h.clause_type}_: {h.text_summary}")
                            for e in h.entities:
                                unit = f" {e['unit']}" if e.get("unit") else ""
                                st.caption(f"↳ {e['entity_type']}: {e['value']}{unit}")
                st.session_state.messages.append({"role": "assistant", "content": result.answer, "hits": result.hits})
        st.rerun()

with demo_tab:
    st.caption(
        "The same underlying question about Cree's contract term, run through a naive "
        "chunk-RAG baseline and through ChronoDoc -- showing the exact failure ChronoDoc "
        "catches: a stale value returned with total confidence."
    )

    @st.cache_data(ttl=3600)
    def run_comparison():
        naive_answer, naive_chunks = answer_naive(NAIVE_RAG_QUESTION, top_k=1)
        chronodoc_result = run_query(CHRONODOC_QUESTION, conn=get_graph_connection())
        return naive_answer, naive_chunks, chronodoc_result

    if st.button("Run comparison", icon=":material/play_arrow:"):
        with st.spinner("Running both systems..."):
            naive_answer, naive_chunks, chronodoc_result = run_comparison()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(":material/warning: Before: plain chunk-RAG")
            st.caption(f"Q: {NAIVE_RAG_QUESTION}")
            with st.container(border=True):
                st.write(naive_answer)
                for c in naive_chunks:
                    st.caption(f"Retrieved: [{c.doc_id}, page {c.page_no}, distance {c.distance:.3f}]")

        with col2:
            st.subheader(":material/check_circle: After: ChronoDoc")
            st.caption(f"Q: {CHRONODOC_QUESTION}")
            with st.container(border=True):
                st.write(chronodoc_result.error or chronodoc_result.answer)
                for h in chronodoc_result.hits:
                    status = "CURRENT" if h.is_current else "SUPERSEDED"
                    st.caption(f"[{h.doc_id}, page {h.page_ref}, {status}] {h.clause_type}")

        st.info(
            "Plain chunk-RAG retrieved the 2014 original's term clause -- ranked ahead of the "
            "2020 amendment's -- and answered with total confidence using only that stale value, "
            "with no signal a newer document exists. ChronoDoc walked the SUPERSEDES chain to the "
            "current document, then checked the SAME_AS cluster and explicitly flagged that the "
            "term changed, citing both documents by page.",
            icon=":material/lightbulb:",
        )
