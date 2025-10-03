# app.py
# Streamlit UI for PubMed RAG (Pinecone + SPLADE + RRF + MedCPT + Gemini)

import streamlit as st
from datetime import date

# import your pipeline function
from rag_query import answer_with_query_expansion

# Updated page title
st.set_page_config(page_title="PubMed RAG — Vitamin D Specialized", layout="wide")

st.title("PubMed RAG — Vitamin D Specialized")
st.caption(
    "Corpus limited to Vitamin D papers • Dense + Sparse retrieval → RRF fusion → "
    "MedCPT rerank → Gemini answer"
)

with st.form("params"):
    st.subheader("Query")
    user_query = st.text_area(
        "Research question",
        value="What is the effect of vitamin D on cancer?",
        help="Enter any natural-language question. (Note: index is Vitamin D papers only.)",
        height=100,
    )

    st.subheader("Time window")
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input(
            "Start year (inclusive)", min_value=1800, max_value=2100, value=2020, step=1
        )
    with c2:
        end_year = st.number_input(
            "End year (inclusive)", min_value=1800, max_value=2100, value=date.today().year, step=1
        )

    st.subheader("Retrieval parameters")
    c3, c4, c5 = st.columns(3)
    with c3:
        n_subq = st.number_input(
            "Number of sub-questions (n_subq)", min_value=1, max_value=10, value=3, step=1,
            help="How many retrieval sub-questions to generate for broader coverage."
        )
    with c4:
        k_dense = st.number_input(
            "Top-k dense per sub-question (k_dense)", min_value=1, max_value=50, value=3, step=1,
            help="How many dense vector hits to take from Pinecone per sub-question."
        )
    with c5:
        k_sparse = st.number_input(
            "Top-k sparse per sub-question (k_sparse)", min_value=1, max_value=50, value=6, step=1,
            help="How many SPLADE (sparse) hits to take from Pinecone per sub-question."
        )

    st.subheader("Fusion & rerank")
    c6, c7 = st.columns(2)
    with c6:
        top_n_rrf = st.number_input(
            "Pool size after RRF (top_n_rrf)", min_value=1, max_value=200, value=30, step=1,
            help="How many docs to keep after pooling across sub-questions with RRF."
        )
    with c7:
        top_n_rerank = st.number_input(
            "Final rerank size (top_n_rerank)", min_value=1, max_value=100, value=10, step=1,
            help="How many candidates to pass through MedCPT reranker."
        )

    st.subheader("Model choices")
    c8, c9 = st.columns(2)
    with c8:
        llm_for_subq = st.text_input(
            "LLM for sub-question generation (llm_for_subq)",
            value="gemini-2.5-flash-lite",
            help="Vertex AI model to generate retrieval sub-questions."
        )
    with c9:
        llm_for_answer = st.text_input(
            "LLM for final answer (llm_for_answer)",
            value="gemini-2.5-flash",
            help="Vertex AI model to compose the final Markdown answer."
        )

    st.subheader("Weights & decoding")
    c10, c11, c12 = st.columns(3)
    with c10:
        w_dense = st.number_input(
            "Weight for dense list (w_dense)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="RRF weight for dense results."
        )
    with c11:
        w_sparse = st.number_input(
            "Weight for sparse list (w_sparse)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="RRF weight for sparse results."
        )
    with c12:
        temperature = st.slider(
            "Answer temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Higher = more varied wording; lower = more deterministic."
        )

    run = st.form_submit_button("Run search & generate answer")

if run:
    if not user_query.strip():
        st.error("Please enter a research question.")
    elif int(start_year) > int(end_year):
        st.error("Start year must be ≤ end year.")
    else:
        with st.spinner("Retrieving, pooling, reranking, and generating…"):
            try:
                answer = answer_with_query_expansion(
                    user_query=user_query,
                    start_year=int(start_year),
                    end_year=int(end_year),
                    n_subq=int(n_subq),
                    k_dense=int(k_dense),
                    k_sparse=int(k_sparse),
                    top_n_rerank=int(top_n_rerank),
                    top_n_rrf=int(top_n_rrf),
                    llm_for_subq=llm_for_subq.strip(),
                    llm_for_answer=llm_for_answer.strip(),
                    temperature=float(temperature),
                    w_dense=float(w_dense),
                    w_sparse=float(w_sparse),
                )
                st.success("Done!")
                st.markdown("### Answer")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Note: Index contains only Vitamin D papers you ingested.")
