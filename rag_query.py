# =============================================================================
# # RAG Pipeline over PubMed with Pinecone + SPLADE + RRF + MedCPT + Gemini
# =============================================================================
# This script:
# 1) Initializes Vertex AI auth/session
# 2) Sets up Pinecone (dense + sparse) and encoders
# 3) Defines RRF fusion for dense/sparse lists
# 4) Retrieves hybrid candidates for a query (dense ⊕ sparse → RRF)
# 5) Builds LLM prompts and generation with Gemini
# 6) Generates sub-questions, pools across them with RRF
# 7) Re-ranks with MedCPT cross-encoder
# 8) Returns a concise Markdown answer
# =============================================================================


# -----------------------------------------------------------------------------
# ## 1) Imports & Environment
# -----------------------------------------------------------------------------
import os
import vertexai
from google.oauth2 import service_account

# Vertex AI credentials (local keyfile) + init
creds = service_account.Credentials.from_service_account_file(
    #YOUR JSON FILE PATH
)
vertexai.init(
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION"),
    credentials=creds,
    api_transport="grpc",
)

# Standard libs & utilities
import os
import time
import math
from datetime import date, timedelta
import xml.etree.ElementTree as ET
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import re
from markdown import markdown
from bs4 import BeautifulSoup

# Vector DB + LangChain
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore")

# Load env once
load_dotenv(override=True)


# -----------------------------------------------------------------------------
# ## 2) Pinecone: create/connect dense & sparse indexes + encoders
# -----------------------------------------------------------------------------
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Create dense index
if not pc.has_index("dense-pubmed"):
    pc.create_index(
        name="dense-pubmed",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
dense_index = pc.Index("dense-pubmed")
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
dense_vector_store = PineconeVectorStore(index=dense_index, embedding=embeddings)

# Create sparse index
if not pc.has_index("sparse-pubmed"):
    pc.create_index(
        name="sparse-pubmed",
        vector_type="sparse",
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
sparse_index = pc.Index("sparse-pubmed")

# SPLADE sparse encoder (for query/doc)
from sentence_transformers import SparseEncoder
import torch
# Load the SPLADE sparse encoder
# In your init cell, replace the encoder load:
sparse_encoder = SparseEncoder("NeuML/pubmedbert-base-splade", device="cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# ## 3) MedCPT Cross-Encoder Reranker (used after retrieval/fusion)
# -----------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def rerank_medcpt(pairs, candidates, top_n: int = 10):
    """
    candidates: list of dicts with fields:
      - id (pmid)
      - text (full_text or text)
      - title (optional), doi (optional)
    Returns: top_n list sorted by MedCPT score (desc)
    """
    # Build (query, text) pairs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device)
    scores = []

    enc = tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1)  # higher => more relevant
    scores.extend(logits.detach().cpu().tolist())

    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [
        {
            "id": candidates[i]["id"],
            "score": float(scores[i]),
            "title": candidates[i].get("title", ""),
            "year": candidates[i].get("year", ""),
            "text": candidates[i]["text"],
            "doi": candidates[i].get("doi", ""),
        }
        for i in order[:top_n]
    ]


# -----------------------------------------------------------------------------
# ## 4) RRF (Reciprocal Rank Fusion) across dense/sparse lists
# -----------------------------------------------------------------------------
from collections import defaultdict

def _rrf_merge_ranked_lists_weighted(lists, weights=None, k=60, id_key="id"):
    """
    lists: list of ranked lists, each: [ {id, ...}, {id, ...}, ... ] (best→worst)
    weights: same length as lists; default = 1.0 for each list
    Returns a single ranked list (deduped) with:
      - rrf_score: summed (weighted) RRF
      - hybrid_best_rank: best (lowest) rank seen across lists
      - hybrid_best_list: which list contributed that best rank (e.g., "dense"/"sparse")
    """
    if weights is None:
        weights = [1.0] * len(lists)

    total = defaultdict(float)   # doc_id -> summed score
    best_rank = {}               # doc_id -> (rank, which_list)
    payload = {}                 # doc_id -> representative dict

    for li, (L, w) in enumerate(zip(lists, weights)):
        for r, item in enumerate(L, start=1):
            did = item.get(id_key)
            if not did:
                continue
            inc = w * (1.0 / (k + r))
            total[did] += inc

            # keep representative payload + best local rank
            if (did not in best_rank) or (r < best_rank[did][0]):
                best_rank[did] = (r, li)
                payload[did] = item

    fused = []
    for did, score in total.items():
        it = dict(payload[did])  # shallow copy so we can annotate
        it["rrf_score"] = float(score)
        it["hybrid_best_rank"] = best_rank[did][0]
        it["hybrid_best_list"] = best_rank[did][1]  # 0/1 index into lists
        fused.append(it)

    # sort by global RRF, tiebreak by better local rank
    fused.sort(key=lambda x: (-x["rrf_score"], x["hybrid_best_rank"]))
    return fused


# -----------------------------------------------------------------------------
# ## 5) Per-query hybrid retrieval: dense + sparse → RRF → hybrid list
# -----------------------------------------------------------------------------
def get_hybrid_candidates(
    query: str,
    k_dense: int = 3,
    k_sparse: int = 6,
    start_year: int | None = None,
    end_year: int   | None = None,
    k_rrf_hybrid: int = 60,             # RRF k for within-subQ fusion
    w_dense: float = 1.0,
    w_sparse: float = 1.0,
):
    """
    Runs dense (LangChain PineconeVectorStore) and sparse (SPLADE→Pinecone) for *this* sub-question,
    then RRF-fuses them into a single ranked list to return.

    Returns: ranked list of dicts with fields:
      id, text, title, year, doi, source (from the representative item),
      rrf_score (hybrid), hybrid_best_rank, hybrid_best_list
    """

    # ---- Build year filter (one-sided bounds allowed) ----
    flt = {}
    if start_year and end_year:
        flt["year"] = {"$gte": start_year, "$lte": end_year}
    elif start_year:
        flt["year"] = {"$gte": start_year}
    elif end_year:
        flt["year"] = {"$lte": end_year}

    # ---- Dense (LangChain vector store) ----
    dense_hits = dense_vector_store.similarity_search(query, k=k_dense, filter=flt)
    dense = []
    for res in dense_hits:
        md = getattr(res, "metadata", {}) or {}
        dense.append({
            "text": getattr(res, "page_content", "") or md.get("text", ""),
            "title": md.get("title", ""),
            "id": str(md.get("_id", "")),
            "year": md.get("year", "N/A"),
            "doi": md.get("doi", ""),
            "source": "dense"
        })

    # ---- Sparse (SPLADE→Pinecone) ----
    def _to_sparse_values(sp_tensor):
        sp = sp_tensor.coalesce()
        return {
            "indices": sp.indices().cpu().numpy().flatten().tolist(),
            "values":  sp.values().cpu().numpy().tolist()
        }

    sp_q = sparse_encoder.encode_query([query])[0]
    sv = _to_sparse_values(sp_q)
    res = sparse_index.query(
        top_k=k_sparse,
        sparse_vector=sv,
        include_metadata=True,
        filter=flt,
    )

    sparse = []
    for m in res["matches"]:
        md = m.get("metadata", {}) or {}
        sparse.append({
            "text": md.get("full_text") or md.get("text", ""),
            "title": md.get("title", ""),
            "id": str(m.get("id", "")),
            "year": md.get("year", "N/A"),
            "doi": md.get("doi", ""),
            "source": "sparse"
        })

    # ---- Hybrid RRF fusion (dense ⊕ sparse) ----
    hybrid = _rrf_merge_ranked_lists_weighted(
        lists=[dense, sparse],
        weights=[w_dense, w_sparse],
        k=k_rrf_hybrid,
        id_key="id",
    )

    return hybrid


# -----------------------------------------------------------------------------
# ## 6) Gemini prompting & answer assembly
# -----------------------------------------------------------------------------
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- Build context from your reranked list (id=PMID, doi, title, text) ---
def _build_context_from_reranked(reranked):
    blocks = []
    for h in reranked:
        pmid = str(h.get("id", ""))
        year = h.get("year", "N/A")
        text = (h.get("text") or "").strip()
        blocks.append(
            f"[PMID:{pmid}; Year:{year}]\nEXCERPT:\n{text}"
        )
    return "\n\n---\n\n".join(blocks)

# --- Reusable prompt template (Markdown mode) ---
PROMPT_MD = """You are a careful biomedical literature assistant.

TASK:
Answer the QUESTION using ONLY the EXCERPTS. Do not use prior knowledge.
Write a concise, complete Markdown answer.

HARD REQUIREMENTS:
1) Every factual sentence MUST end with bracketed tags in this exact form:
   [PMID:<pmid>, Year:<year>]
   - If multiple sources support a sentence, include multiple tags separated by a single space.
2) Do NOT add any sections like “Key references”, “Sources”, “Further reading”, figures, tables, or footnotes.
3) Do NOT mention the existence of EXCERPTS or that you are an AI model.
4) If EXCERPTS are insufficient to support an essential claim, explicitly say:
   “Insufficient evidence in provided excerpts.” [PMID:N/A, Year:N/A]
5) Stay under 250–350 words unless the QUESTION asks for more depth.
6) End your output with the token </ANSWER> and do not write anything after it.

STRUCTURE:
- 2-sentence executive summary (each sentence with citations).
- Short bulleted findings (each bullet is one sentence, each with citations).
- 1-sentence bottom line (with citations).

QUESTION:
{question}

EXCERPTS (the only sources you may use):
{context}

Begin your answer now. Remember: every sentence needs bracketed tags; no extra sections; end with </ANSWER>.

"""

def answer_with_gemini(question: str, reranked: list, model_name: str = "gemini-2.5-flash",
                                      temperature: float = 0) -> str:
    """
    Returns a Markdown answer where every sentence ends with [PMID:...; DOI:...]
    """
    context = _build_context_from_reranked(reranked)
    prompt = PROMPT_MD.format(question=question, context=context)

    model = GenerativeModel(model_name)
    cfg = GenerationConfig(temperature=temperature)

    resp = model.generate_content(prompt, generation_config=cfg)
    return getattr(resp, "text", str(resp))


# -----------------------------------------------------------------------------
# ## 7) Sub-question generation (recall expansion)
# -----------------------------------------------------------------------------
def generate_subquestions(query,model_name = "gemini-2.5-flash",question_number = 3):
    prompt = f"""
    You are assisting biomedical literature retrieval.

    GOAL
    Decompose the user's query into EXACTLY {question_number} short, diverse retrieval sub-questions that
    maximize recall. Use clinical PICO structure when applicable and diversify angles to avoid
    synonyms.

    GUIDELINES
    - Each sub-question ≤ 12 words, no bullets/numbering.
    - Prefer concrete facets: Population/Problem, Intervention/Exposure (or vitamin/supplement),
    Comparator (e.g., placebo/low-dose/none), Outcome (e.g., incidence/mortality/biomarker).
    - Vary dimensions across the {question_number} questions (e.g., population, cancer subtype, dose/duration,
    mechanism/biomarker, study design).
    - Avoid yes/no phrasing; avoid repeating near-synonyms.

    INPUT
    User query: {query}

    OUTPUT
    Return ONLY a JSON array with {question_number} strings, e.g. ["...", "...", "..."].
    """
    model = GenerativeModel(model_name)
    cfg = GenerationConfig(temperature=0.3)
    resp = model.generate_content(prompt, generation_config=cfg)
    text = getattr(resp, "text", str(resp)).strip()
    try:
        qs = json.loads(text)
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
    except Exception:
        # Fallback: line-scrape if JSON fails
        qs = [s.strip("-• ").strip() for s in text.splitlines() if s.strip()]
    qs.append(query)
    return qs


# -----------------------------------------------------------------------------
# ## 8) RRF pooling across sub-questions (global fusion)
# -----------------------------------------------------------------------------
from collections import defaultdict
def rrf_pool(results_by_subq, k=60):
    total = defaultdict(float)      # sum across sub-questions
    best  = {}                      # doc_id -> (best_inc, best_subq, best_rank, payload)

    for subq, docs in results_by_subq.items():
        for r, item in enumerate(docs, start=1):
            did = item.get("id")
            if not did:
                continue
            inc = 1.0 / (k + r)
            total[did] += inc
            if (did not in best) or (inc > best[did][0]):
                payload = dict(item)
                payload["best_subq"] = subq
                payload["best_rank"] = r
                best[did] = (inc, subq, r, payload)

    fused = []
    for did, (inc, subq, r, payload) in best.items():
        payload = dict(payload)
        payload["rrf_score"] = float(total[did])
        fused.append(payload)

    fused.sort(key=lambda p: (-p["rrf_score"], p["best_rank"]))
    pairs = [[p["best_subq"], p["text"]] for p in fused]
    return pairs, fused


# -----------------------------------------------------------------------------
# ## 9) Concurrent retrieval for sub-questions → RRF pool
# -----------------------------------------------------------------------------
def retrieve_for_subquestions_concurrent_with_rrf(subqs: list[str],
                                         start_year: int = 1800, end_year: int = 2025,
                                         k_dense: int = 3, k_sparse: int = 6, top_n_rrf: int = 30, w_dense: float = 1.0, w_sparse: float = 1.0):
    """
    Runs get_hybrid_candidates concurrently for each sub-question, then pools results.
    Otherwise, just concat, add subq/subq_rank, then dedup by id/doi/text head.
    """
    results_by_subq: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(3, len(subqs))) as ex:
        futs = {ex.submit(get_hybrid_candidates, q, k_dense, k_sparse, start_year = start_year, end_year = end_year, w_dense = w_dense, w_sparse = w_sparse): q
                for q in subqs}
        for fut in as_completed(futs):
            q = futs[fut]
            docs = fut.result()  # already deduped within hybrid
            # annotate local rank so RRF can use it if desired
            for rank, d in enumerate(docs, start=1):
                d = dict(d)
                d["subq"] = q
                d["subq_rank"] = rank
                docs[rank-1] = d
            results_by_subq[q] = docs

    pairs, candidates = rrf_pool(results_by_subq, k=60)
    return pairs[:top_n_rrf], candidates[:top_n_rrf]


# -----------------------------------------------------------------------------
# ## 10) One-call pipeline: subq → concurrent retrieve → MedCPT rerank → Gemini answer
# -----------------------------------------------------------------------------
def answer_with_query_expansion(user_query: str,
                                start_year: int, end_year: int,
                                n_subq: int = 3,
                                k_dense: int = 3,
                                k_sparse: int = 6,
                                top_n_rerank: int = 8,
                                top_n_rrf: int = 30,
                                llm_for_subq: str = "gemini-2.5-flash",
                                llm_for_answer: str = "gemini-2.5-pro",
                                temperature: float = 0.2,
                                w_dense: float = 1.0,
                                w_sparse: float = 1.0) -> dict:
    """
    Returns: {"subquestions": [...], "pooled": [...], "reranked": [...], "answer": markdown}
    - Sub-questions generated by LLM
    - Pooled (and possibly RRF-fused) candidates across sub-questions
    - MedCPT rerank done against the ORIGINAL user_query (not sub-questions)
    - Final plain text answer from your Gemini prompt
    """
    subqs = generate_subquestions(query = user_query, question_number = n_subq, model_name=llm_for_subq)
    pairs,candidates = retrieve_for_subquestions_concurrent_with_rrf(subqs, k_dense=k_dense, k_sparse=k_sparse,
                                                                     top_n_rrf = top_n_rrf,
                                                                     start_year = start_year, end_year = end_year, w_dense = w_dense, w_sparse = w_sparse)
    reranked = rerank_medcpt(pairs, candidates, top_n=top_n_rerank)
    md = answer_with_gemini(user_query, reranked, model_name=llm_for_answer, temperature= temperature)
    md = md.replace("</ANSWER>", "").strip()
    html = markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def main():
    user_query = "what is the effect of vitamin D on cancer"
    start_year = 2020
    end_year = 2025
    n_subq = 3
    k_dense = 3
    k_sparse = 6
    top_n_rerank = 10
    top_n_rrf = 30
    llm_for_subq = "gemini-2.5-flash-lite"
    llm_for_answer = "gemini-2.5-flash"
    w_dense = 1.0
    w_sparse = 1.0
    temperature = 0.2
    answer = answer_with_query_expansion(user_query = user_query,
                                         start_year = start_year, end_year = end_year,
                                         n_subq = n_subq, k_dense = k_dense, k_sparse = k_sparse,
                                         top_n_rerank = top_n_rerank, top_n_rrf = top_n_rrf,
                                         llm_for_subq = llm_for_subq, llm_for_answer = llm_for_answer,
                                         temperature = temperature, w_dense = w_dense, w_sparse = w_sparse)
    print(answer)
if __name__ == "__main__":
    main()