# =============================================================================
# # PubMed → Pinecone Ingestion
# =============================================================================
# This script:
# 1) Loads env and sets up Pinecone (dense + sparse)
# 2) Loads dense (HF embeddings) and sparse (SPLADE) encoders
# 3) Defines PubMed helpers (ESearch + EFetch + parsing)
# 4) Upserts to Pinecone (dense vectors and sparse vectors)
# 5) Runs a full ingestion pipeline for a search term
# =============================================================================


# -----------------------------------------------------------------------------
# ## 1) Standard libs, utilities, and env
# -----------------------------------------------------------------------------
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

# Load env once
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# ## 2) Pinecone setup (dense & sparse) + LangChain vector store
# -----------------------------------------------------------------------------
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore")

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

# -----------------------------------------------------------------------------
# ## 3) Sparse encoder (SPLADE)
# -----------------------------------------------------------------------------
from sentence_transformers import SparseEncoder
import torch
# Load the SPLADE sparse encoder
# In your init cell, replace the encoder load:
sparse_encoder = SparseEncoder("NeuML/pubmedbert-base-splade", device="cuda" if torch.cuda.is_available() else "cpu")

# Timer starts here to reflect the original code position
t0 = time.time()  # Start timer

# -----------------------------------------------------------------------------
# ## 4) PubMed / NCBI E-Utilities constants
# -----------------------------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESearch_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFetch_URL  = f"{EUTILS_BASE}/efetch.fcgi"

# -----------------------------------------------------------------------------
# ## 5) Rate limiter & chunking utilities
# -----------------------------------------------------------------------------
# ---------------- Rate limiter ----------------
class RateLimiter:
    def __init__(self, max_calls, period=1.0):
        self._lock = threading.Lock()
        self._tokens = max_calls
        self._last = time.time()
        self._period = period
        self._rate = max_calls / period

    def wait(self):
        with self._lock:
            now = time.time()
            # refill
            self._tokens += (now - self._last) * self._rate
            cap = self._rate * self._period
            if self._tokens > cap:
                self._tokens = cap
            self._last = now

            if self._tokens < 1:
                sleep_for = (1 - self._tokens) / self._rate
                time.sleep(max(0.0, sleep_for))
                self._last = time.time()
                self._tokens = 0
            self._tokens -= 1

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# -----------------------------------------------------------------------------
# ## 6) PubMed helpers: search IDs across date ranges (handles 10k cap)
# -----------------------------------------------------------------------------
def get_pubmed_ids_for_abstract(term, api_key=None, page_size=10000, max_workers=6,
                                datetype="pdat", start_date: date | None = None,
                                end_date: date | None = None):
    """
    Fetch ALL PMIDs for a free-text TERM constrained to abstracts (e.g., "vitamin d[AB]")
    across a date range. Automatically splits the range when ESearch count exceeds
    10k IDs (PubMed per-query limit), then pages within each subrange.
"""
    limiter = RateLimiter(max_calls=(10 if api_key else 3), period=1.0)

    if start_date is None:
        start_date = date(1800, 1, 1)
    if end_date is None:
        end_date = date.today()

    # Global cumulative counter for this query
    cumulative_ids_seen: set[str] = set()
    cumulative_ids_count: int = 0
    count_lock = threading.Lock()

    def esearch_count(mindate_str: str, maxdate_str: str) -> int:
        limiter.wait()
        params = {
            "db": "pubmed",
            "retmax": 0,
            "retmode": "xml",
            "datetype": datetype,
            "mindate": mindate_str,
            "maxdate": maxdate_str,
            "term": term,
        }
        if api_key:
            params["api_key"] = api_key
        resp = requests.get(ESearch_URL, params=params, timeout=90)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        return int(root.findtext(".//Count", default="0"))

    def fetch_range(mindate_d: date, maxdate_d: date) -> list[str]:
        nonlocal cumulative_ids_count
        mindate_str = f"{mindate_d.year}/{mindate_d.month:02d}/{mindate_d.day:02d}"
        maxdate_str = f"{maxdate_d.year}/{maxdate_d.month:02d}/{maxdate_d.day:02d}"

        total = esearch_count(mindate_str, maxdate_str)
        if total == 0:
            return []

        if total > 10000:
            # Split recursively when above 10k cap
            delta_days = (maxdate_d - mindate_d).days
            if delta_days <= 0:
                return []
            mid = mindate_d + timedelta(days=delta_days // 2)
            left_end = mid
            right_start = mid + timedelta(days=1)
            left_ids = fetch_range(mindate_d, left_end)
            right_ids = fetch_range(right_start, maxdate_d)
            return list(dict.fromkeys(left_ids + right_ids))

        offsets = list(range(0, total, page_size))

        def fetch_page(offset):
            limiter.wait()
            p = {
                "db": "pubmed",
                "retstart": offset,
                "retmax": page_size,
                "retmode": "xml",
                "datetype": datetype,
                "mindate": mindate_str,
                "maxdate": maxdate_str,
                "term": term,
            }
            if api_key:
                p["api_key"] = api_key
            r = requests.get(ESearch_URL, params=p, timeout=90)
            r.raise_for_status()
            tree = ET.fromstring(r.content)
            return [e.text for e in tree.findall(".//IdList/Id")]

        range_ids: list[str] = []
        desc_term = term if len(term) <= 40 else term[:37] + "..."
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_page, off): off for off in offsets}
            pbar = tqdm(total=len(futures), desc=f"PMID pages for query: {desc_term}")
            for fut in as_completed(futures):
                off = futures[fut]
                try:
                    ids = fut.result()
                    with count_lock:
                        new_ids = [pid for pid in ids if pid not in cumulative_ids_seen]
                        if new_ids:
                            cumulative_ids_seen.update(new_ids)
                            cumulative_ids_count += len(new_ids)
                        pbar.set_postfix_str(f"ids: {cumulative_ids_count:,}")
                    range_ids.extend(new_ids)
                except Exception as e:
                    print(f"Failed to fetch page at offset {off}: {e}")
                finally:
                    pbar.update(1)
            pbar.close()

        return list(dict.fromkeys(range_ids))

    return fetch_range(start_date, end_date)

# -----------------------------------------------------------------------------
# ## 7) PubMed parsing helpers (EFetch XML → dict)
# -----------------------------------------------------------------------------
def _text_or_none(elem):
    if elem is None:
        return None
    return "".join(elem.itertext()).strip() or None

import re

def _extract_year_from_pubdate(article_elem) -> str | None:
    """
    Try several places where PubMed stores the publication year.
    Returns a 4-digit string like '2021' or None if not found.
    Priority:
      1) MedlineCitation/Article/Journal/JournalIssue/PubDate/Year
      2) .../PubDate/MedlineDate (regex first 4-digit year)
      3) MedlineCitation/Article/ArticleDate/Year
      4) PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year
    """
    # 1) Standard Year
    y = article_elem.find(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
    y = _text_or_none(y)
    if y and re.fullmatch(r"\d{4}", y):
        return y

    # 2) MedlineDate like "1998 Jan-Feb" or "2019 Spring"
    md = article_elem.find(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")
    md = _text_or_none(md)
    if md:
        m = re.search(r"(19|20)\d{2}", md)
        if m:
            return m.group(0)

    # 3) ArticleDate block
    ay = article_elem.find(".//MedlineCitation/Article/ArticleDate/Year")
    ay = _text_or_none(ay)
    if ay and re.fullmatch(r"\d{4}", ay):
        return ay

    # 4) PubMed indexing date
    py = article_elem.find(".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year")
    py = _text_or_none(py)
    if py and re.fullmatch(r"\d{4}", py):
        return py

    return None

def parse_pubmed_article(article_elem):
    """
    Parse one <PubmedArticle> Element into a dict with title/abstract/pmid/doi/authors.
    """
    pmid = _text_or_none(article_elem.find(".//MedlineCitation/PMID")) or ""

    title = _text_or_none(article_elem.find(".//MedlineCitation/Article/ArticleTitle")) or "N/A"

    abs_parts = []
    for at in article_elem.findall(".//MedlineCitation/Article/Abstract/AbstractText"):
        t = _text_or_none(at)
        if t:
            label = at.get("Label")
            abs_parts.append(f"{label}: {t}" if label else t)
    abstract = " ".join(abs_parts) if abs_parts else None

    doi = None
    for aid in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if (aid.get("IdType") or "").lower() == "doi":
            doi = _text_or_none(aid)
            break
    doi = doi or "N/A"

    names = []
    for a in article_elem.findall(".//MedlineCitation/Article/AuthorList/Author"):
        coll = a.find("CollectiveName")
        if coll is not None:
            t = _text_or_none(coll)
            if t:
                names.append(t)
            continue
        fore = _text_or_none(a.find("ForeName")) or _text_or_none(a.find("GivenName")) or ""
        last = _text_or_none(a.find("LastName")) or ""
        nm = (f"{fore} {last}").strip()
        if nm:
            names.append(nm)
    authors = ", ".join(names) if names else "N/A"
    year = _extract_year_from_pubdate(article_elem) or "N/A"

    return {"pmid": pmid, "title": title, "abstract": abstract, "doi": doi, "authors": authors, "year": year}

def _join_text_for_index(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("abstract") or "").strip()
    return f"{title}, {abstract}" if abstract else title

# -----------------------------------------------------------------------------
# ## 8) Upsert helpers: sparse & dense records to Pinecone
# -----------------------------------------------------------------------------
def upsert_records_sparse(docs: list[dict], sparse_index, batch_size: int = 100) -> int:
    texts = [_join_text_for_index(d) for d in docs]
    ids   = [d["pmid"] for d in docs]
    metas = [{"full_text": _join_text_for_index(d), "title": d["title"], "doi": d["doi"], "authors": d["authors"], "year": int(d["year"])} for d in docs]
    total = 0
    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i:i+batch_size]
        sp_list = sparse_encoder.encode_document(chunk_texts)   # Fixed: use sparse_encoder
        vectors = []
        for j, sp in enumerate(sp_list):
            k = i + j
            sp = sp.coalesce()
            sparse_dict = {
                "indices": sp.indices().cpu().numpy().flatten().tolist(),  # List of int indices
                "values": sp.values().cpu().numpy().tolist()               # List of float values
            }
            vectors.append({
                "id": ids[k],
                "sparse_values": sparse_dict,                     # ← sparse payload
                "metadata": metas[k],
            })
        if vectors:
            sparse_index.upsert(vectors=vectors)
            total += len(vectors)
    return total

def upsert_records_dense(docs: list[dict], dense_vector_store, batch_size: int = 100) -> int:
    buf_dense, total = [], 0
    for d in docs:
        pmid = d.get("pmid")
        if not pmid:
            continue
        buf_dense.append(Document(
            page_content=_join_text_for_index(d),
            metadata={"_id": pmid,"title": d.get("title"), "doi": d.get("doi"), "authors": d.get("authors"), "year": int(d.get("year"))}
        ))
        if len(buf_dense) >= batch_size:
            dense_vector_store.add_documents(documents = buf_dense)
            total += len(buf_dense)
            buf_dense.clear()
    if buf_dense:
        dense_vector_store.add_documents(documents = buf_dense)
        total += len(buf_dense)
    return total

# -----------------------------------------------------------------------------
# ## 9) EFetch → parse → upsert pipeline
# -----------------------------------------------------------------------------
def efetch_parse_and_upsert_pinecone(pmids, sparse_index, dense_vector_store, email, api_key=None,
                                     ids_per_call=200, max_workers=8, max_retries=3, retry_delay=2.0):
    """
    EFetch PMIDs in batches, parse fields, and upsert *sparse text* to Pinecone
    (managed sparse model). Uses POST per NCBI best practice.
    """
    rps = 10 if api_key else 3
    limiter = RateLimiter(rps, 1.0)

    base_params = {
        "db": "pubmed",
        "retmode": "xml",
        "tool": "pubmed-efetch-parser",
        "email": email,
    }
    if api_key:
        base_params["api_key"] = api_key

    def fetch_batch(id_batch):
        for attempt in range(1, max_retries + 1):
            try:
                limiter.wait()
                data = base_params | {"id": ",".join(id_batch)}
                resp = requests.post(EFetch_URL, data=data, timeout=90)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                return [parse_pubmed_article(e) for e in root.findall(".//PubmedArticle")]
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(retry_delay * (2 ** (attempt - 1)))
        return []

    total_upserts = 0
    batches = list(chunked(pmids, ids_per_call))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_batch, b): i for i, b in enumerate(batches)}
        pbar = tqdm(total=len(futures), desc="EFetch batches → Pinecone")
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                docs = fut.result() or []
                filtered_docs = [d for d in docs if d.get("pmid") and (d.get("abstract") or "").strip()]
                if filtered_docs:
                    dense_up = upsert_records_dense(filtered_docs, dense_vector_store, batch_size=100)
                    sparse_up = upsert_records_sparse(filtered_docs, sparse_index, batch_size=100)
                    total_upserts += dense_up  
            except Exception as e:
                print(f"Failed EFetch batch #{idx}: {e}")
            finally:
                pbar.set_postfix_str(f"upserts: {total_upserts:,}")
                pbar.update(1)
        pbar.close()
    return total_upserts

# -----------------------------------------------------------------------------
# ## 10) User/config knobs & run
# -----------------------------------------------------------------------------
def main():
    EMAIL = "haidangnguyen1815@gmail.com"
    API_KEY = os.getenv("PUBMED_API_KEY") 

    # Throughput knobs
    ESEARCH_PAGE_SIZE = 10000   # page size for ESearch (per-query cap is 10k)
    ESEARCH_WORKERS   = 6
    EFETCH_IDS_PER    = 200     # safer per NCBI guidance; uses POST
    EFETCH_WORKERS    = 8

    # Query for vitamin D in abstracts only. You can adjust term as needed.
    SEARCH_TERM = "vitamin D[Title/Abstract]"
    START_DATE = date(1800, 1, 1)    # earliest reasonable bound
    END_DATE   = date.today()        # up to today

    # ESearch all PMIDs matching the term in abstracts, automatically splitting by date
    pmids = get_pubmed_ids_for_abstract(
        term=SEARCH_TERM,
        api_key=API_KEY,
        page_size=ESEARCH_PAGE_SIZE,
        max_workers=ESEARCH_WORKERS,
        datetype="pdat",
        start_date=START_DATE,
        end_date=END_DATE,
    )
    print(f"PMIDs found for query '{SEARCH_TERM}': {len(pmids):,}")

    if pmids:
        upserts = efetch_parse_and_upsert_pinecone(
            pmids=pmids,
            sparse_index=sparse_index,  # Fixed: pass the index, not encoder
            dense_vector_store=dense_vector_store,
            email=EMAIL,
            api_key=API_KEY,
            ids_per_call=EFETCH_IDS_PER,
            max_workers=EFETCH_WORKERS,
            max_retries=3,
            retry_delay=2.0,
        )
        print(f"New documents inserted (upserts): {upserts:,}")

    dt_sec = time.time() - t0
    print(f"Done in {dt_sec:.1f}s")
if __name__ == "__main__":
    main()
