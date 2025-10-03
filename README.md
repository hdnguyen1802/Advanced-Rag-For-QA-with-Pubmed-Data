## PubMed RAG — Vitamin D Specialized

A retrieval-augmented generation (RAG) pipeline tailored to PubMed literature about Vitamin D. It builds a hybrid retriever over Pinecone using both dense embeddings and SPLADE sparse representations, fuses results via RRF, reranks with MedCPT, and generates a concise, citation-constrained answer with Vertex AI Gemini. A Streamlit UI is provided for interactive exploration.

### Key Features
- **Hybrid retrieval**: Dense (`NeuML/pubmedbert-base-embeddings`) + Sparse (SPLADE `NeuML/pubmedbert-base-splade`).
- **Reciprocal Rank Fusion (RRF)** to combine dense and sparse results per sub-question, and pool across sub-questions.
- **Query expansion**: Gemini decomposes the user query into diverse sub-questions to improve recall.
- **Reranking**: `ncbi/MedCPT-Cross-Encoder` re-scores top candidates against the original query.
- **Answering**: Gemini composes a Markdown answer where every sentence is bracket-cited with `[PMID:..., Year:...]`.
- **Streamlit UI**: Adjustable parameters and time-window filtering.

### Repository Layout
- `pubmed_to_pinecone.py`: End-to-end PubMed ingestion into Pinecone (dense + sparse indices).
- `rag_query.py`: Retrieval, fusion, reranking, and answer generation pipeline.
- `app.py`: Streamlit UI to run the pipeline interactively.
- `requirements.txt`: Python dependencies.

## Prerequisites
- Python 3.10+
- A Google Cloud project with Vertex AI enabled and access to Gemini models.
- A Google Cloud Service Account key (JSON) with Vertex AI permissions.
- A Pinecone account and API key (v3 SDK, serverless indexes supported).
- Optional but recommended: PubMed (NCBI) API Key to increase rate limits.
- GPU is optional; if available, SPLADE and MedCPT will use CUDA.

## Installation
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the project root with the following variables (values are examples):
```bash
# Pinecone
PINECONE_API_KEY=pcn-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Vertex AI
PROJECT_ID=my-gcp-project-id
LOCATION=us-central1

# Optional: NCBI / PubMed
PUBMED_API_KEY=your_ncbi_api_key
```

Vertex AI authentication: place your service account key JSON on disk. The current code in `rag_query.py` loads the file directly:
```python
creds = service_account.Credentials.from_service_account_file(
    # YOUR JSON FILE PATH
)
```
Edit this to point to your JSON path, or modify it to read from an environment variable like `GOOGLE_APPLICATION_CREDENTIALS` and load accordingly.

Pinecone: the scripts create (if missing) two indexes with fixed names and specs:
- Dense index: `dense-pubmed` (dimension 768, cosine, AWS us-east-1)
- Sparse index: `sparse-pubmed` (vector_type=sparse, dotproduct, AWS us-east-1)

Ensure your Pinecone project supports these serverless specs and region.

## Ingest PubMed → Pinecone
This step fetches PubMed abstracts, parses metadata, and upserts both dense and sparse representations.

Defaults are tuned for a Vitamin D corpus, but you can change the term and date ranges in `pubmed_to_pinecone.py`.

Run:
```bash
python pubmed_to_pinecone.py
```
What it does:
- Searches PMIDs across the date range, splitting automatically if >10k results.
- Uses EFetch to retrieve records in batches (POST), parses XML to extract title, abstract, DOI, authors, year.
- Upserts:
  - Dense vectors via LangChain `PineconeVectorStore` using `NeuML/pubmedbert-base-embeddings`.
  - Sparse vectors via SPLADE (`NeuML/pubmedbert-base-splade`).

Notes:
- Only records with abstracts are upserted.
- Environment `PUBMED_API_KEY` increases E-Utilities rate limits.

## Run the Streamlit App
```bash
streamlit run app.py
```

In the UI you can:
- Enter a research question (e.g., “What is the effect of vitamin D on cancer?”).
- Set the time window.
- Tune retrieval parameters: number of sub-questions, top-k dense/sparse, weights, pool sizes.
- Choose Gemini models for sub-question generation and final answering.

On submit, the app will:
1) Generate sub-questions with Gemini.
2) Retrieve dense and sparse candidates per sub-question and fuse with RRF.
3) Pool across sub-questions, then rerank with MedCPT against the original query.
4) Build an excerpt context and ask Gemini for a concise Markdown answer with inline citations.

## Programmatic Use
You can call the pipeline directly from Python:
```python
from rag_query import answer_with_query_expansion

answer_text = answer_with_query_expansion(
    user_query="what is the effect of vitamin D on cancer",
    start_year=2020,
    end_year=2025,
    n_subq=3,
    k_dense=3,
    k_sparse=6,
    top_n_rerank=10,
    top_n_rrf=30,
    llm_for_subq="gemini-2.5-flash-lite",
    llm_for_answer="gemini-2.5-flash",
    temperature=0.2,
    w_dense=1.0,
    w_sparse=1.0,
)
print(answer_text)
```

## Performance & Cost Notes
- Ingestion can be network and CPU heavy; consider narrower date ranges or smaller batches.
- Gemini and Pinecone operations incur costs; monitor usage in respective dashboards.



