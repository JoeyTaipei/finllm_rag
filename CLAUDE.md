# finllm-kgi

RAG-based financial QA system targeting KGI Bank documents.

## Project Structure

```
C:\Projects\FinLLM\
├── src\
│   ├── ingestion\       # PDF loading & chunking
│   ├── retrieval\       # Hybrid BM25 + vector retrieval, RRF reranker
│   ├── generation\      # Claude-based answer generation
│   └── evaluation\      # Eval harness
├── data\
│   ├── eval\            # Evaluation datasets
│   └── raw_pdfs\        # Source KGI Bank PDFs
├── scripts\             # One-off utility scripts
├── tests\               # pytest test suite
├── app\                 # Streamlit frontend
├── outputs\             # Generated outputs (gitignored)
├── .env                 # API keys (gitignored)
├── .env.example         # Key template
├── .gitignore
├── CLAUDE.md
└── requirements.txt
```

## Architecture

**Ingestion:** PDFs in `data/raw_pdfs/` are loaded with `pypdf`, chunked, and stored in ChromaDB (dense vectors via `sentence-transformers`) alongside an in-memory BM25 index (sparse, tokenized with `jieba` for Chinese text).

**Retrieval → Generation:** A query runs parallel BM25 + ChromaDB similarity search (top-20 each). Results are merged via Reciprocal Rank Fusion (`src/retrieval/reranker.py`) to produce top-5 documents, which are passed as context to Claude (`claude-sonnet-4-6`) for answer generation. All runs are traced via LangSmith.

## Key Commands

```powershell
# Install dependencies
.venv\Scripts\pip install -r requirements.txt

# Run Streamlit app
.venv\Scripts\streamlit run app\main.py

# Run tests
.venv\Scripts\pytest tests\
```
