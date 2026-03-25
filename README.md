# 🧬 Biomedical RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system for biomedical research, focused on **Thyroid Cancer**, **Lung Cancer**, and **Colon Cancer**. Built with hybrid dense + sparse retrieval, intent-aware query routing, a LangGraph agent with live PubMed fallback, and full RAGAS evaluation infrastructure.

---

## 📌 Project Overview

This system takes a corpus of biomedical research papers and enables semantic question answering grounded in retrieved evidence. It was designed to demonstrate end-to-end ML engineering: from raw data cleaning to embedding, hybrid retrieval, answer generation, automated evaluation, and agentic fallback for corpus gaps.

**Key capabilities:**
- Hybrid retrieval combining FAISS dense search (OpenAI `text-embedding-3-large`) and BM25 sparse search
- Intent routing across 18 biomedical query families (mutations, resistance, pharmacokinetics, immunotherapy, etc.)
- Cancer-type hard gating — results are filtered to the queried cancer type
- Entity-level reranking for genes, drugs, checkpoints, and pathways
- LangGraph agent — tries local index first, falls back to live PubMed search (NCBI Entrez) when corpus gaps are detected
- Strict grounded answer generation with mandatory citations and evidence gating
- Full RAGAS evaluation (Faithfulness 0.75, Answer Relevancy 0.73, Context Utilization 0.62)
- Centralized config (`config.yaml`), 67 unit + integration tests, Makefile orchestration

---

## 🏗️ Architecture

```
Raw CSV (7,000 docs)
        │
        ▼
01_clean_and_enrich.py      ← deduplication via SHA1 content hash → ~2,000 unique docs
        │
        ▼
02_clean_phase1.py          ← parallel cleaning, citation removal, boilerplate stripping
        │
        ▼
03_filter_docs.py           ← drop docs < 200 words
        │
        ▼
06_make_highconf_csv_strict.py  ← label confidence filtering → 484 high-confidence docs
        │
        ▼
04_chunk_papers.py          ← section-aware chunking (320 tokens, 80 token overlap) → 4,061 chunks
        │
        ▼
05_filter_chunks.py         ← spacing repair, reference block removal
05b_filter_chunks_pass2.py  ← table/junk detection, secondary filtering
        │
        ▼
06_build_faiss_openai.py    ← embed with text-embedding-3-large, build FAISS IndexFlatIP
07_add_content_hash_to_meta.py  ← MD5 content hashing for dedup at retrieval time
08_tag_reference_lists.py   ← score and tag reference-list chunks
09_build_bm25.py            ← build BM25Okapi index
        │
        ▼
app/entities.py             ← shared entity patterns + cancer term dictionaries
app/query_router.py         ← intent detection, entity extraction, cancer type extraction
app/retrieve_faiss.py       ← hybrid retrieval: FAISS + BM25 + reranking + cancer gating
app/agent.py                ← LangGraph agent: local search → PubMed fallback → generate
        │
        ▼
eval/01–06                  ← RAGAS evaluation pipeline
```

---

## 📊 Deduplication Story

| Stage | Documents |
|-------|-----------|
| Raw input | ~7,000 |
| After SHA1 content-hash dedup | ~2,000 unique |
| After label confidence filtering | 484 high-confidence |
| Chunks produced | 4,061 |

The original dataset had significant duplication (~70%). SHA1 fingerprinting on normalized text caught exact and near-exact duplicates before any expensive processing.

---

## 📈 Evaluation Results (RAGAS)

Evaluated on 30 queries across all 3 cancer types using GPT-4o-mini for answer generation and `text-embedding-3-large` for embeddings.

| Metric | Baseline | After Improvements | Change |
|--------|----------|--------------------|--------|
| Faithfulness | 0.64 | **0.75** | +17% |
| Answer Relevancy | 0.57 | **0.73** | +28% |
| Context Utilization | 0.47 | **0.62** | +32% |

**Improvements made:**
- Strict grounded generation prompt with mandatory per-sentence citations
- Evidence gate — refuses to answer if key entities are missing from retrieved chunks
- AE gate — special handling for adverse effects queries
- Clinical vs preclinical mismatch detection
- LangGraph agent with live PubMed fallback for corpus gaps (e.g. BRAF/thyroid, checkpoint/thyroid queries)

Evaluation pipeline: `make eval` or manually:
```
eval/01_run_retrieval.py → eval/04_build_ragas_dataset.py → eval/05_generate_answers.py → eval/06_run_ragas.py
```

---

## 🗂️ Data Sourcing

The raw dataset (`data/biomedical.csv`) is **not included** in this repo due to size (~180MB). The pre-built FAISS index and BM25 index are also excluded.

**To reproduce the dataset:**

The corpus was assembled from PubMed abstracts and full-text articles across three cancer types using the NCBI Entrez API:

```python
from Bio import Entrez
Entrez.email = "your@email.com"

# Example: fetch lung cancer papers
handle = Entrez.esearch(db="pubmed", term="NSCLC EGFR treatment", retmax=500)
record = Entrez.read(handle)
pmids = record["IdList"]
```

Search terms used:
- **Lung Cancer:** `NSCLC treatment`, `EGFR lung cancer`, `KRAS lung resistance`, `immunotherapy lung cancer`
- **Colon Cancer:** `colorectal cancer KRAS`, `CRC BRAF`, `MEK inhibitor colorectal`, `microsatellite instability`
- **Thyroid Cancer:** `papillary thyroid carcinoma`, `BRAF thyroid`, `lenvatinib thyroid`, `RET thyroid cancer`

**To rebuild the indices from scratch:**
```bash
# Place your biomedical.csv in data/
make ingest
```

This runs the full 9-step ingest pipeline and produces:
- `outputs/index_openai/faiss.index`
- `outputs/index_openai/meta_tagged_v2.jsonl`
- `outputs/index_openai/bm25.pkl`

---

## 🗂️ Project Structure

```
biomed_rag/
├── app/
│   ├── entities.py              # Shared entity patterns + cancer term dicts
│   ├── query_router.py          # Intent routing + entity extraction
│   ├── retrieve_faiss.py        # Hybrid retrieval pipeline
│   ├── agent.py                 # LangGraph agent with PubMed fallback
│   ├── config.py                # Config loader
│   └── streamlit_app.py         # Chat UI + RAGAS eval dashboard
├── eval/
│   ├── 01_run_retrieval.py
│   ├── 02_health_report.py
│   ├── 03_entity_hit_rate.py
│   ├── 04_build_ragas_dataset.py
│   ├── 05_generate_answers.py
│   ├── 06_run_ragas.py
│   ├── queries_ragas.jsonl      # 30 evaluation queries
│   └── queries_manual.jsonl
├── ingest/
│   ├── 01_clean_and_enrich.py
│   ├── 02_clean_phase1.py
│   ├── 03_filter_docs.py
│   ├── 04_chunk_papers.py
│   ├── 05_filter_chunks.py
│   ├── 05b_filter_chunks_pass2.py
│   ├── 05c_filter_by_label_confidence.py
│   ├── 06_build_faiss_openai.py
│   ├── 06_make_highconf_csv_strict.py
│   ├── 07_add_content_hash_to_meta.py
│   ├── 08_tag_reference_lists.py
│   └── 09_build_bm25.py
├── tests/
│   ├── test_config.py           # 20 config validation tests
│   ├── test_query_router.py     # 25 intent + entity extraction tests
│   └── test_retriever.py        # 22 retriever unit + integration tests
├── outputs/
│   └── index_openai/            # FAISS index, BM25 index, metadata (not committed)
├── config.yaml                  # All tunable parameters
├── Makefile                     # Pipeline orchestration
├── conftest.py                  # Pytest path setup
├── .env                         # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/karishma17mehta/biomed_rag.git
cd biomed_rag

# 2. Create and activate virtual environment
python3 -m venv .venv-1
source .venv-1/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
echo "export OPENAI_API_KEY=your-key-here" >> .env
source .env
```

---

## 🚀 Quick Start

```bash
# Run the Streamlit app (requires pre-built index)
make app

# Run all tests
make test

# Run full evaluation
make eval

# Rebuild indices from scratch (requires data/biomedical.csv)
make ingest
```

---

## 🔍 Running a Query

```python
# Use the agent (local index + PubMed fallback)
from app.agent import run_agent

result = run_agent("What is the role of BRAF mutations in papillary thyroid carcinoma?")
print(result["answer"])
print(f"Used PubMed: {result['used_pubmed']}")

# Or use the retriever directly
from app.retrieve_faiss import retrieve

hits = retrieve("How do KRAS mutations contribute to therapy resistance in colon cancer?", top_k=10)
for h in hits:
    print(h["score"], h["section"], h["paper_id"])
    print(h["text"][:300])
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector Store | FAISS `IndexFlatIP` |
| Sparse Retrieval | BM25Okapi (`rank_bm25`) |
| Agent Framework | LangGraph |
| Live Search | NCBI Entrez API (PubMed) |
| Answer Generation | GPT-4o-mini |
| Evaluation | RAGAS 0.1.21 |
| UI | Streamlit |
| Language | Python 3.12 |
| Config | YAML (`config.yaml`) |
| Tests | pytest (67 tests) |

---

## 🚀 Roadmap

- [x] **Phase 1:** Streamlit app — chat interface + RAGAS eval dashboard
- [x] **Phase 2:** LangGraph agent — local FAISS search + live PubMed search via NCBI Entrez API
- [ ] **Phase 3:** Deploy to Hugging Face Spaces
- [ ] **Phase 4:** Host dataset on Hugging Face Datasets

---

## 👩‍💻 Author

**Karishma Mehta**  
MS Business Analytics, UC Davis  
[GitHub](https://github.com/karishma17mehta) · [LinkedIn](https://linkedin.com/in/karishma17mehta)
