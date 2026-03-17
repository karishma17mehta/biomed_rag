# 🧬 Biomedical RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system for biomedical research, focused on **Thyroid Cancer**, **Lung Cancer**, and **Colon Cancer**. Built with hybrid dense + sparse retrieval, intent-aware query routing, and full RAGAS evaluation infrastructure.

---

## 📌 Project Overview

This system takes a corpus of biomedical research papers and enables semantic question answering grounded in retrieved evidence. It was designed to demonstrate end-to-end ML engineering: from raw data cleaning to embedding, hybrid retrieval, answer generation, and automated evaluation.

**Key capabilities:**
- Hybrid retrieval combining FAISS dense search (OpenAI `text-embedding-3-large`) and BM25 sparse search
- Intent routing across 18 biomedical query families (mutations, resistance, pharmacokinetics, immunotherapy, etc.)
- Cancer-type hard gating — results are filtered to the queried cancer type
- Entity-level reranking for genes, drugs, checkpoints, and pathways
- Full RAGAS evaluation (Faithfulness ~0.9, Answer Relevancy ~0.6)

---

## 🏗️ Architecture

```
Raw CSV (7,000 docs)
        │
        ▼
01_clean_and_enrich.py     ← deduplication via SHA1 content hash → ~2,000 unique docs
        │
        ▼
02_clean_phase1.py         ← parallel cleaning, citation removal, boilerplate stripping
        │
        ▼
03_filter_docs.py          ← drop docs < 200 words
        │
        ▼
06_make_highconf_csv_strict.py  ← label confidence filtering → 484 high-confidence docs
        │
        ▼
04_chunk_papers.py         ← section-aware chunking (320 tokens, 80 token overlap) → 4,061 chunks
        │
        ▼
05_filter_chunks.py        ← spacing repair, reference block removal
05b_filter_chunks_pass2.py ← table/junk detection, secondary filtering
        │
        ▼
06_build_faiss_openai.py   ← embed with text-embedding-3-large, build FAISS IndexFlatIP
07_add_content_hash_to_meta.py ← MD5 content hashing for dedup at retrieval time
08_tag_reference_lists.py  ← score and tag reference-list chunks
09_build_bm25.py           ← build BM25Okapi index
        │
        ▼
app/query_router.py        ← intent detection, entity extraction, cancer type extraction
app/retrieve_faiss.py      ← hybrid retrieval: FAISS + BM25 + reranking + cancer gating
        │
        ▼
eval/01–06                 ← RAGAS evaluation pipeline
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

Evaluated on 30 queries across all 3 cancer types using GPT-4o-mini for answer generation.

| Metric | Score |
|--------|-------|
| Faithfulness | ~0.90 |
| Answer Relevancy | ~0.60 |
| Context Relevance | varies by intent |

Evaluation pipeline: `eval/01_run_retrieval.py` → `eval/04_build_ragas_dataset.py` → `eval/05_generate_answers.py` → `eval/06_run_ragas.py`

---

## 🗂️ Project Structure

```
biomed_rag/
├── app/
│   ├── query_router.py          # Intent routing + entity extraction
│   ├── retrieve_faiss.py        # Hybrid retrieval pipeline
│   └── 00_smoke_test_faiss.py   # Index smoke test
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
├── outputs/
│   └── index_openai/
│       ├── faiss.index
│       ├── meta_tagged_v2.jsonl
│       └── bm25.pkl
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
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY=your-key-here
# or add to a .env file (never commit this)
```

> **Note:** The raw data file (`data/biomedical.csv`, ~180MB) is not included in this repo. The pre-built FAISS index and BM25 index are also not included due to size. Run the ingest pipeline to regenerate them.

---

## 🔍 Running a Query

```python
from app.retrieve_faiss import retrieve

results = retrieve(
    "How do KRAS mutations contribute to therapy resistance in colon cancer?",
    top_k=10
)

for r in results:
    print(r["score"], r["section"], r["paper_id"])
    print(r["text"][:300])
```

---

## 🚀 Roadmap

- [ ] **Phase 1:** Streamlit app — chat interface + RAGAS eval dashboard
- [ ] **Phase 2:** LangGraph agent — local FAISS search + live PubMed search via NCBI Entrez API
- [ ] **Phase 3:** Deploy to Hugging Face Spaces
- [ ] **Phase 4:** Host dataset on Hugging Face Datasets

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector Store | FAISS `IndexFlatIP` |
| Sparse Retrieval | BM25Okapi (`rank_bm25`) |
| Answer Generation | GPT-4o-mini |
| Evaluation | RAGAS 0.4.x |
| Language | Python 3.12 |

---

## 👩‍💻 Author

**Karishma Mehta**  
MS Business Analytics, UC Davis  
[GitHub](https://github.com/karishma17mehta) · [LinkedIn](https://linkedin.com/in/karishma17mehta)