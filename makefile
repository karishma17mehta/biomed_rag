# Makefile — BioMed RAG Pipeline
# Usage:
#   make ingest       — run full ingest pipeline (clean → chunk → embed → index)
#   make eval         — run full evaluation pipeline (retrieve → RAGAS)
#   make app          — launch Streamlit app
#   make test         — run all tests
#   make clean-cache  — remove __pycache__ and .pyc files
#   make help         — show this message

PYTHON     := .venv-1/bin/python3
STREAMLIT  := .venv-1/bin/streamlit

.PHONY: help ingest eval app test clean-cache \
        ingest-clean ingest-chunk ingest-embed ingest-index \
        eval-retrieve eval-ragas

# ─────────────────────────────────────────────────────────────
# Default target
# ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "BioMed RAG Pipeline"
	@echo "────────────────────────────────────────────"
	@echo "  make ingest        Full ingest pipeline"
	@echo "  make ingest-clean  Steps 01-03: clean + filter docs"
	@echo "  make ingest-chunk  Steps 04-05: chunk + filter chunks"
	@echo "  make ingest-embed  Step  06:    build FAISS index"
	@echo "  make ingest-index  Steps 07-09: hash + BM25 + tag refs"
	@echo ""
	@echo "  make eval          Full evaluation pipeline"
	@echo "  make eval-retrieve Step 01: run retrieval"
	@echo "  make eval-ragas    Step 06: run RAGAS scoring"
	@echo ""
	@echo "  make app           Launch Streamlit app"
	@echo "  make test          Run all tests"
	@echo "  make clean-cache   Remove __pycache__ and .pyc files"
	@echo "────────────────────────────────────────────"
	@echo ""

# ─────────────────────────────────────────────────────────────
# Ingest pipeline
# ─────────────────────────────────────────────────────────────
ingest: ingest-clean ingest-chunk ingest-embed ingest-index
	@echo "✅ Ingest pipeline complete."

ingest-clean:
	@echo "── Step 01: Clean and enrich ──"
	$(PYTHON) -m ingest.01_clean_and_enrich
	@echo "── Step 02: Clean phase 1 ──"
	$(PYTHON) -m ingest.02_clean_phase1
	@echo "── Step 03: Filter docs (min length) ──"
	$(PYTHON) -m ingest.03_filter_docs

ingest-chunk:
	@echo "── Step 04: Chunk papers ──"
	$(PYTHON) -m ingest.04_chunk_papers
	@echo "── Step 05: Filter chunks ──"
	$(PYTHON) -m ingest.05_filter_chunks
	@echo "── Step 05b: Filter chunks pass 2 ──"
	$(PYTHON) -m ingest.05b_filter_chunks_pass2
	@echo "── Step 05c: Filter by label confidence ──"
	$(PYTHON) -m ingest.05c_filter_by_label_confidence
	@echo "── Step 06: Make high-confidence CSV ──"
	$(PYTHON) -m ingest.06_make_highconf_csv_strict

ingest-embed:
	@echo "── Step 06: Build FAISS index (OpenAI embeddings) ──"
	@echo "    ⚠️  This step calls the OpenAI API and may take several minutes."
	$(PYTHON) -m ingest.06_build_faiss_openai

ingest-index:
	@echo "── Step 07: Add content hashes ──"
	$(PYTHON) -m ingest.07_add_content_hash_to_meta
	@echo "── Step 08: Tag reference lists ──"
	$(PYTHON) -m ingest.08_tag_reference_lists
	@echo "── Step 09: Build BM25 index ──"
	$(PYTHON) -m ingest.09_build_bm25

# ─────────────────────────────────────────────────────────────
# Eval pipeline
# ─────────────────────────────────────────────────────────────
eval: eval-retrieve eval-ragas
	@echo "✅ Evaluation pipeline complete."

eval-retrieve:
	@echo "── Eval 01: Run retrieval ──"
	$(PYTHON) -m eval.01_run_retrieval
	@echo "── Eval 02: Health report ──"
	$(PYTHON) -m eval.02_health_report
	@echo "── Eval 03: Entity hit rate ──"
	$(PYTHON) -m eval.03_entity_hit_rate
	@echo "── Eval 04: Build RAGAS dataset ──"
	$(PYTHON) -m eval.04_build_ragas_dataset
	@echo "── Eval 05: Generate answers ──"
	$(PYTHON) -m eval.05_generate_answers

eval-ragas:
	@echo "── Eval 06: Run RAGAS scoring ──"
	@echo "    ⚠️  This step calls the OpenAI API."
	$(PYTHON) -m eval.06_run_ragas

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app:
	@echo "── Launching Streamlit app ──"
	$(STREAMLIT) run app/streamlit_app.py

# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────
test:
	@echo "── Running tests ──"
	$(PYTHON) -m pytest tests/ -v

# ─────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────
clean-cache:
	@echo "── Removing __pycache__ and .pyc files ──"
	find . -type d -name "__pycache__" -not -path "./.venv*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.venv*" -delete 2>/dev/null || true
	@echo "✅ Cache cleared."