# tests/test_retriever.py
"""
Integration-lite tests for app/retrieve_faiss.py

These tests check output FORMAT and routing logic without
making OpenAI API calls or requiring the FAISS index.
Index-dependent tests are skipped if index files are missing.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
INDEX_EXISTS = Path("outputs/index_openai/faiss.index").exists()
skip_if_no_index = pytest.mark.skipif(
    not INDEX_EXISTS,
    reason="FAISS index not present — run ingest pipeline first"
)


# ─────────────────────────────────────────────────────────────
# Unit tests — no index needed
# ─────────────────────────────────────────────────────────────
class TestBodyText:
    def test_extracts_body_after_separator(self):
        from app.retrieve_faiss import body_text
        full = "[ paper_000001 | RESULTS ] --- This is the body text."
        result = body_text(full)
        assert result == "This is the body text."

    def test_returns_full_text_if_no_separator(self):
        from app.retrieve_faiss import body_text
        result = body_text("No separator here")
        assert result == "No separator here"

    def test_empty_string_returns_empty(self):
        from app.retrieve_faiss import body_text
        assert body_text("") == ""


class TestEntityOverlapRatio:
    def test_full_overlap(self):
        from app.retrieve_faiss import entity_overlap_ratio
        ratio = entity_overlap_ratio(["KRAS", "EGFR"], "KRAS mutation and EGFR inhibitor resistance")
        assert ratio == 1.0

    def test_partial_overlap(self):
        from app.retrieve_faiss import entity_overlap_ratio
        ratio = entity_overlap_ratio(["KRAS", "BRAF"], "KRAS mutation in colorectal cancer")
        assert ratio == 0.5

    def test_no_overlap(self):
        from app.retrieve_faiss import entity_overlap_ratio
        ratio = entity_overlap_ratio(["BRAF"], "colorectal cancer treatment")
        assert ratio == 0.0

    def test_empty_entities_returns_zero(self):
        from app.retrieve_faiss import entity_overlap_ratio
        assert entity_overlap_ratio([], "some text") == 0.0


class TestCancerKeywordMismatchPenalty:
    def test_no_penalty_when_requested_present(self):
        from app.retrieve_faiss import cancer_keyword_mismatch_penalty
        penalty = cancer_keyword_mismatch_penalty("Lung_Cancer", "nsclc treatment with erlotinib")
        assert penalty == 0.0

    def test_penalty_when_other_cancer_dominates(self):
        from app.retrieve_faiss import cancer_keyword_mismatch_penalty
        # requesting lung but text is full of thyroid terms
        text = "thyroid papillary follicular medullary anaplastic cancer"
        penalty = cancer_keyword_mismatch_penalty("Lung_Cancer", text)
        assert penalty > 0.0

    def test_no_penalty_for_empty_requested(self):
        from app.retrieve_faiss import cancer_keyword_mismatch_penalty
        assert cancer_keyword_mismatch_penalty("", "any text") == 0.0


class TestCancerEvidenceAdjust:
    def test_positive_delta_when_requested_present(self):
        from app.retrieve_faiss import cancer_evidence_adjust
        delta, scores = cancer_evidence_adjust(
            "Lung_Cancer",
            "nsclc patients treated with erlotinib showed improved pfs in lung adenocarcinoma"
        )
        assert delta > 0

    def test_negative_delta_when_other_cancer_dominates(self):
        from app.retrieve_faiss import cancer_evidence_adjust
        delta, scores = cancer_evidence_adjust(
            "Lung_Cancer",
            "thyroid carcinoma papillary ptc follicular medullary atc"
        )
        assert delta < 0

    def test_returns_tuple(self):
        from app.retrieve_faiss import cancer_evidence_adjust
        result = cancer_evidence_adjust("Colon_Cancer", "colorectal cancer crc")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestEntityBonusScore:
    def test_returns_zero_for_no_entities(self):
        from app.retrieve_faiss import entity_bonus_score
        assert entity_bonus_score([], "some text") == 0.0

    def test_returns_positive_for_hit(self):
        from app.retrieve_faiss import entity_bonus_score
        score = entity_bonus_score(["KRAS"], "KRAS mutation in colorectal cancer")
        assert score > 0.0

    def test_capped_at_max(self):
        from app.retrieve_faiss import entity_bonus_score
        from app.config import CFG
        cap = CFG["retrieval"]["entity_bonus_cap"]
        # many entities all present
        ents = ["KRAS", "BRAF", "EGFR", "ALK", "RET", "MET", "NRAS", "PIK3CA", "PTEN", "TP53"]
        score = entity_bonus_score(ents, " ".join(e.lower() for e in ents))
        assert score <= cap


# ─────────────────────────────────────────────────────────────
# Integration tests — require FAISS index
# ─────────────────────────────────────────────────────────────
class TestRetrieveOutputFormat:
    @skip_if_no_index
    def test_returns_list(self):
        from app.retrieve_faiss import retrieve
        results = retrieve("EGFR mutations in lung cancer", top_k=5)
        assert isinstance(results, list)

    @skip_if_no_index
    def test_respects_top_k(self):
        from app.retrieve_faiss import retrieve
        results = retrieve("KRAS colon cancer", top_k=3)
        assert len(results) <= 3

    @skip_if_no_index
    def test_each_result_has_required_keys(self):
        from app.retrieve_faiss import retrieve
        results = retrieve("EGFR mutations in NSCLC", top_k=3)
        required_keys = {"score", "text", "paper_id", "section", "intent"}
        for r in results:
            for key in required_keys:
                assert key in r, f"Missing key: {key}"

    @skip_if_no_index
    def test_results_sorted_by_score_descending(self):
        from app.retrieve_faiss import retrieve
        results = retrieve("KRAS mutations in colorectal cancer", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @skip_if_no_index
    def test_cancer_gating_lung(self):
        from app.retrieve_faiss import retrieve
        results = retrieve("EGFR mutations in lung cancer NSCLC", top_k=5)
        # all results should have lung cancer evidence
        for r in results:
            text = r.get("text", "").lower()
            assert any(k in text for k in ["lung", "nsclc", "pulmonary"]), \
                f"Expected lung cancer content, got: {text[:100]}"

    @skip_if_no_index
    def test_dedup_max_per_paper(self):
        from app.retrieve_faiss import retrieve
        results = retrieve("EGFR NSCLC treatment", top_k=10, max_per_paper=1)
        paper_ids = [r["paper_id"] for r in results]
        assert len(paper_ids) == len(set(paper_ids)), "Duplicate paper_ids found"