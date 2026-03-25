# tests/test_query_router.py
"""
Unit tests for app/query_router.py
Tests intent routing, entity extraction, and cancer type extraction.
"""

import pytest
from app.query_router import infer_intent, extract_entities, extract_cancer_type


# ─────────────────────────────────────────────────────────────
# Intent routing
# ─────────────────────────────────────────────────────────────

class TestInferIntent:
    def test_resistance_intent(self):
        q = "What mechanisms drive resistance to EGFR inhibitors?"
        assert infer_intent(q) == "resistance"

    def test_mutations_intent(self):
        q = "What is the role of KRAS mutations in colon cancer?"
        assert infer_intent(q) == "mutations"

    def test_trial_design_intent(self):
        q = "What phase III randomized trials exist for NSCLC?"
        assert infer_intent(q) == "trial_design"

    def test_efficacy_outcomes_intent(self):
        # Use ORR which unambiguously triggers efficacy_outcomes
        q = "What is the ORR and response rate for this treatment?"
        assert infer_intent(q) == "efficacy_outcomes"

    def test_adverse_events_intent(self):
        q = "What grade 3 toxicity and adverse events were observed?"
        assert infer_intent(q) == "adverse_events"

    def test_immune_therapy_intent(self):
        # Use checkpoint term without PD abbreviation to avoid pharmacokinetics match
        q = "Which immune checkpoint inhibitors are used in cancer treatment?"
        assert infer_intent(q) == "immune_therapy"

    def test_models_cell_lines_intent(self):
        q = "What xenograft models are used to study colorectal cancer?"
        assert infer_intent(q) == "models_cell_lines"

    def test_returns_string(self):
        assert isinstance(infer_intent("any query"), str)

    def test_empty_query_returns_general(self):
        assert infer_intent("") == "general"


# ─────────────────────────────────────────────────────────────
# Entity extraction
# ─────────────────────────────────────────────────────────────

class TestExtractEntities:
    def test_extracts_gene_kras(self):
        q = "How do KRAS mutations affect colon cancer?"
        assert "KRAS" in extract_entities(q)

    def test_extracts_gene_egfr(self):
        q = "What is the role of EGFR in lung cancer?"
        assert "EGFR" in extract_entities(q)

    def test_extracts_gene_braf(self):
        q = "What is the role of BRAF mutations in papillary thyroid carcinoma?"
        assert "BRAF" in extract_entities(q)

    def test_extracts_checkpoint_pd1(self):
        q = "Which PD-1 inhibitors are used in lung cancer?"
        ents = extract_entities(q)
        assert any("PD" in e for e in ents)

    def test_extracts_drug(self):
        q = "What is the efficacy of erlotinib in NSCLC?"
        ents = extract_entities(q)
        assert any("erlotinib" in e.lower() for e in ents)

    def test_returns_list(self):
        assert isinstance(extract_entities("KRAS mutation"), list)

    def test_empty_query_returns_empty(self):
        assert extract_entities("") == []

    def test_no_false_positives_common_words(self):
        q = "What is the role of DNA in cancer?"
        ents = extract_entities(q)
        # DNA is in GENE_STOP so should not appear
        assert "DNA" not in ents


# ─────────────────────────────────────────────────────────────
# Cancer type extraction
# ─────────────────────────────────────────────────────────────

class TestExtractCancerType:
    def test_detects_colon_cancer(self):
        q = "How do KRAS mutations drive resistance in colorectal cancer?"
        assert extract_cancer_type(q) == "Colon_Cancer"

    def test_detects_lung_cancer(self):
        q = "What are the treatment outcomes in NSCLC patients?"
        assert extract_cancer_type(q) == "Lung_Cancer"

    def test_detects_thyroid_cancer(self):
        q = "What is the role of BRAF in papillary thyroid carcinoma?"
        assert extract_cancer_type(q) == "Thyroid_Cancer"

    def test_detects_colon_via_crc(self):
        assert extract_cancer_type("CRC prognosis") == "Colon_Cancer"

    def test_detects_thyroid_via_ptc(self):
        assert extract_cancer_type("PTC recurrence factors") == "Thyroid_Cancer"

    def test_detects_lung_via_nsclc(self):
        assert extract_cancer_type("NSCLC survival") == "Lung_Cancer"

    def test_no_cancer_returns_empty(self):
        assert extract_cancer_type("What are biomarkers?") == ""

    def test_returns_string(self):
        assert isinstance(extract_cancer_type("lung cancer"), str)