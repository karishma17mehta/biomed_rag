# tests/test_config.py
"""
Unit tests for app/config.py
Tests that config.yaml loads correctly and all expected keys exist.
"""

import pytest
from app.config import CFG, retrieval, generation, agent_cfg, models, paths


class TestConfigLoads:
    def test_cfg_is_dict(self):
        assert isinstance(CFG, dict)

    def test_all_sections_present(self):
        for section in ["models", "retrieval", "chunking", "agent", "generation", "evaluation", "paths"]:
            assert section in CFG, f"Missing section: {section}"


class TestModelsConfig:
    def test_embedding_model_set(self):
        assert models.embedding == "text-embedding-3-large"

    def test_generator_model_set(self):
        assert models.generator == "gpt-4o-mini"

    def test_temperature_is_zero(self):
        assert models.generator_temperature == 0


class TestRetrievalConfig:
    def test_top_k_positive(self):
        assert retrieval.top_k > 0

    def test_bm25_div_positive(self):
        assert retrieval.bm25_div > 0

    def test_rerank_weight_in_range(self):
        assert 0 < retrieval.rerank_weight < 1

    def test_bm25_weights_ordered(self):
        # weight with entities should be >= weight without
        assert retrieval.bm25_weight_with_entities >= retrieval.bm25_weight_no_entities

    def test_cancer_penalties_negative(self):
        assert retrieval.cancer_absent_strong_penalty < 0
        assert retrieval.cancer_absent_weak_penalty < 0
        assert retrieval.cancer_generic_penalty < 0


class TestGenerationConfig:
    def test_max_contexts_positive(self):
        assert generation.max_contexts > 0

    def test_sentence_range_valid(self):
        assert generation.min_answer_sentences < generation.max_answer_sentences

    def test_score_weights_positive(self):
        assert generation.entity_score_weight > 0
        assert generation.cancer_score_weight > 0
        assert generation.intent_score_weight > 0

    def test_clinical_bonuses_correct_sign(self):
        assert generation.clinical_match_bonus > 0
        assert generation.clinical_mismatch_penalty < 0

    def test_ae_bonuses_correct_sign(self):
        assert generation.ae_match_bonus > 0
        assert generation.ae_mismatch_penalty < 0


class TestAgentConfig:
    def test_threshold_in_range(self):
        assert 0 < agent_cfg.local_score_threshold < 2

    def test_min_hits_positive(self):
        assert agent_cfg.local_min_hits > 0

    def test_pubmed_results_positive(self):
        assert agent_cfg.pubmed_max_results > 0


class TestPathsConfig:
    def test_all_paths_present(self):
        assert paths.faiss_index
        assert paths.meta_file
        assert paths.bm25_file

    def test_paths_are_strings(self):
        assert isinstance(paths.faiss_index, str)
        assert isinstance(paths.meta_file, str)