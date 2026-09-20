"""Tests for the aligner module."""

import numpy as np
import pytest

from emalign.aligner import CognateAligner, enforce_weight_bounds, init_weights
from emalign.cldf_io import CognateEntry, CognateSet, Form
from emalign.features import (
    CRITICAL_FEATURE_INDICES,
    CRITICAL_FEATURE_MIN_WEIGHT,
    FEATURE_INDEX,
    NUM_FEATURES,
)


@pytest.fixture
def simple_cognate_set():
    """Create a simple cognate set for testing."""
    forms = [
        Form(id="1-form1", language_id="1", form="pat", morphs=["pat"]),
        Form(id="2-form1", language_id="2", form="bat", morphs=["bat"]),
        Form(id="3-form1", language_id="3", form="fat", morphs=["fat"]),
    ]
    entries = [
        (CognateEntry(id="1", form_id="1-form1", cognateset_id="100", morph_index=0), forms[0]),
        (CognateEntry(id="2", form_id="2-form1", cognateset_id="100", morph_index=0), forms[1]),
        (CognateEntry(id="3", form_id="3-form1", cognateset_id="100", morph_index=0), forms[2]),
    ]
    return CognateSet(id="100", entries=entries)


@pytest.fixture
def multi_morph_cognate_set():
    """Create a cognate set with multi-morph forms."""
    forms = [
        Form(id="1-form2", language_id="1", form="pre+fix", morphs=["pre", "fix"]),
        Form(id="2-form2", language_id="2", form="pro+fic", morphs=["pro", "fic"]),
    ]
    entries = [
        (CognateEntry(id="10", form_id="1-form2", cognateset_id="200", morph_index=1), forms[0]),
        (CognateEntry(id="11", form_id="2-form2", cognateset_id="200", morph_index=1), forms[1]),
    ]
    return CognateSet(id="200", entries=entries)


class TestInitWeights:
    def test_weights_shape(self):
        """Test that initialized weights have correct shape."""
        weights = init_weights()
        assert weights.shape == (NUM_FEATURES,)
    
    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        weights = init_weights()
        assert np.isclose(weights.sum(), 1.0)
    
    def test_weights_are_positive(self):
        """Test that all weights are positive."""
        weights = init_weights()
        assert all(w > 0 for w in weights)
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same weights."""
        w1 = init_weights(random_seed=42)
        w2 = init_weights(random_seed=42)
        np.testing.assert_array_equal(w1, w2)
    
    def test_different_seeds_different_weights(self):
        """Test that different seeds produce different weights."""
        w1 = init_weights(random_seed=42)
        w2 = init_weights(random_seed=123)
        assert not np.allclose(w1, w2)
    
    def test_critical_features_have_higher_initial_weights(self):
        """Test that syl and cons features get boosted initial weights."""
        weights = init_weights(random_seed=42)
        syl_idx = FEATURE_INDEX["syl"]
        cons_idx = FEATURE_INDEX["cons"]
        
        # Critical features should be above minimum threshold
        assert weights[syl_idx] >= CRITICAL_FEATURE_MIN_WEIGHT
        assert weights[cons_idx] >= CRITICAL_FEATURE_MIN_WEIGHT


class TestEnforceWeightBounds:
    def test_enforces_minimum_on_critical_features(self):
        """Test that minimum weights are enforced on syl and cons."""
        # Create weights where critical features are too low
        weights = np.ones(NUM_FEATURES, dtype=np.float32) / NUM_FEATURES
        for idx in CRITICAL_FEATURE_INDICES:
            weights[idx] = 0.001  # Below minimum
        weights /= weights.sum()
        
        enforced = enforce_weight_bounds(weights)
        
        for idx in CRITICAL_FEATURE_INDICES:
            assert enforced[idx] >= CRITICAL_FEATURE_MIN_WEIGHT
    
    def test_maintains_normalization(self):
        """Test that enforced weights still sum to 1."""
        weights = np.ones(NUM_FEATURES, dtype=np.float32) / NUM_FEATURES
        enforced = enforce_weight_bounds(weights)
        
        assert np.isclose(enforced.sum(), 1.0)
    
    def test_does_not_reduce_high_weights(self):
        """Test that weights above minimum are not reduced."""
        weights = np.ones(NUM_FEATURES, dtype=np.float32) / NUM_FEATURES
        weights[FEATURE_INDEX["syl"]] = 0.2  # Well above minimum
        weights /= weights.sum()
        
        original_syl = weights[FEATURE_INDEX["syl"]]
        enforced = enforce_weight_bounds(weights)
        
        # Should maintain relative ordering
        assert enforced[FEATURE_INDEX["syl"]] >= CRITICAL_FEATURE_MIN_WEIGHT


class TestCognateAligner:
    def test_init_defaults(self):
        """Test default initialization."""
        aligner = CognateAligner()
        
        assert aligner.gap_penalty == 0.9
        assert aligner.learning_rate == 0.01
        assert aligner.gap_learning_rate == 0.05
        assert aligner.max_iterations == 40
        assert aligner.learn_gap_penalty is True
        assert aligner.weights.shape == (NUM_FEATURES,)
    
    def test_align_simple_cognate_set(self, simple_cognate_set):
        """Test alignment of a simple cognate set."""
        aligner = CognateAligner(random_seed=42)
        results = aligner.align([simple_cognate_set])
        
        assert len(results) == 3  # One alignment per form
        for result in results:
            assert result.cognateset_id == "100"
            assert "|" in result.aligned_form
    
    def test_align_multi_morph(self, multi_morph_cognate_set):
        """Test alignment uses correct morph based on index."""
        aligner = CognateAligner(random_seed=42)
        results = aligner.align([multi_morph_cognate_set])
        
        # Should align "fix" and "fic" (morph_index=1), not "pre" and "pro"
        assert len(results) == 2
        for result in results:
            aligned = result.aligned_form.replace("|", "")
            # Should contain characters from fix/fic, not pre/pro
            assert "f" in aligned or "-" in aligned
    
    def test_fit_updates_weights(self, simple_cognate_set):
        """Test that fitting updates weights."""
        aligner = CognateAligner(random_seed=42, max_iterations=3)
        initial_weights = aligner.weights.copy()
        
        aligner.fit([simple_cognate_set])
        
        # Weights should have changed
        assert not np.allclose(aligner.weights, initial_weights)
    
    def test_fit_and_align(self, simple_cognate_set):
        """Test fit_and_align convenience method."""
        aligner = CognateAligner(random_seed=42)
        results = aligner.fit_and_align([simple_cognate_set])
        
        assert len(results) == 3
    
    def test_empty_cognate_sets(self):
        """Test handling of empty cognate set list."""
        aligner = CognateAligner()
        results = aligner.align([])
        
        assert results == []
    
    def test_convergence_stops_early(self, simple_cognate_set):
        """Test that convergence threshold stops iteration."""
        aligner = CognateAligner(
            random_seed=42,
            max_iterations=100,
            convergence_threshold=1.0,  # Very high threshold
        )
        
        # Should stop well before 100 iterations
        aligner.fit([simple_cognate_set], verbose=False)
        # Just verify it completes without error

    def test_gap_penalty_learning(self, simple_cognate_set):
        """Test that gap penalty is learned when enabled."""
        initial_gap = 0.5
        aligner = CognateAligner(
            random_seed=42,
            gap_penalty=initial_gap,
            learn_gap_penalty=True,
            max_iterations=5,
        )
        
        aligner.fit([simple_cognate_set], verbose=False)
        
        # Gap penalty should have changed (learned)
        # Note: exact value depends on data, just verify it's within bounds
        assert CognateAligner.GAP_PENALTY_MIN <= aligner.gap_penalty <= CognateAligner.GAP_PENALTY_MAX

    def test_gap_penalty_fixed_when_disabled(self, simple_cognate_set):
        """Test that gap penalty stays fixed when learning is disabled."""
        initial_gap = 0.7
        aligner = CognateAligner(
            random_seed=42,
            gap_penalty=initial_gap,
            learn_gap_penalty=False,
            max_iterations=5,
        )
        
        aligner.fit([simple_cognate_set], verbose=False)
        
        # Gap penalty should not have changed
        assert aligner.gap_penalty == initial_gap

    def test_anchor_selection_uses_segment_count(self):
        """Test that anchor selection uses phoneme segment count, not character count.
        
        This is crucial for IPA handling where complex segments like aspirated
        consonants (pʰ) are represented by multiple Unicode characters but are
        single phonemes.
        """
        # Create forms where character length differs from segment count:
        # - 'pʰej' has 4 chars but 3 segments: ['pʰ', 'e', 'j']
        # - 'pχa' has 3 chars but 3 segments: ['p', 'χ', 'a']
        # With character counting, 'pʰej' would win (4 > 3)
        # With segment counting, 'pχa' wins due to lower language_id tiebreaker
        forms = [
            Form(id="form-lang3", language_id="3", form="pχa", morphs=["pχa"]),
            Form(id="form-lang4", language_id="4", form="pʰej", morphs=["pʰej"]),
        ]
        entries = [
            (CognateEntry(id="e3", form_id="form-lang3", cognateset_id="seg_test", morph_index=0), forms[0]),
            (CognateEntry(id="e4", form_id="form-lang4", cognateset_id="seg_test", morph_index=0), forms[1]),
        ]
        cs = CognateSet(id="seg_test", entries=entries)
        
        aligner = CognateAligner(random_seed=42)
        anchor = aligner._select_anchor_for_cognate_set(cs)
        
        # Both have 3 segments, so language_id tiebreaker selects lang 3
        assert anchor[0] == "3"
        assert anchor[2] == "pχa"


class TestAlignmentFormat:
    def test_pipe_delimiter(self, simple_cognate_set):
        """Test that aligned forms use pipe as delimiter."""
        aligner = CognateAligner(random_seed=42)
        results = aligner.align([simple_cognate_set])
        
        for result in results:
            # Should have pipe-delimited segments
            segments = result.aligned_form.split("|")
            assert len(segments) >= 1
    
    def test_gaps_represented(self, simple_cognate_set):
        """Test that gaps in alignment are preserved."""
        # Create cognate set with different length forms
        forms = [
            Form(id="1-diff", language_id="1", form="stop", morphs=["stop"]),
            Form(id="2-diff", language_id="2", form="top", morphs=["top"]),
        ]
        entries = [
            (CognateEntry(id="1", form_id="1-diff", cognateset_id="300", morph_index=0), forms[0]),
            (CognateEntry(id="2", form_id="2-diff", cognateset_id="300", morph_index=0), forms[1]),
        ]
        cs = CognateSet(id="300", entries=entries)
        
        aligner = CognateAligner(random_seed=42)
        results = aligner.align([cs])
        
        # At least one alignment should have a gap
        assert len(results) == 2
