"""Tests for the alignment module."""

import numpy as np
import pytest

from emalign.alignment import (
    AlignedPair,
    GAP,
    alignment_probability,
    collect_alignment_statistics,
    weighted_levenshtein_align,
)
from emalign.features import NUM_FEATURES


@pytest.fixture
def uniform_weights():
    """Uniform feature weights."""
    return np.ones(NUM_FEATURES, dtype=np.float32) / NUM_FEATURES


class TestWeightedLevenshteinAlign:
    def test_identical_sequences(self, uniform_weights):
        """Test alignment of identical sequences."""
        seq = ["p", "a", "t"]
        result = weighted_levenshtein_align(seq, seq, uniform_weights)
        
        assert result.seq1 == seq
        assert result.seq2 == seq
        assert result.score == 0.0
    
    def test_empty_sequences(self, uniform_weights):
        """Test alignment of empty sequences."""
        result = weighted_levenshtein_align([], [], uniform_weights)
        
        assert result.seq1 == []
        assert result.seq2 == []
        assert result.score == 0.0
    
    def test_one_empty_sequence(self, uniform_weights):
        """Test alignment when one sequence is empty."""
        seq = ["p", "a"]
        result = weighted_levenshtein_align(seq, [], uniform_weights, gap_penalty=1.0)
        
        assert len(result.seq1) == 2
        assert all(s == GAP for s in result.seq2)
        assert result.score == 2.0  # Two gaps
    
    def test_different_length_sequences(self, uniform_weights):
        """Test alignment of sequences with different lengths."""
        seq1 = ["p", "a", "t"]
        seq2 = ["b", "a"]
        result = weighted_levenshtein_align(seq1, seq2, uniform_weights, gap_penalty=1.0)
        
        # Should have at least one gap
        assert GAP in result.seq1 or GAP in result.seq2
        assert len(result.seq1) == len(result.seq2)
    
    def test_alignment_is_symmetric_length(self, uniform_weights):
        """Test that aligned sequences have equal length."""
        seq1 = ["m", "a", "n"]
        seq2 = ["p", "a", "n", "s"]
        result = weighted_levenshtein_align(seq1, seq2, uniform_weights)
        
        assert len(result.seq1) == len(result.seq2)


class TestCollectAlignmentStatistics:
    def test_empty_alignments(self):
        """Test statistics from empty alignment list."""
        stats = collect_alignment_statistics([])
        
        np.testing.assert_array_equal(stats.feature_diffs_sum, np.zeros(NUM_FEATURES))
        assert stats.num_substitutions == 0
        assert stats.num_gaps == 0
    
    def test_single_alignment_with_identical_segments(self):
        """Test statistics from alignment of identical segments."""
        alignment = AlignedPair(seq1=["p"], seq2=["p"], score=0.0)
        stats = collect_alignment_statistics([alignment])
        
        np.testing.assert_array_equal(stats.feature_diffs_sum, np.zeros(NUM_FEATURES))
        assert stats.num_substitutions == 1
        assert stats.num_gaps == 0
    
    def test_gaps_are_skipped(self):
        """Test that gaps are not counted in substitution statistics."""
        alignment = AlignedPair(seq1=["p", GAP], seq2=[GAP, "t"], score=2.0)
        stats = collect_alignment_statistics([alignment])
        
        assert stats.num_substitutions == 0  # Both positions have gaps
        assert stats.num_gaps == 2  # Both positions are gaps


class TestAlignmentProbability:
    def test_zero_score_high_probability(self):
        """Test that zero score gives highest log-probability."""
        alignment = AlignedPair(seq1=["p"], seq2=["p"], score=0.0)
        weights = np.ones(NUM_FEATURES) / NUM_FEATURES
        
        log_prob = alignment_probability(alignment, weights)
        assert log_prob == 0.0
    
    def test_higher_score_lower_probability(self):
        """Test that higher scores give lower log-probability."""
        align1 = AlignedPair(seq1=["p"], seq2=["p"], score=0.0)
        align2 = AlignedPair(seq1=["p"], seq2=["b"], score=1.0)
        weights = np.ones(NUM_FEATURES) / NUM_FEATURES
        
        prob1 = alignment_probability(align1, weights)
        prob2 = alignment_probability(align2, weights)
        
        assert prob1 > prob2
