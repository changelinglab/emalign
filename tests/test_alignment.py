"""Tests for the alignment module."""

import numpy as np
import pytest

from emalign.alignment import (
    AlignedPair,
    GAP,
    alignment_probability,
    collect_alignment_statistics,
    merge_pairwise_alignments,
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


class TestMergePairwiseAlignments:
    """Tests for multi-sequence alignment merging."""
    
    def test_single_alignment_no_gaps(self):
        """Test merging a single alignment with no gaps."""
        anchor = ["s", "i"]
        aligned = AlignedPair(seq1=["s", "i"], seq2=["s", "i"], score=0.0)
        
        result = merge_pairwise_alignments([aligned], anchor)
        
        assert len(result) == 2  # guide + 1 target
        assert result[0] == ["s", "i"]  # guide
        assert result[1] == ["s", "i"]  # target
    
    def test_different_gap_positions_merged(self):
        """Test that different gap insertions are merged consistently."""
        anchor = ["s", "i"]
        # Target 1: s-x-i (gap inserted between s and i on anchor side)
        align1 = AlignedPair(seq1=["s", GAP, "i"], seq2=["s", "x", "i"], score=1.0)
        # Target 2: s-i (no gap)
        align2 = AlignedPair(seq1=["s", "i"], seq2=["s", "i"], score=0.0)
        
        result = merge_pairwise_alignments([align1, align2], anchor)
        
        assert len(result) == 3  # guide + 2 targets
        # Guide should have gap to accommodate both alignments
        assert result[0] == ["s", GAP, "i"]  # guide with gap
        assert result[1] == ["s", "x", "i"]  # target 1 with x
        assert result[2] == ["s", GAP, "i"]  # target 2 with gap
    
    def test_x_not_in_vowel_column(self):
        """
        Test that /x/ (consonant) and vowels don't share columns.
        
        This is a regression test for the issue where 'sxɯ' forms
        had /x/ incorrectly appearing in a column with vowels.
        
        Scenario: anchor is 'sxɯ' (3 segments), targets are 'si' (2 segments)
        The pairwise alignment aligns s-s, then x aligns with gap, ɯ aligns with i.
        """
        anchor = ["s", "x", "ɯ"]  # 'sxɯ' - anchor with /x/
        # Target 1: 'si' aligns as s|x|-  vs  s|-|i (pairwise)
        align1 = AlignedPair(
            seq1=["s", "x", "ɯ"], 
            seq2=["s", GAP, "i"], 
            score=1.0
        )
        # Target 2: 'sɐ' aligns as s|x|ɯ vs s|-|ɐ (pairwise)
        align2 = AlignedPair(
            seq1=["s", "x", "ɯ"], 
            seq2=["s", GAP, "ɐ"],
            score=1.0
        )
        
        result = merge_pairwise_alignments([align1, align2], anchor)
        
        # All results should have same length
        assert all(len(r) == len(result[0]) for r in result)
        
        guide = result[0]
        target1 = result[1]
        target2 = result[2]
        
        # Find column with x in guide
        x_col = None
        for i, seg in enumerate(guide):
            if seg == "x":
                x_col = i
                break
        
        assert x_col is not None, "x should be in guide"
        
        # In the x column, targets should have gaps (not vowels)
        assert target1[x_col] == GAP, f"Column with x should have gap in target1, got {target1[x_col]}"
        assert target2[x_col] == GAP, f"Column with x should have gap in target2, got {target2[x_col]}"
        
        # Vowels should be in different column (the ɯ column)
        vowel_col = None
        for i, seg in enumerate(guide):
            if seg == "ɯ":
                vowel_col = i
                break
        
        assert vowel_col is not None, "ɯ should be in guide"
        assert vowel_col != x_col, "x and vowels should be in different columns"
        assert target1[vowel_col] == "i", f"Expected 'i' in vowel column, got {target1[vowel_col]}"
        assert target2[vowel_col] == "ɐ", f"Expected 'ɐ' in vowel column, got {target2[vowel_col]}"
    
    def test_empty_alignments_returns_anchor(self):
        """Test that empty alignment list returns just the anchor."""
        anchor = ["p", "a", "t"]
        result = merge_pairwise_alignments([], anchor)
        
        assert result == [anchor]
