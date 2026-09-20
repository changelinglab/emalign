"""Tests for the features module."""

import numpy as np
import pytest

from emalign.features import (
    NUM_FEATURES,
    feature_diff_vector,
    feature_distance,
    get_features,
    segment_ipa,
)


class TestGetFeatures:
    def test_known_segment(self):
        """Test feature extraction for a known IPA segment."""
        features = get_features("p")
        assert features is not None
        assert features.shape == (NUM_FEATURES,)
        assert all(f in {-1, 0, 1} for f in features)
    
    def test_unknown_segment_returns_none(self):
        """Test that unknown segments return None."""
        features = get_features("🔥")
        assert features is None
    
    def test_identical_segments_have_same_features(self):
        """Test that identical segments produce identical features."""
        f1 = get_features("t")
        f2 = get_features("t")
        assert f1 is not None and f2 is not None
        np.testing.assert_array_equal(f1, f2)


class TestSegmentIpa:
    def test_simple_word(self):
        """Test segmentation of a simple IPA word."""
        segs = segment_ipa("pat")
        assert len(segs) >= 3  # At least p, a, t
    
    def test_empty_string(self):
        """Test segmentation of empty string."""
        segs = segment_ipa("")
        assert segs == []
    
    def test_complex_segment(self):
        """Test segmentation with affricates."""
        segs = segment_ipa("t͡ʃ")
        # Should be treated as one segment
        assert len(segs) >= 1


class TestFeatureDistance:
    def test_identical_segments_zero_distance(self):
        """Test that identical segments have zero distance."""
        weights = np.ones(NUM_FEATURES) / NUM_FEATURES
        dist = feature_distance("p", "p", weights)
        assert dist == 0.0
    
    def test_different_segments_positive_distance(self):
        """Test that different segments have positive distance."""
        weights = np.ones(NUM_FEATURES) / NUM_FEATURES
        dist = feature_distance("p", "b", weights)
        assert dist > 0.0
    
    def test_unknown_segment_penalty(self):
        """Test that unknown segments receive a penalty."""
        weights = np.ones(NUM_FEATURES) / NUM_FEATURES
        dist = feature_distance("p", "🔥", weights)
        assert dist == 2.0  # Max penalty


class TestFeatureDiffVector:
    def test_identical_segments_zero_diff(self):
        """Test that identical segments have zero difference vector."""
        diff = feature_diff_vector("k", "k")
        assert diff is not None
        np.testing.assert_array_equal(diff, np.zeros(NUM_FEATURES))
    
    def test_different_segments_nonzero_diff(self):
        """Test that different segments have nonzero difference."""
        diff = feature_diff_vector("p", "m")
        assert diff is not None
        assert diff.sum() > 0
    
    def test_unknown_segment_returns_none(self):
        """Test that unknown segments return None."""
        diff = feature_diff_vector("p", "🔥")
        assert diff is None
