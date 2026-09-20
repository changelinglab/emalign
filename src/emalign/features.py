"""
Phoneme feature extraction using PanPhon.

This module provides the interface to PanPhon's 24 articulatory features
for computing phoneme similarity.
"""

import numpy as np
import panphon
from functools import lru_cache

# PanPhon's 24 articulatory features
FEATURE_NAMES = [
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid",
    "voi", "sg", "cg", "ant", "cor", "distr", "lab", "hi", "lo",
    "back", "round", "velaric", "tense", "long", "hitone", "hireg",
]
NUM_FEATURES = len(FEATURE_NAMES)

# Feature name to index mapping
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Critical features that distinguish major phoneme classes (vowels/consonants/glides)
# These should have the HIGHEST weights to prevent cross-class alignments
CRITICAL_FEATURES = {"syl", "cons"}
CRITICAL_FEATURE_INDICES = [FEATURE_INDEX[f] for f in CRITICAL_FEATURES]
CRITICAL_FEATURE_MIN_WEIGHT = 0.15  # Minimum weight for each critical feature (15% of total)

# Syllabicity mismatch penalty: consonants and vowels should NEVER align
# This penalty is added when [syl] differs, making C-V alignment always worse than a gap
SYLLABICITY_MISMATCH_PENALTY = 2.0

# Global feature table instance
_ft: panphon.FeatureTable | None = None


def get_feature_table() -> panphon.FeatureTable:
    """Get or create the global PanPhon feature table."""
    global _ft
    if _ft is None:
        _ft = panphon.FeatureTable()
    return _ft


@lru_cache(maxsize=4096)
def get_features(segment: str) -> np.ndarray | None:
    """
    Get the feature vector for a phoneme segment.
    
    Args:
        segment: An IPA segment string.
        
    Returns:
        A numpy array of shape (NUM_FEATURES,) with values in {-1, 0, 1},
        or None if the segment is not found in PanPhon.
    """
    ft = get_feature_table()
    features = ft.word_fts(segment)
    if not features:
        return None
    # Use first segment's features (handles complex segments)
    fts = features[0]
    return np.array([fts[name] for name in FEATURE_NAMES], dtype=np.float32)


def segment_ipa(form: str) -> list[str]:
    """
    Segment an IPA string into individual phoneme segments.
    
    Args:
        form: An IPA form string.
        
    Returns:
        A list of IPA segment strings.
    """
    ft = get_feature_table()
    return ft.ipa_segs(form)


def feature_distance(
    seg1: str,
    seg2: str,
    weights: np.ndarray | None = None,
) -> float:
    """
    Compute the weighted feature distance between two segments.
    
    The distance is computed as:
        distance = sum(weights * |features1 - features2|) + syllabicity_penalty
    
    A large penalty is added when segments differ in syllabicity (i.e., when
    aligning a consonant with a vowel), making such alignments always worse
    than inserting a gap.
    
    If one segment is unknown, returns a penalty value.
    
    Args:
        seg1: First IPA segment.
        seg2: Second IPA segment.
        weights: Feature weights of shape (NUM_FEATURES,). If None, uniform weights.
        
    Returns:
        The weighted distance between segments.
    """
    if weights is None:
        weights = np.ones(NUM_FEATURES, dtype=np.float32) / NUM_FEATURES
    
    f1 = get_features(seg1)
    f2 = get_features(seg2)
    
    # Penalty for unknown segments
    if f1 is None or f2 is None:
        return 2.0  # Max possible distance with normalized weights
    
    diff = np.abs(f1 - f2)
    distance = float(np.dot(weights, diff))
    
    # Add syllabicity mismatch penalty: C-V alignments are linguistically wrong
    syl_idx = FEATURE_INDEX["syl"]
    if f1[syl_idx] != f2[syl_idx]:
        distance += SYLLABICITY_MISMATCH_PENALTY
    
    return distance


def feature_diff_vector(seg1: str, seg2: str) -> np.ndarray | None:
    """
    Compute the absolute feature difference vector between two segments.
    
    Args:
        seg1: First IPA segment.
        seg2: Second IPA segment.
        
    Returns:
        Array of shape (NUM_FEATURES,) with |features1 - features2|,
        or None if either segment is unknown.
    """
    f1 = get_features(seg1)
    f2 = get_features(seg2)
    if f1 is None or f2 is None:
        return None
    return np.abs(f1 - f2)
