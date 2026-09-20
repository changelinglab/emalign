"""
Pairwise sequence alignment with weighted phonetic features.

This module implements the weighted Levenshtein alignment algorithm where
substitution costs are computed based on PanPhon feature differences.
"""

import numpy as np
from dataclasses import dataclass

from emalign.features import (
    feature_distance,
    feature_diff_vector,
    segment_ipa,
    NUM_FEATURES,
)

# Gap symbol for alignments
GAP = "-"


@dataclass
class AlignedPair:
    """A pair of aligned sequences."""
    seq1: list[str]  # Segments from first sequence (may include GAP)
    seq2: list[str]  # Segments from second sequence (may include GAP)
    score: float     # Alignment score (lower is better)


def weighted_levenshtein_align(
    seq1: list[str],
    seq2: list[str],
    weights: np.ndarray,
    gap_penalty: float = 1.0,
) -> AlignedPair:
    """
    Compute optimal alignment using weighted Levenshtein distance.
    
    The substitution cost between segments s1 and s2 is:
        cost = weights · |features(s1) - features(s2)|
    
    Args:
        seq1: First sequence of IPA segments.
        seq2: Second sequence of IPA segments.
        weights: Feature weights of shape (NUM_FEATURES,).
        gap_penalty: Cost for inserting a gap.
        
    Returns:
        AlignedPair with aligned sequences and total score.
    """
    n, m = len(seq1), len(seq2)
    
    # DP matrix for costs
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    
    # Initialize first row and column with gap penalties
    for i in range(1, n + 1):
        dp[i, 0] = i * gap_penalty
    for j in range(1, m + 1):
        dp[0, j] = j * gap_penalty
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = feature_distance(seq1[i - 1], seq2[j - 1], weights)
            dp[i, j] = min(
                dp[i - 1, j - 1] + sub_cost,  # Substitution/match
                dp[i - 1, j] + gap_penalty,    # Delete from seq1
                dp[i, j - 1] + gap_penalty,    # Insert to seq1
            )
    
    # Traceback to recover alignment
    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = feature_distance(seq1[i - 1], seq2[j - 1], weights)
            if np.isclose(dp[i, j], dp[i - 1, j - 1] + sub_cost):
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and np.isclose(dp[i, j], dp[i - 1, j] + gap_penalty):
            aligned1.append(seq1[i - 1])
            aligned2.append(GAP)
            i -= 1
        else:
            aligned1.append(GAP)
            aligned2.append(seq2[j - 1])
            j -= 1
    
    return AlignedPair(
        seq1=list(reversed(aligned1)),
        seq2=list(reversed(aligned2)),
        score=dp[n, m],
    )


def align_to_anchor(
    anchor_seq: list[str],
    target_seq: list[str],
    weights: np.ndarray,
    gap_penalty: float = 1.0,
) -> AlignedPair:
    """
    Align a target sequence to an anchor sequence.
    
    This is a convenience wrapper around weighted_levenshtein_align that
    returns the alignment in a consistent format (anchor, target).
    """
    return weighted_levenshtein_align(anchor_seq, target_seq, weights, gap_penalty)


def collect_alignment_statistics(
    alignments: list[AlignedPair],
) -> tuple[np.ndarray, int]:
    """
    Collect feature difference statistics from a set of alignments.
    
    This computes the sum of feature differences for all aligned segment pairs
    (excluding gaps), used for the M-step of EM.
    
    Args:
        alignments: List of aligned pairs.
        
    Returns:
        Tuple of (feature_diffs_sum, num_pairs) where feature_diffs_sum is
        the sum of |f1 - f2| over all aligned pairs and num_pairs is the count.
    """
    feature_diffs_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    num_pairs = 0
    
    for alignment in alignments:
        for s1, s2 in zip(alignment.seq1, alignment.seq2):
            if s1 == GAP or s2 == GAP:
                continue
            diff = feature_diff_vector(s1, s2)
            if diff is not None:
                feature_diffs_sum += diff
                num_pairs += 1
    
    return feature_diffs_sum, num_pairs


def alignment_probability(alignment: AlignedPair, weights: np.ndarray) -> float:
    """
    Compute the probability of an alignment given feature weights.
    
    Uses a log-linear model where:
        P(alignment) ∝ exp(-score)
    
    Args:
        alignment: An aligned pair.
        weights: Feature weights.
        
    Returns:
        Log-probability of the alignment (unnormalized).
    """
    return -alignment.score
