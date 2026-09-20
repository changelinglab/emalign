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


@dataclass
class AlignmentStatistics:
    """Statistics collected from a set of alignments."""
    feature_diffs_sum: np.ndarray  # Sum of feature diffs for substitutions
    num_substitutions: int          # Number of substitution pairs
    num_gaps: int                   # Total gap count (insertions + deletions)
    total_positions: int            # Total alignment positions
    avg_substitution_cost: float    # Average cost of substitutions


def collect_alignment_statistics(
    alignments: list[AlignedPair],
    weights: np.ndarray | None = None,
) -> AlignmentStatistics:
    """
    Collect feature difference and gap statistics from alignments.
    
    This computes statistics needed for the M-step of EM, including
    feature differences for substitutions and gap counts.
    
    Args:
        alignments: List of aligned pairs.
        weights: Current feature weights (for computing substitution costs).
        
    Returns:
        AlignmentStatistics with feature diffs, substitution/gap counts.
    """
    feature_diffs_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    num_substitutions = 0
    num_gaps = 0
    total_positions = 0
    total_sub_cost = 0.0
    
    for alignment in alignments:
        for s1, s2 in zip(alignment.seq1, alignment.seq2):
            total_positions += 1
            if s1 == GAP or s2 == GAP:
                num_gaps += 1
            else:
                diff = feature_diff_vector(s1, s2)
                if diff is not None:
                    feature_diffs_sum += diff
                    num_substitutions += 1
                    if weights is not None:
                        total_sub_cost += float(np.dot(weights, diff))
    
    avg_sub_cost = total_sub_cost / max(num_substitutions, 1)
    
    return AlignmentStatistics(
        feature_diffs_sum=feature_diffs_sum,
        num_substitutions=num_substitutions,
        num_gaps=num_gaps,
        total_positions=total_positions,
        avg_substitution_cost=avg_sub_cost,
    )


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


def merge_pairwise_alignments(
    alignments: list[AlignedPair],
    anchor_segs: list[str],
) -> list[list[str]]:
    """
    Merge multiple pairwise alignments into consistent multi-sequence columns.
    
    This takes a set of pairwise alignments (anchor vs target) and produces
    a merged alignment where all sequences share the same column structure.
    The approach:
    1. Build a "guide" from the anchor that includes all gaps induced by targets
    2. Map each target's alignment to the guide positions
    
    Args:
        alignments: List of AlignedPair objects from pairwise alignment.
        anchor_segs: The original (ungapped) anchor segments.
        
    Returns:
        List of aligned target sequences, all with consistent columns.
        The first entry is the merged anchor (guide).
    """
    if not alignments:
        return [anchor_segs]
    
    # Each alignment has anchor (seq1) with gaps inserted at different positions
    # We need to merge these into a single guide sequence
    
    # Track gap positions relative to original anchor positions
    # For each alignment, record where gaps were inserted
    anchor_len = len(anchor_segs)
    
    # gap_positions[i] = set of alignments that have a gap before anchor position i
    # We need to track cumulative gaps before each anchor position
    max_gaps_before = [0] * (anchor_len + 1)  # gaps before position 0, 1, ..., n
    
    alignment_gap_maps = []  # For each alignment: list of (anchor_pos, gap_count)
    
    for aligned in alignments:
        # Walk through aligned anchor (seq1) and count gaps before each position
        gaps_before = [0] * (anchor_len + 1)
        anchor_idx = 0
        cumulative_gaps = 0
        
        for seg in aligned.seq1:
            if seg == GAP:
                cumulative_gaps += 1
            else:
                gaps_before[anchor_idx] = cumulative_gaps
                anchor_idx += 1
        # Gaps after last anchor position
        gaps_before[anchor_len] = cumulative_gaps - gaps_before[anchor_len - 1] if anchor_len > 0 else cumulative_gaps
        
        alignment_gap_maps.append(gaps_before)
        
        # Update max gaps needed before each position
        for i in range(anchor_len + 1):
            max_gaps_before[i] = max(max_gaps_before[i], gaps_before[i])
    
    # Build the guide: for each anchor position, insert max_gaps_before[i] gaps
    guide = []
    for i, seg in enumerate(anchor_segs):
        guide.extend([GAP] * max_gaps_before[i])
        guide.append(seg)
    guide.extend([GAP] * max_gaps_before[anchor_len])
    
    # Now map each target sequence to the guide
    results = [guide]  # First entry is the guide/anchor
    
    for aligned, gap_map in zip(alignments, alignment_gap_maps):
        target_aligned = []
        target_iter = iter(aligned.seq2)
        anchor_idx = 0
        
        for i, seg in enumerate(anchor_segs):
            # Add gaps to match guide's gaps before this position
            need_gaps = max_gaps_before[i]
            have_gaps = gap_map[i]
            
            # Copy the gaps from target that correspond to anchor gaps
            for _ in range(have_gaps):
                target_aligned.append(next(target_iter, GAP))
            
            # Add extra gaps if guide has more gaps than this alignment
            for _ in range(need_gaps - have_gaps):
                target_aligned.append(GAP)
            
            # Add the target segment aligned with this anchor position
            target_aligned.append(next(target_iter, GAP))
            anchor_idx += 1
        
        # Handle trailing gaps
        need_trailing = max_gaps_before[anchor_len]
        have_trailing = gap_map[anchor_len]
        for _ in range(have_trailing):
            target_aligned.append(next(target_iter, GAP))
        for _ in range(need_trailing - have_trailing):
            target_aligned.append(GAP)
        
        # Consume any remaining target segments (shouldn't happen in correct alignment)
        for seg in target_iter:
            target_aligned.append(seg)
        
        results.append(target_aligned)
    
    return results
