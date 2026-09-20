"""
Main cognate alignment class with EM-based weight learning.

This module implements the CognateAligner class which:
1. Selects anchor languages using a heuristic
2. Learns feature weights via expectation-maximization with SGD
3. Produces optimal alignments for all cognate sets
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from emalign.alignment import (
    AlignedPair,
    AlignmentStatistics,
    align_to_anchor,
    alignment_probability,
    collect_alignment_statistics,
    GAP,
)
from emalign.cldf_io import AlignmentResult, CognateSet, iter_cognate_morphs
from emalign.features import (
    NUM_FEATURES,
    segment_ipa,
    CRITICAL_FEATURE_INDICES,
    CRITICAL_FEATURE_MIN_WEIGHT,
)


@dataclass
class CognateSetAlignments:
    """Alignments for a single cognate set."""
    cognateset_id: str
    anchor_lang: str
    anchor_morph: str
    alignments: list[tuple[str, str, str, AlignedPair]]  # (lang, form_id, morph, alignment)


def init_weights(random_seed: int | None = None) -> np.ndarray:
    """
    Initialize feature weights with small random perturbations.
    
    Critical features (syl, cons) receive the HIGHEST initial weights to ensure
    proper distinction between vowels, consonants, and glides.
    
    Args:
        random_seed: Optional seed for reproducibility.
        
    Returns:
        Weights array of shape (NUM_FEATURES,).
    """
    rng = np.random.default_rng(random_seed)
    
    # Start with small base weights for non-critical features
    weights = np.full(NUM_FEATURES, 0.02, dtype=np.float64)
    noise = rng.uniform(-0.005, 0.005, NUM_FEATURES)
    weights += noise
    
    # Set critical features (syl, cons) to high values - these should dominate
    for idx in CRITICAL_FEATURE_INDICES:
        weights[idx] = 0.20 + rng.uniform(-0.01, 0.01)
    
    # Normalize to sum to 1
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return weights.astype(np.float32)


def enforce_weight_bounds(weights: np.ndarray) -> np.ndarray:
    """
    Enforce minimum weight bounds on critical features.
    
    This prevents the EM algorithm from down-weighting features that are
    essential for distinguishing phoneme classes (vowels vs consonants).
    
    Uses iterative projection to ensure critical features maintain minimum
    weights even after normalization.
    
    Args:
        weights: Current weights array.
        
    Returns:
        Weights with bounds enforced and renormalized.
    """
    weights = weights.copy().astype(np.float64)
    
    # Iteratively enforce bounds and renormalize
    for _ in range(5):  # Usually converges in 1-2 iterations
        needs_adjustment = False
        for idx in CRITICAL_FEATURE_INDICES:
            if weights[idx] < CRITICAL_FEATURE_MIN_WEIGHT:
                weights[idx] = CRITICAL_FEATURE_MIN_WEIGHT
                needs_adjustment = True
        if not needs_adjustment:
            break
        
        # Renormalize by adjusting non-critical weights
        critical_sum = sum(weights[idx] for idx in CRITICAL_FEATURE_INDICES)
        non_critical_indices = [i for i in range(NUM_FEATURES) if i not in CRITICAL_FEATURE_INDICES]
        non_critical_sum = sum(weights[i] for i in non_critical_indices)
        
        if non_critical_sum > 0:
            target_non_critical = 1.0 - critical_sum
            scale = target_non_critical / non_critical_sum
            for i in non_critical_indices:
                weights[i] *= scale
    
    # Final normalization to ensure sum is exactly 1
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return weights.astype(np.float32)


class CognateAligner:
    """
    EM-based cognate aligner with anchor language selection.
    
    This class learns feature weights and gap penalty from cognate data and
    produces alignments by selecting anchor languages and aligning other forms
    to them.
    """
    
    # Bounds for gap penalty learning
    GAP_PENALTY_MIN = 0.1
    GAP_PENALTY_MAX = 2.0
    
    def __init__(
        self,
        gap_penalty: float = 0.5,
        learning_rate: float = 0.01,
        gap_learning_rate: float = 0.05,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        learn_gap_penalty: bool = True,
        random_seed: int | None = None,
    ):
        """
        Initialize the aligner.
        
        Args:
            gap_penalty: Initial cost for gaps in alignment.
            learning_rate: SGD learning rate for feature weight updates.
            gap_learning_rate: SGD learning rate for gap penalty updates.
            max_iterations: Maximum EM iterations.
            convergence_threshold: Stop when weight change is below this.
            learn_gap_penalty: If True, learn gap penalty empirically.
            random_seed: Random seed for reproducibility.
        """
        self.gap_penalty = gap_penalty
        self.learning_rate = learning_rate
        self.gap_learning_rate = gap_learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.learn_gap_penalty = learn_gap_penalty
        self.random_seed = random_seed
        
        self.weights = init_weights(random_seed)
        self._rng = np.random.default_rng(random_seed)
    
    def _select_anchor_for_cognate_set(
        self,
        cognate_set: CognateSet,
    ) -> tuple[str, str, str] | None:
        """
        Select the anchor language and morph for a cognate set.
        
        Heuristic: Choose the language with the longest morph, breaking ties
        by language frequency in the dataset.
        
        Args:
            cognate_set: A cognate set with forms.
            
        Returns:
            Tuple of (language_id, form_id, morph) for the anchor, or None if empty.
        """
        candidates = []
        for entry, morph in iter_cognate_morphs(cognate_set):
            form = next(f for e, f in cognate_set.entries if e.id == entry.id)
            candidates.append((form.language_id, entry.form_id, morph, len(morph)))
        
        if not candidates:
            return None
        
        # Sort by morph length descending, then by language_id for stability
        candidates.sort(key=lambda x: (-x[3], x[0]))
        return candidates[0][:3]
    
    def _align_cognate_set(
        self,
        cognate_set: CognateSet,
    ) -> CognateSetAlignments | None:
        """
        Align all forms in a cognate set to a selected anchor.
        
        Args:
            cognate_set: A cognate set with forms.
            
        Returns:
            CognateSetAlignments with all pairwise alignments, or None if < 2 forms.
        """
        anchor = self._select_anchor_for_cognate_set(cognate_set)
        if anchor is None:
            return None
        
        anchor_lang, anchor_form_id, anchor_morph = anchor
        anchor_segs = segment_ipa(anchor_morph)
        
        alignments = []
        for entry, morph in iter_cognate_morphs(cognate_set):
            form = next(f for e, f in cognate_set.entries if e.id == entry.id)
            target_segs = segment_ipa(morph)
            aligned = align_to_anchor(anchor_segs, target_segs, self.weights, self.gap_penalty)
            alignments.append((form.language_id, entry.form_id, morph, aligned))
        
        if len(alignments) < 2:
            return None
        
        return CognateSetAlignments(
            cognateset_id=cognate_set.id,
            anchor_lang=anchor_lang,
            anchor_morph=anchor_morph,
            alignments=alignments,
        )
    
    def _compute_mean_log_prob(
        self,
        cognate_sets: list[CognateSet],
    ) -> float:
        """Compute mean log-probability of alignments over all cognate sets."""
        total_log_prob = 0.0
        count = 0
        for cs in cognate_sets:
            cs_alignments = self._align_cognate_set(cs)
            if cs_alignments is None:
                continue
            for _, _, _, aligned in cs_alignments.alignments:
                total_log_prob += alignment_probability(aligned, self.weights)
                count += 1
        return total_log_prob / max(count, 1)
    
    def _e_step(
        self,
        cognate_sets: list[CognateSet],
    ) -> list[AlignedPair]:
        """
        E-step: Compute alignments with current weights.
        
        Returns all aligned pairs from the cognate sets.
        """
        all_alignments = []
        for cs in cognate_sets:
            cs_alignments = self._align_cognate_set(cs)
            if cs_alignments is None:
                continue
            for _, _, _, aligned in cs_alignments.alignments:
                all_alignments.append(aligned)
        return all_alignments
    
    def _m_step_sgd(
        self,
        alignments: list[AlignedPair],
    ) -> tuple[np.ndarray, float]:
        """
        M-step with SGD: Update weights and gap penalty based on alignment statistics.
        
        The gradient is computed from feature differences in alignments.
        Features with larger differences should have lower weights to reduce
        their contribution to the distance metric, encouraging more
        "phonologically natural" alignments.
        
        Gap penalty is adjusted based on comparing gap cost to average
        substitution cost - if gaps are too cheap relative to substitutions,
        increase gap penalty, and vice versa.
        
        Critical features (syl, cons) have minimum weight bounds enforced to
        prevent vowel-consonant misalignments.
        
        Args:
            alignments: Current alignments from E-step.
            
        Returns:
            Tuple of (updated_weights, updated_gap_penalty).
        """
        stats = collect_alignment_statistics(alignments, self.weights)
        
        if stats.num_substitutions == 0:
            return self.weights, self.gap_penalty
        
        # Update feature weights
        avg_diffs = stats.feature_diffs_sum / stats.num_substitutions
        gradient = avg_diffs - avg_diffs.mean()
        new_weights = self.weights - self.learning_rate * gradient
        
        # Project to simplex (ensure non-negative and sum to 1)
        new_weights = np.maximum(new_weights, 1e-6)
        new_weights /= new_weights.sum()
        
        # Enforce minimum bounds on critical features (syl, cons)
        new_weights = enforce_weight_bounds(new_weights)
        
        # Update gap penalty if learning is enabled
        new_gap_penalty = self.gap_penalty
        if self.learn_gap_penalty and stats.total_positions > 0:
            # Compute gap ratio (what fraction of alignment positions are gaps)
            gap_ratio = stats.num_gaps / stats.total_positions
            
            # Target: gaps should be used when substitution would be expensive
            # If gap_penalty < avg_sub_cost, gaps are too cheap -> increase
            # If gap_penalty > avg_sub_cost, gaps are too expensive -> decrease
            # But we also consider the gap ratio - too many gaps suggests penalty too low
            
            # Gradient: positive if gap_penalty should increase
            # Use a target gap ratio around 15-25% as reasonable for cognates
            target_gap_ratio = 0.20
            ratio_error = gap_ratio - target_gap_ratio
            
            # Also compare gap cost to substitution cost
            cost_gradient = stats.avg_substitution_cost - self.gap_penalty
            
            # Combined gradient: increase gap if ratio too high OR if cheaper than subs
            combined_gradient = 0.5 * ratio_error - 0.5 * cost_gradient
            
            new_gap_penalty = self.gap_penalty + self.gap_learning_rate * combined_gradient
            
            # Enforce bounds
            new_gap_penalty = max(self.GAP_PENALTY_MIN, 
                                  min(self.GAP_PENALTY_MAX, new_gap_penalty))
        
        return new_weights, new_gap_penalty
    
    def fit(
        self,
        cognate_sets: list[CognateSet],
        verbose: bool = False,
    ) -> "CognateAligner":
        """
        Fit the aligner by learning feature weights and gap penalty via EM.
        
        Args:
            cognate_sets: List of cognate sets to learn from.
            verbose: If True, print progress information.
            
        Returns:
            self (for method chaining).
        """
        for iteration in range(self.max_iterations):
            old_weights = self.weights.copy()
            old_gap_penalty = self.gap_penalty
            
            # E-step: compute alignments
            alignments = self._e_step(cognate_sets)
            
            if not alignments:
                if verbose:
                    print(f"Iteration {iteration + 1}: No alignments produced")
                break
            
            # M-step: update weights and gap penalty
            self.weights, self.gap_penalty = self._m_step_sgd(alignments)
            
            # Check convergence
            weight_change = np.abs(self.weights - old_weights).max()
            gap_change = abs(self.gap_penalty - old_gap_penalty)
            
            if verbose:
                mean_prob = self._compute_mean_log_prob(cognate_sets)
                print(
                    f"Iteration {iteration + 1}: "
                    f"mean_log_prob={mean_prob:.4f}, "
                    f"max_weight_change={weight_change:.6f}, "
                    f"gap_penalty={self.gap_penalty:.4f}"
                )
            
            if weight_change < self.convergence_threshold and gap_change < self.convergence_threshold:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def align(self, cognate_sets: list[CognateSet]) -> list[AlignmentResult]:
        """
        Generate alignments for all cognate sets.
        
        Args:
            cognate_sets: List of cognate sets to align.
            
        Returns:
            List of AlignmentResult objects for CLDF output.
        """
        results = []
        
        for cs in cognate_sets:
            cs_alignments = self._align_cognate_set(cs)
            if cs_alignments is None:
                continue
            
            for lang_id, form_id, morph, aligned in cs_alignments.alignments:
                # Format aligned sequence with pipe delimiters
                aligned_form = "|".join(aligned.seq2)
                results.append(AlignmentResult(
                    form_id=form_id,
                    cognateset_id=cs.id,
                    aligned_form=aligned_form,
                ))
        
        return results
    
    def fit_and_align(
        self,
        cognate_sets: list[CognateSet],
        verbose: bool = False,
    ) -> list[AlignmentResult]:
        """
        Fit the model and generate alignments in one call.
        
        Args:
            cognate_sets: List of cognate sets.
            verbose: If True, print progress.
            
        Returns:
            List of AlignmentResult objects.
        """
        self.fit(cognate_sets, verbose=verbose)
        return self.align(cognate_sets)


def select_best_anchor_language(
    cognate_sets: list[CognateSet],
    aligner: CognateAligner,
    candidate_languages: list[str] | None = None,
) -> tuple[str, float, np.ndarray, float]:
    """
    Select the anchor language that maximizes mean alignment probability.
    
    This implements the anchor selection algorithm described in AGENTS.md:
    try each language as anchor, learn weights, and select the one with
    highest mean probability.
    
    Args:
        cognate_sets: List of cognate sets.
        aligner: Base aligner (will be cloned for each candidate).
        candidate_languages: Optional list of language IDs to try. If None,
            extracts all languages from the cognate sets.
            
    Returns:
        Tuple of (best_language_id, best_mean_prob, best_weights, best_gap_penalty).
    """
    # Collect all languages
    if candidate_languages is None:
        langs = set()
        for cs in cognate_sets:
            for entry, form in cs.entries:
                langs.add(form.language_id)
        candidate_languages = sorted(langs)
    
    best_lang = None
    best_prob = float("-inf")
    best_weights = None
    best_gap_penalty = aligner.gap_penalty
    
    for lang in candidate_languages:
        # Create aligner that forces this language as anchor
        test_aligner = CognateAligner(
            gap_penalty=aligner.gap_penalty,
            learning_rate=aligner.learning_rate,
            gap_learning_rate=aligner.gap_learning_rate,
            max_iterations=aligner.max_iterations,
            convergence_threshold=aligner.convergence_threshold,
            learn_gap_penalty=aligner.learn_gap_penalty,
            random_seed=aligner.random_seed,
        )
        
        # Filter cognate sets to those containing this language
        filtered_sets = [
            cs for cs in cognate_sets
            if any(f.language_id == lang for _, f in cs.entries)
        ]
        
        if not filtered_sets:
            continue
        
        # Fit and evaluate
        test_aligner.fit(filtered_sets, verbose=False)
        mean_prob = test_aligner._compute_mean_log_prob(filtered_sets)
        
        if mean_prob > best_prob:
            best_lang = lang
            best_prob = mean_prob
            best_weights = test_aligner.weights.copy()
            best_gap_penalty = test_aligner.gap_penalty
    
    return best_lang, best_prob, best_weights, best_gap_penalty
