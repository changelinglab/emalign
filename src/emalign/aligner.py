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
    align_to_anchor,
    alignment_probability,
    collect_alignment_statistics,
    GAP,
)
from emalign.cldf_io import AlignmentResult, CognateSet, iter_cognate_morphs
from emalign.features import NUM_FEATURES, segment_ipa


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
    
    Weights are initialized as 1/24 + uniform noise in [-0.01, 0.01].
    
    Args:
        random_seed: Optional seed for reproducibility.
        
    Returns:
        Weights array of shape (NUM_FEATURES,).
    """
    rng = np.random.default_rng(random_seed)
    base = 1.0 / NUM_FEATURES
    noise = rng.uniform(-0.01, 0.01, NUM_FEATURES)
    weights = base + noise
    # Normalize to sum to 1
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return weights.astype(np.float32)


class CognateAligner:
    """
    EM-based cognate aligner with anchor language selection.
    
    This class learns feature weights from cognate data and produces
    alignments by selecting anchor languages and aligning other forms to them.
    """
    
    def __init__(
        self,
        gap_penalty: float = 1.0,
        learning_rate: float = 0.01,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        random_seed: int | None = None,
    ):
        """
        Initialize the aligner.
        
        Args:
            gap_penalty: Cost for gaps in alignment.
            learning_rate: SGD learning rate for weight updates.
            max_iterations: Maximum EM iterations.
            convergence_threshold: Stop when weight change is below this.
            random_seed: Random seed for reproducibility.
        """
        self.gap_penalty = gap_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
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
    ) -> np.ndarray:
        """
        M-step with SGD: Update weights based on alignment statistics.
        
        The gradient is computed from feature differences in alignments.
        Features with larger differences should have lower weights to reduce
        their contribution to the distance metric, encouraging more
        "phonologically natural" alignments.
        
        Args:
            alignments: Current alignments from E-step.
            
        Returns:
            Updated weights.
        """
        feature_diffs_sum, num_pairs = collect_alignment_statistics(alignments)
        
        if num_pairs == 0:
            return self.weights
        
        # Gradient: features with higher average diff should be down-weighted
        avg_diffs = feature_diffs_sum / num_pairs
        
        # Update rule: decrease weight for high-diff features
        # w_new = w - lr * (avg_diff - mean(avg_diff))
        # This pushes weights toward features that vary less across cognates
        gradient = avg_diffs - avg_diffs.mean()
        new_weights = self.weights - self.learning_rate * gradient
        
        # Project to simplex (ensure non-negative and sum to 1)
        new_weights = np.maximum(new_weights, 1e-6)
        new_weights /= new_weights.sum()
        
        return new_weights.astype(np.float32)
    
    def fit(
        self,
        cognate_sets: list[CognateSet],
        verbose: bool = False,
    ) -> "CognateAligner":
        """
        Fit the aligner by learning feature weights via EM.
        
        Args:
            cognate_sets: List of cognate sets to learn from.
            verbose: If True, print progress information.
            
        Returns:
            self (for method chaining).
        """
        for iteration in range(self.max_iterations):
            old_weights = self.weights.copy()
            
            # E-step: compute alignments
            alignments = self._e_step(cognate_sets)
            
            if not alignments:
                if verbose:
                    print(f"Iteration {iteration + 1}: No alignments produced")
                break
            
            # M-step: update weights
            self.weights = self._m_step_sgd(alignments)
            
            # Check convergence
            weight_change = np.abs(self.weights - old_weights).max()
            
            if verbose:
                mean_prob = self._compute_mean_log_prob(cognate_sets)
                print(
                    f"Iteration {iteration + 1}: "
                    f"mean_log_prob={mean_prob:.4f}, "
                    f"max_weight_change={weight_change:.6f}"
                )
            
            if weight_change < self.convergence_threshold:
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
) -> tuple[str, float, np.ndarray]:
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
        Tuple of (best_language_id, best_mean_prob, best_weights).
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
    
    for lang in candidate_languages:
        # Create aligner that forces this language as anchor
        test_aligner = CognateAligner(
            gap_penalty=aligner.gap_penalty,
            learning_rate=aligner.learning_rate,
            max_iterations=aligner.max_iterations,
            convergence_threshold=aligner.convergence_threshold,
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
    
    return best_lang, best_prob, best_weights
