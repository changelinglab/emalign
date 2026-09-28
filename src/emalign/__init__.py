"""
emalign: EM-based alignment of cognate sets in comparative dictionaries.

This package provides tools for generating phoneme alignments of cognate forms
using expectation-maximization to learn feature weights from PanPhon's 24
articulatory features.

Usage:
    # As a library
    from emalign import align
    alignments = align("path/to/cldf/", "output.csv", verbose=True)

    # Or using components directly
    from emalign import CognateAligner, load_cldf_dataset, write_alignments
    forms, cognate_sets, languages = load_cldf_dataset("path/to/cldf/")
    aligner = CognateAligner()
    alignments = aligner.fit_and_align(cognate_sets)
    write_alignments(alignments, "output.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emalign.cldf_io import AlignmentResult

__version__ = "0.1.0"

from emalign.aligner import CognateAligner, select_best_anchor_language
from emalign.cldf_io import (
    AlignmentResult,
    CognateSet,
    Form,
    Language,
    load_cldf_dataset,
    write_alignments,
)


def align(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    langs: str | list[str] | set[str] | None = None,
    gap_penalty: float = 0.9,
    no_learn_gap: bool = False,
    gap_learning_rate: float = 0.05,
    learning_rate: float = 0.01,
    max_iterations: int = 40,
    convergence_threshold: float = 1e-4,
    seed: int | None = None,
    select_anchor: bool = False,
    verbose: bool = False,
) -> list[AlignmentResult]:
    """
    Generate phoneme alignments for cognate sets using EM-learned feature weights.

    This function provides a programmatic API equivalent to the `emalign` CLI.
    It loads a CLDF dataset, learns feature weights via EM, generates alignments,
    and optionally writes the results to a CSV file.

    Args:
        input_path: Path to CLDF dataset (metadata JSON or directory containing it).
        output_path: Path for output alignments CSV file. If None, results are
            only returned, not written to disk.
        langs: Language filter. Can be:
            - A comma-separated string: "lang1,lang2,lang3"
            - A list: ["lang1", "lang2", "lang3"]
            - A set: {"lang1", "lang2", "lang3"}
            Values can be language IDs or Glottocodes.
            If None, all languages are included.
        gap_penalty: Initial cost for gaps in alignment (default: 0.9).
        no_learn_gap: If True, use fixed gap penalty instead of learning it.
        gap_learning_rate: SGD learning rate for gap penalty updates (default: 0.05).
        learning_rate: SGD learning rate for feature weight updates (default: 0.01).
        max_iterations: Maximum EM iterations (default: 40).
        convergence_threshold: Stop when weight change is below this (default: 1e-4).
        seed: Random seed for reproducibility.
        select_anchor: If True, try all languages as anchors and select the best.
        verbose: If True, print progress information.

    Returns:
        List of AlignmentResult objects, each containing:
            - form_id: The ID of the aligned form
            - cognateset_id: The ID of the cognate set
            - aligned_form: Pipe-delimited aligned segments (e.g., "p|a|t|-")

    Raises:
        FileNotFoundError: If the CLDF dataset path does not exist.

    Example:
        >>> from emalign import align
        >>> # Basic usage
        >>> alignments = align("data/cldf_tangkhulic/", "output.csv")
        >>> # With options
        >>> alignments = align(
        ...     "data/cldf_tangkhulic/",
        ...     verbose=True,
        ...     seed=42,
        ...     max_iterations=20,
        ... )
        >>> # Filter by languages
        >>> alignments = align(
        ...     "data/cldf_tangkhulic/",
        ...     langs="kach1286,chal1279.1",
        ...     select_anchor=True,
        ... )
    """
    # Parse language filter
    language_ids: set[str] | None = None
    if langs is not None:
        if isinstance(langs, str):
            language_ids = {lang.strip() for lang in langs.split(",")}
        elif isinstance(langs, (list, set)):
            language_ids = set(langs)
        else:
            raise TypeError(
                f"langs must be str, list, or set, not {type(langs).__name__}"
            )

    # Load data
    if verbose:
        print(f"Loading CLDF dataset from {input_path}...")
        if language_ids:
            print(f"Filtering to languages: {', '.join(sorted(language_ids))}")

    forms, cognate_sets, languages = load_cldf_dataset(input_path, language_ids)

    if verbose:
        lang_count = len({f.language_id for f in forms.values()})
        print(
            f"Loaded {len(forms)} forms from {lang_count} languages "
            f"and {len(cognate_sets)} cognate sets."
        )

    # Create aligner
    aligner = CognateAligner(
        gap_penalty=gap_penalty,
        learning_rate=learning_rate,
        gap_learning_rate=gap_learning_rate,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        learn_gap_penalty=not no_learn_gap,
        random_seed=seed,
    )

    # Optionally select best anchor language
    if select_anchor:
        if verbose:
            print("Selecting optimal anchor language...")
        best_lang, best_prob, best_weights, best_gap = select_best_anchor_language(
            cognate_sets, aligner
        )
        if verbose:
            print(
                f"Best anchor language: {best_lang} "
                f"(mean_log_prob={best_prob:.4f}, gap_penalty={best_gap:.4f})"
            )
        aligner.weights = best_weights
        aligner.gap_penalty = best_gap

    # Fit model and generate alignments
    if verbose:
        print("Learning feature weights and gap penalty...")

    alignments = aligner.fit_and_align(cognate_sets, verbose=verbose)

    if verbose:
        print(f"Generated {len(alignments)} alignments.")
        print(f"Final gap penalty: {aligner.gap_penalty:.4f}")

    # Write output if path provided
    if output_path is not None:
        write_alignments(alignments, output_path)
        if verbose:
            print(f"Wrote alignments to {output_path}")

    return alignments


__all__ = [
    # Main entry point (CLI-equivalent API)
    "align",
    # Core classes
    "CognateAligner",
    # Data structures
    "AlignmentResult",
    "CognateSet",
    "Form",
    "Language",
    # I/O functions
    "load_cldf_dataset",
    "write_alignments",
    # Utilities
    "select_best_anchor_language",
    # Version
    "__version__",
]
