"""
Command-line interface for emalign.

Usage:
    emalign <input_cldf_path> <output_csv_path> [options]
"""

import argparse
import sys
from pathlib import Path

from emalign.aligner import CognateAligner, select_best_anchor_language
from emalign.cldf_io import load_cldf_dataset, write_alignments


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="emalign",
        description="Generate phoneme alignments for cognate sets using EM-learned feature weights.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to CLDF dataset (metadata JSON or directory containing it).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path for output alignments CSV file.",
    )
    parser.add_argument(
        "-l", "--langs",
        type=str,
        default=None,
        help="Comma-separated list of language IDs or Glottocodes to include. "
             "If not specified, all languages are included.",
    )
    parser.add_argument(
        "--gap-penalty",
        type=float,
        default=0.9,
        help="Initial gap penalty for alignment (default: 0.9).",
    )
    parser.add_argument(
        "--no-learn-gap",
        action="store_true",
        help="Disable learning gap penalty (use fixed value).",
    )
    parser.add_argument(
        "--gap-learning-rate",
        type=float,
        default=0.05,
        help="SGD learning rate for gap penalty (default: 0.05).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate for feature weights (default: 0.01).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=40,
        help="Maximum EM iterations (default: 40).",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-4,
        help="Convergence threshold for weight change (default: 1e-4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--select-anchor",
        action="store_true",
        help="Try all languages as anchors and select the best one.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    
    args = parser.parse_args(argv)
    
    # Parse language filter
    language_ids: set[str] | None = None
    if args.langs:
        language_ids = {lang.strip() for lang in args.langs.split(",")}
    
    # Load data
    if args.verbose:
        print(f"Loading CLDF dataset from {args.input}...")
        if language_ids:
            print(f"Filtering to languages: {', '.join(sorted(language_ids))}")
    
    try:
        forms, cognate_sets, languages = load_cldf_dataset(args.input, language_ids)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        lang_count = len({f.language_id for f in forms.values()})
        print(f"Loaded {len(forms)} forms from {lang_count} languages and {len(cognate_sets)} cognate sets.")
    
    # Create aligner
    aligner = CognateAligner(
        gap_penalty=args.gap_penalty,
        learning_rate=args.learning_rate,
        gap_learning_rate=args.gap_learning_rate,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        learn_gap_penalty=not args.no_learn_gap,
        random_seed=args.seed,
    )
    
    # Optionally select best anchor language
    if args.select_anchor:
        if args.verbose:
            print("Selecting optimal anchor language...")
        best_lang, best_prob, best_weights, best_gap = select_best_anchor_language(
            cognate_sets, aligner
        )
        if args.verbose:
            print(f"Best anchor language: {best_lang} (mean_log_prob={best_prob:.4f}, gap_penalty={best_gap:.4f})")
        aligner.weights = best_weights
        aligner.gap_penalty = best_gap
    
    # Fit model and generate alignments
    if args.verbose:
        print("Learning feature weights and gap penalty...")
    
    alignments = aligner.fit_and_align(cognate_sets, verbose=args.verbose)
    
    if args.verbose:
        print(f"Generated {len(alignments)} alignments.")
        print(f"Final gap penalty: {aligner.gap_penalty:.4f}")
    
    # Write output
    write_alignments(alignments, args.output)
    
    if args.verbose:
        print(f"Wrote alignments to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
