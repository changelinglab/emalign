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
        "--gap-penalty",
        type=float,
        default=1.0,
        help="Gap penalty for alignment (default: 1.0).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum EM iterations (default: 10).",
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
    
    # Load data
    if args.verbose:
        print(f"Loading CLDF dataset from {args.input}...")
    
    try:
        forms, cognate_sets = load_cldf_dataset(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Loaded {len(forms)} forms and {len(cognate_sets)} cognate sets.")
    
    # Create aligner
    aligner = CognateAligner(
        gap_penalty=args.gap_penalty,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        random_seed=args.seed,
    )
    
    # Optionally select best anchor language
    if args.select_anchor:
        if args.verbose:
            print("Selecting optimal anchor language...")
        best_lang, best_prob, best_weights = select_best_anchor_language(
            cognate_sets, aligner
        )
        if args.verbose:
            print(f"Best anchor language: {best_lang} (mean_log_prob={best_prob:.4f})")
        aligner.weights = best_weights
    
    # Fit model and generate alignments
    if args.verbose:
        print("Learning feature weights and generating alignments...")
    
    alignments = aligner.fit_and_align(cognate_sets, verbose=args.verbose)
    
    if args.verbose:
        print(f"Generated {len(alignments)} alignments.")
    
    # Write output
    write_alignments(alignments, args.output)
    
    if args.verbose:
        print(f"Wrote alignments to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
