#!/usr/bin/env python3
"""
Visualize cognate alignments from emalign output.

This script reads an alignments.csv file and displays each cognate set
as a nicely formatted table with proper handling of combining characters.

Usage:
    python visemalign.py alignments.csv
    python visemalign.py alignments.csv --forms forms.csv --languages languages.csv
"""

import argparse
import csv
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def display_width(s: str) -> int:
    """
    Calculate the display width of a string, accounting for combining characters.
    
    Combining characters (like ͡ U+0361) have zero display width.
    Wide characters (CJK, etc.) have width 2.
    """
    width = 0
    for char in s:
        category = unicodedata.category(char)
        if category in ("Mn", "Me", "Mc"):  # Mark, Nonspacing / Enclosing / Spacing Combining
            continue  # Zero width
        east_asian_width = unicodedata.east_asian_width(char)
        if east_asian_width in ("F", "W"):  # Fullwidth or Wide
            width += 2
        else:
            width += 1
    return width


def pad_to_width(s: str, target_width: int, align: str = "left") -> str:
    """
    Pad a string to a target display width, accounting for combining characters.
    
    Args:
        s: The string to pad.
        target_width: The desired display width.
        align: "left", "right", or "center".
        
    Returns:
        The padded string.
    """
    current_width = display_width(s)
    padding_needed = max(0, target_width - current_width)
    
    if align == "right":
        return " " * padding_needed + s
    elif align == "center":
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return " " * left_pad + s + " " * right_pad
    else:  # left
        return s + " " * padding_needed


def load_alignments(filepath: Path) -> dict[str, list[dict]]:
    """
    Load alignments from CSV and group by cognateset ID.
    
    Returns:
        Dict mapping cognateset_id -> list of alignment records.
    """
    cognate_groups: dict[str, list[dict]] = defaultdict(list)
    
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cogset_id = row["Cognateset_ID"]
            cognate_groups[cogset_id].append({
                "id": row["ID"],
                "form_id": row["Form_ID"],
                "aligned_form": row["Aligned_Form"],
            })
    
    return dict(cognate_groups)


def load_forms(filepath: Path) -> dict[str, dict]:
    """Load forms from CLDF forms.csv file."""
    forms = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            forms[row["ID"]] = {
                "language_id": row.get("Language_ID", ""),
                "form": row.get("Form", ""),
                "description": row.get("Description", ""),
            }
    return forms


def load_languages(filepath: Path) -> dict[str, str]:
    """Load language names from CLDF languages.csv file."""
    languages = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            languages[row["ID"]] = row.get("Name", row["ID"])
    return languages


def format_cognate_table(
    cogset_id: str,
    alignments: list[dict],
    forms: dict[str, dict] | None = None,
    languages: dict[str, str] | None = None,
) -> str:
    """
    Format a single cognate set as a table.
    
    Args:
        cogset_id: The cognate set ID.
        alignments: List of alignment records for this cognate set.
        forms: Optional dict of form metadata.
        languages: Optional dict of language names.
        
    Returns:
        Formatted table as a string.
    """
    lines = []
    
    # Parse aligned forms into segments
    parsed_alignments = []
    for alignment in alignments:
        segments = alignment["aligned_form"].split("|")
        form_id = alignment["form_id"]
        
        # Get metadata if available
        lang_name = ""
        original_form = ""
        if forms and form_id in forms:
            lang_id = forms[form_id]["language_id"]
            original_form = forms[form_id]["form"]
            if languages and lang_id in languages:
                lang_name = languages[lang_id]
            else:
                lang_name = lang_id
        
        parsed_alignments.append({
            "form_id": form_id,
            "segments": segments,
            "lang_name": lang_name,
            "original_form": original_form,
        })
    
    if not parsed_alignments:
        return f"Cognate Set {cogset_id}: (empty)\n"
    
    # Determine the maximum number of alignment positions
    max_positions = max(len(a["segments"]) for a in parsed_alignments)
    
    # Calculate column widths for each position
    col_widths = []
    for pos in range(max_positions):
        max_width = 1  # Minimum width
        for alignment in parsed_alignments:
            if pos < len(alignment["segments"]):
                seg = alignment["segments"][pos]
                max_width = max(max_width, display_width(seg))
        col_widths.append(max_width)
    
    # Calculate label column width
    if any(a["lang_name"] for a in parsed_alignments):
        label_width = max(display_width(a["lang_name"]) for a in parsed_alignments)
        label_width = max(label_width, 8)  # Minimum for "Language"
    else:
        label_width = max(display_width(a["form_id"]) for a in parsed_alignments)
        label_width = max(label_width, 7)  # Minimum for "Form ID"
    
    # Build header
    header_label = "Language" if any(a["lang_name"] for a in parsed_alignments) else "Form ID"
    lines.append(f"╔{'═' * (label_width + 2)}╦{'═' * (sum(col_widths) + len(col_widths) * 3 - 1)}╗")
    lines.append(f"║ Cognate Set: {cogset_id}{' ' * max(0, label_width + sum(col_widths) + len(col_widths) * 3 - len(f'Cognate Set: {cogset_id}') - 1)}║")
    lines.append(f"╠{'═' * (label_width + 2)}╬{'╦'.join('═' * (w + 2) for w in col_widths)}╣")
    
    # Column headers (position numbers)
    pos_headers = [pad_to_width(str(i + 1), col_widths[i], "center") for i in range(max_positions)]
    lines.append(f"║ {pad_to_width(header_label, label_width)} ║ {' ║ '.join(pos_headers)} ║")
    lines.append(f"╠{'═' * (label_width + 2)}╬{'╬'.join('═' * (w + 2) for w in col_widths)}╣")
    
    # Data rows
    for alignment in parsed_alignments:
        label = alignment["lang_name"] if alignment["lang_name"] else alignment["form_id"]
        cells = []
        for pos in range(max_positions):
            if pos < len(alignment["segments"]):
                seg = alignment["segments"][pos]
            else:
                seg = ""
            cells.append(pad_to_width(seg, col_widths[pos], "center"))
        
        lines.append(f"║ {pad_to_width(label, label_width)} ║ {' ║ '.join(cells)} ║")
    
    # Footer
    lines.append(f"╚{'═' * (label_width + 2)}╩{'╩'.join('═' * (w + 2) for w in col_widths)}╝")
    
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="visemalign",
        description="Visualize cognate alignments from emalign output.",
    )
    parser.add_argument(
        "alignments",
        type=Path,
        help="Path to alignments.csv file.",
    )
    parser.add_argument(
        "--forms",
        type=Path,
        default=None,
        help="Path to CLDF forms.csv for additional metadata.",
    )
    parser.add_argument(
        "--languages",
        type=Path,
        default=None,
        help="Path to CLDF languages.csv for language names.",
    )
    parser.add_argument(
        "--cogset",
        type=str,
        default=None,
        help="Display only the specified cognate set ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cognate sets displayed.",
    )
    
    args = parser.parse_args(argv)
    
    # Load alignments
    if not args.alignments.exists():
        print(f"Error: File not found: {args.alignments}", file=sys.stderr)
        return 1
    
    alignments = load_alignments(args.alignments)
    
    if not alignments:
        print("No alignments found in file.", file=sys.stderr)
        return 1
    
    # Load optional metadata
    forms = None
    languages = None
    
    if args.forms and args.forms.exists():
        forms = load_forms(args.forms)
    
    if args.languages and args.languages.exists():
        languages = load_languages(args.languages)
    
    # Filter by cognate set if specified
    if args.cogset:
        if args.cogset not in alignments:
            print(f"Error: Cognate set {args.cogset} not found.", file=sys.stderr)
            return 1
        alignments = {args.cogset: alignments[args.cogset]}
    
    # Apply limit
    cogset_ids = list(alignments.keys())
    if args.limit:
        cogset_ids = cogset_ids[:args.limit]
    
    # Display tables
    for i, cogset_id in enumerate(cogset_ids):
        if i > 0:
            print()  # Blank line between tables
        table = format_cognate_table(
            cogset_id,
            alignments[cogset_id],
            forms=forms,
            languages=languages,
        )
        print(table)
    
    # Summary
    print(f"\nDisplayed {len(cogset_ids)} of {len(alignments)} cognate sets.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
