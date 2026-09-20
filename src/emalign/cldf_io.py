"""
CLDF data loading and output generation.

This module handles reading comparative dictionaries in CLDF format and
writing alignment results to CLDF-compliant CSV files.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pycldf import Dataset


@dataclass
class Form:
    """A lexical form with its metadata."""
    id: str
    language_id: str
    form: str
    morphs: list[str]  # Individual morphs split by '+'


@dataclass
class CognateEntry:
    """A cognate set entry linking a form to a cognate set."""
    id: str
    form_id: str
    cognateset_id: str
    morph_index: int


@dataclass
class CognateSet:
    """A cognate set with all associated forms."""
    id: str
    entries: list[tuple[CognateEntry, Form]]


def load_cldf_dataset(path: str | Path) -> tuple[dict[str, Form], list[CognateSet]]:
    """
    Load a CLDF dataset from the given path.
    
    Args:
        path: Path to the CLDF metadata JSON file or directory containing it.
        
    Returns:
        A tuple of (forms_dict, cognate_sets) where forms_dict maps form IDs
        to Form objects and cognate_sets is a list of CognateSet objects.
    """
    path = Path(path)
    if path.is_dir():
        metadata_files = list(path.glob("*-metadata.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No CLDF metadata file found in {path}")
        path = metadata_files[0]
    
    dataset = Dataset.from_metadata(path)
    
    # Load forms
    forms: dict[str, Form] = {}
    for row in dataset["FormTable"]:
        form_str = row["Form"]
        morphs = form_str.split("+") if form_str else []
        forms[row["ID"]] = Form(
            id=row["ID"],
            language_id=str(row["Language_ID"]),
            form=form_str,
            morphs=morphs,
        )
    
    # Load cognates and group by cognateset
    cognate_groups: dict[str, list[CognateEntry]] = {}
    for row in dataset["CognateTable"]:
        entry = CognateEntry(
            id=str(row["ID"]),
            form_id=row["Form_ID"],
            cognateset_id=str(row["Cognateset_ID"]),
            morph_index=row["Morph_Index"],
        )
        if entry.cognateset_id not in cognate_groups:
            cognate_groups[entry.cognateset_id] = []
        cognate_groups[entry.cognateset_id].append(entry)
    
    # Build cognate sets with form references
    cognate_sets: list[CognateSet] = []
    for cogset_id, entries in cognate_groups.items():
        form_entries = []
        for entry in entries:
            if entry.form_id in forms:
                form_entries.append((entry, forms[entry.form_id]))
        if form_entries:
            cognate_sets.append(CognateSet(id=cogset_id, entries=form_entries))
    
    return forms, cognate_sets


@dataclass
class AlignmentResult:
    """Result of aligning a single form within a cognate set."""
    form_id: str
    cognateset_id: str
    aligned_form: str  # Pipe-delimited segments


def write_alignments(
    alignments: list[AlignmentResult],
    output_path: str | Path,
) -> None:
    """
    Write alignment results to a CLDF-compliant CSV file.
    
    Args:
        alignments: List of AlignmentResult objects to write.
        output_path: Path for the output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Form_ID", "Cognateset_ID", "Aligned_Form"])
        for i, alignment in enumerate(alignments, start=1):
            writer.writerow([
                i,
                alignment.form_id,
                alignment.cognateset_id,
                alignment.aligned_form,
            ])


def iter_cognate_morphs(cognate_set: CognateSet) -> Iterator[tuple[CognateEntry, str]]:
    """
    Iterate over the morphs in a cognate set, yielding (entry, morph) pairs.
    
    Each entry is paired with the specific morph indicated by its morph_index.
    """
    for entry, form in cognate_set.entries:
        if 0 <= entry.morph_index < len(form.morphs):
            yield entry, form.morphs[entry.morph_index]
