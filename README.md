# emalign-phonology

A Python package for aligning cognate sets in comparative dictionaries using articulatory features, with weights learned via expectation maximization.

[![PyPI version](https://badge.fury.io/py/emalign-phonology.svg)](https://badge.fury.io/py/emalign-phonology)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`emalign-phonology` generates phoneme alignments for cognate forms in CLDF-formatted comparative dictionaries. It uses:

- **PanPhon's 24 articulatory features** to compute phoneme similarity
- **Expectation-Maximization (EM)** with SGD to learn optimal feature weights
- **Anchor-based alignment** to handle multi-language cognate sets efficiently

The package can be used both as a **command-line utility** and as a **Python library** with an API that mirrors the CLI interface.

## Installation

```bash
pip install emalign-phonology
```

Or for development:

```bash
git clone https://github.com/changelinglab/emalign.git
cd emalign
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Basic usage
emalign input_cldf/ output_alignments.csv

# With options
emalign input_cldf/ output.csv --verbose --seed 42 --max-iterations 20

# Select optimal anchor language
emalign input_cldf/ output.csv --select-anchor --verbose
```

### Python API (Recommended)

```python
from emalign import align

# Simple usage - equivalent to CLI
alignments = align("path/to/cldf/", "output.csv", verbose=True)

# With options - mirrors CLI arguments
alignments = align(
    "path/to/cldf/",
    "output.csv",
    langs="kach1286,chal1279.1",  # Filter by language
    seed=42,
    max_iterations=20,
    select_anchor=True,
    verbose=True,
)

# Without writing to file - just get the results
alignments = align("path/to/cldf/")
for a in alignments:
    print(f"{a.form_id}: {a.aligned_form}")
```

## Usage

### Command Line Interface

```bash
emalign <input_cldf_path> <output_csv_path> [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-l, --langs LIST` | Comma-separated list of language IDs or Glottocodes to include | All languages |
| `--gap-penalty FLOAT` | Initial gap penalty for alignment | 0.9 |
| `--no-learn-gap` | Disable learning gap penalty (use fixed value) | Learn gap |
| `--gap-learning-rate FLOAT` | SGD learning rate for gap penalty | 0.05 |
| `--learning-rate FLOAT` | SGD learning rate for feature weights | 0.01 |
| `--max-iterations INT` | Maximum EM iterations | 40 |
| `--convergence-threshold FLOAT` | Stop when weight change is below this | 1e-4 |
| `--seed INT` | Random seed for reproducibility | None |
| `--select-anchor` | Try all languages as anchors and select the best | False |
| `-v, --verbose` | Print progress information | False |

### Python Library API

#### High-Level API: `align()`

The `align()` function provides a CLI-equivalent interface for programmatic use:

```python
from emalign import align

alignments = align(
    input_path,           # Path to CLDF dataset
    output_path=None,     # Optional: write results to CSV
    langs=None,           # Language filter (string, list, or set)
    gap_penalty=0.9,
    no_learn_gap=False,
    gap_learning_rate=0.05,
    learning_rate=0.01,
    max_iterations=40,
    convergence_threshold=1e-4,
    seed=None,
    select_anchor=False,
    verbose=False,
)
```

#### Low-Level API

For more control, use the component classes directly:

```python
from emalign import (
    CognateAligner,
    load_cldf_dataset,
    write_alignments,
    select_best_anchor_language,
)

# Load CLDF data
forms, cognate_sets, languages = load_cldf_dataset("path/to/cldf/")

# Create and configure aligner
aligner = CognateAligner(
    gap_penalty=0.9,
    learning_rate=0.01,
    max_iterations=40,
    random_seed=42,
)

# Optionally select the best anchor language
best_lang, best_prob, best_weights, best_gap = select_best_anchor_language(
    cognate_sets, aligner
)
aligner.weights = best_weights
aligner.gap_penalty = best_gap

# Fit model and generate alignments
alignments = aligner.fit_and_align(cognate_sets, verbose=True)

# Write output
write_alignments(alignments, "alignments.csv")
```

### Language Filtering

You can restrict alignments to a subset of languages using either language IDs or Glottocodes:

```bash
# CLI: Filter by language IDs
emalign input_cldf/ output.csv --langs 1,2,3,4

# CLI: Filter by Glottocodes
emalign input_cldf/ output.csv --langs kach1286,chal1279.1,east2902
```

```python
# Python: Various filter formats
align("data/", langs="kach1286,chal1279.1")      # String
align("data/", langs=["kach1286", "chal1279.1"]) # List
align("data/", langs={"kach1286", "chal1279.1"}) # Set
```

## Data Formats

### Input Format

The input must be a CLDF dataset with:

- `forms.csv`: Lexical forms with `ID`, `Language_ID`, `Form` columns
- `cognates.csv`: Cognate judgments with `Form_ID`, `Cognateset_ID`, `Morph_Index` columns
- `languages.csv`: Language metadata with `ID`, `Name`, `Glottocode` columns
- `*-metadata.json`: CLDF metadata file

Forms should be in IPA with morphs separated by `+` (e.g., `pre+fix`).

### Output Format

The output is a CLDF-compliant CSV with columns:

| Column | Description |
|--------|-------------|
| `ID` | Unique alignment identifier |
| `Form_ID` | Reference to the original form |
| `Cognateset_ID` | Reference to the cognate set |
| `Aligned_Form` | Pipe-delimited aligned segments (e.g., `p\|a\|t\|-`) |

## Algorithm

1. **Initialization**: Feature weights initialized with small random perturbations, with critical features (syllabic, consonantal) receiving higher initial weights
2. **E-step**: Compute optimal alignments using weighted Levenshtein distance based on articulatory feature differences
3. **M-step**: Update weights via SGD based on alignment statistics; optionally learn gap penalty
4. **Iteration**: Repeat until convergence or max iterations reached

The alignment cost between segments is computed as:

```
cost = weights · |features(seg1) - features(seg2)|
```

where `features()` returns PanPhon's 24-dimensional articulatory feature vector.

## API Reference

### Classes

- **`CognateAligner`**: Main aligner class with EM-based weight learning
- **`AlignmentResult`**: Dataclass holding alignment results (form_id, cognateset_id, aligned_form)
- **`CognateSet`**: Dataclass representing a cognate set with its entries
- **`Form`**: Dataclass representing a lexical form
- **`Language`**: Dataclass representing a language with metadata

### Functions

- **`align()`**: High-level function providing CLI-equivalent API
- **`load_cldf_dataset()`**: Load a CLDF dataset from path
- **`write_alignments()`**: Write alignment results to CSV
- **`select_best_anchor_language()`**: Find optimal anchor language for alignment

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@software{emalign,
  title = {emalign-phonology: EM-based cognate alignment},
  author = {Changeling Lab},
  url = {https://github.com/changelinglab/emalign},
  year = {2024}
}
```
