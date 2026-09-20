# emalign

A Python package for aligning cognate sets in comparative dictionaries using articulatory features, with weights learned via expectation maximization.

## Overview

`emalign` generates phoneme alignments for cognate forms in CLDF-formatted comparative dictionaries. It uses:

- **PanPhon's 24 articulatory features** to compute phoneme similarity
- **Expectation-Maximization (EM)** with SGD to learn optimal feature weights
- **Anchor-based alignment** to handle multi-language cognate sets efficiently

## Installation

```bash
pip install emalign
```

Or for development:

```bash
git clone https://github.com/yourusername/emalign.git
cd emalign
pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
# Basic usage
emalign input_cldf/ output_alignments.csv

# With options
emalign input_cldf/ output.csv --verbose --seed 42 --max-iterations 20

# Select optimal anchor language
emalign input_cldf/ output.csv --select-anchor --verbose
```

### Options

- `--gap-penalty FLOAT`: Gap penalty for alignment (default: 1.0)
- `--learning-rate FLOAT`: SGD learning rate (default: 0.01)
- `--max-iterations INT`: Maximum EM iterations (default: 10)
- `--convergence-threshold FLOAT`: Stop when weight change is below this (default: 1e-4)
- `--seed INT`: Random seed for reproducibility
- `--select-anchor`: Try all languages as anchors and select the best
- `-v, --verbose`: Print progress information

### Python API

```python
from emalign import CognateAligner, load_cldf_dataset, write_alignments

# Load CLDF data
forms, cognate_sets = load_cldf_dataset("path/to/cldf/")

# Create and fit aligner
aligner = CognateAligner(
    gap_penalty=1.0,
    learning_rate=0.01,
    max_iterations=10,
    random_seed=42,
)

# Generate alignments
alignments = aligner.fit_and_align(cognate_sets, verbose=True)

# Write output
write_alignments(alignments, "alignments.csv")
```

## Input Format

The input must be a CLDF dataset with:

- `forms.csv`: Lexical forms with `ID`, `Language_ID`, `Form` columns
- `cognates.csv`: Cognate judgments with `Form_ID`, `Cognateset_ID`, `Morph_Index` columns
- `cldf-metadata.json`: CLDF metadata file

Forms should be in IPA with morphs separated by `+` (e.g., `pre+fix`).

## Output Format

The output is a CLDF-compliant CSV with columns:

- `ID`: Unique alignment identifier
- `Form_ID`: Reference to the original form
- `Cognateset_ID`: Reference to the cognate set
- `Aligned_Form`: Pipe-delimited aligned segments (e.g., `p|a|t|-`)

## Algorithm

1. **Initialization**: Feature weights initialized as 1/24 with small random perturbations
2. **E-step**: Compute optimal alignments using weighted Levenshtein distance
3. **M-step**: Update weights via SGD based on alignment statistics
4. **Iteration**: Repeat until convergence or max iterations reached

The alignment cost between segments is computed as:
```
cost = weights · |features(seg1) - features(seg2)|
```

## License

MIT License
