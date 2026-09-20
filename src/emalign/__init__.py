"""
emalign: EM-based alignment of cognate sets in comparative dictionaries.

This package provides tools for generating phoneme alignments of cognate forms
using expectation-maximization to learn feature weights from PanPhon's 24
articulatory features.
"""

__version__ = "0.1.0"

from emalign.aligner import CognateAligner
from emalign.cldf_io import Language, load_cldf_dataset, write_alignments

__all__ = ["CognateAligner", "Language", "load_cldf_dataset", "write_alignments"]
