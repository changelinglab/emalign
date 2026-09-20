"""Tests for the CLDF I/O module."""

import csv
import tempfile
from pathlib import Path

import pytest

from emalign.cldf_io import (
    AlignmentResult,
    CognateEntry,
    CognateSet,
    Form,
    iter_cognate_morphs,
    load_cldf_dataset,
    write_alignments,
)


# Path to test data
TEST_DATA_PATH = Path(__file__).parent.parent / "data" / "cldf_tangkhulic"


class TestLoadCldfDataset:
    @pytest.mark.skipif(
        not TEST_DATA_PATH.exists(),
        reason="Test data not available"
    )
    def test_load_tangkhulic(self):
        """Test loading the Tangkhulic CLDF dataset."""
        forms, cognate_sets = load_cldf_dataset(TEST_DATA_PATH)
        
        assert len(forms) > 0
        assert len(cognate_sets) > 0
        
        # Check form structure
        first_form = next(iter(forms.values()))
        assert first_form.id is not None
        assert first_form.language_id is not None
        assert first_form.form is not None
        assert isinstance(first_form.morphs, list)
    
    @pytest.mark.skipif(
        not TEST_DATA_PATH.exists(),
        reason="Test data not available"
    )
    def test_cognate_set_structure(self):
        """Test structure of loaded cognate sets."""
        forms, cognate_sets = load_cldf_dataset(TEST_DATA_PATH)
        
        for cs in cognate_sets[:5]:  # Check first 5
            assert cs.id is not None
            assert len(cs.entries) > 0
            for entry, form in cs.entries:
                assert entry.form_id == form.id
    
    def test_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_cldf_dataset("/nonexistent/path")


class TestWriteAlignments:
    def test_write_and_read(self):
        """Test writing alignments to CSV and reading back."""
        alignments = [
            AlignmentResult(form_id="1", cognateset_id="100", aligned_form="p|a|t"),
            AlignmentResult(form_id="2", cognateset_id="100", aligned_form="b|a|t"),
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            write_alignments(alignments, output_path)
            
            # Read back and verify
            with open(output_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 2
            assert rows[0]["ID"] == "1"
            assert rows[0]["Form_ID"] == "1"
            assert rows[0]["Cognateset_ID"] == "100"
            assert rows[0]["Aligned_Form"] == "p|a|t"
        finally:
            output_path.unlink()
    
    def test_creates_parent_directory(self):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "alignments.csv"
            
            alignments = [
                AlignmentResult(form_id="1", cognateset_id="100", aligned_form="t|e|s|t"),
            ]
            
            write_alignments(alignments, output_path)
            
            assert output_path.exists()


class TestIterCognateMorphs:
    def test_yields_correct_morph(self):
        """Test that iterator yields the morph at morph_index."""
        forms = [
            Form(id="f1", language_id="1", form="a+b+c", morphs=["a", "b", "c"]),
            Form(id="f2", language_id="2", form="x+y", morphs=["x", "y"]),
        ]
        entries = [
            (CognateEntry(id="1", form_id="f1", cognateset_id="cs1", morph_index=1), forms[0]),
            (CognateEntry(id="2", form_id="f2", cognateset_id="cs1", morph_index=0), forms[1]),
        ]
        cs = CognateSet(id="cs1", entries=entries)
        
        result = list(iter_cognate_morphs(cs))
        
        assert len(result) == 2
        assert result[0][1] == "b"  # morph_index=1 of ["a", "b", "c"]
        assert result[1][1] == "x"  # morph_index=0 of ["x", "y"]
    
    def test_skips_invalid_morph_index(self):
        """Test that invalid morph indices are skipped."""
        form = Form(id="f1", language_id="1", form="a+b", morphs=["a", "b"])
        entry = CognateEntry(id="1", form_id="f1", cognateset_id="cs1", morph_index=5)  # Out of bounds
        cs = CognateSet(id="cs1", entries=[(entry, form)])
        
        result = list(iter_cognate_morphs(cs))
        
        assert len(result) == 0
