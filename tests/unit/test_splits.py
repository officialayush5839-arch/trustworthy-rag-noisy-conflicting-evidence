"""
Unit tests for split disjointness, integrity, and SHA-256 manifest verification.
"""

import json
from pathlib import Path
from trustworthy_rag.data.ingestion import DatasetIngestionEngine


def test_split_disjointness():
    """Verify zero overlap between dev, validation, and test question IDs."""
    engine = DatasetIngestionEngine()
    dev_qs = engine.parse_questions_jsonl("data/frozen/questions_dev.jsonl")
    val_qs = engine.parse_questions_jsonl("data/frozen/questions_val.jsonl")
    test_qs = engine.parse_questions_jsonl("data/frozen/questions_test.jsonl")

    dev_ids = {q.question_id for q in dev_qs}
    val_ids = {q.question_id for q in val_qs}
    test_ids = {q.question_id for q in test_qs}

    assert len(dev_ids.intersection(val_ids)) == 0, "Data leakage between dev and validation!"
    assert len(dev_ids.intersection(test_ids)) == 0, "Data leakage between dev and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage between validation and test!"


def test_manifest_sha256_verification():
    """Verify that all files in data/frozen match their SHA-256 hashes in manifest_sha256.json."""
    manifest_path = Path("data/splits/manifest_sha256.json")
    assert manifest_path.is_file(), "Manifest file missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    engine = DatasetIngestionEngine()
    for filename, expected_hash in manifest.items():
        file_path = Path("data/frozen") / filename
        assert file_path.is_file(), f"Target file {file_path} missing"
        actual_hash = engine.compute_file_hash(file_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {filename}!"
