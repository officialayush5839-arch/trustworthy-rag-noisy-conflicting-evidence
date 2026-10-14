"""
Unit tests for DatasetIngestionEngine and schema validation.
"""

from pathlib import Path
import pytest
from trustworthy_rag.data.ingestion import DatasetIngestionEngine


def test_parse_valid_frozen_documents():
    engine = DatasetIngestionEngine()
    docs = engine.parse_documents_jsonl("data/frozen/documents.jsonl")

    assert len(docs) == 9
    for doc in docs:
        assert doc.doc_id.startswith("doc_")
        assert len(doc.sha256_hash) == 64
        assert doc.corpus_name == "curated_rag_benchmark"


def test_parse_valid_frozen_questions():
    engine = DatasetIngestionEngine()
    dev_qs = engine.parse_questions_jsonl("data/frozen/questions_dev.jsonl")
    val_qs = engine.parse_questions_jsonl("data/frozen/questions_val.jsonl")
    test_qs = engine.parse_questions_jsonl("data/frozen/questions_test.jsonl")

    assert len(dev_qs) == 50
    assert len(val_qs) == 100
    assert len(test_qs) == 250

    for q in dev_qs:
        assert q.split == "dev"
        assert len(q.gold_answers) > 0


def test_parse_nonexistent_file():
    engine = DatasetIngestionEngine()
    with pytest.raises(FileNotFoundError):
        engine.parse_documents_jsonl("data/nonexistent.jsonl")
