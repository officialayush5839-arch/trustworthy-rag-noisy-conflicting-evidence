"""
Integration test for the conventional RAG baseline pipeline.
"""

from scripts.run_baseline import run_baseline_pipeline


def test_baseline_pipeline_execution():
    """Verify that the baseline pipeline runs on the validation split and emits valid JSONL logs."""
    output_file = run_baseline_pipeline(split="validation", limit=5)

    assert output_file.is_file()
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5

    for line in lines:
        assert '"query_id":' in line
        assert '"raw_response":' in line
        assert '"decision":' in line
