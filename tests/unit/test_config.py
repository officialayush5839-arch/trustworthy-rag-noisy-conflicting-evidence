"""
Unit tests for configuration loaders and validation.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from trustworthy_rag.config import ExperimentConfig, load_config


def test_load_default_config():
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "default_trustworthy_rag"
    assert config.retrieval.sparse_method == "bm25"
    assert config.retrieval.top_k == 10
    assert config.policy.tau_ans_rel == 0.65
    assert len(config.random_seeds) == 5


def test_load_nonexistent_config_fails():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_invalid_policy_thresholds():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_name="test",
            policy={"tau_ans_rel": 1.5},  # type: ignore
        )
