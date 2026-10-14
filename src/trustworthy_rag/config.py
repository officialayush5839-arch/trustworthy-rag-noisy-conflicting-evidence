"""
Configuration schemas and loaders for the Trustworthy RAG platform.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import yaml


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sparse_method: str = "bm25"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 10
    rrf_k: int = 60


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str = Field("local", description="local | gemini | openai")
    model_identifier: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_template_path: Optional[str] = None


class PolicyThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    tau_ans_rel: float = Field(0.65, ge=0.0, le=1.0)
    tau_cov_min: float = Field(0.60, ge=0.0, le=1.0)
    tau_clar_min: float = Field(0.40, ge=0.0, le=1.0)
    tau_con_max: float = Field(0.70, ge=0.0, le=1.0)
    tau_cit_min: float = Field(0.70, ge=0.0, le=1.0)


class PerturbationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition: str = "clean"
    severity: float = 0.0
    random_seed: int = 42


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    dataset_name: str = "ragtruth"
    split_dir: str = "data/frozen"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    policy: PolicyThresholds = Field(default_factory=PolicyThresholds)
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)
    random_seeds: List[int] = Field(default_factory=lambda: [42, 137, 2024, 7, 99])
    output_dir: str = "experiments/runs"


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)
