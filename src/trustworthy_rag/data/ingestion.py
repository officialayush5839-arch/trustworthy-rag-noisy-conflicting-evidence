"""
Benchmark ingestion and dataset loader engine.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

logger = logging.getLogger(__name__)

from trustworthy_rag.schemas import Document, Question


class DatasetIngestionEngine:
    """Ingests and validates raw benchmark datasets with cryptographic fingerprinting."""

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Compute the SHA-256 hash of a file."""
        path = Path(file_path)
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def parse_documents_jsonl(self, file_path: str | Path) -> List[Document]:
        """Parse and validate documents from a JSONL file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {path}")

        documents: List[Document] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    if "sha256_hash" not in data:
                        content = data.get("text", "")
                        data["sha256_hash"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    doc = Document(**data)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to parse Document at line {line_no} in {path}: {e}")
                    raise ValueError(f"Schema violation at line {line_no}: {e}") from e

        return documents

    def parse_questions_jsonl(self, file_path: str | Path) -> List[Question]:
        """Parse and validate questions from a JSONL file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Question file not found: {path}")

        questions: List[Question] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    q = Question(**data)
                    questions.append(q)
                except Exception as e:
                    logger.error(f"Failed to parse Question at line {line_no} in {path}: {e}")
                    raise ValueError(f"Schema violation at line {line_no}: {e}") from e

        return questions

    def export_jsonl(self, records: List[Document] | List[Question], output_path: str | Path) -> str:
        """Export validated records to JSONL and return its SHA-256 checksum."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

        return self.compute_file_hash(out)
