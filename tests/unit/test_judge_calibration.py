"""
Unit test for human calibration of automated claim evaluation judges.
Verifies inter-annotator agreement (Cohen's kappa >= 0.70).
"""

import json
from pathlib import Path


def compute_cohens_kappa(labels_a, labels_b):
    """Compute Cohen's kappa coefficient between two raters."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    categories = sorted(list(set(labels_a).union(set(labels_b))))

    # Observed agreement
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement by chance
    p_e = 0.0
    for cat in categories:
        count_a = sum(1 for x in labels_a if x == cat)
        count_b = sum(1 for x in labels_b if x == cat)
        p_e += (count_a / n) * (count_b / n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def test_human_judge_calibration_kappa():
    gold_path = Path("data/annotations/human_gold_sample.json")
    assert gold_path.is_file(), "Missing human gold annotation file"

    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    human_labels = [s["human_label"] for s in samples]
    auto_labels = [s["automated_label"] for s in samples]

    kappa = compute_cohens_kappa(human_labels, auto_labels)
    assert kappa >= 0.70, f"Expected kappa >= 0.70, got {kappa:.3f}"
