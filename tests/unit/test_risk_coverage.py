"""
Unit tests for RiskCoverageCompiler and bootstrap confidence intervals.
"""

from trustworthy_rag.evaluation.risk_coverage import RiskCoverageCompiler


def test_risk_coverage_curve_calculation():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    errors = [0.0, 0.0, 0.0, 1.0, 1.0]

    cov, risk, aurcc = RiskCoverageCompiler.compute_curve(scores, errors)

    assert len(cov) == 5
    assert cov[-1] == 1.0
    assert risk[0] == 0.0  # Top score had 0 error
    assert risk[-1] == 2.0 / 5.0  # Full coverage error rate is 40%
    assert 0.0 <= aurcc <= 1.0


def test_bootstrap_ci():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    errors = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    mean_val, lower, upper = RiskCoverageCompiler.bootstrap_ci(scores, errors, n_bootstraps=50)

    assert 0.0 <= lower <= mean_val <= upper <= 1.0
