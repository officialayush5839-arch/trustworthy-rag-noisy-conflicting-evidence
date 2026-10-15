"""
Risk-Coverage curve compilation and Area Under Risk-Coverage Curve (AURCC) calculation.
"""

from typing import List, Tuple
import numpy as np


class RiskCoverageCompiler:
    """Computes empirical Risk-Coverage Pareto curves and selective error metrics."""

    @staticmethod
    def compute_curve(
        scores: List[float],
        errors: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute coverage array, selective risk array, and Area Under the Risk-Coverage Curve (AURCC).

        Args:
            scores: Model confidence or evidence quality scores for each query.
            errors: Binary or continuous error values (e.g. 1.0 - faithfulness) for each query.
        Returns:
            Tuple of (coverage_array, risk_array, aurcc_scalar).
        """
        if not scores or len(scores) != len(errors):
            return np.array([0.0]), np.array([0.0]), 0.0

        n = len(scores)
        # Sort descending by decision score
        sorted_indices = np.argsort(-np.array(scores))
        sorted_errors = np.array(errors)[sorted_indices]

        cumulative_errors = np.cumsum(sorted_errors)
        coverage_steps = np.arange(1, n + 1) / n
        selective_risk = cumulative_errors / np.arange(1, n + 1)

        # Prepend point at zero coverage with risk equal to top-1 risk for clean curve
        cov_curve = np.concatenate(([0.0], coverage_steps))
        risk_curve = np.concatenate(([selective_risk[0]], selective_risk))

        # Integrate curve: AURCC = int_0^1 Risk(c) dc
        aurcc = float(np.trapezoid(risk_curve, cov_curve))
        return coverage_steps, selective_risk, aurcc

    @staticmethod
    def bootstrap_ci(
        scores: List[float],
        errors: List[float],
        n_bootstraps: int = 500,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Compute mean AURCC with bootstrap confidence interval (lower, upper)."""
        rng = np.random.RandomState(seed)
        n = len(scores)
        bootstrapped_aurccs: List[float] = []

        scores_arr = np.array(scores)
        errors_arr = np.array(errors)

        for _ in range(n_bootstraps):
            indices = rng.choice(n, size=n, replace=True)
            _, _, aurcc = RiskCoverageCompiler.compute_curve(
                scores_arr[indices].tolist(),
                errors_arr[indices].tolist(),
            )
            bootstrapped_aurccs.append(aurcc)

        mean_val = float(np.mean(bootstrapped_aurccs))
        lower_bound = float(np.percentile(bootstrapped_aurccs, 100 * (alpha / 2)))
        upper_bound = float(np.percentile(bootstrapped_aurccs, 100 * (1 - alpha / 2)))

        return mean_val, lower_bound, upper_bound
