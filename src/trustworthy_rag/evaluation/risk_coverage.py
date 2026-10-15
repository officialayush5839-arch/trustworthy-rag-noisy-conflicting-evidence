"""
Risk-Coverage curve compilation and Area Under Risk-Coverage Curve (AURCC) calculation.
Provides strict input validation and non-degenerate numerical trapezoidal integration.
"""

from __future__ import annotations

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
        Raises:
            ValueError: If inputs are empty, lengths differ, or values are non-finite.
        """
        if not scores or not errors:
            raise ValueError("Input 'scores' and 'errors' arrays must not be empty.")

        if len(scores) != len(errors):
            raise ValueError(
                f"Mismatch in array lengths: {len(scores)} scores vs {len(errors)} errors."
            )

        scores_arr = np.asarray(scores, dtype=np.float64)
        errors_arr = np.asarray(errors, dtype=np.float64)

        if not np.all(np.isfinite(scores_arr)) or not np.all(np.isfinite(errors_arr)):
            raise ValueError("Input arrays must contain finite numerical values (no NaN or Inf).")

        n = len(scores_arr)
        # Sort descending by decision score
        sorted_indices = np.argsort(-scores_arr)
        sorted_errors = errors_arr[sorted_indices]

        cumulative_errors = np.cumsum(sorted_errors)
        coverage_steps = np.arange(1, n + 1) / n
        selective_risk = cumulative_errors / np.arange(1, n + 1)

        # Prepend point at zero coverage with risk equal to top-1 risk for continuous curve
        cov_curve = np.concatenate(([0.0], coverage_steps))
        risk_curve = np.concatenate(([selective_risk[0]], selective_risk))

        # Integrate curve: AURCC = int_0^1 Risk(c) dc
        if hasattr(np, "trapezoid"):
            aurcc = float(np.trapezoid(risk_curve, cov_curve))
        else:
            aurcc = float(np.trapz(risk_curve, cov_curve))

        return coverage_steps, selective_risk, round(aurcc, 6)

    @staticmethod
    def bootstrap_ci(
        scores: List[float],
        errors: List[float],
        n_bootstraps: int = 500,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Compute mean AURCC with bootstrap confidence interval (lower, upper)."""
        if not scores or len(scores) != len(errors):
            raise ValueError("Cannot bootstrap with empty or mismatched arrays.")

        rng = np.random.RandomState(seed)
        n = len(scores)
        bootstrapped_aurccs: List[float] = []

        scores_arr = np.asarray(scores, dtype=np.float64)
        errors_arr = np.asarray(errors, dtype=np.float64)

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

        return round(mean_val, 6), round(lower_bound, 6), round(upper_bound, 6)
