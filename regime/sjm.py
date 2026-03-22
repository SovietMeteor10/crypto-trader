"""
Statistical Jump Model for regime detection.
Based on Cortese, Kolm and Lindstrom (2023), Digital Finance.

Finds K regimes by minimising a penalised assignment problem:
  min_{z} sum_t ||x_t - mu_{z_t}||^2 + lambda * sum_t 1[z_t != z_{t-1}]

Solved exactly via dynamic programming (Viterbi-style algorithm).
This is NOT probabilistic — no EM, no hidden states, no distribution assumptions.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SJMResult:
    regimes: np.ndarray          # integer array of regime labels 0..K-1
    centroids: np.ndarray        # K x n_features centroid matrix
    n_jumps: int                 # number of regime transitions
    within_ss: float             # within-regime sum of squares (lower = better fit)
    regime_counts: dict          # {regime_id: count}
    regime_means: dict           # {regime_id: mean_return} for labelling


class StatisticalJumpModel:
    """
    Statistical Jump Model with dynamic programming solver.

    Parameters
    ----------
    n_regimes : int
        Number of regimes K (default 3: bull/neutral/bear)
    jump_penalty : float
        Lambda — cost of switching regime. Higher = fewer switches.
        Typical range: 0.01 to 10.0. Start with 0.1.
    max_iter : int
        Maximum alternating optimisation iterations
    random_state : int
        Seed for centroid initialisation
    """

    def __init__(
        self,
        n_regimes: int = 3,
        jump_penalty: float = 0.1,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.result_: Optional[SJMResult] = None

    def _dp_assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Dynamic programming assignment step (vectorized over K).
        O(T * K^2) time, O(T * K) space.
        """
        T, _ = X.shape
        K = len(centroids)
        lam = self.jump_penalty

        # Cost of assigning time t to regime k: shape (T, K)
        assign_cost = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)

        dp = np.full((T, K), np.inf)
        parent = np.zeros((T, K), dtype=np.int32)

        dp[0, :] = assign_cost[0, :]

        # Forward pass — vectorized inner loop over K
        for t in range(1, T):
            # transition_cost[j, k] = cost of going from regime j to regime k
            # = dp[t-1, j] + lam * (j != k)
            prev = dp[t - 1, :]  # shape (K,)
            # Broadcasting: prev[:, None] + lam gives (K, K) with jump penalty
            # then subtract lam on diagonal (staying = no penalty)
            trans = prev[:, np.newaxis] + lam  # (K, K): trans[j,k] = prev[j] + lam
            np.fill_diagonal(trans, prev)       # staying: no jump cost
            best_prev = trans.min(axis=0)       # (K,) min over source regimes
            dp[t, :] = best_prev + assign_cost[t, :]
            parent[t, :] = trans.argmin(axis=0)

        # Backtrack
        regimes = np.zeros(T, dtype=int)
        regimes[T - 1] = np.argmin(dp[T - 1, :])
        for t in range(T - 2, -1, -1):
            regimes[t] = parent[t + 1, regimes[t + 1]]

        return regimes

    def _update_centroids(self, X: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Update centroids as mean of assigned points."""
        K = self.n_regimes
        centroids = np.zeros((K, X.shape[1]))
        for k in range(K):
            mask = regimes == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                # Empty regime — reinitialise randomly
                rng = np.random.default_rng(self.random_state + k)
                centroids[k] = X[rng.choice(len(X))]
        return centroids

    def fit(self, X: np.ndarray) -> "StatisticalJumpModel":
        """
        Fit the SJM to feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
            Standardised feature matrix (zero mean, unit variance)
        """
        T, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Initialise centroids using k-means++ style
        centroids = np.zeros((self.n_regimes, n_features))
        centroids[0] = X[rng.choice(T)]
        for k in range(1, self.n_regimes):
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :k, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / dists.sum()
            centroids[k] = X[rng.choice(T, p=probs)]

        # Alternating optimisation
        regimes = None
        prev_regimes = None

        for iteration in range(self.max_iter):
            regimes = self._dp_assign(X, centroids)
            centroids = self._update_centroids(X, regimes)

            if prev_regimes is not None and np.array_equal(regimes, prev_regimes):
                break
            prev_regimes = regimes.copy()

        self.centroids_ = centroids

        # Compute diagnostics
        within_ss = sum(
            np.sum((X[regimes == k] - centroids[k]) ** 2)
            for k in range(self.n_regimes)
            if (regimes == k).sum() > 0
        )
        n_jumps = int((np.diff(regimes) != 0).sum())
        regime_counts = {k: int((regimes == k).sum()) for k in range(self.n_regimes)}

        self.result_ = SJMResult(
            regimes=regimes,
            centroids=centroids,
            n_jumps=n_jumps,
            within_ss=within_ss,
            regime_counts=regime_counts,
            regime_means={},  # filled after labelling
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new observations to regimes.
        For online/walk-forward use: assigns each point to nearest centroid
        (greedy, no jump penalty — call fit() on expanding/rolling window instead).
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        dists = np.sum((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def label_regimes(self, regimes: np.ndarray, returns: np.ndarray) -> dict:
        """
        Label regimes by mean return:
        highest mean return = bull (0), middle = neutral (1), lowest = bear (2)
        Returns mapping: {original_label: 'bull'|'neutral'|'bear'}
        """
        K = self.n_regimes
        mean_returns = {}
        for k in range(K):
            mask = regimes == k
            if mask.sum() > 0:
                mean_returns[k] = returns[mask].mean()
            else:
                mean_returns[k] = 0.0

        sorted_by_return = sorted(mean_returns.keys(), key=lambda k: mean_returns[k], reverse=True)
        if K == 2:
            labels = ['bull', 'bear']
        else:
            labels = ['bull', 'neutral', 'bear']
        return {orig: labels[i] for i, orig in enumerate(sorted_by_return[:K])}
