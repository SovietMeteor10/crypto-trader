"""
GARCH(1,1)-EVT for dynamic tail risk estimation and position sizing.

Pipeline:
1. Fit GARCH(1,1) with GJR specification and Student-t innovations
   to SOL daily returns (constructed from 4H bars)
2. Extract standardised residuals
3. Fit GPD to residuals exceeding the 95th percentile threshold
4. Compute conditional ES_99 = GARCH_vol * GPD_ES_99_of_residuals
5. Derive maximum leverage = loss_budget / conditional_ES_99

References:
- Ardia et al. (2019): GJR-GARCH with Student-t for crypto
- Ke, Yang and Tan (2022): Dynamic PoT outperforms static GARCH-EVT
- Bhat et al. (2025): BTC 99% VaR = -13.6%, ES = -22.1% unconditional
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EVTResult:
    conditional_es_99: float      # Current conditional ES at 99% confidence
    conditional_var_99: float     # Current conditional VaR at 99%
    garch_vol: float              # Current GARCH volatility estimate
    gpd_xi: float                 # GPD shape parameter (tail index)
    gpd_beta: float               # GPD scale parameter
    max_leverage: float           # Derived max leverage given loss budget
    n_exceedances: int            # Number of tail observations used


class GARCHEVTSizer:
    """
    Dynamic position sizer using GARCH(1,1)-EVT tail risk estimation.

    Fits GARCH on a rolling window of daily returns (derived from 4H bars).
    Uses Peaks-over-Threshold GPD for tail fitting on standardised residuals.
    Recomputes daily (every 6 bars at 4H frequency).

    Parameters
    ----------
    loss_budget_pct : float
        Maximum acceptable daily loss as fraction of equity (e.g. 0.05 = 5%)
    max_leverage_cap : float
        Hard cap on leverage regardless of EVT estimate (e.g. 5.0)
    min_leverage : float
        Minimum leverage to avoid zero positions in calm regimes (e.g. 0.5)
    garch_window : int
        Number of daily returns for GARCH fitting (e.g. 252 = 1 year)
    threshold_quantile : float
        Quantile for GPD threshold selection (e.g. 0.95)
    """

    def __init__(
        self,
        loss_budget_pct: float = 0.05,
        max_leverage_cap: float = 5.0,
        min_leverage: float = 0.3,
        garch_window: int = 252,
        threshold_quantile: float = 0.95,
    ):
        self.loss_budget_pct = loss_budget_pct
        self.max_leverage_cap = max_leverage_cap
        self.min_leverage = min_leverage
        self.garch_window = garch_window
        self.threshold_quantile = threshold_quantile

    def _fit_gjr_garch(self, returns: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit GJR-GARCH(1,1) using quasi-maximum likelihood.
        Returns (current_vol, standardised_residuals).
        """
        try:
            from arch import arch_model
            model = arch_model(
                returns * 100,  # scale for numerical stability
                vol='Garch',
                p=1, o=1, q=1,  # o=1 gives GJR (asymmetric) term
                dist='StudentsT',
                mean='Zero',
            )
            result = model.fit(disp='off', show_warning=False)
            conditional_vol = result.conditional_volatility / 100
            std_resid = returns / conditional_vol
            current_vol = float(conditional_vol.iloc[-1])
            return current_vol, std_resid.values

        except Exception:
            # Fallback: EWMA volatility (lambda=0.94, RiskMetrics standard)
            ewma_var = np.zeros(len(returns))
            ewma_var[0] = returns[0]**2
            lam = 0.94
            for t in range(1, len(returns)):
                ewma_var[t] = lam * ewma_var[t-1] + (1-lam) * returns[t-1]**2
            ewma_vol = np.sqrt(ewma_var)
            std_resid = returns / np.where(ewma_vol > 0, ewma_vol, 1e-8)
            current_vol = float(ewma_vol[-1])
            return current_vol, std_resid

    def _fit_gpd(
        self,
        std_resid: np.ndarray,
        threshold_quantile: float,
    ) -> Tuple[float, float, float]:
        """
        Fit GPD to exceedances of std_resid above the threshold.
        Returns (xi, beta, threshold).
        """
        losses = -std_resid  # positive = loss
        threshold = np.quantile(losses, threshold_quantile)
        exceedances = losses[losses > threshold] - threshold
        n_exc = len(exceedances)

        if n_exc < 10:
            return 0.2, np.std(losses) * 0.5, threshold

        try:
            from scipy.stats import genpareto
            xi, loc, beta = genpareto.fit(exceedances, floc=0)
            xi = np.clip(xi, -0.5, 1.0)
            beta = max(beta, 1e-8)
            return float(xi), float(beta), float(threshold)
        except Exception:
            mean_exc = np.mean(exceedances)
            var_exc = np.var(exceedances)
            if var_exc > 0:
                xi = 0.5 * (mean_exc**2 / var_exc - 1)
                beta = 0.5 * mean_exc * (mean_exc**2 / var_exc + 1)
            else:
                xi, beta = 0.2, mean_exc
            return float(np.clip(xi, -0.5, 1.0)), float(max(beta, 1e-8)), float(threshold)

    def _gpd_es(
        self,
        xi: float,
        beta: float,
        threshold: float,
        n_total: int,
        n_exceedances: int,
        confidence: float = 0.99,
    ) -> Tuple[float, float]:
        """Compute VaR and ES from GPD fit."""
        p_exceed = n_exceedances / n_total
        alpha = confidence

        if p_exceed <= 0 or p_exceed <= (1 - alpha):
            return threshold * 1.5, threshold * 2.0

        if abs(xi) < 1e-6:
            var_q = threshold + beta * np.log(p_exceed / (1 - alpha))
            es_q = var_q + beta
        else:
            factor = (p_exceed / (1 - alpha))**xi
            var_q = threshold + (beta / xi) * (factor - 1)
            if xi < 1:
                es_q = (var_q + beta - xi * threshold) / (1 - xi)
            else:
                es_q = var_q * 3

        return float(max(var_q, 0)), float(max(es_q, 0))

    def compute(
        self,
        daily_returns: pd.Series,
        equity: float,
        price: float,
    ) -> EVTResult:
        """Compute EVT-based position size."""
        returns = daily_returns.dropna().values
        if len(returns) < 30:
            return EVTResult(
                conditional_es_99=0.20,
                conditional_var_99=0.136,
                garch_vol=0.04,
                gpd_xi=0.20,
                gpd_beta=0.02,
                max_leverage=self.loss_budget_pct / 0.20,
                n_exceedances=0,
            )

        returns = returns[-self.garch_window:]

        current_vol, std_resid = self._fit_gjr_garch(returns)
        xi, beta, threshold = self._fit_gpd(std_resid, self.threshold_quantile)

        losses = -std_resid
        n_exc = int((losses > threshold).sum())
        var_std, es_std = self._gpd_es(
            xi, beta, threshold,
            n_total=len(std_resid),
            n_exceedances=n_exc,
        )

        conditional_var = current_vol * var_std
        conditional_es = current_vol * es_std

        conditional_es = np.clip(conditional_es, 0.02, 0.50)
        conditional_var = np.clip(conditional_var, 0.01, 0.35)

        max_leverage = self.loss_budget_pct / conditional_es
        max_leverage = np.clip(max_leverage, self.min_leverage, self.max_leverage_cap)

        return EVTResult(
            conditional_es_99=float(conditional_es),
            conditional_var_99=float(conditional_var),
            garch_vol=float(current_vol),
            gpd_xi=float(xi),
            gpd_beta=float(beta),
            max_leverage=float(max_leverage),
            n_exceedances=n_exc,
        )

    def compute_position_size(
        self,
        daily_returns: pd.Series,
        equity: float,
        price: float,
    ) -> float:
        """Returns position size in base currency units."""
        evt = self.compute(daily_returns, equity, price)
        notional = equity * evt.max_leverage
        return notional / price


def build_daily_returns(ohlcv_4h: pd.DataFrame) -> pd.Series:
    """Construct daily returns from 4H OHLCV data."""
    daily_close = ohlcv_4h['close'].resample('1D').last().dropna()
    return daily_close.pct_change().dropna()
