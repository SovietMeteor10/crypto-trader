"""SizerModule — position sizing given signal and portfolio state."""

import numpy as np


class SizerModule:
    def __init__(
        self,
        method: str = "fixed_fractional",
        fraction: float = 0.02,
        leverage: float = 3.0,
        max_position_pct: float = 0.95,
    ):
        if method not in ("fixed_fractional", "kelly", "fixed_usdt"):
            raise ValueError(f"Unknown sizing method: {method}")
        self.method = method
        self.fraction = fraction
        self.leverage = leverage
        self.max_position_pct = max_position_pct
        self._trade_history: list[float] = []

    def record_trade(self, pnl_pct: float) -> None:
        """Record a completed trade PnL for Kelly calculation."""
        self._trade_history.append(pnl_pct)

    def compute_size(
        self,
        signal: int,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        if signal == 0:
            return 0.0

        if self.method == "fixed_fractional":
            notional = equity * self.fraction * self.leverage
        elif self.method == "fixed_usdt":
            notional = self.fraction * self.leverage
        elif self.method == "kelly":
            notional = self._kelly_notional(equity)
        else:
            notional = equity * self.fraction * self.leverage

        # Cap at max_position_pct of equity
        max_notional = equity * self.max_position_pct * self.leverage
        notional = min(notional, max_notional)

        size = notional / price
        return abs(size)

    def _kelly_notional(self, equity: float) -> float:
        """Half-Kelly sizing. Falls back to fixed_fractional if < 20 trades."""
        if len(self._trade_history) < 20:
            return equity * self.fraction * self.leverage

        wins = [t for t in self._trade_history if t > 0]
        losses = [t for t in self._trade_history if t < 0]

        if not wins or not losses:
            return equity * self.fraction * self.leverage

        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return equity * self.fraction * self.leverage

        # Kelly fraction: W - (1-W)/R where R = avg_win/avg_loss
        R = avg_win / avg_loss
        kelly_f = win_rate - (1 - win_rate) / R

        # Half-Kelly, clamped to [0, fraction]
        half_kelly = max(0.0, kelly_f / 2)
        half_kelly = min(half_kelly, self.max_position_pct)

        return equity * half_kelly * self.leverage

    def check_liquidation_risk(
        self,
        entry_price: float,
        position_size: float,
        equity: float,
        leverage: float,
        direction: int,
    ) -> dict:
        notional = position_size * entry_price
        margin = notional / leverage

        # Liquidation occurs when loss = margin (simplified, ignoring maintenance margin)
        # For longs: liq_price = entry * (1 - 1/leverage)
        # For shorts: liq_price = entry * (1 + 1/leverage)
        if direction == 1:
            liq_price = entry_price * (1 - 1 / leverage)
        else:
            liq_price = entry_price * (1 + 1 / leverage)

        distance_pct = abs(entry_price - liq_price) / entry_price

        return {
            "liquidation_price": liq_price,
            "distance_pct": distance_pct,
            "safe": distance_pct >= 0.15,
        }
