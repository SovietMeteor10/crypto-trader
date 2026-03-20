"""LiveAdapter — same interface as BacktestEngine, routes to ccxt live."""

import json
import os
from datetime import datetime, timezone

import ccxt

from .data_module import DataModule
from .cost_module import CostModule
from .sizer_module import SizerModule
from .signal_module import SignalModule


class LiveAdapter:
    """
    Same conceptual interface as BacktestEngine.run() but routes to live Binance Futures.
    """

    def __init__(
        self,
        data_module: DataModule,
        cost_module: CostModule,
        sizer_module: SizerModule,
        signal_module: SignalModule,
        params: dict,
        api_key: str,
        api_secret: str,
        symbol: str,
        timeframe: str,
        dry_run: bool = True,
    ):
        self.data_module = data_module
        self.cost_module = cost_module
        self.sizer_module = sizer_module
        self.signal_module = signal_module
        self.params = params
        self.symbol = symbol
        self.timeframe = timeframe
        self.dry_run = dry_run

        self._exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        })

        self._log_file = "live_log.jsonl"
        self._current_position = None  # {direction, size, entry_price}

    def run_bar(self) -> dict:
        """
        Called once per closed bar. Fetches latest OHLCV, generates signal,
        computes size, places order if changed.
        """
        try:
            # Fetch recent data for signal generation
            now = datetime.now(timezone.utc)
            start = (now - self._timeframe_delta() * 200).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

            data = self.data_module.get_ohlcv(self.symbol, self.timeframe, start, end)
            signal = self.signal_module.generate(data, self.params)
            current_signal = int(signal.iloc[-1])

            state = self.get_state()
            equity = state.get("equity", 0)
            price = data["close"].iloc[-1]

            action = "hold"
            order_id = None

            current_dir = 0
            if self._current_position:
                current_dir = self._current_position["direction"]

            if current_signal != current_dir:
                # Close existing position if any
                if self._current_position is not None:
                    action = "close_and_open" if current_signal != 0 else "close"
                    if not self.dry_run:
                        side = "buy" if self._current_position["direction"] == -1 else "sell"
                        order = self._exchange.create_market_order(
                            self.symbol, side, self._current_position["size"],
                        )
                        order_id = order.get("id")
                    self._current_position = None

                # Open new position if signal is non-zero
                if current_signal != 0:
                    vol = data["close"].pct_change().std() * (8760 ** 0.5)
                    size = self.sizer_module.compute_size(
                        current_signal, equity, price, vol,
                    )
                    if size > 0:
                        action = "open" if action == "hold" else action
                        if not self.dry_run:
                            side = "buy" if current_signal == 1 else "sell"
                            order = self._exchange.create_market_order(
                                self.symbol, side, size,
                            )
                            order_id = order.get("id")
                        self._current_position = {
                            "direction": current_signal,
                            "size": size,
                            "entry_price": price,
                        }

            result = {
                "timestamp": now.isoformat(),
                "signal": current_signal,
                "current_position": self._current_position,
                "action_taken": action,
                "order_id": order_id,
                "price": price,
                "equity": equity,
                "dry_run": self.dry_run,
            }

            self._log(result)
            return result

        except Exception as e:
            error_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "signal": None,
                "current_position": self._current_position,
                "action_taken": "error",
                "order_id": None,
            }
            self._log(error_result)
            return error_result

    def get_state(self) -> dict:
        """Returns current position, equity, open PnL from Binance API."""
        try:
            if self.dry_run:
                return {
                    "equity": 1000.0,
                    "position": self._current_position,
                    "open_pnl": 0.0,
                }

            balance = self._exchange.fetch_balance()
            equity = float(balance.get("total", {}).get("USDT", 0))

            positions = self._exchange.fetch_positions([self.symbol])
            open_pnl = 0.0
            for pos in positions:
                if float(pos.get("contracts", 0)) > 0:
                    open_pnl = float(pos.get("unrealizedPnl", 0))

            return {
                "equity": equity,
                "position": self._current_position,
                "open_pnl": open_pnl,
            }
        except Exception as e:
            return {"equity": 0, "position": None, "open_pnl": 0, "error": str(e)}

    def _log(self, data: dict) -> None:
        with open(self._log_file, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _timeframe_delta(self):
        from datetime import timedelta
        mapping = {
            "1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15), "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1), "2h": timedelta(hours=2),
            "4h": timedelta(hours=4), "1d": timedelta(days=1),
        }
        return mapping.get(self.timeframe, timedelta(hours=1))
