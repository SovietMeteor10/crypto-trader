"""BacktestEngine — orchestrates walk-forward, returns ResultsBundle."""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import optuna
import pandas as pd

from .data_module import DataModule
from .cost_module import CostModule
from .sizer_module import SizerModule
from .signal_module import SignalModule

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ResultsBundle:
    symbol: str
    timeframe: str
    start: str
    end: str
    strategy_name: str
    params: dict
    equity_curve: pd.Series
    trades: pd.DataFrame
    monthly_returns: pd.Series
    split: str
    window_id: int = None


class BacktestEngine:
    def __init__(
        self,
        data_module: DataModule,
        cost_module: CostModule,
        sizer_module: SizerModule,
    ):
        self.data_module = data_module
        self.cost_module = cost_module
        self.sizer_module = sizer_module

    def run(
        self,
        signal_module: SignalModule,
        params: dict,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        initial_equity: float = 1000.0,
        split_label: str = "train",
    ) -> ResultsBundle:
        data = self.data_module.get_ohlcv(symbol, timeframe, start, end)
        signal = signal_module.generate(data, params)
        signal_module.validate_output(signal, data)

        cash = initial_equity
        equity_series = []
        trades = []
        position = None  # {direction, size, entry_price, entry_time, open_fee, funding_cost}

        # Compute rolling volatility for sizer (20-period annualised)
        returns = data["close"].pct_change()
        tf_hours = self._timeframe_to_hours(timeframe)
        periods_per_year = 8760 / tf_hours
        rolling_vol = returns.rolling(20).std() * np.sqrt(periods_per_year)
        rolling_vol = rolling_vol.fillna(returns.std() * np.sqrt(periods_per_year))

        # FIX: Start from bar 1. Signal at bar i-1 is acted on at bar i's close.
        # Bar 0 has no prior signal so we skip it.
        # This eliminates look-ahead bias: signal uses data up to T, trade enters at T+1.
        equity_series.append(initial_equity)  # bar 0: no action

        for i in range(1, len(data)):
            sig = signal.iloc[i - 1]   # FIX: use PREVIOUS bar's signal
            price = data["close"].iloc[i]
            ts = data.index[i]
            vol = rolling_vol.iloc[i] if rolling_vol.iloc[i] > 0 else 0.01

            # Check if position needs closing (signal changed or flipped)
            if position is not None:
                should_close = False
                if sig == 0:
                    should_close = True
                elif sig != position["direction"]:
                    should_close = True

                if should_close:
                    close_result = self.cost_module.apply_close(
                        position["entry_price"], price, position["size"],
                        symbol, position["direction"],
                    )
                    pnl_raw = (close_result["fill_price"] - position["entry_price"]) * \
                              position["size"] * position["direction"]
                    pnl_net = pnl_raw - close_result["fee_usdt"] - position.get("funding_cost", 0)

                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": position["entry_price"],
                        "exit_price": close_result["fill_price"],
                        "size": position["size"],
                        "direction": position["direction"],
                        "pnl_usdt": pnl_net,
                        "pnl_pct": pnl_net / cash * 100 if cash > 0 else 0,
                        "cost_usdt": close_result["fee_usdt"] + position.get("open_fee", 0),
                        "funding_cost_usdt": position.get("funding_cost", 0),
                    })

                    cash += pnl_net
                    self.sizer_module.record_trade(pnl_net / max(cash, 1) * 100)
                    position = None

            # Open new position if signal is non-zero and we are flat
            if sig != 0 and position is None:
                size = self.sizer_module.compute_size(sig, cash, price, vol)
                if size > 0:
                    # Liquidation check
                    liq = self.sizer_module.check_liquidation_risk(
                        price, size, cash, self.sizer_module.leverage, sig,
                    )
                    if not liq["safe"]:
                        logger.warning(
                            f"Skipping trade at {ts}: liquidation too close "
                            f"({liq['distance_pct']:.1%} < 15%)"
                        )
                    else:
                        open_result = self.cost_module.apply_open(price, size, symbol, sig)
                        cash -= open_result["fee_usdt"]

                        position = {
                            "direction": sig,
                            "size": size,
                            "entry_price": open_result["fill_price"],
                            "entry_time": ts,
                            "open_fee": open_result["fee_usdt"],
                            "funding_cost": 0.0,
                        }

            # FIX: Mark-to-market equity includes open position value
            if position is not None:
                mtm_pnl = (price - position["entry_price"]) * \
                           position["size"] * position["direction"]
                equity_series.append(cash + mtm_pnl)
            else:
                equity_series.append(cash)

        # Close any remaining position at last bar
        if position is not None:
            last_price = data["close"].iloc[-1]
            last_ts = data.index[-1]
            close_result = self.cost_module.apply_close(
                position["entry_price"], last_price, position["size"],
                symbol, position["direction"],
            )
            pnl_raw = (close_result["fill_price"] - position["entry_price"]) * \
                      position["size"] * position["direction"]
            pnl_net = pnl_raw - close_result["fee_usdt"] - position.get("funding_cost", 0)
            trades.append({
                "entry_time": position["entry_time"],
                "exit_time": last_ts,
                "entry_price": position["entry_price"],
                "exit_price": close_result["fill_price"],
                "size": position["size"],
                "direction": position["direction"],
                "pnl_usdt": pnl_net,
                "pnl_pct": pnl_net / cash * 100 if cash > 0 else 0,
                "cost_usdt": close_result["fee_usdt"] + position.get("open_fee", 0),
                "funding_cost_usdt": position.get("funding_cost", 0),
            })
            cash += pnl_net
            # Update last equity to reflect final close
            equity_series[-1] = cash

        eq_series = pd.Series(equity_series, index=data.index, name="equity")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_price", "exit_price",
            "size", "direction", "pnl_usdt", "pnl_pct", "cost_usdt", "funding_cost_usdt",
        ])

        # Monthly returns (decimal, not percentage — 5% = 0.05)
        monthly = eq_series.resample("ME").last()
        monthly_ret = monthly.pct_change().dropna()

        return ResultsBundle(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategy_name=signal_module.name,
            params=params,
            equity_curve=eq_series,
            trades=trades_df,
            monthly_returns=monthly_ret,
            split=split_label,
        )

    def run_walk_forward(
        self,
        signal_module: SignalModule,
        symbol: str,
        timeframe: str,
        full_start: str,
        full_end: str,
        train_months: int = 6,
        test_months: int = 2,
        gap_weeks: int = 2,
        n_optuna_trials: int = 30,
        initial_equity: float = 1000.0,
    ) -> list[ResultsBundle]:
        start_dt = pd.Timestamp(full_start)
        end_dt = pd.Timestamp(full_end)

        train_delta = pd.DateOffset(months=train_months)
        test_delta = pd.DateOffset(months=test_months)
        gap_delta = timedelta(weeks=gap_weeks)
        step = test_delta

        windows = []
        current = start_dt

        while True:
            train_start = current
            train_end = train_start + train_delta
            test_start = train_end + gap_delta
            test_end = test_start + test_delta

            if test_end > end_dt:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current = current + step

        results = []
        for idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            tr_s_str = tr_s.strftime("%Y-%m-%d")
            tr_e_str = tr_e.strftime("%Y-%m-%d")
            te_s_str = te_s.strftime("%Y-%m-%d")
            te_e_str = te_e.strftime("%Y-%m-%d")

            print(
                f"Window {idx + 1}/{len(windows)}: "
                f"train {tr_s_str} to {tr_e_str}, test {te_s_str} to {te_e_str}"
            )

            best_params, best_sharpe = self._optimise(
                signal_module, symbol, timeframe,
                tr_s_str, tr_e_str, n_optuna_trials, initial_equity,
            )
            print(f"  best train Sharpe: {best_sharpe:.2f}, params: {best_params}")

            test_bundle = self.run(
                signal_module, best_params, symbol, timeframe,
                te_s_str, te_e_str, initial_equity, split_label="walkforward",
            )
            test_bundle.window_id = idx + 1
            results.append(test_bundle)

        return results

    def run_three_split(
        self,
        signal_module: SignalModule,
        params: dict,
        symbol: str,
        timeframe: str,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        holdout_start: str,
        holdout_end: str,
        initial_equity: float = 1000.0,
    ) -> dict[str, ResultsBundle]:
        train = self.run(signal_module, params, symbol, timeframe,
                         train_start, train_end, initial_equity, "train")
        val = self.run(signal_module, params, symbol, timeframe,
                       val_start, val_end, initial_equity, "validation")
        holdout = self.run(signal_module, params, symbol, timeframe,
                           holdout_start, holdout_end, initial_equity, "holdout")
        return {"train": train, "validation": val, "holdout": holdout}

    def _optimise(
        self,
        signal_module: SignalModule,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        n_trials: int,
        initial_equity: float,
    ) -> tuple[dict, float]:
        space = signal_module.parameter_space

        def objective(trial):
            params = {}
            for name, spec in space.items():
                if spec[0] == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif spec[0] == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif spec[0] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec[1])
            try:
                bundle = self.run(
                    signal_module, params, symbol, timeframe,
                    start, end, initial_equity, "optimisation",
                )
                if len(bundle.monthly_returns) < 2:
                    return -10.0
                sharpe = (bundle.monthly_returns.mean() / bundle.monthly_returns.std()
                          * np.sqrt(12)) if bundle.monthly_returns.std() > 0 else 0
                return sharpe
            except Exception:
                return -10.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params, study.best_value

    @staticmethod
    def _timeframe_to_hours(timeframe: str) -> float:
        mapping = {
            "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
            "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12,
            "1d": 24, "1w": 168,
        }
        return mapping.get(timeframe, 1)
