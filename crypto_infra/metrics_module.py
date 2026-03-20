"""MetricsModule — computes all required metrics from ResultsBundle."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine, ResultsBundle
from .signal_module import SignalModule


@dataclass
class MetricsBundle:
    # Return metrics
    total_return_pct: float
    annualised_return_pct: float
    monthly_return_mean: float
    monthly_return_median: float
    monthly_return_std: float
    monthly_return_worst: float
    monthly_return_best: float
    monthly_return_skewness: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    max_consecutive_losing_months: int

    # Trade metrics
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_trade_duration_hours: float

    # Cost metrics
    total_fees_usdt: float
    total_funding_usdt: float
    total_slippage_usdt: float
    cost_drag_annualised_pct: float

    # Correlation
    btc_correlation: float

    # Flags
    flag_overfit: bool
    flag_insufficient_trades: bool
    flag_high_btc_correlation: bool
    flag_negative_skew: bool
    flag_long_drawdown: bool
    flag_consecutive_losses: bool
    flag_fragile_params: bool

    # Summary
    passes_all_checks: bool


class MetricsModule:
    def compute(
        self,
        bundle: ResultsBundle,
        btc_returns: pd.Series = None,
        train_sharpe: float = None,
    ) -> MetricsBundle:
        eq = bundle.equity_curve
        trades = bundle.trades
        monthly = bundle.monthly_returns

        # Return metrics
        total_return_pct = (eq.iloc[-1] / eq.iloc[0] - 1) * 100 if len(eq) > 0 else 0
        days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400 if len(eq) > 1 else 1
        years = max(days / 365.25, 1 / 365.25)
        annualised_return_pct = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100

        m_mean = monthly.mean() if len(monthly) > 0 else 0
        m_median = monthly.median() if len(monthly) > 0 else 0
        m_std = monthly.std() if len(monthly) > 1 else 0
        m_worst = monthly.min() if len(monthly) > 0 else 0
        m_best = monthly.max() if len(monthly) > 0 else 0
        m_skew = float(monthly.skew()) if len(monthly) > 2 else 0

        # Sharpe (annualised from monthly)
        if m_std > 0 and len(monthly) > 1:
            sharpe = (m_mean / m_std) * np.sqrt(12)
        else:
            sharpe = 0.0

        # Sortino
        downside = monthly[monthly < 0]
        if len(downside) > 0:
            downside_std = downside.std()
            sortino = (m_mean / downside_std) * np.sqrt(12) if downside_std > 0 else 0
        else:
            sortino = float("inf") if m_mean > 0 else 0

        # Max drawdown
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax * 100
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Max drawdown duration
        in_dd = eq < cummax
        dd_duration = 0
        max_dd_duration = 0
        for val in in_dd:
            if val:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        tf_hours = BacktestEngine._timeframe_to_hours(bundle.timeframe)
        max_dd_days = int(max_dd_duration * tf_hours / 24)

        # Max consecutive losing months
        max_consec = 0
        current_consec = 0
        for ret in monthly:
            if ret < 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Trade metrics
        n_trades = len(trades)
        if n_trades > 0:
            wins = trades[trades["pnl_usdt"] > 0]
            losses = trades[trades["pnl_usdt"] <= 0]
            win_rate = len(wins) / n_trades
            avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
            gross_profit = wins["pnl_usdt"].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses["pnl_usdt"].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            if "entry_time" in trades.columns and "exit_time" in trades.columns:
                durations = pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
                avg_duration_hours = durations.mean().total_seconds() / 3600
            else:
                avg_duration_hours = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_duration_hours = 0

        # Cost metrics
        total_fees = trades["cost_usdt"].sum() if n_trades > 0 and "cost_usdt" in trades.columns else 0
        total_funding = trades["funding_cost_usdt"].sum() if n_trades > 0 and "funding_cost_usdt" in trades.columns else 0
        total_slippage = 0  # slippage is embedded in cost_usdt via fill_price
        avg_equity = eq.mean() if len(eq) > 0 else 1
        total_costs = total_fees + total_funding
        cost_drag = (total_costs / avg_equity) * (1 / years) * 100 if avg_equity > 0 else 0

        # BTC correlation
        btc_corr = 0.0
        if btc_returns is not None and len(eq) > 1:
            strat_returns = eq.pct_change().dropna()
            common = strat_returns.index.intersection(btc_returns.index)
            if len(common) > 10:
                btc_corr = float(strat_returns.loc[common].corr(btc_returns.loc[common]))

        # Flags
        flag_overfit = False
        if train_sharpe is not None and train_sharpe > 0:
            flag_overfit = sharpe < 0.5 * train_sharpe

        flag_insufficient = n_trades < 50
        flag_btc_corr = abs(btc_corr) > 0.7
        flag_neg_skew = m_skew < -0.5
        flag_long_dd = max_dd_days > 60
        flag_consec = max_consec > 2
        flag_fragile = False  # set externally

        all_flags = [
            flag_overfit, flag_insufficient, flag_btc_corr, flag_neg_skew,
            flag_long_dd, flag_consec, flag_fragile,
        ]
        passes = not any(all_flags)

        return MetricsBundle(
            total_return_pct=round(total_return_pct, 4),
            annualised_return_pct=round(annualised_return_pct, 4),
            monthly_return_mean=round(m_mean, 4),
            monthly_return_median=round(m_median, 4),
            monthly_return_std=round(m_std, 4),
            monthly_return_worst=round(m_worst, 4),
            monthly_return_best=round(m_best, 4),
            monthly_return_skewness=round(m_skew, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4) if sortino != float("inf") else 999.0,
            max_drawdown_pct=round(max_dd, 4),
            max_drawdown_duration_days=max_dd_days,
            max_consecutive_losing_months=max_consec,
            total_trades=n_trades,
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 4),
            avg_loss_pct=round(avg_loss, 4),
            profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
            avg_trade_duration_hours=round(avg_duration_hours, 2),
            total_fees_usdt=round(total_fees, 4),
            total_funding_usdt=round(total_funding, 4),
            total_slippage_usdt=round(total_slippage, 4),
            cost_drag_annualised_pct=round(cost_drag, 4),
            btc_correlation=round(btc_corr, 4),
            flag_overfit=flag_overfit,
            flag_insufficient_trades=flag_insufficient,
            flag_high_btc_correlation=flag_btc_corr,
            flag_negative_skew=flag_neg_skew,
            flag_long_drawdown=flag_long_dd,
            flag_consecutive_losses=flag_consec,
            flag_fragile_params=flag_fragile,
            passes_all_checks=passes,
        )

    def run_perturbation_test(
        self,
        signal_module: SignalModule,
        params: dict,
        backtest_engine: BacktestEngine,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        perturbation_pct: float = 0.10,
    ) -> dict:
        # Run base
        base_bundle = backtest_engine.run(
            signal_module, params, symbol, timeframe, start, end,
        )
        base_monthly = base_bundle.monthly_returns
        if len(base_monthly) < 2 or base_monthly.std() == 0:
            base_sharpe = 0.0
        else:
            base_sharpe = (base_monthly.mean() / base_monthly.std()) * np.sqrt(12)

        perturbed_sharpes = {}
        max_drop = 0.0

        for name, value in params.items():
            if not isinstance(value, (int, float)):
                continue

            low_val = value * (1 - perturbation_pct)
            high_val = value * (1 + perturbation_pct)

            sharpes = []
            for pval in [low_val, high_val]:
                p = params.copy()
                if isinstance(value, int):
                    p[name] = max(1, int(round(pval)))
                else:
                    p[name] = pval

                try:
                    b = backtest_engine.run(
                        signal_module, p, symbol, timeframe, start, end,
                    )
                    m = b.monthly_returns
                    if len(m) < 2 or m.std() == 0:
                        s = 0.0
                    else:
                        s = (m.mean() / m.std()) * np.sqrt(12)
                except Exception:
                    s = 0.0
                sharpes.append(round(s, 4))

            perturbed_sharpes[name] = sharpes

            for s in sharpes:
                if base_sharpe > 0:
                    drop = (base_sharpe - s) / base_sharpe * 100
                    max_drop = max(max_drop, drop)

        return {
            "base_sharpe": round(base_sharpe, 4),
            "perturbed_sharpes": perturbed_sharpes,
            "max_sharpe_drop_pct": round(max_drop, 2),
            "fragile": max_drop > 30,
        }

    def format_summary(self, metrics: MetricsBundle, title: str = "") -> str:
        lines = []
        if title:
            lines.append(f"=== {title} ===")
        lines.append(f"Total Return:       {metrics.total_return_pct:.2f}%")
        lines.append(f"Annualised Return:  {metrics.annualised_return_pct:.2f}%")
        lines.append(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
        lines.append(f"Sortino Ratio:      {metrics.sortino_ratio:.2f}")
        lines.append(f"Max Drawdown:       {metrics.max_drawdown_pct:.2f}%")
        lines.append(f"Max DD Duration:    {metrics.max_drawdown_duration_days} days")
        lines.append(f"Total Trades:       {metrics.total_trades}")
        lines.append(f"Win Rate:           {metrics.win_rate:.1%}")
        lines.append(f"Profit Factor:      {metrics.profit_factor:.2f}")
        lines.append(f"Avg Win:            {metrics.avg_win_pct:.2f}%")
        lines.append(f"Avg Loss:           {metrics.avg_loss_pct:.2f}%")
        lines.append(f"Monthly Mean:       {metrics.monthly_return_mean:.2f}%")
        lines.append(f"Monthly Worst:      {metrics.monthly_return_worst:.2f}%")
        lines.append(f"Monthly Best:       {metrics.monthly_return_best:.2f}%")
        lines.append(f"Monthly Skewness:   {metrics.monthly_return_skewness:.2f}")
        lines.append(f"BTC Correlation:    {metrics.btc_correlation:.2f}")
        lines.append(f"Total Fees:         ${metrics.total_fees_usdt:.2f}")
        lines.append(f"Cost Drag (ann.):   {metrics.cost_drag_annualised_pct:.2f}%")
        lines.append("")
        lines.append("Flags:")
        flags = [
            ("Overfit", metrics.flag_overfit),
            ("Insufficient Trades", metrics.flag_insufficient_trades),
            ("High BTC Correlation", metrics.flag_high_btc_correlation),
            ("Negative Skew", metrics.flag_negative_skew),
            ("Long Drawdown", metrics.flag_long_drawdown),
            ("Consecutive Losses", metrics.flag_consecutive_losses),
            ("Fragile Params", metrics.flag_fragile_params),
        ]
        for name, flag in flags:
            status = "FAIL" if flag else "PASS"
            lines.append(f"  {name:.<30} {status}")
        lines.append(f"\nAll Checks: {'PASS' if metrics.passes_all_checks else 'FAIL'}")
        return "\n".join(lines)

    def compare(
        self,
        bundles: dict[str, MetricsBundle],
    ) -> str:
        headers = ["Metric"] + list(bundles.keys())
        rows = [
            ("Sharpe", [f"{b.sharpe_ratio:.2f}" for b in bundles.values()]),
            ("Total Return %", [f"{b.total_return_pct:.2f}" for b in bundles.values()]),
            ("Max Drawdown %", [f"{b.max_drawdown_pct:.2f}" for b in bundles.values()]),
            ("Win Rate", [f"{b.win_rate:.1%}" for b in bundles.values()]),
            ("Trades", [str(b.total_trades) for b in bundles.values()]),
            ("Profit Factor", [f"{b.profit_factor:.2f}" for b in bundles.values()]),
            ("Monthly Mean %", [f"{b.monthly_return_mean:.2f}" for b in bundles.values()]),
            ("Monthly Worst %", [f"{b.monthly_return_worst:.2f}" for b in bundles.values()]),
        ]

        col_widths = [max(len(h), 18) for h in headers]
        for label, vals in rows:
            col_widths[0] = max(col_widths[0], len(label))
            for i, v in enumerate(vals):
                col_widths[i + 1] = max(col_widths[i + 1], len(v))

        def fmt_row(cells):
            return " | ".join(c.ljust(w) for c, w in zip(cells, col_widths))

        lines = [fmt_row(headers)]
        lines.append("-+-".join("-" * w for w in col_widths))
        for label, vals in rows:
            lines.append(fmt_row([label] + vals))

        return "\n".join(lines)
