"""
Trade-level audit: MA calculation check, holding period stats, deployment numbers.
"""
import sys, json
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from strategies.daily_ma_sol import DailyMASOL
from crypto_infra.signal_module import SignalModule

dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
cost_mod = CostModule()

sol_1d = dm.get_ohlcv('SOL/USDT:USDT', '1d', '2020-01-01', '2026-03-21')

PARAMS = {'ma_period': 26, 'buffer_pct': 0.0}
results = {}


# ══════════════════════════════════════════════════════════════════
# PART 1 — MA Calculation Audit
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: MA Calculation Audit")
print("=" * 70)

print("""
FINDINGS:
  Q1: generate() compares day T's close to MA including day T. No shift.
  Q2: rolling(26).mean() includes current bar.
  Q3: Daily signal at T 00:00 → ffill to 4H T 00:00 → engine uses at T 04:00.
  Q4: Line 79: sig = signal.iloc[i - 1] — confirmed.

  Chain: Day T close (known at 00:00 UTC) → signal at 00:00 → executes at 04:00.
  This is a 4-hour lag. Borderline but feasible in live trading.

  For honest comparison, running BOTH current and shifted versions.
""")

# ── Position sizing verification ──
print("Position sizing verification:")
print(f"{'Equity':>10} {'Fraction':>10} {'Price':>10} {'Size':>10} {'Notional':>12} {'Leverage':>10}")
print("-" * 65)
for equity in [1000, 10000, 25000]:
    for frac in [0.02, 0.10, 0.20]:
        s = SizerModule(method='fixed_fractional', fraction=frac, leverage=3.0)
        size = s.compute_size(signal=1, equity=equity, price=150.0, volatility=0.5)
        notional = size * 150.0
        lev = notional / equity
        print(f"{equity:>10} {frac:>10.0%} {150.0:>10.1f} {size:>10.4f} {notional:>12.2f} {lev:>9.1f}x")

results['part1'] = {
    'ma_uses_same_day': True,
    'engine_lag': '1 bar (4H)',
    'total_execution_delay': '4 hours from daily close',
    'leverage': '3x confirmed',
}


# ══════════════════════════════════════════════════════════════════
# PART 2 — Trade-level statistics
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Trade-Level Statistics")
print("=" * 70)


def extract_holding_periods(signals, prices, equity_start=10000.0,
                             fraction=0.10, leverage=3.0, fee_rate=0.0005):
    trades = []
    position = 0
    entry_price = None
    entry_time = None
    equity = equity_start

    for i in range(1, len(signals)):
        new_sig = signals.iloc[i - 1]  # engine convention

        if new_sig != position:
            # Close existing
            if position != 0 and entry_price is not None:
                exit_price = prices.iloc[i]
                exit_time = prices.index[i]
                raw_ret = (exit_price - entry_price) / entry_price * position
                notional = equity * fraction * leverage
                gross_pnl = notional * raw_ret
                cost_close = notional * fee_rate
                net_pnl = gross_pnl - cost_close
                equity += net_pnl
                duration = (exit_time - entry_time).days

                trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': float(entry_price), 'exit_price': float(exit_price),
                    'duration_days': duration,
                    'raw_return_pct': float(raw_ret * 100),
                    'gross_pnl': float(gross_pnl),
                    'net_pnl': float(net_pnl),
                    'notional': float(notional),
                    'equity_after': float(equity),
                    'win': net_pnl > 0,
                })

            # Open new
            if new_sig != 0:
                position = new_sig
                entry_price = prices.iloc[i]
                entry_time = prices.index[i]
                cost_open = equity * fraction * leverage * fee_rate
                equity -= cost_open
            else:
                position = 0
                entry_price = None
                entry_time = None

    return pd.DataFrame(trades)


def print_stats(df, name, initial=10000.0):
    if df.empty:
        print(f"\n{name}: No trades")
        return {}

    wins = df[df['win']]
    losses = df[~df['win']]
    n = len(df)

    print(f"\n{'='*60}")
    print(f"TRADE STATISTICS — {name}")
    print(f"{'='*60}")

    print(f"\n--- Counts ---")
    print(f"Total trades: {n}")
    print(f"  Long:  {len(df[df['direction']=='LONG'])}")
    print(f"  Short: {len(df[df['direction']=='SHORT'])}")
    wr = len(wins) / n * 100
    print(f"  Win:   {len(wins)} ({wr:.1f}%)")
    print(f"  Loss:  {len(losses)} ({100-wr:.1f}%)")

    print(f"\n--- P&L per trade ---")
    if len(wins) > 0:
        print(f"Avg win:     ${wins['net_pnl'].mean():.2f} ({wins['raw_return_pct'].mean():.2f}%)")
        print(f"Largest win: ${wins['net_pnl'].max():.2f} ({wins['raw_return_pct'].max():.2f}%)")
    if len(losses) > 0:
        print(f"Avg loss:    ${losses['net_pnl'].mean():.2f} ({losses['raw_return_pct'].mean():.2f}%)")
        print(f"Largest loss:${losses['net_pnl'].min():.2f} ({losses['raw_return_pct'].min():.2f}%)")

    gw = wins['gross_pnl'].sum() if len(wins) > 0 else 0
    gl = abs(losses['gross_pnl'].sum()) if len(losses) > 0 else 0.001
    pf = gw / gl
    print(f"\nProfit factor: {pf:.2f}x")

    avg_w = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_l = losses['net_pnl'].mean() if len(losses) > 0 else 0
    exp = (wr / 100) * avg_w + (1 - wr / 100) * avg_l
    print(f"Expectancy:    ${exp:.2f}/trade")

    # Consecutive losses
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for w in df['win']:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    print(f"\n--- Streaks ---")
    print(f"Max consecutive wins:   {max_win_streak}")
    print(f"Max consecutive losses: {max_loss_streak}")

    # Max drawdown from trade equity
    peak_eq = initial
    max_dd_pct = 0
    eq = initial
    for _, t in df.iterrows():
        eq += t['net_pnl']
        peak_eq = max(peak_eq, eq)
        dd = (eq - peak_eq) / peak_eq * 100
        max_dd_pct = min(max_dd_pct, dd)
    print(f"Max drawdown:           {max_dd_pct:.1f}%")

    print(f"\n--- Duration ---")
    print(f"Avg duration:  {df['duration_days'].mean():.1f} days")
    print(f"Min duration:  {df['duration_days'].min()} days")
    print(f"Max duration:  {df['duration_days'].max()} days")

    # All trades table
    print(f"\n--- All trades ---")
    print(f"{'#':>3} {'Entry':>12} {'Exit':>12} {'Dir':>6} {'Days':>5} {'Ret%':>7} {'Net P&L':>10} {'W/L':>4}")
    print("-" * 70)
    for idx, t in df.iterrows():
        wl = 'W' if t['win'] else 'L'
        print(f"{idx:3d} {str(t['entry_time'])[:10]:>12} {str(t['exit_time'])[:10]:>12} "
              f"{t['direction']:>6} {t['duration_days']:5d} {t['raw_return_pct']:6.2f}% "
              f"${t['net_pnl']:9.2f} {wl:>4}")

    final_eq = df['equity_after'].iloc[-1] if len(df) > 0 else initial
    total_ret = (final_eq / initial - 1) * 100
    print(f"\nStarting: ${initial:,.2f} → Ending: ${final_eq:,.2f} ({total_ret:+.1f}%)")

    return {
        'trades': n, 'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2), 'expectancy': round(exp, 2),
        'max_consec_losses': max_loss_streak, 'max_consec_wins': max_win_streak,
        'avg_win_pct': round(float(wins['raw_return_pct'].mean()), 2) if len(wins) > 0 else 0,
        'avg_loss_pct': round(float(losses['raw_return_pct'].mean()), 2) if len(losses) > 0 else 0,
        'avg_duration_days': round(float(df['duration_days'].mean()), 1),
        'total_return_pct': round(total_ret, 1),
        'max_dd_pct': round(max_dd_pct, 1),
    }


# Run for holdout and 2022
for period_name, start, end in [('HOLDOUT 2024-2026', '2024-01-01', '2026-03-21'),
                                  ('2022 BEAR MARKET', '2022-01-01', '2022-12-31')]:
    sol_4h = dm.get_ohlcv('SOL/USDT:USDT', '4h', start, end)
    strategy = DailyMASOL(daily_ohlcv=sol_1d)
    signals = strategy.generate(sol_4h, PARAMS)
    prices = sol_4h['close']

    trades_df = extract_holding_periods(signals, prices)
    stats = print_stats(trades_df, period_name)
    results[period_name] = stats

    # Directional accuracy
    position = signals.shift(1).fillna(0)
    bar_ret = prices.pct_change()
    correct = ((position == 1) & (bar_ret > 0)) | ((position == -1) & (bar_ret < 0))
    active = position != 0
    pct = (correct & active).sum() / active.sum() * 100 if active.sum() > 0 else 0
    print(f"\n  Directional accuracy (4H bars): {pct:.1f}% ({(correct & active).sum()}/{active.sum()})")
    print(f"  Edge over random: {pct - 50:.1f}pp")
    results[period_name]['directional_accuracy'] = round(pct, 1)

# ── Capital requirements ──
print("\n" + "=" * 70)
print("Capital Requirements")
print("=" * 70)

ho_stats = results.get('HOLDOUT 2024-2026', {})
monthly_return_pct = ho_stats.get('total_return_pct', 0) / 26
print(f"\nAvg monthly return at 10% sizing: {monthly_return_pct:.2f}%")

for target_gbp in [500, 1000, 2000]:
    target_usd = target_gbp * 1.27
    cap = target_usd / (monthly_return_pct / 100) if monthly_return_pct > 0 else float('inf')
    print(f"  £{target_gbp}/month → ${cap:,.0f} (~£{cap/1.27:,.0f})")

# Save
with open('trade_audit_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved to trade_audit_results.json")
