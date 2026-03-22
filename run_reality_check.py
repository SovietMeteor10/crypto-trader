"""
Deep reality check — 8 checks (A-H) to find the honest Sharpe.
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

PARAMS = {'ma_period': 26, 'buffer_pct': 0.0}
HO = ('2024-01-01', '2026-03-21')
BEAR = ('2022-01-01', '2022-12-31')
FULL = ('2021-01-01', '2026-03-21')

results = {}


def run_strat(symbol, params, start, end, fraction=0.02, leverage=3.0):
    sizer = SizerModule(method='fixed_fractional', fraction=fraction, leverage=leverage)
    eng = BacktestEngine(dm, cost_mod, sizer)
    daily = dm.get_ohlcv(symbol, '1d', '2020-01-01', '2026-03-21')
    sig = DailyMASOL(daily_ohlcv=daily)
    return eng.run(sig, params, symbol, '4h', start, end, 1000, 'check')


def sharpe_monthly(b):
    m = b.monthly_returns
    if len(m) < 2 or m.std() == 0: return 0.0
    return float((m.mean() / m.std()) * np.sqrt(12))


def sharpe_daily(b):
    eq = b.equity_curve
    d = eq.resample('1D').last().pct_change().dropna()
    if len(d) < 2 or d.std() == 0: return 0.0
    return float((d.mean() / d.std()) * np.sqrt(252))


def sharpe_4h(b):
    r = b.equity_curve.pct_change().dropna()
    if len(r) < 2 or r.std() == 0: return 0.0
    return float((r.mean() / r.std()) * np.sqrt(252 * 6))


def max_dd(b):
    eq = b.equity_curve
    peak = eq.cummax()
    return float(((eq - peak) / peak).min() * 100)


# ══════════════════════════════════════════════════════════════════
# CHECK A — Daily Sharpe vs Monthly Sharpe
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHECK A: Daily vs Monthly Sharpe")
print("=" * 70)

for label, (s, e) in [('Holdout', HO), ('2022 Bear', BEAR), ('Full', FULL)]:
    b = run_strat('SOL/USDT:USDT', PARAMS, s, e)
    sm, sd, s4 = sharpe_monthly(b), sharpe_daily(b), sharpe_4h(b)
    ret = (b.equity_curve.iloc[-1] / 1000 - 1) * 100
    print(f"\n  {label}:")
    print(f"    Monthly Sharpe: {sm:.2f}")
    print(f"    Daily Sharpe:   {sd:.2f}")
    print(f"    4H Bar Sharpe:  {s4:.2f}")
    print(f"    Total return:   {ret:+.1f}%")

results['A'] = {}
for label, (s, e) in [('holdout', HO), ('bear_2022', BEAR), ('full', FULL)]:
    b = run_strat('SOL/USDT:USDT', PARAMS, s, e)
    results['A'][label] = {
        'monthly': round(sharpe_monthly(b), 2),
        'daily': round(sharpe_daily(b), 2),
        'bar_4h': round(sharpe_4h(b), 2),
    }

# ══════════════════════════════════════════════════════════════════
# CHECK B — ETH generalisation
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK B: Cross-asset generalisation (BTC, ETH, SOL)")
print("=" * 70)

assets = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
periods = [('Holdout', HO), ('2022', BEAR)]

print(f"\n{'Asset':<18} {'Period':>8} {'Mo Sharpe':>10} {'Dy Sharpe':>10} {'Return':>8} {'MaxDD':>7} {'WinMo':>6}")
print("-" * 70)

results['B'] = {}
for sym in assets:
    label = sym.split('/')[0]
    results['B'][label] = {}
    for pname, (s, e) in periods:
        b = run_strat(sym, PARAMS, s, e)
        sm, sd = sharpe_monthly(b), sharpe_daily(b)
        ret = (b.equity_curve.iloc[-1] / 1000 - 1) * 100
        dd = max_dd(b)
        m = b.monthly_returns
        wm = f"{(m > 0).sum()}/{len(m)}"
        print(f"{label:<18} {pname:>8} {sm:10.2f} {sd:10.2f} {ret:7.1f}% {dd:6.1f}% {wm:>6}")
        results['B'][label][pname] = {'monthly': round(sm, 2), 'daily': round(sd, 2),
                                       'return': round(ret, 1), 'max_dd': round(dd, 1)}

# ══════════════════════════════════════════════════════════════════
# CHECK C — Choppy market stress test
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK C: Choppy market stress test")
print("=" * 70)

sol_daily = dm.get_ohlcv('SOL/USDT:USDT', '1d', '2021-01-01', '2026-03-21')
close = sol_daily['close']

choppy = []
window = 90
for i in range(0, len(close) - window, 15):
    w = close.iloc[i:i + window]
    net = abs(w.iloc[-1] / w.iloc[0] - 1)
    rvol = w.pct_change().std() * np.sqrt(252)
    ma = w.rolling(26).mean()
    crosses = (np.sign(w - ma).diff().abs() > 0).sum()
    choppiness = crosses / (net + 0.01)
    choppy.append({
        'start': w.index[0], 'end': w.index[-1],
        'net_change': round(float(net) * 100, 1),
        'rvol': round(float(rvol) * 100, 1),
        'crosses': int(crosses),
        'choppiness': round(float(choppiness), 1),
    })

choppy_df = pd.DataFrame(choppy).sort_values('choppiness', ascending=False)
top5 = choppy_df.head(5)

print(f"\nTop 5 choppiest 90-day windows:")
print(f"{'Start':>12} {'End':>12} {'Net Chg':>8} {'Crosses':>8} {'Choppy':>7}")

results['C'] = []
for _, row in top5.iterrows():
    s = str(row['start'])[:10]
    e = str(row['end'])[:10]
    print(f"{s:>12} {e:>12} {row['net_change']:7.1f}% {row['crosses']:8d} {row['choppiness']:7.1f}")

    b = run_strat('SOL/USDT:USDT', PARAMS, s, e)
    ret = (b.equity_curve.iloc[-1] / 1000 - 1) * 100
    dd = max_dd(b)
    sd = sharpe_daily(b)
    print(f"  → Return: {ret:+.1f}%, MaxDD: {dd:.1f}%, Daily Sharpe: {sd:.2f}")
    results['C'].append({
        'start': s, 'end': e, 'net_change': row['net_change'],
        'crosses': row['crosses'], 'return': round(ret, 1),
        'max_dd': round(dd, 1), 'daily_sharpe': round(sd, 2),
    })

# ══════════════════════════════════════════════════════════════════
# CHECK D — Position sizing Sharpe stability
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK D: Position sizing Sharpe stability")
print("=" * 70)

print(f"\n{'Frac':>6} {'Mo Sharpe':>10} {'Dy Sharpe':>10} {'4H Sharpe':>10} {'Return':>8} {'MaxDD':>7} {'Worst Mo':>9}")
print("-" * 65)

results['D'] = []
for frac in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
    b = run_strat('SOL/USDT:USDT', PARAMS, *HO, fraction=frac)
    sm, sd, s4 = sharpe_monthly(b), sharpe_daily(b), sharpe_4h(b)
    ret = (b.equity_curve.iloc[-1] / 1000 - 1) * 100
    dd = max_dd(b)
    wm = b.monthly_returns.min() * 100
    print(f"{frac:5.0%} {sm:10.2f} {sd:10.2f} {s4:10.2f} {ret:7.1f}% {dd:6.1f}% {wm:8.2f}%")
    results['D'].append({
        'fraction': frac, 'monthly': round(sm, 2), 'daily': round(sd, 2),
        'bar_4h': round(s4, 2), 'return': round(ret, 1),
        'max_dd': round(dd, 1), 'worst_month': round(wm, 2),
    })

# ══════════════════════════════════════════════════════════════════
# CHECK E — Parameter sensitivity across periods
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK E: Parameter sensitivity (daily Sharpe)")
print("=" * 70)

periods_e = {'2022': BEAR, '2023': ('2023-01-01', '2023-12-31'),
             'Holdout': HO, 'Full': FULL}
ma_list = [10, 15, 20, 26, 30, 50, 100]

print(f"\n{'MA':<6}", end='')
for pn in periods_e: print(f"{pn:>10}", end='')
print()

results['E'] = {}
for ma in ma_list:
    print(f"{ma:<6}", end='')
    results['E'][ma] = {}
    for pn, (s, e) in periods_e.items():
        b = run_strat('SOL/USDT:USDT', {'ma_period': ma, 'buffer_pct': 0.0}, s, e)
        sd = sharpe_daily(b)
        print(f"{sd:10.2f}", end='')
        results['E'][ma][pn] = round(sd, 2)
    print()

# ══════════════════════════════════════════════════════════════════
# CHECK F — WF Sharpe distribution (capped at 20)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK F: Walk-forward Sharpe distribution")
print("=" * 70)

# Re-run WF for daily MA
sizer_wf = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
eng_wf = BacktestEngine(dm, cost_mod, sizer_wf)
sig_wf = DailyMASOL(daily_ohlcv=sol_daily)

wf = eng_wf.run_walk_forward(
    sig_wf, 'SOL/USDT:USDT', '4h', '2021-01-01', HO[1],
    train_months=6, test_months=2, gap_weeks=2,
    n_optuna_trials=15, initial_equity=1000,
)

# Compute daily Sharpe for each window, cap at 20
wf_daily_sharpes = []
for r in wf:
    sd = sharpe_daily(r)
    wf_daily_sharpes.append(min(sd, 20))

wf_arr = np.array(wf_daily_sharpes)
print(f"\n  WF Daily Sharpe (capped at 20):")
print(f"    Mean:   {wf_arr.mean():.2f}")
print(f"    Median: {np.median(wf_arr):.2f}")
print(f"    Std:    {wf_arr.std():.2f}")
print(f"    Min:    {wf_arr.min():.2f}")
print(f"    Max:    {wf_arr.max():.2f}")
print(f"    25th:   {np.percentile(wf_arr, 25):.2f}")
print(f"    75th:   {np.percentile(wf_arr, 75):.2f}")
print(f"    % positive: {(wf_arr > 0).mean()*100:.0f}%")

results['F'] = {
    'mean': round(float(wf_arr.mean()), 2),
    'median': round(float(np.median(wf_arr)), 2),
    'std': round(float(wf_arr.std()), 2),
    'min': round(float(wf_arr.min()), 2),
    'max': round(float(wf_arr.max()), 2),
    'pct_positive': round(float((wf_arr > 0).mean() * 100), 0),
    'values': [round(float(s), 2) for s in wf_daily_sharpes],
}

# ══════════════════════════════════════════════════════════════════
# CHECK G — Naive benchmarks
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK G: Naive benchmarks")
print("=" * 70)

sol_4h = dm.get_ohlcv('SOL/USDT:USDT', '4h', *HO)

class ConstSig(SignalModule):
    def __init__(self, v): self._v = v
    @property
    def name(self): return f"const_{self._v}"
    @property
    def parameter_space(self): return {}
    def generate(self, data, params): return pd.Series(self._v, index=data.index)

class RandDailySig(SignalModule):
    def __init__(self, seed, daily_idx):
        rng = np.random.RandomState(seed)
        self._daily_sig = pd.Series(rng.choice([-1, 1], size=len(daily_idx)), index=daily_idx)
    @property
    def name(self): return "rand_daily"
    @property
    def parameter_space(self): return {}
    def generate(self, data, params):
        return self._daily_sig.reindex(data.index, method='ffill').fillna(0).astype(int)

sizer_g = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
eng_g = BacktestEngine(dm, cost_mod, sizer_g)

benchmarks = [
    ('Always long', ConstSig(1)),
    ('Always short', ConstSig(-1)),
    ('Random daily (42)', RandDailySig(42, sol_daily.index)),
    ('Random daily (123)', RandDailySig(123, sol_daily.index)),
    ('Random daily (999)', RandDailySig(999, sol_daily.index)),
    ('Daily MA (26)', sig_wf),
]

print(f"\n{'Benchmark':<22} {'Mo Sharpe':>10} {'Dy Sharpe':>10} {'Return':>8}")
print("-" * 55)

results['G'] = {}
for name, sig in benchmarks:
    p = PARAMS if 'MA' in name else {}
    b = eng_g.run(sig, p, 'SOL/USDT:USDT', '4h', *HO, 1000, 'bench')
    sm, sd = sharpe_monthly(b), sharpe_daily(b)
    ret = (b.equity_curve.iloc[-1] / 1000 - 1) * 100
    print(f"{name:<22} {sm:10.2f} {sd:10.2f} {ret:7.1f}%")
    results['G'][name] = {'monthly': round(sm, 2), 'daily': round(sd, 2), 'return': round(ret, 1)}

# ══════════════════════════════════════════════════════════════════
# CHECK H — Realistic live trading simulation
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK H: Realistic live trading constraints")
print("=" * 70)

# H1: Extra slippage (add 5bp per side via higher cost)
# Simulate by running with zero-cost then subtracting estimated slippage
b_base = run_strat('SOL/USDT:USDT', PARAMS, *HO)
base_ret = (b_base.equity_curve.iloc[-1] / 1000 - 1) * 100
base_sd = sharpe_daily(b_base)
n_trades = len(b_base.trades)

# Estimated extra slippage: 5bp * 2 sides * n_trades * avg_notional
avg_equity = b_base.equity_curve.mean()
avg_notional = avg_equity * 0.02 * 3
slip_total = n_trades * 2 * 0.0005 * avg_notional
slip_pct = slip_total / 1000 * 100

# H2: Execution delay — shift signal by 2 bars instead of 1
# Modify strategy to add extra shift
class DelayedMASOL(SignalModule):
    def __init__(self, daily, delay=2):
        self._daily = daily
        self._delay = delay
    @property
    def name(self): return "delayed_ma"
    @property
    def parameter_space(self): return {}
    def generate(self, data, params):
        d = self._daily['close']
        ma = d.rolling(26).mean()
        sig = pd.Series(0, index=d.index)
        sig[d > ma] = 1
        sig[d < ma] = -1
        sig_4h = sig.reindex(data.index, method='ffill').fillna(0)
        # Extra delay: shift by delay-1 additional bars (engine already shifts 1)
        return sig_4h.shift(self._delay - 1).fillna(0).astype(int)

delayed_sig = DelayedMASOL(sol_daily, delay=2)
b_delayed = eng_g.run(delayed_sig, {}, 'SOL/USDT:USDT', '4h', *HO, 1000, 'delayed')
delayed_ret = (b_delayed.equity_curve.iloc[-1] / 1000 - 1) * 100
delayed_sd = sharpe_daily(b_delayed)

# H3: Funding rate estimate
# 49% long * avg 0.01% per 8H = 0.049% per 8H for long side
# 51% short * avg -0.005% per 8H = -0.026% per 8H for short side
# Net per 8H: 0.049% - 0.026% = 0.023% drag
# Days in holdout: ~810
# Total funding: 810 * 3 * 0.00023 * avg_notional
funding_drag = 810 * 3 * 0.00023 * avg_notional
funding_pct = funding_drag / 1000 * 100

adjusted_ret = base_ret - slip_pct - funding_pct
# Adjusted Sharpe: approximate by reducing mean return proportionally
ratio = adjusted_ret / base_ret if base_ret > 0 else 1
adjusted_sd = base_sd * ratio

print(f"\n  Baseline:              return={base_ret:+.1f}%, daily Sharpe={base_sd:.2f}")
print(f"  + 5bp extra slippage:  drag={slip_pct:.1f}%")
print(f"  + 8H execution delay:  return={delayed_ret:+.1f}%, daily Sharpe={delayed_sd:.2f}")
print(f"  + Funding costs:       drag={funding_pct:.1f}%")
print(f"\n  ADJUSTED (all drags):  return={adjusted_ret:+.1f}%, approx daily Sharpe={adjusted_sd:.2f}")

results['H'] = {
    'baseline_return': round(base_ret, 1), 'baseline_daily_sharpe': round(base_sd, 2),
    'slippage_drag_pct': round(slip_pct, 1),
    'delayed_return': round(delayed_ret, 1), 'delayed_sharpe': round(delayed_sd, 2),
    'funding_drag_pct': round(funding_pct, 1),
    'adjusted_return': round(adjusted_ret, 1), 'adjusted_sharpe': round(adjusted_sd, 2),
}

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("REALITY CHECK SUMMARY")
print("=" * 70)

ho_daily = results['A']['holdout']['daily']
print(f"\n  HONEST SHARPE (daily, holdout):     {ho_daily}")
print(f"  Reported Sharpe (monthly, holdout): {results['A']['holdout']['monthly']}")
print(f"  Inflation factor:                   {results['A']['holdout']['monthly'] / ho_daily:.1f}x")

with open('reality_check_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved to reality_check_results.json")
