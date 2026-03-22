"""
Daily MA Enhancement: Trade frequency, Stop losses, Leverage scaling.
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

dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
cost_mod = CostModule()

PERIODS = {
    'Holdout': ('2024-01-01', '2026-03-21'),
    '2022': ('2022-01-01', '2022-12-31'),
}
results = {}


def extract_trades(signals, prices, equity=10000.0, fraction=0.10, leverage=3.0, fee_rate=0.0005):
    """Extract holding periods and compute stats."""
    trades = []
    pos = 0; ep = None; et = None; eq = equity
    for i in range(1, len(signals)):
        ns = signals.iloc[i-1]
        if ns != pos:
            if pos != 0 and ep is not None:
                xp = prices.iloc[i]; xt = prices.index[i]
                rr = (xp - ep) / ep * pos
                notional = eq * fraction * leverage
                gpnl = notional * rr; cost = notional * fee_rate
                npnl = gpnl - cost; eq += npnl
                trades.append({'dir': 'L' if pos == 1 else 'S', 'ret': float(rr*100),
                               'npnl': float(npnl), 'gpnl': float(gpnl), 'dur': (xt-et).days,
                               'win': npnl > 0})
            if ns != 0:
                pos = ns; ep = prices.iloc[i]; et = prices.index[i]
                eq -= eq * fraction * leverage * fee_rate
            else:
                pos = 0; ep = None
    return pd.DataFrame(trades), eq


def stats(df, eq_start=10000.0, eq_end=None):
    if df.empty: return {'trades': 0}
    w = df[df['win']]; l = df[~df['win']]
    n = len(df); wr = len(w)/n*100
    gw = w['gpnl'].sum() if len(w) > 0 else 0
    gl = abs(l['gpnl'].sum()) if len(l) > 0 else 0.001
    pf = gw/gl
    mcl = 0; cl = 0
    for v in df['win']:
        cl = 0 if v else cl + 1; mcl = max(mcl, cl)
    ret = ((eq_end or eq_start)/eq_start - 1)*100
    return {'trades': n, 'wr': round(wr,1), 'pf': round(pf,2), 'mcl': mcl,
            'avg_win': round(float(w['ret'].mean()),2) if len(w)>0 else 0,
            'avg_loss': round(float(l['ret'].mean()),2) if len(l)>0 else 0,
            'return': round(ret,1), 'tpm': round(n/(len(df)*0.33 if len(df)>0 else 1),1)}


def run_engine(symbol, params, start, end, fraction=0.10):
    sizer = SizerModule(method='fixed_fractional', fraction=fraction, leverage=3.0)
    eng = BacktestEngine(dm, cost_mod, sizer)
    daily = dm.get_ohlcv(symbol, '1d', '2020-01-01', '2026-03-21')
    sig = DailyMASOL(daily_ohlcv=daily)
    b = eng.run(sig, params, symbol, '4h', start, end, 10000, 'test')
    eq = b.equity_curve
    m = b.monthly_returns
    peak = eq.cummax(); dd = ((eq-peak)/peak).min()*100
    sharpe_d = 0
    d = eq.resample('1D').last().pct_change().dropna()
    if len(d) > 1 and d.std() > 0: sharpe_d = float(d.mean()/d.std()*np.sqrt(252))
    return {
        'return': round(float((eq.iloc[-1]/10000-1)*100),1),
        'max_dd': round(float(dd),1),
        'worst_mo': round(float(m.min()*100),2) if len(m)>0 else 0,
        'mean_mo': round(float(m.mean()*100),2) if len(m)>0 else 0,
        'sharpe_d': round(sharpe_d,2),
        'trades': len(b.trades),
    }


def print_table(rows, title):
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    print(f"{'Config':<30} {'WR%':>5} {'PF':>6} {'MCL':>4} {'MaxDD':>7} {'WrstMo':>7} {'MnMo':>7} {'Ret':>8} {'T/Mo':>5}")
    print("-"*90)
    for r in rows:
        print(f"{r['name']:<30} {r.get('wr',''):>5} {r.get('pf',''):>6} {r.get('mcl',''):>4} "
              f"{r.get('max_dd',''):>6}% {r.get('worst_mo',''):>6}% {r.get('mean_mo',''):>6}% "
              f"{r.get('return',''):>7}% {r.get('tpm',''):>5}")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Trade Frequency
# ══════════════════════════════════════════════════════════════════
print("=" * 90)
print("EXPERIMENT 1: Trade Frequency")
print("=" * 90)

for pname, (s, e) in PERIODS.items():
    rows = []

    # Baseline
    sol_4h = dm.get_ohlcv('SOL/USDT:USDT', '4h', s, e)
    sol_1d = dm.get_ohlcv('SOL/USDT:USDT', '1d', '2020-01-01', '2026-03-21')
    strat = DailyMASOL(daily_ohlcv=sol_1d)
    sigs = strat.generate(sol_4h, {'ma_period': 26, 'buffer_pct': 0.0})
    tdf, eq_end = extract_trades(sigs, sol_4h['close'])
    months = max((pd.Timestamp(e) - pd.Timestamp(s)).days / 30, 1)
    st = stats(tdf, 10000, eq_end)
    eng_r = run_engine('SOL/USDT:USDT', {'ma_period': 26, 'buffer_pct': 0.0}, s, e)
    st.update({'name': 'Baseline (SOL MA26)', 'max_dd': eng_r['max_dd'],
               'worst_mo': eng_r['worst_mo'], 'mean_mo': eng_r['mean_mo'],
               'return': eng_r['return'], 'tpm': round(st['trades']/months, 1)})
    rows.append(st)

    # 1A: Multi-asset
    total_pnl = None
    total_trades = 0
    for sym in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
        d4h = dm.get_ohlcv(sym, '4h', s, e)
        d1d = dm.get_ohlcv(sym, '1d', '2020-01-01', '2026-03-21')
        sig_a = DailyMASOL(daily_ohlcv=d1d)
        sigs_a = sig_a.generate(d4h, {'ma_period': 26, 'buffer_pct': 0.0})
        tdf_a, _ = extract_trades(sigs_a, d4h['close'], fraction=0.033)  # 10%/3 per asset
        total_trades += len(tdf_a)
        pos = sigs_a.shift(1).fillna(0)
        ret = d4h['close'].pct_change()
        pnl = pos * ret * 10000 * 0.033 * 3
        cost_s = sigs_a.diff().abs().fillna(0) * 10000 * 0.033 * 3 * 0.0005
        net = pnl - cost_s
        if total_pnl is None: total_pnl = net
        else: total_pnl = total_pnl.add(net, fill_value=0)

    eq_ma = 10000 + total_pnl.cumsum()
    mo_ma = eq_ma.resample('ME').last().pct_change().dropna()
    dd_ma = ((eq_ma - eq_ma.cummax()) / eq_ma.cummax()).min() * 100
    rows.append({'name': '1A: Multi (BTC+ETH+SOL)', 'trades': total_trades,
                 'wr': '', 'pf': '', 'mcl': '',
                 'max_dd': round(float(dd_ma), 1),
                 'worst_mo': round(float(mo_ma.min()*100), 2) if len(mo_ma)>0 else 0,
                 'mean_mo': round(float(mo_ma.mean()*100), 2) if len(mo_ma)>0 else 0,
                 'return': round(float((eq_ma.iloc[-1]/10000-1)*100), 1),
                 'tpm': round(total_trades/months, 1)})

    # 1B: Shorter MA
    for ma in [10, 15, 20]:
        sigs_b = strat.generate(sol_4h, {'ma_period': ma, 'buffer_pct': 0.0})
        tdf_b, eq_b = extract_trades(sigs_b, sol_4h['close'])
        st_b = stats(tdf_b, 10000, eq_b)
        eng_b = run_engine('SOL/USDT:USDT', {'ma_period': ma, 'buffer_pct': 0.0}, s, e)
        st_b.update({'name': f'1B: SOL MA{ma}', 'max_dd': eng_b['max_dd'],
                     'worst_mo': eng_b['worst_mo'], 'mean_mo': eng_b['mean_mo'],
                     'return': eng_b['return'], 'tpm': round(st_b['trades']/months, 1)})
        rows.append(st_b)

    # 1C: Dual MA crossover (fast=10, slow=26)
    fast_ma = sol_1d['close'].ewm(span=10, adjust=False).mean()
    slow_ma = sol_1d['close'].ewm(span=26, adjust=False).mean()
    dual_sig = pd.Series(0, index=sol_1d.index)
    dual_sig[fast_ma > slow_ma] = 1
    dual_sig[fast_ma < slow_ma] = -1
    dual_4h = dual_sig.reindex(sol_4h.index, method='ffill').fillna(0).astype(int)
    tdf_c, eq_c = extract_trades(dual_4h, sol_4h['close'])
    st_c = stats(tdf_c, 10000, eq_c)
    st_c.update({'name': '1C: Dual MA (10/26)', 'max_dd': '', 'worst_mo': '', 'mean_mo': '',
                 'return': round((eq_c/10000-1)*100, 1), 'tpm': round(st_c['trades']/months, 1)})
    rows.append(st_c)

    print_table(rows, f"EXPERIMENT 1 — {pname}")
    results[f'exp1_{pname}'] = rows

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Stop Losses
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("EXPERIMENT 2: Stop Losses")
print("=" * 90)

def apply_stop_loss(signals, prices, stop_pct):
    """Apply fixed % stop loss. After stop, go flat until next daily signal change."""
    out = signals.copy()
    pos = 0; ep = None; stopped = False
    for i in range(1, len(signals)):
        ns = signals.iloc[i-1]
        p = prices.iloc[i]
        if ns != pos and not stopped:
            pos = ns; ep = p; stopped = False
        if pos != 0 and ep is not None:
            loss = (p - ep) / ep * pos
            if loss < -stop_pct/100:
                out.iloc[i] = 0; stopped = True; pos = 0; ep = None
                continue
        if stopped and ns != 0:
            if signals.iloc[max(0,i-7):i].diff().abs().sum() > 0:  # signal changed recently
                stopped = False; pos = ns; ep = p
        out.iloc[i] = pos
    return out

for pname, (s, e) in PERIODS.items():
    rows = []
    sol_4h = dm.get_ohlcv('SOL/USDT:USDT', '4h', s, e)
    sigs_base = strat.generate(sol_4h, {'ma_period': 26, 'buffer_pct': 0.0})
    tdf_base, eq_base = extract_trades(sigs_base, sol_4h['close'])
    st_base = stats(tdf_base, 10000, eq_base)
    st_base.update({'name': 'Baseline (no stop)', 'max_dd': '', 'worst_mo': '',
                    'mean_mo': '', 'return': round((eq_base/10000-1)*100,1),
                    'tpm': round(st_base['trades']/months, 1)})
    rows.append(st_base)

    # 2A: Fixed stop
    for sp in [3, 5, 8, 10]:
        sigs_s = apply_stop_loss(sigs_base, sol_4h['close'], sp)
        tdf_s, eq_s = extract_trades(sigs_s, sol_4h['close'])
        st_s = stats(tdf_s, 10000, eq_s)
        st_s.update({'name': f'2A: Stop {sp}%', 'max_dd': '', 'worst_mo': '',
                     'mean_mo': '', 'return': round((eq_s/10000-1)*100,1),
                     'tpm': round(st_s['trades']/months,1)})
        rows.append(st_s)

    # 2C: Time stop
    for days in [3, 5, 7]:
        def apply_time_stop(signals, prices, max_bars):
            out = signals.copy(); pos = 0; entry_bar = 0
            for i in range(1, len(signals)):
                ns = signals.iloc[i-1]
                if ns != pos:
                    pos = ns; entry_bar = i
                if pos != 0 and (i - entry_bar) >= max_bars * 6:
                    # Check if still profitable
                    if i > entry_bar:
                        ret = (prices.iloc[i] - prices.iloc[entry_bar]) / prices.iloc[entry_bar] * pos
                        if ret < 0:
                            out.iloc[i] = 0; pos = 0
                out.iloc[i] = pos if pos != 0 else out.iloc[i]
            return out
        sigs_t = apply_time_stop(sigs_base, sol_4h['close'], days)
        tdf_t, eq_t = extract_trades(sigs_t, sol_4h['close'])
        st_t = stats(tdf_t, 10000, eq_t)
        st_t.update({'name': f'2C: Time {days}d', 'max_dd': '', 'worst_mo': '',
                     'mean_mo': '', 'return': round((eq_t/10000-1)*100,1),
                     'tpm': round(st_t['trades']/months,1)})
        rows.append(st_t)

    print_table(rows, f"EXPERIMENT 2 — {pname}")
    results[f'exp2_{pname}'] = rows

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Leverage Scaling
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("EXPERIMENT 3: Leverage Scaling")
print("=" * 90)

for pname, (s, e) in PERIODS.items():
    rows = []
    for frac in [0.10, 0.20, 0.33, 0.50, 0.67, 1.00]:
        eff_lev = frac * 3
        r = run_engine('SOL/USDT:USDT', {'ma_period': 26, 'buffer_pct': 0.0}, s, e, fraction=frac)
        # Also get trade stats
        sol_4h = dm.get_ohlcv('SOL/USDT:USDT', '4h', s, e)
        sigs_l = strat.generate(sol_4h, {'ma_period': 26, 'buffer_pct': 0.0})
        tdf_l, eq_l = extract_trades(sigs_l, sol_4h['close'], fraction=frac)
        st_l = stats(tdf_l, 10000, eq_l)
        st_l.update({'name': f'Lev {eff_lev:.1f}x (frac={frac:.0%})', 'max_dd': r['max_dd'],
                     'worst_mo': r['worst_mo'], 'mean_mo': r['mean_mo'],
                     'return': r['return'], 'tpm': round(st_l['trades']/months,1)})
        rows.append(st_l)

    print_table(rows, f"EXPERIMENT 3 — {pname}")
    results[f'exp3_{pname}'] = rows

    # Find max safe leverage
    for r in rows:
        if r['max_dd'] != '' and float(str(r['max_dd']).replace('%','')) < -15:
            break
        safe = r
    print(f"\n  Max safe leverage ({pname}): {safe['name']} "
          f"(DD={safe['max_dd']}%, worst mo={safe['worst_mo']}%)")

# Save
with open('enhance_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Strategy log
entry = "\n\n## Enhancement Experiments\n"
for k, rows in results.items():
    entry += f"\n### {k}\n"
    for r in rows:
        entry += f"  {r['name']}: WR={r.get('wr','')}%, PF={r.get('pf','')}, DD={r.get('max_dd','')}%, Ret={r.get('return','')}%\n"

with open('STRATEGY_LOG.md', 'r') as f:
    content = f.read()
with open('STRATEGY_LOG.md', 'w') as f:
    f.write(content + entry)

print("\nSaved to enhance_results.json and STRATEGY_LOG.md")
