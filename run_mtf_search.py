"""
Multi-timeframe MA search: standalone, hierarchy, confidence, selection.
"""
import sys, json, time
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
from crypto_infra.data_module import DataModule

dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

SYMBOLS = ['SOL/USDT:USDT', 'BTC/USDT:USDT']
PERIODS = {'Holdout': ('2024-01-01', '2026-03-21'), '2022': ('2022-01-01', '2022-12-31')}
FEE = 0.0005
FRAC = 0.10
LEV = 3.0


def backtest_ma(close_series, ma_period, fraction=FRAC):
    """Pure MA backtest with shift(1) to prevent lookahead. Returns metrics dict."""
    close = close_series
    ma = close.rolling(ma_period).mean()
    raw = (close > ma).astype(int) * 2 - 1
    signal = raw.shift(1).fillna(0)  # CRITICAL: use prior bar
    position = signal.shift(1).fillna(0)  # execute bar after signal
    ret = close.pct_change()

    notional = 1000 * fraction * LEV
    gpnl = position * ret * notional
    cost_s = signal.diff().abs().fillna(0) * notional * FEE
    net = gpnl - cost_s
    equity = 1000 + net.cumsum()

    # Daily metrics
    d = equity.resample('1D').last().pct_change().dropna()
    sd = float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 1 and d.std() > 0 else 0
    m = equity.resample('ME').last().pct_change().dropna()
    dd = float((equity / equity.cummax() - 1).min() * 100)
    wd = float(d.min() * 100) if len(d) > 0 else 0
    total_cost = float(cost_s.sum())
    months = max(len(m), 1)
    cost_monthly = total_cost / months

    # Trades
    trades = []
    pos = 0; ep = None
    for i in range(2, len(signal)):
        ns = int(signal.iloc[i])
        if ns != pos:
            if pos != 0 and ep is not None:
                xp = close.iloc[i]
                rr = (xp - ep) / ep * pos
                trades.append({'ret': float(rr), 'win': rr > 0})
            if ns != 0: pos = ns; ep = close.iloc[i]
            else: pos = 0; ep = None

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['ret', 'win'])
    wr = float(tdf['win'].mean() * 100) if len(tdf) > 0 else 0
    mcl = 0; cl = 0
    for w in tdf.get('win', []):
        cl = 0 if w else cl + 1; mcl = max(mcl, cl)
    if len(tdf) > 0:
        w_sum = tdf[tdf['win']]['ret'].sum()
        l_sum = abs(tdf[~tdf['win']]['ret'].sum()) if len(tdf[~tdf['win']]) > 0 else 0.001
        pf = float(w_sum / l_sum) if l_sum > 0 else 99
    else:
        pf = 0
    tpm = len(tdf) / months
    d5 = int((d < -0.05).sum())
    total_ret = float((equity.iloc[-1] / 1000 - 1) * 100)

    return {
        'sharpe_d': round(sd, 2), 'return': round(total_ret, 1),
        'max_dd': round(dd, 1), 'worst_day': round(wd, 2),
        'days_5pct': d5, 'wr': round(wr, 1), 'pf': round(pf, 2),
        'mcl': mcl, 'tpm': round(tpm, 1), 'trades': len(tdf),
        'worst_mo': round(float(m.min() * 100), 2) if len(m) > 0 else 0,
        'mean_mo': round(float(m.mean() * 100), 2) if len(m) > 0 else 0,
        'cost_mo': round(cost_monthly, 2),
    }


def hierarchy_backtest(daily_close, entry_close, dir_ma=26, entry_ma=12, fraction=FRAC):
    """Daily direction + shorter TF entry. Both shifted(1)."""
    d_ma = daily_close.rolling(dir_ma).mean()
    d_sig = ((daily_close > d_ma).astype(int) * 2 - 1).shift(1)

    e_ma = entry_close.rolling(entry_ma).mean()
    e_sig = ((entry_close > e_ma).astype(int) * 2 - 1).shift(1)

    d_aligned = d_sig.reindex(entry_close.index, method='ffill').fillna(0)

    combined = pd.Series(0, index=entry_close.index)
    combined[(e_sig == 1) & (d_aligned == 1)] = 1
    combined[(e_sig == -1) & (d_aligned == -1)] = -1

    position = combined.shift(1).fillna(0)
    ret = entry_close.pct_change()
    notional = 1000 * fraction * LEV
    gpnl = position * ret * notional
    cost_s = combined.diff().abs().fillna(0) * notional * FEE
    net = gpnl - cost_s
    equity = 1000 + net.cumsum()

    d = equity.resample('1D').last().pct_change().dropna()
    sd = float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 1 and d.std() > 0 else 0
    m = equity.resample('ME').last().pct_change().dropna()
    dd = float((equity / equity.cummax() - 1).min() * 100)
    wd = float(d.min() * 100) if len(d) > 0 else 0
    months = max(len(m), 1)

    trades = []
    pos = 0; ep = None
    for i in range(2, len(combined)):
        ns = int(combined.iloc[i])
        if ns != pos:
            if pos != 0 and ep is not None:
                rr = (entry_close.iloc[i] - ep) / ep * pos
                trades.append({'ret': float(rr), 'win': rr > 0})
            if ns != 0: pos = ns; ep = entry_close.iloc[i]
            else: pos = 0; ep = None

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['ret', 'win'])
    wr = float(tdf['win'].mean() * 100) if len(tdf) > 0 else 0
    mcl = 0; cl = 0
    for w in tdf.get('win', []):
        cl = 0 if w else cl + 1; mcl = max(mcl, cl)
    if len(tdf) > 0:
        w_sum = tdf[tdf['win']]['ret'].sum()
        l_sum = abs(tdf[~tdf['win']]['ret'].sum()) if len(tdf[~tdf['win']]) > 0 else 0.001
        pf = float(w_sum / l_sum) if l_sum > 0 else 99
    else:
        pf = 0

    return {
        'sharpe_d': round(sd, 2), 'return': round(float((equity.iloc[-1]/1000-1)*100), 1),
        'max_dd': round(dd, 1), 'worst_day': round(wd, 2),
        'wr': round(wr, 1), 'pf': round(pf, 2), 'mcl': mcl,
        'tpm': round(len(tdf) / months, 1), 'trades': len(tdf),
        'worst_mo': round(float(m.min()*100), 2) if len(m) > 0 else 0,
        'mean_mo': round(float(m.mean()*100), 2) if len(m) > 0 else 0,
        'cost_mo': round(float(cost_s.sum() / months), 2),
    }


# ══════════════════════════════════════════════════════════════════
# PART 1 — Standalone timeframe testing
# ══════════════════════════════════════════════════════════════════
print("=" * 100)
print("PART 1: Standalone Timeframe Testing")
print("=" * 100)

tf_ma_combos = {
    '1h':  [24, 48, 72, 120, 168],
    '4h':  [6, 12, 18, 30, 42],
    '1d':  [10, 15, 20, 26, 50],
}
cal_days = {'1h': 1/24, '4h': 1/6, '1d': 1}

p1_results = {}
for symbol in SYMBOLS:
    label = symbol.split('/')[0]
    for pname, (s, e) in PERIODS.items():
        key = f'{label}_{pname}'
        rows = []
        for tf, mas in tf_ma_combos.items():
            ohlcv = dm.get_ohlcv(symbol, tf, s, e)
            for ma in mas:
                r = backtest_ma(ohlcv['close'], ma)
                cd = round(ma * cal_days[tf], 1)
                r.update({'tf': tf, 'ma': ma, 'cal_days': cd})
                rows.append(r)
        p1_results[key] = rows

        print(f"\n{label} — {pname}")
        print(f"{'TF':>4} {'MA':>4} {'CalD':>5} {'Sharpe':>7} {'WR%':>5} {'PF':>6} {'MCL':>4} "
              f"{'T/Mo':>5} {'MaxDD':>7} {'WrstD':>7} {'MnMo':>6} {'CostMo':>7} {'Ret':>8}")
        print("-" * 95)
        for r in rows:
            print(f"{r['tf']:>4} {r['ma']:>4} {r['cal_days']:>5} {r['sharpe_d']:>7} "
                  f"{r['wr']:>5} {r['pf']:>6} {r['mcl']:>4} {r['tpm']:>5} "
                  f"{r['max_dd']:>6}% {r['worst_day']:>6}% {r['mean_mo']:>5}% "
                  f"${r['cost_mo']:>6} {r['return']:>7}%")

with open('mtf_part1.json', 'w') as f:
    json.dump(p1_results, f, indent=2, default=str)
print("\nPart 1 saved.")

# ══════════════════════════════════════════════════════════════════
# PART 2 — Hierarchy (daily direction + shorter entry)
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 2: Hierarchy (Daily Direction + Shorter Entry)")
print("=" * 100)

hierarchy_combos = [
    ('4h', 12), ('4h', 6), ('4h', 18),
    ('1h', 24), ('1h', 48), ('1h', 72), ('1h', 120),
]

p2_results = {}
for symbol in SYMBOLS:
    label = symbol.split('/')[0]
    for pname, (s, e) in PERIODS.items():
        key = f'{label}_{pname}'
        daily = dm.get_ohlcv(symbol, '1d', '2020-01-01', '2026-03-21')
        rows = []
        for etf, ema in hierarchy_combos:
            entry_ohlcv = dm.get_ohlcv(symbol, etf, s, e)
            r = hierarchy_backtest(daily['close'], entry_ohlcv['close'],
                                    dir_ma=26, entry_ma=ema)
            r.update({'entry_tf': etf, 'entry_ma': ema})
            rows.append(r)
        p2_results[key] = rows

        print(f"\n{label} — {pname} (Daily MA26 direction filter)")
        print(f"{'EntryTF':>8} {'EntryMA':>8} {'Sharpe':>7} {'WR%':>5} {'PF':>6} {'MCL':>4} "
              f"{'T/Mo':>5} {'MaxDD':>7} {'WrstD':>7} {'MnMo':>6} {'CostMo':>7} {'Ret':>8}")
        print("-" * 90)
        for r in rows:
            print(f"{r['entry_tf']:>8} {r['entry_ma']:>8} {r['sharpe_d']:>7} "
                  f"{r['wr']:>5} {r['pf']:>6} {r['mcl']:>4} {r['tpm']:>5} "
                  f"{r['max_dd']:>6}% {r['worst_day']:>6}% {r['mean_mo']:>5}% "
                  f"${r['cost_mo']:>6} {r['return']:>7}%")

with open('mtf_part2.json', 'w') as f:
    json.dump(p2_results, f, indent=2, default=str)
print("\nPart 2 saved.")

# ══════════════════════════════════════════════════════════════════
# PART 3 — Confidence weighting
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 3: Confidence Weighting")
print("=" * 100)

p3_results = {}
for symbol in SYMBOLS:
    label = symbol.split('/')[0]
    for pname, (s, e) in PERIODS.items():
        d1d = dm.get_ohlcv(symbol, '1d', '2020-01-01', '2026-03-21')
        d4h = dm.get_ohlcv(symbol, '4h', s, e)
        d1h = dm.get_ohlcv(symbol, '1h', s, e)

        # Signals (all shifted)
        ds = ((d1d['close'] > d1d['close'].rolling(26).mean()).astype(int)*2-1).shift(1)
        ds_4h = ds.reindex(d4h.index, method='ffill').fillna(0)

        h4s = ((d4h['close'] > d4h['close'].rolling(12).mean()).astype(int)*2-1).shift(1)

        h1s = ((d1h['close'] > d1h['close'].rolling(48).mean()).astype(int)*2-1).shift(1)
        h1s_4h = h1s.reindex(d4h.index, method='ffill').fillna(0)

        direction = ds_4h
        score = (ds_4h == direction).astype(float) + \
                (h4s == direction).astype(float) + \
                (h1s_4h == direction).astype(float)
        score[direction == 0] = 0

        # Variable position sizing
        pos_size = direction * score * FRAC * LEV  # 0, 0.3, 0.6, 0.9
        pos_exec = pos_size.shift(1).fillna(0)
        ret = d4h['close'].pct_change()
        gpnl = pos_exec * ret * 1000
        cost_s = pos_size.diff().abs().fillna(0) * 1000 * FEE
        net = gpnl - cost_s
        equity = 1000 + net.cumsum()

        d = equity.resample('1D').last().pct_change().dropna()
        sd = float(d.mean()/d.std()*np.sqrt(252)) if len(d)>1 and d.std()>0 else 0
        m = equity.resample('ME').last().pct_change().dropna()
        dd = float((equity/equity.cummax()-1).min()*100)

        key = f'{label}_{pname}'
        p3_results[key] = {
            'sharpe_d': round(sd, 2),
            'return': round(float((equity.iloc[-1]/1000-1)*100), 1),
            'max_dd': round(dd, 1),
            'mean_mo': round(float(m.mean()*100), 2) if len(m)>0 else 0,
            'worst_mo': round(float(m.min()*100), 2) if len(m)>0 else 0,
        }
        print(f"\n{label} — {pname}: Sharpe={sd:.2f}, Ret={p3_results[key]['return']}%, "
              f"DD={p3_results[key]['max_dd']}%, MnMo={p3_results[key]['mean_mo']}%")

with open('mtf_part3.json', 'w') as f:
    json.dump(p3_results, f, indent=2, default=str)
print("\nPart 3 saved.")

# ══════════════════════════════════════════════════════════════════
# PART 4 — Selection
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 4: Paper Trading Configuration Selection")
print("=" * 100)

# Collect all candidates from Parts 1 and 2 (holdout only for scoring)
candidates = []

for sym in SYMBOLS:
    label = sym.split('/')[0]
    # Part 1 standalone
    for r in p1_results.get(f'{label}_Holdout', []):
        r2022 = None
        for r22 in p1_results.get(f'{label}_2022', []):
            if r22['tf'] == r['tf'] and r22['ma'] == r['ma']:
                r2022 = r22; break
        candidates.append({
            'name': f'{label} {r["tf"]} MA{r["ma"]} standalone',
            'type': 'standalone', **r,
            'sharpe_2022': r2022['sharpe_d'] if r2022 else 0,
            'dd_2022': r2022['max_dd'] if r2022 else -99,
        })
    # Part 2 hierarchy
    for r in p2_results.get(f'{label}_Holdout', []):
        r2022 = None
        for r22 in p2_results.get(f'{label}_2022', []):
            if r22['entry_tf'] == r['entry_tf'] and r22['entry_ma'] == r['entry_ma']:
                r2022 = r22; break
        candidates.append({
            'name': f'{label} Daily26+{r["entry_tf"]} MA{r["entry_ma"]}',
            'type': 'hierarchy', **r,
            'sharpe_2022': r2022['sharpe_d'] if r2022 else 0,
            'dd_2022': r2022['max_dd'] if r2022 else -99,
        })

# Score each
for c in candidates:
    score = 0
    if c['tpm'] >= 15: score += 1
    if c['wr'] >= 60: score += 1
    if c['mcl'] <= 5: score += 1
    if c['worst_day'] >= -8: score += 1
    if c['sharpe_d'] >= 1.5: score += 1
    if c['sharpe_2022'] >= 0.5: score += 1
    c['score'] = score

candidates.sort(key=lambda x: (-x['score'], -x['sharpe_d']))

print(f"\nTop 15 candidates (sorted by score then Sharpe):")
print(f"{'Name':<35} {'Score':>5} {'Sharpe':>7} {'WR%':>5} {'PF':>6} {'MCL':>4} "
      f"{'T/Mo':>5} {'WrstD':>7} {'Sh2022':>7} {'CostMo':>7}")
print("-" * 100)
for c in candidates[:15]:
    print(f"{c['name']:<35} {c['score']:>5} {c['sharpe_d']:>7} {c['wr']:>5} "
          f"{c['pf']:>6} {c['mcl']:>4} {c['tpm']:>5} {c['worst_day']:>6}% "
          f"{c['sharpe_2022']:>7} ${c.get('cost_mo',0):>6}")

best = candidates[0]
print(f"\n{'='*60}")
print(f"SELECTED: {best['name']}")
print(f"{'='*60}")
print(f"  Daily Sharpe:       {best['sharpe_d']}")
print(f"  Trades/month:       {best['tpm']}")
print(f"  Win rate:           {best['wr']}%")
print(f"  Profit factor:      {best['pf']}x")
print(f"  Max consec losses:  {best['mcl']}")
print(f"  Worst single day:   {best['worst_day']}%")
print(f"  Max drawdown:       {best['max_dd']}%")
print(f"  Monthly mean:       {best['mean_mo']}%")
print(f"  Monthly cost drag:  ${best.get('cost_mo',0)}")
print(f"  2022 Sharpe:        {best['sharpe_2022']}")
print(f"  Score:              {best['score']}/6")

# Capital for £1k/mo
if best['mean_mo'] > 0:
    cap = 1270 / (best['mean_mo'] / 100)
    print(f"  Capital for £1k/mo: ~${cap:,.0f} (~£{cap/1.27:,.0f})")

with open('mtf_part4.json', 'w') as f:
    json.dump({'selected': best, 'top10': candidates[:10]}, f, indent=2, default=str)

# Save all
with open('mtf_all_results.json', 'w') as f:
    json.dump({'part1': p1_results, 'part2': p2_results, 'part3': p3_results,
               'selected': best}, f, indent=2, default=str)

print("\nAll results saved.")
