"""
Market structure statistical characterisation.
Analyses A-D plus cross-asset tests.
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/market_structure")
OUTPUT_DIR = Path("/home/ubuntu/projects/crypto-trader/outputs/research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = {
    'P1_strong_bear': ('2022-01-01', '2022-06-30'),
    'P2_bear_ftx': ('2022-07-01', '2022-12-31'),
    'P3_recovery': ('2023-01-01', '2023-06-30'),
    'P4_early_bull': ('2023-07-01', '2023-12-31'),
    'P5_etf_bull': ('2024-01-01', '2024-06-30'),
    'P6_correction': ('2024-07-01', '2024-12-31'),
}


# ── Analysis A: Predictive Regression Table ─────────────────────

def predictive_table(features: pd.DataFrame, horizons=[1, 4, 24, 72]) -> pd.DataFrame:
    returns = features['ret_1h']
    exclude = {'close', 'ret_1h', 'ret_4h', 'ret_24h', 'oi', 'volume'}
    feature_cols = [c for c in features.columns if c not in exclude]

    results = []
    for h in horizons:
        fwd_ret = returns.shift(-h)
        for col in feature_cols:
            x = features[col].copy()
            df = pd.DataFrame({'x': x, 'y': fwd_ret}).dropna()
            if len(df) < 100:
                continue
            # Winsorise
            lo, hi = df['x'].quantile(0.01), df['x'].quantile(0.99)
            x_w = df['x'].clip(lo, hi)
            std = x_w.std()
            if std == 0:
                continue
            x_std = (x_w - x_w.mean()) / std
            slope, intercept, r, p, se = stats.linregress(x_std, df['y'])
            tstat = slope / se if se > 0 else 0
            results.append({
                'feature': col, 'horizon': f'{h}H',
                'beta': slope, 'tstat': tstat, 'pvalue': p,
                'r2_pct': r**2 * 100, 'n': len(df),
                'significant': abs(tstat) > 2.0,
            })

    df_r = pd.DataFrame(results)
    df_r['abs_tstat'] = df_r['tstat'].abs()
    return df_r.sort_values('abs_tstat', ascending=False)


# ── Analysis B: Hypothesis Tests ────────────────────────────────

def test_hypothesis_binary(features, condition_col, horizons=[4, 24, 72], condition_value=1):
    """Test whether a binary condition predicts returns."""
    returns = features['ret_1h']
    mask = features[condition_col] == condition_value
    n_events = int(mask.sum())

    results = {'condition': condition_col, 'n_events': n_events}
    for h in horizons:
        fwd_ret = returns.shift(-h)
        cond_ret = fwd_ret[mask].dropna()
        all_ret = fwd_ret.dropna()
        if len(cond_ret) < 5:
            results[f'mean_ret_{h}h'] = np.nan
            continue
        results[f'mean_ret_{h}h'] = float(cond_ret.mean())
        results[f'baseline_{h}h'] = float(all_ret.mean())
        t, p = stats.ttest_ind(cond_ret, all_ret, equal_var=False)
        results[f'tstat_{h}h'] = float(t)
        results[f'pval_{h}h'] = float(p)
        results[f'significant_{h}h'] = bool(p < 0.05)
        # Fraction followed by >1% decline within 24h
        if h == 24:
            fwd_24 = returns.shift(-24)
            decline = (fwd_24[mask].dropna() < -0.01).mean()
            results['pct_1pct_decline_24h'] = float(decline * 100)
    return results


def test_smart_money(features, horizons=[4, 24]):
    """H2: smart_dumb_div > 0 vs < 0."""
    if 'smart_dumb_div' not in features.columns:
        return {'error': 'smart_dumb_div not available'}
    returns = features['ret_1h']
    div = features['smart_dumb_div']

    results = {}
    for h in horizons:
        fwd = returns.shift(-h)
        pos_mask = div > 0
        neg_mask = div < 0
        pos_ret = fwd[pos_mask].dropna()
        neg_ret = fwd[neg_mask].dropna()
        if len(pos_ret) < 50 or len(neg_ret) < 50:
            continue
        t, p = stats.ttest_ind(pos_ret, neg_ret, equal_var=False)
        results[f'pos_mean_{h}h'] = float(pos_ret.mean())
        results[f'neg_mean_{h}h'] = float(neg_ret.mean())
        results[f'diff_{h}h'] = float(pos_ret.mean() - neg_ret.mean())
        results[f'tstat_{h}h'] = float(t)
        results[f'pval_{h}h'] = float(p)
        results[f'significant_{h}h'] = bool(p < 0.05)
        results[f'n_pos'] = int(pos_mask.sum())
        results[f'n_neg'] = int(neg_mask.sum())
    return results


def test_funding_consistency(features, years=[2022, 2023, 2024]):
    """H3: Test funding extreme → reversal, per year."""
    if 'funding_ext_long' not in features.columns:
        return {'error': 'funding_ext_long not available'}

    results = {}
    returns = features['ret_1h']

    for year in years:
        mask_year = features.index.year == year
        f_year = features[mask_year]
        ret_year = returns[mask_year]

        for cond, label in [('funding_ext_long', 'ext_long'), ('funding_ext_short', 'ext_short')]:
            if cond not in f_year.columns:
                continue
            m = f_year[cond] == 1
            n = int(m.sum())
            for h in [24, 72]:
                fwd = ret_year.shift(-h)
                cond_ret = fwd[m].dropna()
                all_ret = fwd.dropna()
                if len(cond_ret) < 5:
                    continue
                key = f'{label}_{year}_{h}h'
                results[key] = {
                    'n': n,
                    'mean_ret': float(cond_ret.mean()),
                    'baseline': float(all_ret.mean()),
                    'tstat': float(stats.ttest_ind(cond_ret, all_ret, equal_var=False)[0]),
                    'pval': float(stats.ttest_ind(cond_ret, all_ret, equal_var=False)[1]),
                }
    return results


def test_oi_regimes(features, horizons=[4, 24]):
    """H4: ANOVA on OI regime return distributions."""
    regime_cols = ['price_up_oi_up', 'price_up_oi_down', 'price_dn_oi_up', 'price_dn_oi_down']
    if not all(c in features.columns for c in regime_cols):
        return {'error': 'OI regime columns not available'}

    returns = features['ret_1h']
    results = {}
    for h in horizons:
        fwd = returns.shift(-h)
        groups = []
        group_stats = {}
        for col in regime_cols:
            m = features[col] == 1
            g = fwd[m].dropna()
            groups.append(g.values)
            group_stats[col] = {'mean': float(g.mean()), 'n': len(g)}

        if all(len(g) > 20 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            results[f'{h}h'] = {
                'groups': group_stats,
                'f_stat': float(f_stat),
                'p_val': float(p_val),
                'significant': bool(p_val < 0.05),
            }
    return results


# ── Analysis C: Regime-conditional ──────────────────────────────

def regime_conditional(features, top_features, best_horizons):
    """Test top features across 6 regime periods."""
    returns = features['ret_1h']
    results = {}

    for feat, h_str in zip(top_features, best_horizons):
        h = int(h_str.replace('H', ''))
        results[feat] = {}
        for regime_name, (start, end) in REGIMES.items():
            mask = (features.index >= start) & (features.index <= end)
            x = features.loc[mask, feat]
            fwd = returns[mask].shift(-h)
            df = pd.DataFrame({'x': x, 'y': fwd}).dropna()
            if len(df) < 50:
                results[feat][regime_name] = {'tstat': np.nan, 'n': len(df)}
                continue
            lo, hi = df['x'].quantile(0.01), df['x'].quantile(0.99)
            x_w = df['x'].clip(lo, hi)
            std = x_w.std()
            if std == 0:
                results[feat][regime_name] = {'tstat': 0, 'n': len(df)}
                continue
            x_std = (x_w - x_w.mean()) / std
            slope, _, _, p, se = stats.linregress(x_std, df['y'])
            tstat = slope / se if se > 0 else 0
            results[feat][regime_name] = {
                'tstat': float(tstat), 'pval': float(p), 'n': len(df),
            }
    return results


# ── Analysis D: OHLCV vs OHLCV + Structure ─────────────────────

def compare_models(features, target_horizon=4):
    """Compare OHLCV-only vs OHLCV+structure predictive power."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    returns = features['ret_1h']
    fwd_ret = returns.shift(-target_horizon)

    # OHLCV features
    f2 = features.copy()
    f2['rvol_24h'] = returns.rolling(24).std()
    f2['mom_1h'] = features['close'].pct_change(1)
    f2['mom_4h'] = features['close'].pct_change(4)
    f2['mom_24h'] = features['close'].pct_change(24)
    ohlcv_feats = ['ret_1h', 'ret_4h', 'ret_24h', 'rvol_24h', 'mom_1h', 'mom_4h', 'mom_24h']

    # Structure features
    exclude = {'close', 'ret_1h', 'ret_4h', 'ret_24h', 'oi', 'volume',
               'rvol_24h', 'mom_1h', 'mom_4h', 'mom_24h'}
    struct_feats = [c for c in features.columns if c not in exclude]

    common_idx = fwd_ret.dropna().index
    y = fwd_ret.loc[common_idx]

    tscv = TimeSeriesSplit(n_splits=12)
    r2_ohlcv, r2_combined = [], []

    for train_idx, test_idx in tscv.split(y):
        for feat_list, store in [(ohlcv_feats, r2_ohlcv),
                                  (ohlcv_feats + struct_feats, r2_combined)]:
            X = f2[feat_list].loc[common_idx].fillna(0)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)
            score = model.score(X_test_s, y_test)
            store.append(score)

    return {
        'ohlcv_r2_mean': np.mean(r2_ohlcv) * 100,
        'combined_r2_mean': np.mean(r2_combined) * 100,
        'improvement_pct': (np.mean(r2_combined) - np.mean(r2_ohlcv)) * 100,
        'structure_adds_value': bool(np.mean(r2_combined) > np.mean(r2_ohlcv) + 0.0001),
    }


# ── Cross-asset analysis ────────────────────────────────────────

def cross_asset_tests(btc_features, sol_features):
    """Test BTC structure → SOL returns."""
    common = btc_features.index.intersection(sol_features.index)
    if len(common) < 500:
        return {'error': 'insufficient common bars'}

    btc = btc_features.loc[common]
    sol = sol_features.loc[common]
    sol_ret = sol['ret_1h']

    tests = []
    btc_cols = ['squeeze_setup', 'smart_dumb_div', 'funding_ext_long',
                'crowd_long', 'oi_chg_4h', 'taker_momentum']

    for col in btc_cols:
        if col not in btc.columns:
            continue
        for h in [4, 24]:
            fwd = sol_ret.shift(-h)
            x = btc[col]
            df = pd.DataFrame({'x': x, 'y': fwd}).dropna()
            if len(df) < 100:
                continue
            lo, hi = df['x'].quantile(0.01), df['x'].quantile(0.99)
            x_w = df['x'].clip(lo, hi)
            std = x_w.std()
            if std == 0:
                continue
            x_std = (x_w - x_w.mean()) / std
            slope, _, _, p, se = stats.linregress(x_std, df['y'])
            tstat = slope / se if se > 0 else 0
            tests.append({
                'btc_feature': col, 'target': 'SOL_ret',
                'horizon': f'{h}H', 'tstat': tstat, 'pval': p,
                'significant': abs(tstat) > 2.0,
            })

    return pd.DataFrame(tests).sort_values('tstat', key=abs, ascending=False) if tests else {}


# ── Main ────────────────────────────────────────────────────────

def main():
    all_results = {}

    for symbol in ['BTC_USDT_USDT', 'SOL_USDT_USDT']:
        path = CACHE_DIR / f"{symbol}_unified_1h.parquet"
        if not path.exists():
            print(f"Missing: {path}")
            continue

        f = pd.read_parquet(path)
        f.index = pd.DatetimeIndex(f.index)
        label = symbol.split('_')[0]
        print(f"\n{'='*60}")
        print(f"Analysing {label}: {len(f)} bars, {f.shape[1]} features")
        print('='*60)

        # A: Predictive table
        print("\n--- Analysis A: Predictive Regressions ---")
        pred = predictive_table(f)
        sig = pred[pred['significant']]
        print(f"Total significant (|t|>2): {len(sig)} out of {len(pred)}")
        print(f"\nTop 20 by |t-stat|:")
        cols = ['feature', 'horizon', 'tstat', 'r2_pct', 'beta', 'n']
        print(pred[cols].head(20).to_string(index=False))
        all_results[f'{label}_pred_table'] = pred.to_dict('records')

        # Get top 3 features for regime analysis
        top3 = pred.drop_duplicates('feature').head(3)
        top_features = top3['feature'].tolist()
        top_horizons = top3['horizon'].tolist()

        # B: Hypothesis tests
        print("\n--- Analysis B: Hypothesis Tests ---")

        # H1: squeeze
        if 'squeeze_setup' in f.columns:
            h1 = test_hypothesis_binary(f, 'squeeze_setup')
            print(f"\nH1 Squeeze Setup: {h1['n_events']} events")
            for h in [4, 24, 72]:
                k = f'mean_ret_{h}h'
                if k in h1:
                    t = h1.get(f'tstat_{h}h', 0)
                    print(f"  {h}H: mean ret={h1[k]*100:.4f}%, baseline={h1.get(f'baseline_{h}h',0)*100:.4f}%, t={t:.2f}")
            all_results[f'{label}_h1_squeeze'] = h1

        # H2: smart money
        h2 = test_smart_money(f)
        if 'error' not in h2:
            print(f"\nH2 Smart Money Divergence:")
            for h in [4, 24]:
                k = f'diff_{h}h'
                if k in h2:
                    print(f"  {h}H: div>0 mean={h2[f'pos_mean_{h}h']*100:.4f}%, "
                          f"div<0 mean={h2[f'neg_mean_{h}h']*100:.4f}%, "
                          f"diff={h2[k]*100:.4f}%, t={h2[f'tstat_{h}h']:.2f}")
        all_results[f'{label}_h2_smart'] = h2

        # H3: funding consistency
        h3 = test_funding_consistency(f)
        if 'error' not in h3:
            print(f"\nH3 Funding Extremes (per year):")
            for key, val in h3.items():
                sig_str = " ***" if val.get('pval', 1) < 0.05 else ""
                print(f"  {key}: n={val['n']}, mean={val['mean_ret']*100:.4f}%, "
                      f"baseline={val['baseline']*100:.4f}%, t={val['tstat']:.2f}{sig_str}")
        all_results[f'{label}_h3_funding'] = h3

        # H4: OI regimes
        h4 = test_oi_regimes(f)
        if 'error' not in h4:
            print(f"\nH4 OI Regime Classification:")
            for h_key, val in h4.items():
                print(f"  {h_key}: F={val['f_stat']:.2f}, p={val['p_val']:.4f}, sig={val['significant']}")
                for g, gs in val['groups'].items():
                    print(f"    {g}: mean={gs['mean']*100:.4f}%, n={gs['n']}")
        all_results[f'{label}_h4_oi'] = h4

        # C: Regime-conditional
        print(f"\n--- Analysis C: Regime-Conditional (top 3 features) ---")
        rc = regime_conditional(f, top_features, top_horizons)
        for feat, regimes in rc.items():
            print(f"\n  {feat}:")
            for rn, rv in regimes.items():
                sig_str = " ***" if abs(rv.get('tstat', 0)) > 2 else ""
                print(f"    {rn}: t={rv.get('tstat', 0):.2f}, n={rv.get('n', 0)}{sig_str}")
        all_results[f'{label}_regime_cond'] = rc

        # D: Model comparison
        print(f"\n--- Analysis D: OHLCV vs OHLCV+Structure ---")
        try:
            mc = compare_models(f)
            print(f"  OHLCV-only R²: {mc['ohlcv_r2_mean']:.4f}%")
            print(f"  Combined R²:   {mc['combined_r2_mean']:.4f}%")
            print(f"  Improvement:   {mc['improvement_pct']:.4f}%")
            print(f"  Structure adds value: {mc['structure_adds_value']}")
            all_results[f'{label}_model_compare'] = mc
        except Exception as e:
            print(f"  Model comparison failed: {e}")

    # Cross-asset
    btc_path = CACHE_DIR / "BTC_USDT_USDT_unified_1h.parquet"
    sol_path = CACHE_DIR / "SOL_USDT_USDT_unified_1h.parquet"
    if btc_path.exists() and sol_path.exists():
        print(f"\n{'='*60}")
        print("Cross-Asset Analysis: BTC structure → SOL returns")
        print('='*60)
        btc_f = pd.read_parquet(btc_path)
        sol_f = pd.read_parquet(sol_path)
        btc_f.index = pd.DatetimeIndex(btc_f.index)
        sol_f.index = pd.DatetimeIndex(sol_f.index)
        ca = cross_asset_tests(btc_f, sol_f)
        if isinstance(ca, pd.DataFrame) and len(ca) > 0:
            print(ca.to_string(index=False))
            all_results['cross_asset'] = ca.to_dict('records')
        else:
            print("  No significant cross-asset signals")

    # Save results
    def serialise(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialise(v) for v in obj]
        return obj

    out_path = OUTPUT_DIR / "market_structure_results.json"
    with open(out_path, 'w') as fp:
        json.dump(serialise(all_results), fp, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
