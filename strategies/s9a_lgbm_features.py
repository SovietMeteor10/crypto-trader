"""Strategy 9A: GradientBoosting Feature-Based Classifier."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class LGBMFeaturesSignal(SignalModule):
    @property
    def name(self) -> str:
        return "9A_lgbm_features"

    @property
    def parameter_space(self) -> dict:
        return {
            "n_estimators": ("int", 50, 200),
            "max_depth": ("int", 2, 4),
            "forward_bars": ("int", 4, 24),
        }

    @staticmethod
    def _build_features(data: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from OHLCV data. No lookahead."""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        features = pd.DataFrame(index=data.index)

        # Returns over various lookbacks
        for lb in [1, 4, 8, 24, 48]:
            features[f"ret_{lb}"] = close.pct_change(lb)

        # Volatility (std of returns)
        ret_1 = close.pct_change(1)
        for lb in [8, 24]:
            features[f"vol_{lb}"] = ret_1.rolling(lb, min_periods=1).std()

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        features["rsi_14"] = features["rsi_14"].fillna(50.0)

        # MACD histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features["macd_hist"] = macd_line - signal_line

        # Volume z-score (20)
        vol_mean = volume.rolling(20, min_periods=1).mean()
        vol_std = volume.rolling(20, min_periods=1).std()
        features["vol_z_20"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        features["vol_z_20"] = features["vol_z_20"].fillna(0.0)

        # Bar range / ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=1).mean()
        features["range_atr_ratio"] = (high - low) / atr_14.replace(0, np.nan)
        features["range_atr_ratio"] = features["range_atr_ratio"].fillna(1.0)

        # Hour of day — sin/cos encoded
        if hasattr(data.index, 'hour'):
            hour = data.index.hour.values.astype(float)
        else:
            hour = np.zeros(len(data))
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day of week — sin/cos encoded
        if hasattr(data.index, 'dayofweek'):
            dow = data.index.dayofweek.values.astype(float)
        else:
            dow = np.zeros(len(data))
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        return features

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        from sklearn.ensemble import GradientBoostingClassifier

        n_estimators = params["n_estimators"]
        max_depth = params["max_depth"]
        forward_bars = params["forward_bars"]

        features = self._build_features(data)

        # Target: sign of forward N-bar return (no lookahead — only used for training)
        forward_ret = data["close"].shift(-forward_bars) / data["close"] - 1.0

        # Fill NaN in features
        features = features.fillna(0.0)

        # Split: first 70% train, rest predict
        n = len(data)
        train_end = int(n * 0.7)

        result = pd.Series(0, index=data.index, dtype=int)

        # Need enough data for training
        if train_end < 100:
            return result

        # Training set — exclude rows where forward return is NaN
        train_mask = np.arange(n) < train_end
        valid_target = ~forward_ret.isna()
        usable = train_mask & valid_target.values

        X_train = features.iloc[:train_end][usable[:train_end]]
        y_raw = forward_ret.iloc[:train_end][usable[:train_end]]
        y_train = np.sign(y_raw).astype(int)

        # Replace 0 target with 1 (no neutral class for classifier)
        y_train = y_train.replace(0, 1)

        # Map to class labels: -1 -> 0, 1 -> 1
        y_mapped = ((y_train + 1) // 2).astype(int)

        if len(X_train) < 50 or y_mapped.nunique() < 2:
            return result

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            subsample=0.8,
            learning_rate=0.1,
        )
        clf.fit(X_train.values, y_mapped.values)

        # Predict on test portion
        X_test = features.iloc[train_end:]
        if len(X_test) == 0:
            return result

        proba = clf.predict_proba(X_test.values)

        # proba columns correspond to classes; find column indices
        classes = clf.classes_
        idx_long = np.where(classes == 1)[0]
        idx_short = np.where(classes == 0)[0]

        for i in range(len(X_test)):
            p_long = proba[i, idx_long[0]] if len(idx_long) > 0 else 0.0
            p_short = proba[i, idx_short[0]] if len(idx_short) > 0 else 0.0

            if p_long > 0.55:
                result.iloc[train_end + i] = 1
            elif p_short > 0.55:
                result.iloc[train_end + i] = -1
            # else stays 0

        return result
