"""
LightGBM market structure model for crypto directional trading.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


class LGBMTrader:
    def __init__(self, confidence_threshold=0.52, max_depth=4,
                 n_estimators=200, min_child_samples=50,
                 learning_rate=0.05, num_leaves=15):
        self.confidence_threshold = confidence_threshold
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
        }
        self.model = None
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train, X_val, y_val):
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)

        dtrain = lgb.Dataset(X_train, label=y_train_enc)
        dval = lgb.Dataset(X_val, label=y_val_enc, reference=dtrain)

        self.model = lgb.train(
            self.params, dtrain, valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

    def predict_signals(self, X):
        if self.model is None:
            raise RuntimeError("Not trained")
        probs = self.model.predict(X)
        classes = self.label_encoder.classes_

        signals = pd.Series(0, index=X.index)
        for i in range(len(X)):
            max_prob = probs[i].max()
            max_idx = probs[i].argmax()
            predicted = classes[max_idx]
            if max_prob >= self.confidence_threshold and predicted != 0:
                signals.iloc[i] = predicted
        return signals

    def get_feature_importance(self, feature_names):
        imp = self.model.feature_importance(importance_type='gain')
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)

    def check_disguised_momentum(self, fi):
        top5 = fi.head(5).index.tolist()
        keywords = ['ret_', 'mom_', 'ma_cross_', 'body_direction']
        n_mom = sum(1 for f in top5 if any(kw in f for kw in keywords))
        is_disguised = n_mom >= 4
        if is_disguised:
            print("  WARNING: Top features are mostly price returns — disguised momentum")
        return is_disguised
