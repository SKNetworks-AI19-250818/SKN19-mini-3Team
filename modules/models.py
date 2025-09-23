import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

class TreeXGBoostCox :
    def __init__(self, eta=0.1, max_depth=3, subsample=1.0, colsample_bytree=1.0,
                 min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0, tree_method="auto",
                 num_boost_round=100, eval_metric="cox-nloglik"):
        
        self.params = {
            "objective": "survival:cox",
            "eval_metric": eval_metric,
            "eta": eta,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "lambda": reg_lambda,   # L2 정규화
            "alpha": reg_alpha,     # L1 정규화
            "tree_method": tree_method
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.is_fitted = False
        self.baseline_survival = None

    def S0(self, t):
            return float(self.baseline_survival.loc[self.baseline_survival.index <= t].iloc[-1])

    def fit(self, X, y, e) :
        dtrain = xgb.DMatrix(X, label=y, weight=e)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        self.is_fitted = True

        df = pd.DataFrame(X.copy())
        df['Time'] = y
        df['Event'] = e
        cph = CoxPHFitter()
        cph.fit(df, duration_col='Time', event_col='Event')
        self.baseline_survival = cph.baseline_survival_


    def predict(self, X, times) :
        if not self.is_fitted :
            raise RuntimeError("Model must be fitted before prediction. Please run fit() first.")
        
        dmatrix = xgb.DMatrix(X)
        scores = self.model.predict(dmatrix)

        surv = {}
        for t in np.atleast_1d(times):
            surv[t] = np.exp(-self.S0(t) * np.exp(scores))  # Cox 식: S(t|x) = S0(t)^exp(score)
        return surv
    
    def score(self, X, y, e) :
        scores = self.predict(X)
        return concordance_index(y, -scores, e)
    

