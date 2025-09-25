import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
import xgboost as xgb
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

class TreeXGBoostCox(BaseEstimator) :
    def __init__(self, eta=0.1, max_depth=3, subsample=1.0, colsample_bytree=1.0,
                 min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0, tree_method="auto",
                 num_boost_round=100, eval_metric="cox-nloglik"):
        
        # XGBoost Hyper Parameter
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.num_boost_round = num_boost_round
        self.eval_metric = eval_metric

        self.model = None
        self.is_fitted = False
        self.baseline_survival = None

        # XGBoost params dict
        self.params = {
            "objective": "survival:cox",
            "eval_metric": self.eval_metric,
            "eta": self.eta,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda,   # L2 정규화
            "alpha": self.reg_alpha,     # L1 정규화
            "tree_method": self.tree_method
        }

    # 생존함수 : 시간 t에서의 생존 확률을 반환하는 함수
    def S0(self, t):
            return float(self.baseline_survival.loc[self.baseline_survival.index <= t].iloc[-1, 0])

    # 모델 학습
    def fit(self, X, y, e=None) :
        if e is None:
            if not isinstance(y, pd.DataFrame):
                raise ValueError("e가 None일 때 y는 'Time'과 'Alive' 컬럼을 가진 DataFrame이어야 합니다.")
            y_values = y['Time']
            e_values = y['Alive']
        else :
            y_values = y
            e_values = e

        times = y_values
        event = e_values

        dtrain = xgb.DMatrix(X, label=times, weight=e)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        self.is_fitted = True

        df = pd.DataFrame(X.copy())
        df['Time'] = times
        df['Event'] = event
        cph = CoxPHFitter()
        cph.fit(df, duration_col='Time', event_col='Event')
        self.baseline_survival = cph.baseline_survival_

    # 모델 예측 : 위험 점수로 출력된 값을 생존함수를 통해 생존확률을 계산
    def predict(self, X, t) :
        if not self.is_fitted :
            raise RuntimeError("Model must be fitted before prediction. Please run fit() first.")
        
        dmatrix = xgb.DMatrix(X)
        scores = self.model.predict(dmatrix)
        scores = (scores - scores.mean()) / scores.std()

        surv = {}
        for t in np.atleast_1d(t):
            surv[t] = self.S0(t) ** np.exp(scores)  # Cox 식: S(t|x) = S0(t)^exp(score)
        return surv
    
    # 모델 평가 : 생존확률을 통해 이진분류, 정확도를 평가
    def score(self, X_test, time_test, event_test=None, t=115.5, threshold=0.5, show_comparison=False):
        if event_test is None:
            if not isinstance(time_test, pd.DataFrame):
                raise ValueError("e가 None일 때 y는 'Time'과 'Alive' 컬럼을 가진 DataFrame이어야 합니다.")
            y_values = time_test['Time']
            e_values = time_test['Alive']
        else : 
            y_values = time_test
            e_values = event_test

            
        times = y_values
        event = e_values

        # 생존율 예측
        surv_dict = self.predict(X_test, t=t)
        surv_probs = surv_dict[t]

        pred_labels = (surv_probs >= threshold).astype(int)

        true_labels = np.where((times > t) | ((times <= t) & (event == 0)), 1, 0)

        # 정확도 계산
        accuracy = (pred_labels == true_labels).mean()

        comparison = pd.DataFrame({
            'Predicted': pred_labels,
            'Actual': true_labels,
            'Survival_Prob': surv_probs
        })
        if show_comparison is True :
            print(comparison[['Predicted', 'Actual']].value_counts())

        return accuracy
    
    # 잘못 예측한 데이터를 분석용 데이터프레임으로 반환
    def create_mismatch_df(self, X_test, time_test, event_test, t=115.5, threshold=0.5):
        surv_dict = self.predict(X_test, t=t)
        surv_probs = surv_dict[t]  # np.array 형태

        pred_labels = (surv_probs >= threshold).astype(int)

        true_labels = np.where((time_test > t) | ((time_test <= t) & (event_test == 0)), 1, 0)

        comparison = pd.DataFrame({
            'Predicted_Label': pred_labels,
            'Actual_Label': true_labels,
            'Predicted_Survival_Prob': np.round(surv_probs, 2),
            'Time': time_test,
            'Event': event_test
        })

        mismatch_df = comparison[comparison['Predicted_Label'] != comparison['Actual_Label']]

        return mismatch_df
    
    # confusion matrix 반환
    def confusion_matrix(self, X_test, time_test, event_test, t=115.5, threshold=0.5):
        # label : 0 - 사망, 1 - 생존

        surv_dict = self.predict(X_test, t=t)
        surv_probs = surv_dict[t]

        pred_labels = (surv_probs >= threshold).astype(int)

        true_labels = np.where((time_test > t) | ((time_test <= t) & (event_test == 0)), 1, 0)

        tp = np.sum((pred_labels == 0) & (true_labels == 0))
        tn = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 0) & (true_labels == 1))
        fn = np.sum((pred_labels == 1) & (true_labels == 0))

        confusion = np.array([[tn, fp],[fn, tp]])

        return confusion

class TreeRandomForestSurvival(BaseEstimator):
    def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 max_features="sqrt",
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = None
        self.is_fitted = False

    # 모델 학습
    def fit(self, X, y, e=None):
        if e is None:
            if not isinstance(y, pd.DataFrame):
                raise ValueError("e가 None일 때 y는 'Time'과 'Alive' 컬럼을 가진 DataFrame이어야 합니다.")
            y_values = y['Time']
            e_values = y['Alive']
        else :
            y_values = y
            e_values = e

        times = y_values
        event = e_values

        # sksurv는 structured array 필요
        y_struct = Surv.from_arrays(event=event.astype(bool), time=times)

        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y_struct)
        self.is_fitted = True
        return self

    # 모델 예측 : 위험 점수로 출력된 값을 생존함수를 통해 생존확률을 계산
    def predict(self, X, t):

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        surv_funcs = self.model.predict_survival_function(X)

        surv = {}
        for t in np.atleast_1d(t):
            surv[t] = np.array([fn(t) for fn in surv_funcs])  # 각 샘플별 생존확률
        return surv

    # 모델 평가 : 생존확률을 통해 이진분류, 정확도를 평가
    def score(self, X_test, time_test, event_test=None, t=115.5, threshold=0.5, show_comparison=False):

        if event_test is None:
            if not isinstance(time_test, pd.DataFrame):
                raise ValueError("e가 None일 때 y는 'Time'과 'Alive' 컬럼을 가진 DataFrame이어야 합니다.")
            y_values = time_test['Time']
            e_values = time_test['Alive']

        else : 
            y_values = time_test
            e_values = event_test

            
        times = y_values
        event = e_values

        # 생존율 예측
        surv_dict = self.predict(X_test, t=t)
        surv_probs = surv_dict[t]

        pred_labels = (surv_probs >= threshold).astype(int)

        true_labels = np.where((times > t) | ((times <= t) & (event == 0)), 1, 0)

        # 정확도 계산
        accuracy = (pred_labels == true_labels).mean()

        comparison = pd.DataFrame({
            'Predicted': pred_labels,
            'Actual': true_labels,
            'Survival_Prob': surv_probs
        })
        if show_comparison is True :
            print(comparison[['Predicted', 'Actual']].value_counts())

        return accuracy
    
    # 잘못 예측한 데이터를 분석용 데이터프레임으로 반환
    def create_mismatch_df(self, X_test, time_test, event_test, t=115.5, threshold=0.5):
        surv_dict = self.predict(X_test, t=t)
        surv_probs = surv_dict[t]  # np.array 형태

        pred_labels = (surv_probs >= threshold).astype(int)

        true_labels = np.where((time_test > t) | ((time_test <= t) & (event_test == 0)), 1, 0)

        comparison = pd.DataFrame({
            'Predicted_Label': pred_labels,
            'Actual_Label': true_labels,
            'Predicted_Survival_Prob': np.round(surv_probs, 2),
            'Time': time_test,
            'Event': event_test
        })

        mismatch_df = comparison[comparison['Predicted_Label'] != comparison['Actual_Label']]

        return mismatch_df
    
    # confusion matrix 반환
    def confusion_matrix(self, X_test, time_test, event_test, t=115.5, threshold=0.5):
        # label : 0 - 사망, 1 - 생존

        surv_dict = self.predict(X_test, t=t)
        surv_probs = surv_dict[t]

        pred_labels = (surv_probs >= threshold).astype(int)

        true_labels = np.where((time_test > t) | ((time_test <= t) & (event_test == 0)), 1, 0)

        tp = np.sum((pred_labels == 0) & (true_labels == 0))
        tn = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 0) & (true_labels == 1))
        fn = np.sum((pred_labels == 1) & (true_labels == 0))

        confusion = np.array([[tn, fp],[fn, tp]])
        return confusion