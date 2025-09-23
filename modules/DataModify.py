import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# continuous한 컬럼과 categorical한 컬럼을 반환

def return_cols(df, type, boundary=15) :    # type : ('continuous', 'categorical'), boundary : 두 분류를 결정할 서로 다른 요소의 수
    cols = []
    if type == 'continuous' :
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) >= boundary :
                cols.append(col)

    elif type == 'categorical' :
        for col in df.columns:
            if df[col].nunique(dropna=True) < boundary :
                cols.append(col)
    else :
        ValueError('Wrong type.')

    return cols

# SMOTE 사용한 데이터 오버 샘플링

def data_oversampling(df) :
    """
        oversampling 코드 구현
    """
    pass

# 인코딩되기 전의 데이터에서 이상치를 제거

def data_anomaly_edit(df, return_anomaly=False) :
        df_cleaned = df.copy()
        df_anomaly = pd.DataFrame(columns=['Index', 'Column', 'Value'])

        alive_mask = (
            (df_cleaned['Event'].isna()) &
            (df_cleaned['Harvest'].isna()) &
            (df_cleaned['Alive'].isna())
        )
        if alive_mask.any():
            alive_indices = df_cleaned.index[alive_mask]
            alive_rows = pd.DataFrame({
                'Index': alive_indices,
                'Column': 'Alive',
                'Value': 'NA'
            })
            df_anomaly = pd.concat([df_anomaly, alive_rows], ignore_index=True)
            df_cleaned = df_cleaned[~alive_mask]

        cols = ["Phenolics", "NSC", "Lignin"]
        mask = (df_cleaned[cols] < 0).any(axis=1)

        # Phenolics, NSC, Lignin 음수값 처리
        cols = ["Phenolics", "NSC", "Lignin"]
        df_negatives = df_cleaned[cols].copy()
        df_neg_long = df_negatives.stack().reset_index()
        df_neg_long = df_neg_long.rename(columns={'level_0':'Index', 'level_1':'Column', 0:'Value'})
        df_neg_long = df_neg_long[df_neg_long['Value'] < 0]
        
        df_anomaly = pd.concat([df_anomaly, df_neg_long[['Index','Column','Value']]], ignore_index=True)

        df_cleaned['Phenolics'] = df_cleaned['Phenolics'] - df_cleaned['Phenolics'].min()
                
        if return_anomaly is True :
            return df_cleaned, df_anomaly
        else :
            return df_cleaned

# 특성과 라벨, 훈련 데이터와 테스트 데이터를 분리하여 세트로 반환

def split_feature_label(df, test_size=.2) :
    X = df.drop(columns='Alive')
    y = df['Alive']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return [X_train, y_train], [X_test, y_test]     # train set과 test set을 반환

# feature 데이터 스케일링

def scale_data(train_set, test_set) :

    X_train = train_set[0]
    X_test = test_set[0]

    continuous_cols = return_cols(train_set, 'continuous')
    
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    train_set_scaled = [X_train_scaled, train_set[1]]

    X_test_scaled = X_test.copy()
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    test_set_scaled = [X_test_scaled, test_set[1]]

    return train_set_scaled, test_set_scaled   # 값의 종류가 15개가 넘는, continuous 하고 숫자를 가지는 값들만 scaled

# 검열값을 최종 생존율에 따라 랜덤으로 예측한 데이터를 split

def train_test_split_ignore_censored(alive_data, censored_data, test_size=0.236) :
    train_set, test_set = split_feature_label(alive_data)
    dead_censored, alive_censored = split_feature_label(censored_data, test_size=test_size)

    alive_censored[1][:] = 2
    dead_censored[1][:] = 0
    
    train_set[0] = pd.concat([train_set[0], alive_censored[0], dead_censored[0]], axis=0)
    train_set[1] = pd.concat([train_set[1], alive_censored[1], dead_censored[1]])

    return train_set, test_set

# Cox 모델에 사용하기 위하여 X, y, e 형태로 데이터 split

def split_data_X_y_e(df, test_size=.2, random_state=None) :

    y = df[['Time', 'Alive']].copy()
    X = df.drop(columns=['Time', 'Alive'])

    y['Alive'] = np.where(y['Alive'] != 0, 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y['Alive'])

    e_train = y_train['Alive'].copy()
    y_train = y_train['Time'].copy()
    e_test = y_test['Alive'].copy()
    y_test = y_test['Time'].copy()

    return [X_train, y_train, e_train], [X_test, y_test, e_test]