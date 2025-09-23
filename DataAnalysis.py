import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
import platform
from IPython.display import display     # display함수를 import

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest        # 이상치 판별을 위한 Isolation Forest 모델

import DataModify

### 데이터 체크 클래스 선언

class DataCheck() :
    boundary = 15
    
    # 생성 시 기존 데이터를 넣어둠
    def __init__(self, df) :
        self.raw_df = df

    @staticmethod
    def set_categorical_threshold(boundary) :
        DataCheck.boundary = boundary

    # continuous한 컬럼과 categorical한 컬럼을 반환
    @staticmethod
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
    
    # 정보 출력
    def print_info(self, df = None) :
        if df is None :         # 데이터를 전달하지 않으면 기존 데이터를 활용
            df = self.raw_df

        print('------ Data Info -----')
        df.info()
        print('\n----- Data Describe -----')
        display(df.describe())

    # 각 컬럼들의 값 출력
    def print_value_counts(self, df = None) :
        if df is None :         # 데이터를 전달하지 않으면 기존 데이터를 활용
            df = self.raw_df
        
        for col in df.columns:
            if df[col].nunique(dropna=True) > 15 :  # 각기 다른 값이 15개 이상인 Continuous 한 값들은 출력하지 않음
                print(col)
                print('continuous')
                print("-"*20)
                continue

            value_counts = df[col].fillna("NA").value_counts(dropna=False)  # 결측치는 NA로 처리 후 출력
            print(value_counts)
            print("-" * 20)

    # 데이터 to csv
    def save_to_csv(self, df):
        df.to_csv('Tree_Data_processing.csv', index=False)
    

### 데이터 전처리 클래스 선언

class DataPreprocessing () :

    def __init__(self, df) :
        self.raw_df = df        # 수정 전 기본 데이터
        self.df = df            # 수정할 데이터
        self.categories = {}    # 인코딩한 카테고리
        self.drop_cols = []

    # 드랍할 컬럼을 설정
    def set_drop_cols(self, cols):
        self.drop_cols = cols

    # 수정 전의 데이터를 불러옴
    def call_raw_data(self) :
        return self.raw_df
    
    # 수정된 상태의 데이터를 불러옴, 데이터를 나누지 않은 상태로 불러옴
    def call_full_data(self) :
        return self.df
    
    # 데이터 드랍
    def drop_data(self, columns=None) :
        self.df = self.df.drop(columns=columns)
    
    # 결측치 처리
    def fill_na(self) :
        self.df['Harvest'] = self.df['Harvest'].fillna('O') # Harvest는 Binary 한 값
        self.df['Alive'] = self.df['Alive'].fillna('O')     # Alive는 Binary 한 값
        self.df['EMF'] = self.df['EMF'].fillna(0)           # EMF는 균의 비율로, 측정되지 않은 데이터는 균이 없는 것으로 추정

    # categorical한 데이터 encoding
    def category_encoding(self, encoding='label') :
        # encoding : 'label' - 라벨 인코딩, 'onehot' - One-hot 인코딩
        # categories = {'encoding_type':..., '컬럼명':{'요소':'라벨', '요소':'라벨', ...}, ...}
        # categorical한 데이터 = 데이터가 가지는 서로 다른 값이 15개 미만

        categorical_col = DataCheck.return_cols(self.df, 'categorical')       # Categorical한 column의 컬럼명을 선택

        df_encoded = self.df.copy() # 인코딩할 데이터

        if encoding == 'label' : # 라벨 형식으로 인코딩
            self.categories['encoding_type'] = 'label'

            for col in categorical_col:
                unique_vals = df_encoded[col].unique()
                label = {val: i for i, val in enumerate(unique_vals)}
                df_encoded[col] = df_encoded[col].map(label)
                self.categories[col] = label

        elif encoding == 'onehot' : # One-hot 인코딩
            self.categories['encoding_type'] = 'onehot'
            
            for col in categorical_col:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                self.categories[col] = dummies.columns.tolist()

        else:   # 이상한 값이 들어오면 Value Error 발생
            raise ValueError(f"알 수 없는 encoding_type: {encoding}")

        return df_encoded   # 인코딩된 데이터를 반환
    
    # encoding된 데이터 decoding
    def category_decoding(self, df = None, categories=None) :
        if df is None :                 # 넣어준 인자가 없다면 기존 인스턴스에 저장된 데이터를 디코딩 수행
            df_decoded = self.df.copy()
        else:                           # 넣어준 인자가 있다면 넣어준 데이터에 대해 디코딩 수행
            df_decoded = df.copy()

        if categories is not None :
            self.categories = categories

        encoding_type = self.categories.get('encoding_type', None)  # categories 데이터에 저장된 인코딩 타입에 따라 인코딩 타입 설정

        if encoding_type == 'label':    # 라벨 인코딩일 경우
            for col, mapping in self.categories.items():
                if col == 'encoding_type':
                    continue
                # {원본값: 숫자} → {숫자: 원본값}
                reverse_map = {v: k for k, v in mapping.items()}
                df_decoded[col] = df_decoded[col].map(reverse_map)

        elif encoding_type == 'onehot': # One-hot 인코딩일 경우
            for col, dummy_cols in self.categories.items():
                if col == 'encoding_type':
                    continue
                # 각 mapping column 에서 값이 True인 행을 찾아 역으로 mapping
                def decode_row(row):
                    for dummy_col in dummy_cols:
                        if row[dummy_col] == 1:
                            return dummy_col.replace(f"{col}_", "")
                    return None
                
                df_decoded[col] = df_decoded.apply(decode_row, axis=1)
                df_decoded = df_decoded.drop(columns=dummy_cols)

        else:
            raise ValueError(f"알 수 없는 encoding_type: {encoding_type}")

        return df_decoded           # dictionary 형태 categories 를 받아서, 해당 데이터를 기반으로 디코딩 후, 테이블 반환
    
    # 날짜 기재 방식 통일
    def set_date(self):
        def date_modify(x):
            x = str(x)
            if '/' in x:
                return x.split('/')[0]
            else:
                return x 
        self.df['PlantDate'] = self.df['PlantDate'].apply(date_modify)
    
    # 생존, 실험 중단(수확), 사망 이벤트 -> label data로 병합
    def merge_label(self, df) :
        # Event가 NULL인 데이터는 pred_data로 따로 저장
        df.loc[df["Event"] == 1, "Alive"] = 0
        df.loc[df["Harvest"] == 'X', "Alive"] = 1
        df.loc[df["Alive"] == 'X', "Alive"] = 2

        df.drop(columns=['Event','Harvest'], inplace=True)

        return df  # Event, Harvest, Alive를 병합하여 Dead, Harvest, Alive 3개의 카테고리로 분류
    
    # 전처리 과정 일괄 수행 (encoding은 선택)
    def run(self, encoding=None, return_anomaly=False) :

        self.drop_data(self.drop_cols)

        if return_anomaly is True :
            self.df, df_anomaly = DataModify.data_anomaly_edit(self.df, return_anomaly=True)
        else :
            self.df = DataModify.data_anomaly_edit(self.df, return_anomaly=False)

        self.fill_na()
        self.set_date()
        self.df = self.merge_label(self.df)
        
        if encoding is not None :       # 데이터를 인코딩
            self.df = self.category_encoding(encoding)

        if return_anomaly is True :
            return self.df, df_anomaly
        else:
            return self.df
        

### 데이터 시각화 클래스 선언

class DataVisualize () :

    # 생성 시 기존 데이터를 넣어둠
    def __init__(self, df) :
        self.raw_df = df

    # 연속형 값을 가지는 변수들에 대하여 boxplot을 출력
    def show_boxplot_for_continuous_value(self, df=None) :
        if df is None:
            df = self.raw_df
        
        columns = DataCheck.return_cols(df, 'continuous')

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)

        df_melt = df_scaled.melt(var_name="특성", value_name="값")

        plt.figure(figsize=(1.5*len(columns), 6))
        sns.boxplot(x="특성", y="값", data=df_melt)
        plt.title("특성별 값의 분포")
        plt.xticks(rotation=45)
        plt.show()

    # isolation forest 모델을 사용하여 이상치가 있는지, 이상치 점수 분포를 만들고 시각화
    def show_anomaly_score(self, column, df=None, contamination=0.05, random_state=42, threshold=-0.1, return_anomalies=False):
        if df is None:
            df = self.raw_df
        
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)

        # 모델 학습 (DataFrame 특정 컬럼 -> 2D 배열)
        X = df[column].values.reshape(-1, 1)
        iso_forest.fit(X)

        # decision function 점수 계산
        scores = iso_forest.decision_function(X)

        # 점수 분포 시각화
        plt.figure(figsize=(10,6))
        sns.histplot(scores, bins=50, kde=True, color='skyblue')
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        plt.title(f'Isolation Forest Decision Function Scores for {column}')
        plt.xlabel('Decision Function Score')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

        # 원본 DataFrame에 점수 컬럼 추가
        df = df.copy()
        df['IF_score'] = scores

        if return_anomalies is True:
            # threshold 이하인 이상치 행만 반환
            anomalies = df[df['IF_score'] <= threshold]

            return anomalies[[column,'IF_score']]
    
    # 연속적인 값들을 기반으로 correlation 히트맵 출력
    def show_heatmap_for_continuous_value(self, df=None):
        if df is None:
            df = self.raw_df

        # 숫자형 컬럼만 선택
        numeric_df = df.select_dtypes(include=["int64", "float64"])

        # 상관계수 계산
        corr = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap", fontsize=16)
        plt.show()

    # 컬럼 값을 기준으로 누적 막대 그래프 출력
    def show_survival_ratio(self, df, col):
        # Alive 상태별 비율 계산
        ratio_df = df.groupby(col)['Alive'].value_counts(normalize=True).unstack(fill_value=0)

        # 누적 막대 그래프 그리기
        ratio_df.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Set2')

        plt.title(f"{col} 특성별 생존율")
        plt.xlabel(col)
        plt.ylabel("비율(%)")
        plt.ylim(0,1)
        plt.legend(["Dead", "Harvested", "Alive"], title="Alive", bbox_to_anchor=(1.05,1), loc='upper left')
        plt.show()

    # 연속값을 가지는 컬럼에서 특정값을 기준값으로 하여 누적 막대 그래프 출력
    def show_survival_ratio_with_threshold(self, df, col, threshold) :
        group_col = f"{col}_group"
        df[group_col] = df[col].apply(lambda x: f">={threshold}" if x >= threshold else f"<{threshold}")
        
        # Alive 상태 비율 계산
        ratio_df = df.groupby(group_col)['Alive'].value_counts(normalize=True).unstack().fillna(0)
        
        # 그래프 그리기
        plt.figure(figsize=(10,6))
        ratio_df.plot(kind='bar', stacked=True, colormap="Set2")

        plt.title(f"{col} 기준 {threshold}에 따른 Alive 상태 비율")
        plt.xlabel(f"{col} Group")
        plt.ylabel("비율")
        plt.ylim(0,1)
        plt.legend(["Dead", "Harvested", "Alive"], title="Alive", bbox_to_anchor=(1.05,1), loc='upper left')
        plt.xticks(rotation=0)
        plt.show()

    # 시간에 따라 나무 상태의 변화를 나타낸 그래프 출력
    def show_alive_about_time(self, df):
        # 정렬된 Time 목록
        time_points = np.sort(df['Time'].unique())
        
        # 초기 상태: 모든 샘플 Alive=2
        alive_counts = pd.DataFrame({
            'Time': [0],
            0: [0],   # Dead
            1: [0],   # Harvested
            2: [len(df)]  # Alive
        })

        for t in time_points:
            prev = alive_counts.iloc[-1][[0,1,2]].values
        
            current_dead = df[(df['Time']==t) & (df['Alive']==0)].shape[0]
            current_harvest = df[(df['Time']==t) & (df['Alive']==1)].shape[0]
            current_alive = df[(df['Time']==t) & (df['Alive']==2)].shape[0]

            new_counts = [
                prev[0] + current_dead,
                prev[1] + current_harvest,
                prev[2] - (current_dead + current_harvest)
            ]

            alive_counts = pd.concat([alive_counts, pd.DataFrame([[t]+new_counts], columns=['Time',0,1,2])], ignore_index=True)

        plt.figure(figsize=(10,6))
        plt.plot(alive_counts['Time'], alive_counts[0], label='Dead', color='red')
        plt.plot(alive_counts['Time'], alive_counts[1], label='Harvested', color='orange')
        plt.plot(alive_counts['Time'], alive_counts[2], label='Alive', color='green')

        plt.xlabel("Time")
        plt.ylabel("샘플 수")
        plt.title("시간에 따른 나무의 상태")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # 토양의 출처 종과 실제 종 사이의 생존율 그래프 출력
    def show_survival_heatmap_by_soil(self, df):

        # 생존율 계산
        heatmap_data = (
            df[df['Alive'].isin([0,2])]                 # 라벨 1 제거
            .groupby(['Soil', 'Species'])['Alive']
            .apply(lambda x: (x==2).mean())           # 생존(2)의 비율
            .unstack()
        )
        
        plt.figure(figsize=(10,6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={'label':'Survival Rate'}
        )
        plt.title("토양 출처와 종에 따른 생존율")
        plt.xlabel("종")
        plt.ylabel("토양의 출처가 되는 종")
        plt.show()

    # 내재 화합물에 대한 히스토그램 출력
    def show_chemical_histogram(self, df, col):
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=col, bins=30, kde=True)
        plt.title(f"{col}")
        plt.xlabel(col)
        plt.ylabel("Samples")  
        plt.tight_layout()
        plt.show()

    # 내재 화합물 간의 관계를 파악하기 위한 산점도 출력
    def show_chemical_relation_scatter(self, df, versus) :
        for x, y in versus:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=x, y=y, edgecolor='black', hue=x)
            plt.title(f"{y} vs {x}")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
            plt.show()

    # 내재 화합물 간의 관계를 생존율 관련 boxplot으로 출력
    def show_chemical_relation_by_survive(self, df) :
        columns = ['Phenolics', 'Lignin', 'NSC']

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

        for i, col in enumerate(columns):
            sns.boxplot(
                data=df,
                x='Alive',      
                y=col,                
                ax=axes[i]            
            )
            axes[i].set_title(f"생존과 {col}의 관계성")
            axes[i].set_xlabel("생존")
            axes[i].set_ylabel(col)

        plt.tight_layout()
        plt.show()

    # 균 타입별 분포를 히트맵으로 출력
    def show_hitmap_by_Myco(self, df) :

        # 교차표
        crosstab = pd.crosstab(df['Myco'], df['SoilMyco'])

        # 인코딩된 정보가 들어왔을 경우
        if pd.api.types.is_integer_dtype(df['Myco']) :
            # 라벨 맵핑
            myco_map = {0: "AMF (묘목)", 1: "EMF (묘목)"}
            soilmyco_map = {0: "AMF (토양)", 1: "EMF (토양)", 2: "기타/살균 (토양)"}
            # 행/열 라벨 바꾸기
            crosstab.index = crosstab.index.map(myco_map)
            crosstab.columns = crosstab.columns.map(soilmyco_map)

        # heatmap 시각화
        plt.figure(figsize=(9,4))
        sns.heatmap(
            crosstab, 
            annot=True, 
            fmt="d", 
            cmap="YlGnBu",
            # linewidths=1,
            linecolor='black',
            cbar_kws={'label': '빈도수'}
        )
        plt.title("Myco (묘목) vs SoilMyco (토양) 교차표", fontsize=14, fontweight='bold')
        plt.xlabel("SoilMyco", fontsize=12)
        plt.ylabel("Myco", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.show()


