import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import seaborn as sns

def calculate_metrics(confusion):
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # 정확도
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # 정밀도 (Precision)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # 재현율 (Recall)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

    return metrics

def show_alive_about_time(df):
    # Harvested(Alive=1) 데이터는 처음부터 제외
    df = df[df['Alive'] != 1].copy()
    
    # 정렬된 Time 목록
    time_points = np.sort(df['Time'].unique())
    
    # 초기 상태: Alive=2만 카운트
    alive_counts = pd.DataFrame({
        'Time': [0],
        0: [0],   # Dead
        2: [len(df)]  # Alive
    })

    for t in time_points:
        prev = alive_counts.iloc[-1][[0,2]].values

        current_dead = df[(df['Time']==t) & (df['Alive']==0)].shape[0]
        current_alive = df[(df['Time']==t) & (df['Alive']==2)].shape[0]

        new_counts = [
            prev[0] + current_dead,              # Dead 누적
            prev[1] - current_dead               # Alive 감소
        ]

        alive_counts = pd.concat(
            [alive_counts, pd.DataFrame([[t]+new_counts], columns=['Time',0,2])],
            ignore_index=True
        )

    plt.figure(figsize=(10,6))
    plt.plot(alive_counts['Time'], alive_counts[0], label='Dead', color='red')
    plt.plot(alive_counts['Time'], alive_counts[2], label='Alive', color='green')

    plt.xlabel("Time")
    plt.ylabel("샘플 수")
    plt.title("시간에 따른 나무의 상태")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def compare_km_and_model(model, X_test, time_test, event_test, max_time=None):
    # 1. Kaplan-Meier 곡선 (실제)
    kmf = KaplanMeierFitter()
    kmf.fit(time_test, event_observed=event_test)

    # 2. 모델 예측 곡선 (집단 평균)
    surv_funcs = model.model.predict_survival_function(X_test)
    times = surv_funcs[0].x
    surv_matrix = np.array([fn.y for fn in surv_funcs])
    mean_surv = surv_matrix.mean(axis=0)

    # 3. 시각화
    plt.figure(figsize=(10,6))
    kmf.plot_survival_function(ci_show=True, label="Kaplan-Meier (Actual)", color="black")
    plt.step(times, mean_surv, where="post", label="RSF Predicted (Mean)", color="blue")

    if max_time:
        plt.xlim(0, max_time)

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier vs RSF Predicted Survival")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def rsf_score_over_time(rsf, test_set, t_max, n_points=100): 

    t_values = np.linspace(0, t_max, n_points)

    # 각 지표별 값 저장
    scores = []
    for t in t_values:
        s = calculate_metrics(rsf.confusion_matrix(*test_set, t=t))   # <-- 이 부분이 {'Accuracy':..., 'Precision':..., 'Recall':...} 반환
        scores.append(s)

    # 지표 목록
    metrics = scores[0].keys()

    # 그래프 그리기
    plt.figure(figsize=(8,5))
    for metric in metrics:
        values = [s[metric] for s in scores]
        plt.plot(t_values, values, lw=2, label=metric)

    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.ylim(0, 1)   # 비율형 지표이므로 0~1 범위
    plt.title("RSF Multi-Score vs Time")
    plt.grid(True)
    plt.legend()
    plt.show()

def compare_real_vs_pred_alive_dead(model, df_real, X_test, time_points, threshold=0.5):
    """
    실제 데이터와 모델 예측 Alive/Dead 비율을 한 그래프에 겹쳐서 비교
    
    Parameters:
    - model: fit 완료된 TreeRandomForestSurvival 객체
    - df_real: 실제 데이터 DataFrame (Time, Alive 컬럼 필요)
    - X_test: 모델 입력 테스트 데이터
    - time_points: 그래프를 그릴 시간 포인트 리스트
    - threshold: 생존 확률 기준값
    """
    # ------------------------------
    # 1. 실제 데이터 Alive/Dead 비율
    # ------------------------------
    df = df_real[df_real['Alive'] != 1].copy()  # Harvested 제외
    n_samples_real = len(df)
    
    alive_counts_real = pd.DataFrame({
        'Time': [0],
        'Alive': [len(df)],
        'Dead': [0]
    })
    
    for t in np.sort(df['Time'].unique()):
        prev = alive_counts_real.iloc[-1][['Dead','Alive']].values
        current_dead = df[(df['Time']==t) & (df['Alive']==0)].shape[0]
        new_counts = [prev[0]+current_dead, prev[1]-current_dead]
        alive_counts_real = pd.concat(
            [alive_counts_real, pd.DataFrame([[t]+new_counts], columns=['Time','Dead','Alive'])],
            ignore_index=True
        )
    
    # 비율로 변환
    alive_counts_real['Alive'] /= n_samples_real
    alive_counts_real['Dead'] /= n_samples_real

    # ------------------------------
    # 2. 모델 예측 Alive/Dead 비율
    # ------------------------------
    n_samples_pred = len(X_test)
    alive_counts_pred = pd.DataFrame({'Time': [], 'Alive': [], 'Dead': []})
    
    for t in time_points:
        surv_dict = model.predict(X_test, t=t)
        surv_probs = surv_dict[t]
        pred_labels = (surv_probs >= threshold).astype(int)
        alive_counts_pred = pd.concat(
            [alive_counts_pred, pd.DataFrame([[t, (pred_labels==1).mean(), (pred_labels==0).mean()]], columns=['Time','Alive','Dead'])],
            ignore_index=True
        )

    # ------------------------------
    # 3. 시각화 (겹쳐서)
    # ------------------------------
    plt.figure(figsize=(10,6))

    # 실제 데이터
    plt.plot(alive_counts_real['Time'], alive_counts_real['Alive'], label='Real Alive', color='green', linestyle='--')
    plt.plot(alive_counts_real['Time'], alive_counts_real['Dead'], label='Real Dead', color='red', linestyle='--')

    # 모델 예측
    plt.plot(alive_counts_pred['Time'], alive_counts_pred['Alive'], label='Predicted Alive', color='green')
    plt.plot(alive_counts_pred['Time'], alive_counts_pred['Dead'], label='Predicted Dead', color='red')

    plt.xlabel("Time")
    plt.ylabel("Ratio")
    plt.title("Real vs Predicted Alive/Dead Ratios Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def show_confusion_matrix(cm) :
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Greens',
                xticklabels=np.array([0, 1]), 
                yticklabels=np.array([0, 1]))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()