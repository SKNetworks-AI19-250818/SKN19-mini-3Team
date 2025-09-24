import numpy as np

def calculate_metrics(confusion):
    TP = confusion.get('TP', 0)
    TN = confusion.get('TN', 0)
    FP = confusion.get('FP', 0)
    FN = confusion.get('FN', 0)

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
