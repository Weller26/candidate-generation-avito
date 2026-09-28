import numpy as np

def calculate_recall_at_k(predictions: dict, ground_truth: dict, k=50):
    """
    Расчитывает метрику Recall@50 для предсказаний (predictions) 
    и правильных ответов (ground truth).
    """
    recalls = []
    
    for query_id, true_items in ground_truth.items():
        if query_id not in predictions:
            recalls.append(0.0)
            continue

        pred_items = predictions[query_id][:k]

        true_set = set(true_items)
        pred_set = set(pred_items)

        hits = len(true_set.intersection(pred_set))

        recall = hits / len(true_set)
        recalls.append(recall)

    return np.mean(recalls)