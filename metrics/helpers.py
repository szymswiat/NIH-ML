import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def precision_recall_auc_scores(
        targets: np.ndarray,
        preds: np.ndarray
) -> np.ndarray:
    pr_auc_scores = []
    for i in range(targets.shape[1]):
        target, pred = targets[..., i], preds[..., i]
        precision, recall, th = precision_recall_curve(target, pred)
        pr_auc_scores.append(auc(recall, precision))

    return np.array(pr_auc_scores, dtype=float)
