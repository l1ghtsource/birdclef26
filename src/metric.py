import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc_skip_missing(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    scored_mask = y_true.sum(axis=0) > 0
    if not np.any(scored_mask):
        return 0.0
    y_true_scored = y_true[:, scored_mask]
    y_pred_scored = y_pred[:, scored_mask]
    try:
        return float(roc_auc_score(y_true_scored, y_pred_scored, average="macro"))
    except ValueError:
        per_class = []
        for i in range(y_true_scored.shape[1]):
            target = y_true_scored[:, i]
            pred = y_pred_scored[:, i]
            if target.max() == target.min():
                continue
            try:
                per_class.append(roc_auc_score(target, pred))
            except ValueError:
                continue
        if not per_class:
            return 0.0
        return float(np.mean(per_class))
