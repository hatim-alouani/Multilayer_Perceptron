"""Evaluation metrics, computed from scratch.

Convention: positive class = malignant = index 1.
Recall on the malignant class is the metric that actually matters here -- a
false negative means telling someone a malignant tumour is benign.
"""

import numpy as np


def to_labels(A):
    """(n_classes, m) probabilities -> (m,) predicted class indices."""
    return np.argmax(A, axis=0)


def accuracy(Y, A):
    return float(np.mean(to_labels(A) == to_labels(Y)))


def confusion_matrix(y_true, y_pred, positive=1):
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    tn = int(np.sum((y_pred != positive) & (y_true != positive)))
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred, positive=1):
    tp, fp, fn, _ = confusion_matrix(y_true, y_pred, positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def report(y_true, y_pred, positive=1):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, positive)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, positive)
    acc = float(np.mean(y_true == y_pred))
    lines = [
        f"accuracy   : {acc:.4f}",
        f"precision  : {precision:.4f}",
        f"recall     : {recall:.4f}",
        f"f1 score   : {f1:.4f}",
        "confusion matrix (positive = malignant)",
        f"    TP {tp:4d}   FP {fp:4d}",
        f"    FN {fn:4d}   TN {tn:4d}",
    ]
    return "\n".join(lines)
