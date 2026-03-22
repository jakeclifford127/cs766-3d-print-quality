import numpy as np


LABELS = ['optimal', 'under_extruded', 'over_extruded']


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute a confusion matrix.

    Args:
        y_true: list of true labels (strings)
        y_pred: list of predicted labels (strings)
        labels: list of label names in order

    Returns:
        matrix: 2D numpy array (rows=true, cols=predicted)
        labels: list of label names
    """
    if labels is None:
        labels = LABELS

    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=np.int32)

    for true, pred in zip(y_true, y_pred):
        i = label_to_idx[true]
        j = label_to_idx[pred]
        matrix[i, j] += 1

    return matrix, labels


def precision_recall_f1(matrix, labels=None):
    """Compute per-class precision, recall, and F1 from a confusion matrix.

    Args:
        matrix: 2D confusion matrix (rows=true, cols=predicted)
        labels: label names

    Returns:
        dict mapping each label to {precision, recall, f1}
    """
    if labels is None:
        labels = LABELS

    results = {}
    for i, label in enumerate(labels):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    return results


def print_evaluation_report(y_true, y_pred):
    """Print a formatted evaluation report.

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
    """
    matrix, labels = confusion_matrix(y_true, y_pred)
    metrics = precision_recall_f1(matrix, labels)

    # Confusion matrix
    print("\nConfusion Matrix:")
    header = "                " + "  ".join(f"{l:>15}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>15}  " + "  ".join(f"{matrix[i, j]:>15}" for j in range(len(labels)))
        print(row)

    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Label':>15}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 50)
    for label in labels:
        m = metrics[label]
        print(f"{label:>15}  {m['precision']:>10.3f}  {m['recall']:>10.3f}  {m['f1']:>10.3f}")

    # Overall accuracy
    correct = np.trace(matrix)
    total = np.sum(matrix)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nOverall Accuracy: {accuracy:.3f} ({correct}/{total})")

    return metrics
