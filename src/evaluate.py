import numpy as np


LABELS = ['optimal', 'under_extruded', 'over_extruded']


def confusion_matrix(y_true, y_pred, labels=None):
    """Build confusion matrix."""
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
    """Per-class precision, recall, F1."""
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
    """Print formatted eval report."""
    matrix, labels = confusion_matrix(y_true, y_pred)
    metrics = precision_recall_f1(matrix, labels)

    print("\nConfusion Matrix:")
    header = "                " + "  ".join(f"{l:>15}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>15}  " + "  ".join(f"{matrix[i, j]:>15}" for j in range(len(labels)))
        print(row)

    print("\nPer-Class Metrics:")
    print(f"{'Label':>15}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 50)
    for label in labels:
        m = metrics[label]
        print(f"{label:>15}  {m['precision']:>10.3f}  {m['recall']:>10.3f}  {m['f1']:>10.3f}")

    correct = np.trace(matrix)
    total = np.sum(matrix)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nOverall Accuracy: {accuracy:.3f} ({correct}/{total})")

    return metrics
