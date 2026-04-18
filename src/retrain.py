"""Retrain classifier parameters from labeled dataset.

Computes class centroids and shared (pooled) inverse covariance matrix
for the LDA-style Mahalanobis distance classifier. Prints the results
as copy-pasteable numpy arrays for classify.py.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_image
from masking import rgb_to_grayscale, generate_roi_mask
from edge_detection import canny_edge_detection
from classify import extract_features


FEAT_KEYS = [
    'fill_density', 'gradient_entropy', 'gradient_dir_kurtosis',
    'edge_to_gradient_ratio', 'spectral_energy_ratio',
    'edge_density_cv', 'roi_coverage', 'line_spacing_uniformity',
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CLASSES = ['optimal', 'under_extruded', 'over_extruded']


def collect_features():
    """Run pipeline on all labeled images and collect feature vectors."""
    class_features = {c: [] for c in CLASSES}

    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: {cls_dir} not found, skipping")
            continue

        fnames = sorted(f for f in os.listdir(cls_dir) if f.endswith('.png'))
        for fname in fnames:
            path = os.path.join(cls_dir, fname)
            try:
                img = load_image(path)
                grayscale = rgb_to_grayscale(img)
                roi_mask = generate_roi_mask(img, method='auto', cleanup=True)
                edges, magnitude, direction = canny_edge_detection(grayscale)
                features = extract_features(grayscale, edges, magnitude, direction, roi_mask)
                vec = np.array([features[k] for k in FEAT_KEYS])
                class_features[cls].append(vec)
                hough_info = f"hough_count={features.get('hough_line_count', 0):.2f}"
                print(f"  {cls}/{fname}: {hough_info}, uniformity={features['line_spacing_uniformity']:.4f}")
            except Exception as e:
                print(f"  ERROR {cls}/{fname}: {e}")

    return class_features


def compute_centroids(class_features):
    """Compute per-class mean vectors."""
    centroids = {}
    for cls in CLASSES:
        vecs = class_features[cls]
        if len(vecs) == 0:
            print(f"Warning: no features for {cls}")
            continue
        centroids[cls] = np.mean(vecs, axis=0)
    return centroids


def compute_pooled_inv_cov(class_features, reg=0.01):
    """Compute pooled within-class inverse covariance (LDA-style)."""
    n_features = len(FEAT_KEYS)
    S_pooled = np.zeros((n_features, n_features))
    N = 0
    K = len(CLASSES)

    for cls in CLASSES:
        vecs = np.array(class_features[cls])
        if len(vecs) < 2:
            continue
        n_k = len(vecs)
        S_k = np.cov(vecs, rowvar=False, bias=False)  # (n_k-1) normalized
        S_pooled += (n_k - 1) * S_k
        N += n_k

    S_pooled /= (N - K)

    # Regularize for numerical stability
    S_pooled += reg * np.eye(n_features)

    # Invert
    inv_cov = np.linalg.inv(S_pooled)
    return inv_cov


def format_array(arr, name, indent=8):
    """Format a numpy array as a copy-pasteable Python literal."""
    spaces = ' ' * indent
    if arr.ndim == 1:
        vals = ', '.join(f'{v:.4f}' for v in arr)
        return f"'{name}': np.array([{vals}]),"
    elif arr.ndim == 2:
        lines = []
        lines.append(f"np.array([")
        for i, row in enumerate(arr):
            vals = ', '.join(f'{v:>15.8e}' for v in row)
            comma = ',' if i < len(arr) - 1 else ''
            lines.append(f"{spaces}[{vals}]{comma}")
        lines.append(f"{spaces[:-4]}])")
        return '\n'.join(lines)


def main():
    print("=" * 60)
    print("RETRAINING CLASSIFIER")
    print(f"Features: {FEAT_KEYS}")
    print(f"Data dir: {os.path.abspath(DATA_DIR)}")
    print("=" * 60)
    print()

    # Collect features
    print("Collecting features from all images...")
    class_features = collect_features()

    for cls in CLASSES:
        print(f"  {cls}: {len(class_features[cls])} images")
    print()

    # Compute centroids
    centroids = compute_centroids(class_features)
    print("Per-class centroids:")
    for cls in CLASSES:
        print(f"  {cls}: {centroids[cls]}")
    print()

    # Show per-class feature means for analysis
    print("Feature separation analysis:")
    print(f"  {'Feature':<28s} {'Optimal':>10s} {'Under':>10s} {'Over':>10s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
    for i, key in enumerate(FEAT_KEYS):
        vals = [centroids[cls][i] for cls in CLASSES]
        print(f"  {key:<28s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")
    print()

    # Compute inverse covariance
    inv_cov = compute_pooled_inv_cov(class_features)
    print("Shared inverse covariance matrix computed.")
    print()

    # Print copy-pasteable output
    print("=" * 60)
    print("COPY-PASTE INTO classify_layer():")
    print("=" * 60)
    print()
    print("class_centroids = {")
    for cls in CLASSES:
        print(f"    {format_array(centroids[cls], cls)}")
    print("}")
    print()
    print(f"shared_inv_cov = {format_array(inv_cov, 'shared_inv_cov')}")
    print()

    # Quick accuracy check with new parameters
    print("=" * 60)
    print("ACCURACY CHECK (training set)")
    print("=" * 60)
    correct = 0
    total = 0
    per_class = {c: {'correct': 0, 'total': 0} for c in CLASSES}

    for cls in CLASSES:
        for vec in class_features[cls]:
            distances = {}
            for label, mean_vec in centroids.items():
                diff = vec - mean_vec
                distances[label] = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
            pred = min(distances, key=distances.get)
            if pred == cls:
                correct += 1
                per_class[cls]['correct'] += 1
            total += 1
            per_class[cls]['total'] += 1

    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    for cls in CLASSES:
        c = per_class[cls]
        pct = 100 * c['correct'] / c['total'] if c['total'] > 0 else 0
        print(f"  {cls}: {c['correct']}/{c['total']} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
