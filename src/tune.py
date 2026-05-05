"""Parameter tuning with 5-fold CV."""

import os
import sys
import time
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_image
from masking import rgb_to_grayscale, generate_roi_mask, morph_open, morph_close, largest_component_mask, otsu_threshold, rgb_to_hsv, threshold_background
from edge_detection import canny_edge_detection
from classify import (
    extract_features, compute_fill_density, gradient_direction_entropy,
    gradient_magnitude_stats, compute_frequency_profile,
    gradient_direction_kurtosis, compute_edge_density_cv,
    compute_edge_to_gradient_ratio, compute_roi_coverage,
    compute_line_spacing_uniformity, compute_gradient_histogram,
    detect_line_direction, sample_perpendicular_profiles, find_1d_peaks,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CLASSES = ['optimal', 'under_extruded', 'over_extruded']
# 7-feature signature.  fill_density and roi_coverage are paired
# (one is the in-ROI density, the other normalizes against how much
# of the frame is ROI); kept as separate columns for numerical
# stability but conceptually a single feature.
FEAT_KEYS = [
    'fill_density', 'gradient_entropy', 'gradient_dir_kurtosis',
    'edge_to_gradient_ratio', 'spectral_energy_ratio',
    'edge_density_cv', 'roi_coverage', 'line_spacing_uniformity',
]


def load_all_images():
    """Load all labeled images."""
    images = []
    labels = []
    names = []
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.png'):
                continue
            img = load_image(os.path.join(cls_dir, fname))
            images.append(img)
            labels.append(cls)
            names.append(f"{cls}/{fname}")
    return images, labels, names


def extract_features_with_params(image_rgb, sigma=1.0, low_ratio=0.05,
                                  high_ratio=0.15, cleanup_kernel=5,
                                  num_bins=36, num_patches=8,
                                  line_conf_thresh=1.5, peak_prominence=0.1):
    """Feature extraction w/ configurable params."""
    grayscale = rgb_to_grayscale(image_rgb)

    # ROI mask
    gray = rgb_to_grayscale(image_rgb)
    candidates = []
    otsu_mask, _ = otsu_threshold(gray)
    candidates.append(otsu_mask)
    hsv = rgb_to_hsv(image_rgb)
    hsv_mask = threshold_background(hsv)
    candidates.append(hsv_mask)
    inv_mask, _ = otsu_threshold(255.0 - gray)
    candidates.append(inv_mask)

    from masking import _mask_quality
    best_mask = None
    best_score = -1.0
    for candidate in candidates:
        cleaned = morph_open(candidate, kernel_size=cleanup_kernel, iterations=1)
        cleaned = morph_close(cleaned, kernel_size=cleanup_kernel, iterations=1)
        cleaned = largest_component_mask(cleaned)
        score = _mask_quality(cleaned)
        if score > best_score:
            best_score = score
            best_mask = cleaned
    roi_mask = best_mask

    edges, magnitude, direction = canny_edge_detection(
        grayscale, sigma=sigma, low_ratio=low_ratio, high_ratio=high_ratio)

    fill_density = compute_fill_density(edges, roi_mask)
    grad_entropy = gradient_direction_entropy(magnitude, direction, roi_mask)
    grad_stats = gradient_magnitude_stats(magnitude, roi_mask)
    _, dominant_freq, spectral_ratio = compute_frequency_profile(grayscale, roi_mask)
    grad_dir_kurtosis = gradient_direction_kurtosis(magnitude, direction, roi_mask)
    edge_density_cv = compute_edge_density_cv(edges, roi_mask, num_patches=num_patches)

    edge_to_grad = compute_edge_to_gradient_ratio(edges, magnitude, roi_mask)
    roi_cov = compute_roi_coverage(roi_mask)

    # line spacing
    roi_area = np.sum(roi_mask)
    if roi_area == 0:
        line_uniformity = 0.5
    else:
        line_angle, confidence = detect_line_direction(magnitude, direction, roi_mask)
        if confidence < line_conf_thresh:
            line_uniformity = 0.5
        else:
            profiles = sample_perpendicular_profiles(grayscale, roi_mask, line_angle)
            if len(profiles) == 0:
                line_uniformity = 0.5
            else:
                all_spacings = []
                for prof in profiles:
                    peaks = find_1d_peaks(prof, min_prominence_ratio=peak_prominence)
                    if len(peaks) >= 2:
                        all_spacings.extend(np.diff(peaks).tolist())
                if len(all_spacings) < 5:
                    line_uniformity = 0.5
                else:
                    spacings_arr = np.array(all_spacings, dtype=np.float64)
                    mean_sp = np.mean(spacings_arr)
                    if mean_sp < 1e-10:
                        line_uniformity = 0.5
                    else:
                        cv = np.std(spacings_arr) / mean_sp
                        line_uniformity = 1.0 / (1.0 + cv)

    features = {
        'fill_density': fill_density,
        'gradient_entropy': grad_entropy,
        'gradient_dir_kurtosis': grad_dir_kurtosis,
        'edge_to_gradient_ratio': edge_to_grad,
        'spectral_energy_ratio': spectral_ratio,
        'edge_density_cv': edge_density_cv,
        'roi_coverage': roi_cov,
        'line_spacing_uniformity': float(line_uniformity),
    }
    return features


def make_folds(labels, n_folds=5, seed=42):
    """Stratified k-fold split."""
    rng = np.random.RandomState(seed)
    class_indices = {}
    for i, label in enumerate(labels):
        class_indices.setdefault(label, []).append(i)

    folds = [[] for _ in range(n_folds)]
    for cls in CLASSES:
        indices = class_indices.get(cls, [])
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_folds].append(idx)

    return folds


def train_and_test(feature_vecs, labels, train_idx, test_idx, reg=0.01):
    """Train LDA, test on held-out fold."""
    class_vecs = {c: [] for c in CLASSES}
    for i in train_idx:
        class_vecs[labels[i]].append(feature_vecs[i])

    centroids = {}
    for c in CLASSES:
        if len(class_vecs[c]) == 0:
            return 0.0
        centroids[c] = np.mean(class_vecs[c], axis=0)

    # pooled cov
    n_feat = len(FEAT_KEYS)
    S = np.zeros((n_feat, n_feat))
    N = len(train_idx)
    K = len(CLASSES)
    for c in CLASSES:
        vecs = np.array(class_vecs[c])
        if len(vecs) < 2:
            return 0.0
        S += (len(vecs) - 1) * np.cov(vecs, rowvar=False, bias=False)
    S /= (N - K)
    S += reg * np.eye(n_feat)

    try:
        inv_cov = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return 0.0

    correct = 0
    for i in test_idx:
        vec = feature_vecs[i]
        dists = {}
        for lbl, mu in centroids.items():
            diff = vec - mu
            dists[lbl] = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
        if min(dists, key=dists.get) == labels[i]:
            correct += 1

    return correct / len(test_idx) if len(test_idx) > 0 else 0.0


def cv_accuracy(feature_vecs, labels, folds, reg=0.01):
    """Mean cross-validation accuracy."""
    accs = []
    for fold_idx in range(len(folds)):
        test_idx = folds[fold_idx]
        train_idx = []
        for j in range(len(folds)):
            if j != fold_idx:
                train_idx.extend(folds[j])
        acc = train_and_test(feature_vecs, labels, train_idx, test_idx, reg=reg)
        accs.append(acc)
    return np.mean(accs)


def main():
    print("Loading images...")
    images, labels, names = load_all_images()
    n = len(images)
    print(f"Loaded {n} images")

    folds = make_folds(labels, n_folds=5)
    print(f"5-fold CV: {[len(f) for f in folds]} images per fold")
    print()

    # baseline
    print("=" * 60)
    print("BASELINE (current parameters)")
    print("=" * 60)
    baseline_params = dict(sigma=1.0, low_ratio=0.05, high_ratio=0.15,
                           cleanup_kernel=5, num_bins=36, num_patches=8,
                           line_conf_thresh=1.5, peak_prominence=0.1)

    print("Extracting features with default params...")
    t0 = time.time()
    baseline_vecs = []
    for i, img in enumerate(images):
        feats = extract_features_with_params(img, **baseline_params)
        baseline_vecs.append(np.array([feats[k] for k in FEAT_KEYS]))
    print(f"  Done in {time.time()-t0:.1f}s")

    baseline_cv = cv_accuracy(baseline_vecs, labels, folds)
    print(f"  Baseline CV accuracy: {baseline_cv*100:.1f}%")
    print()

    best_params = dict(baseline_params)
    best_cv = baseline_cv

    # group 1: canny params
    print("=" * 60)
    print("GROUP 1: Canny parameters (sigma, low_ratio, high_ratio)")
    print("=" * 60)

    sigma_vals = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    low_vals = [0.03, 0.05, 0.08, 0.10]
    high_vals = [0.10, 0.15, 0.20, 0.25]

    g1_best_cv = 0
    g1_best = {}
    total_combos = len(sigma_vals) * len(low_vals) * len(high_vals)
    combo_num = 0

    for sigma, low_r, high_r in product(sigma_vals, low_vals, high_vals):
        if low_r >= high_r:
            continue
        combo_num += 1
        params = dict(best_params)
        params['sigma'] = sigma
        params['low_ratio'] = low_r
        params['high_ratio'] = high_r

        vecs = []
        for img in images:
            feats = extract_features_with_params(img, **params)
            vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

        acc = cv_accuracy(vecs, labels, folds)
        if acc > g1_best_cv:
            g1_best_cv = acc
            g1_best = {'sigma': sigma, 'low_ratio': low_r, 'high_ratio': high_r}
            print(f"  [{combo_num}] NEW BEST: sigma={sigma}, low={low_r}, high={high_r} -> CV={acc*100:.1f}%")

    if g1_best_cv > best_cv:
        best_params.update(g1_best)
        best_cv = g1_best_cv
    print(f"  Group 1 best: {g1_best} -> {g1_best_cv*100:.1f}%")
    print()

    # group 2: morph cleanup
    print("=" * 60)
    print("GROUP 2: Morphological cleanup kernel")
    print("=" * 60)

    for ks in [3, 5, 7, 9]:
        params = dict(best_params)
        params['cleanup_kernel'] = ks

        vecs = []
        for img in images:
            feats = extract_features_with_params(img, **params)
            vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

        acc = cv_accuracy(vecs, labels, folds)
        marker = " <-- BEST" if acc > best_cv else ""
        print(f"  kernel={ks}: CV={acc*100:.1f}%{marker}")
        if acc > best_cv:
            best_params['cleanup_kernel'] = ks
            best_cv = acc

    print()

    # group 3: feature params
    print("=" * 60)
    print("GROUP 3: Feature extraction parameters")
    print("=" * 60)

    print("  --- num_bins ---")
    for nb in [12, 18, 24, 36, 48, 72]:
        params = dict(best_params)
        params['num_bins'] = nb

        vecs = []
        for img in images:
            feats = extract_features_with_params(img, **params)
            vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

        acc = cv_accuracy(vecs, labels, folds)
        marker = " <-- BEST" if acc > best_cv else ""
        print(f"    bins={nb}: CV={acc*100:.1f}%{marker}")
        if acc > best_cv:
            best_params['num_bins'] = nb
            best_cv = acc

    print("  --- num_patches ---")
    for np_ in [4, 6, 8, 10, 12]:
        params = dict(best_params)
        params['num_patches'] = np_

        vecs = []
        for img in images:
            feats = extract_features_with_params(img, **params)
            vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

        acc = cv_accuracy(vecs, labels, folds)
        marker = " <-- BEST" if acc > best_cv else ""
        print(f"    patches={np_}: CV={acc*100:.1f}%{marker}")
        if acc > best_cv:
            best_params['num_patches'] = np_
            best_cv = acc

    print("  --- line_conf_thresh ---")
    for lct in [0.5, 1.0, 1.2, 1.5, 2.0, 2.5]:
        params = dict(best_params)
        params['line_conf_thresh'] = lct

        vecs = []
        for img in images:
            feats = extract_features_with_params(img, **params)
            vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

        acc = cv_accuracy(vecs, labels, folds)
        marker = " <-- BEST" if acc > best_cv else ""
        print(f"    conf={lct}: CV={acc*100:.1f}%{marker}")
        if acc > best_cv:
            best_params['line_conf_thresh'] = lct
            best_cv = acc

    print("  --- peak_prominence ---")
    for pp in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        params = dict(best_params)
        params['peak_prominence'] = pp

        vecs = []
        for img in images:
            feats = extract_features_with_params(img, **params)
            vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

        acc = cv_accuracy(vecs, labels, folds)
        marker = " <-- BEST" if acc > best_cv else ""
        print(f"    prominence={pp}: CV={acc*100:.1f}%{marker}")
        if acc > best_cv:
            best_params['peak_prominence'] = pp
            best_cv = acc

    print()

    # group 4: regularization
    print("=" * 60)
    print("GROUP 4: Covariance regularization")
    print("=" * 60)

    final_vecs = []
    for img in images:
        feats = extract_features_with_params(img, **best_params)
        final_vecs.append(np.array([feats[k] for k in FEAT_KEYS]))

    for reg in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        acc = cv_accuracy(final_vecs, labels, folds, reg=reg)
        marker = " <-- BEST" if acc > best_cv else ""
        print(f"  reg={reg}: CV={acc*100:.1f}%{marker}")
        if acc > best_cv:
            best_cv = acc
            best_params['reg'] = reg

    print()

    # summary
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline CV: {baseline_cv*100:.1f}%")
    print(f"Tuned CV:    {best_cv*100:.1f}%")
    print(f"Improvement: {(best_cv - baseline_cv)*100:+.1f}%")
    print()
    print("Best parameters:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    reg = best_params.get('reg', 0.01)
    all_idx = list(range(n))

    class_vecs = {c: [] for c in CLASSES}
    for i in range(n):
        class_vecs[labels[i]].append(final_vecs[i])
    centroids = {c: np.mean(class_vecs[c], axis=0) for c in CLASSES}

    n_feat = len(FEAT_KEYS)
    S = np.zeros((n_feat, n_feat))
    N = n
    K = 3
    for c in CLASSES:
        vecs = np.array(class_vecs[c])
        S += (len(vecs) - 1) * np.cov(vecs, rowvar=False, bias=False)
    S /= (N - K)
    S += reg * np.eye(n_feat)
    inv_cov = np.linalg.inv(S)

    correct = 0
    for i in range(n):
        dists = {}
        for lbl, mu in centroids.items():
            diff = final_vecs[i] - mu
            dists[lbl] = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
        if min(dists, key=dists.get) == labels[i]:
            correct += 1
    print(f"\nTraining accuracy with best params: {correct}/{n} ({100*correct/n:.1f}%)")

    print("\n--- COPY-PASTE for classify.py ---")
    print("class_centroids = {")
    for c in CLASSES:
        vals = ', '.join(f'{v:.4f}' for v in centroids[c])
        print(f"    '{c}': np.array([{vals}]),")
    print("}")
    print()
    print("shared_inv_cov = np.array([")
    for i, row in enumerate(inv_cov):
        vals = ', '.join(f'{v:>15.8e}' for v in row)
        comma = ',' if i < len(inv_cov) - 1 else ''
        print(f"    [{vals}]{comma}")
    print("])")


if __name__ == '__main__':
    main()
