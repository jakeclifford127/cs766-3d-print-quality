"""
Main pipeline for single image classification and batch eval.

Usage:
    python pipeline.py <image_path>
    python pipeline.py --batch <data_dir>
"""

import sys
import os
import numpy as np

from convolution import convolve2d_fast
from edge_detection import canny_edge_detection, sobel_gradients, gaussian_blur
from masking import rgb_to_grayscale, generate_roi_mask
from classify import extract_features, classify_layer
from evaluate import print_evaluation_report


def load_image(path, max_dim=512):
    """Load image as RGB, optionally downscale."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    if max_dim is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_single_image(image_rgb, sigma=1.0, low_ratio=0.05, high_ratio=0.15):
    """Run full pipeline on one image."""
    grayscale = rgb_to_grayscale(image_rgb)

    roi_mask = generate_roi_mask(image_rgb, method='auto', cleanup=True)

    edges, magnitude, direction = canny_edge_detection(
        grayscale, sigma=sigma, low_ratio=low_ratio, high_ratio=high_ratio
    )

    features = extract_features(grayscale, edges, magnitude, direction, roi_mask)

    label, confidence = classify_layer(features)

    intermediates = {
        'grayscale': grayscale,
        'roi_mask': roi_mask,
        'edges': edges,
        'magnitude': magnitude,
        'direction': direction,
    }

    return label, confidence, features, intermediates


def visualize_results(image_rgb, intermediates, label, confidence, save_path=None):
    """Show pipeline results as subplots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original')

    axes[0, 1].imshow(intermediates['grayscale'], cmap='gray')
    axes[0, 1].set_title('Grayscale')

    axes[0, 2].imshow(intermediates['roi_mask'], cmap='gray')
    axes[0, 2].set_title('ROI Mask')

    axes[1, 0].imshow(intermediates['magnitude'], cmap='hot')
    axes[1, 0].set_title('Gradient Magnitude')

    axes[1, 1].imshow(intermediates['edges'], cmap='gray')
    axes[1, 1].set_title('Canny Edges')

    # overlay edges on original
    overlay = image_rgb.copy()
    edge_pixels = intermediates['edges'] > 0
    overlay[edge_pixels] = [255, 0, 0]
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Result: {label} ({confidence:.2f})')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def batch_evaluate(data_dir):
    """Evaluate on labeled dataset folders."""
    categories = ['optimal', 'under_extruded', 'over_extruded']
    y_true = []
    y_pred = []

    for category in categories:
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"Warning: directory not found — {cat_dir}")
            continue

        files = [f for f in os.listdir(cat_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for fname in sorted(files):
            fpath = os.path.join(cat_dir, fname)
            try:
                image = load_image(fpath)
                label, confidence, features, _ = process_single_image(image)
                y_true.append(category)
                y_pred.append(label)
                status = "OK" if label == category else "MISS"
                print(f"  [{status}] {fname}: predicted={label} (conf={confidence:.2f}), "
                      f"density={features['fill_density']:.4f}")
            except Exception as e:
                print(f"  [ERR] {fname}: {e}")

    return y_true, y_pred


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py <image_path>         # classify single image")
        print("  python pipeline.py --batch <data_dir>   # batch evaluate")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data'
        print(f"Batch evaluation on: {data_dir}\n")
        y_true, y_pred = batch_evaluate(data_dir)
        if y_true:
            print_evaluation_report(y_true, y_pred)
        else:
            print("No images found. Add images to data/optimal/, data/under_extruded/, data/over_extruded/")
    else:
        image_path = sys.argv[1]
        print(f"Processing: {image_path}")
        image = load_image(image_path)
        label, confidence, features, intermediates = process_single_image(image)

        print(f"\nClassification: {label} (confidence: {confidence:.2f})")
        print("\nFeatures:")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")

        visualize_results(image, intermediates, label, confidence)
