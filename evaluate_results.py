"""
evaluate_results.py
-------------------
Loads saved models + test data from the notebook pipeline and prints:
  - Overall accuracy, precision, recall, F1 for all three models
  - Per-class classification report (KNN & SVM highlighted)
  - Confusion matrix display (text) for KNN and SVM
  - Side-by-side comparison table

Run from the project root:
    python evaluate_results.py
"""

import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── Config (must match notebook) ──────────────────────────────────────────
DATA_DIR      = 'data/raw'
MODELS_DIR    = 'models'
MAX_PER_CLASS = 200
IMAGE_SIZE    = (128, 128)
HIST_BINS     = 32
PCA_VARIANCE  = 0.95
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.10

# ── 1. Load dataset ────────────────────────────────────────────────────────
print("Loading dataset ...")
from src.preprocessing import (
    load_dataset, split_data, LABEL_TO_CLASS
)

images, labels, found_classes = load_dataset(
    data_dir=DATA_DIR,
    image_size=IMAGE_SIZE,
    max_per_class=MAX_PER_CLASS,
)

# Re-map labels to contiguous 0-based (same logic as notebook)
unique_labels       = sorted(np.unique(labels))
remap               = {old: new for new, old in enumerate(unique_labels)}
labels              = np.array([remap[l] for l in labels], dtype=np.int32)
class_names_ordered = [LABEL_TO_CLASS[old] for old in unique_labels]
n_classes           = len(class_names_ordered)
short_names         = [c.replace('Tomato___', '').replace('Tomato__', '').replace('_', ' ')
                       for c in class_names_ordered]

print(f"\nDataset: {len(images)} images  |  {n_classes} classes")

# ── 2. Feature extraction ──────────────────────────────────────────────────
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(
    use_color_histogram  = True,
    histogram_bins       = HIST_BINS,
    histogram_spaces     = ('rgb', 'hsv', 'lab'),
    use_color_statistics = True,
    use_glcm             = True,
    use_lbp              = True,
    use_hu_moments       = True,
)

# ── 3. Split (same seed → identical test set) ──────────────────────────────
(X_train, X_val, X_test,
 y_train, y_val, y_test) = split_data(
    images, labels,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    random_state=RANDOM_STATE,
)

print(f"\nExtracting features for test set ({len(X_test)} images) ...")
F_test = extractor.extract_batch(X_test, desc='Test')

# ── 4. Load PCA reducer ────────────────────────────────────────────────────
from src.dimensionality_reduction import PCAReducer

reducer = PCAReducer.load(f'{MODELS_DIR}/pca_reducer.pkl')
P_test  = reducer.transform(F_test)
print(f"Feature dims after PCA: {P_test.shape[1]}")

# ── 5. Load trained models ─────────────────────────────────────────────────
from src.models import ModelWrapper

model_files = {
    'KNN':           f'{MODELS_DIR}/knn.pkl',
    'SVM':           f'{MODELS_DIR}/svm.pkl',
    'Random Forest': f'{MODELS_DIR}/random_forest.pkl',
}

models = {}
for name, path in model_files.items():
    if Path(path).exists():
        models[name] = ModelWrapper.load(path)
    else:
        print(f"[WARN] Model file not found: {path}")

# ── 6. Evaluate each model ─────────────────────────────────────────────────
DIVIDER = '=' * 72

results_summary = []

for name, wrapper in models.items():
    y_pred = wrapper.predict(P_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
    f1w  = f1_score(y_test, y_pred,        average='weighted', zero_division=0)
    f1m  = f1_score(y_test, y_pred,        average='macro',    zero_division=0)

    results_summary.append({
        'Model':              name,
        'Accuracy':           acc,
        'Precision(weighted)':prec,
        'Recall(weighted)':   rec,
        'F1(weighted)':       f1w,
        'F1(macro)':          f1m,
    })

    print(f"\n{DIVIDER}")
    print(f"  MODEL: {name}")
    print(DIVIDER)
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision(weighted): {prec:.4f}")
    print(f"  Recall   (weighted): {rec:.4f}")
    print(f"  F1-score (weighted): {f1w:.4f}")
    print(f"  F1-score (macro)   : {f1m:.4f}")
    print(f"\n  Per-Class Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=short_names,
        zero_division=0,
        digits=3,
    ))

# ── 7. Confusion matrix (text) for KNN & SVM ──────────────────────────────
for name in ['KNN', 'SVM']:
    if name not in models:
        continue
    wrapper = models[name]
    y_pred  = wrapper.predict(P_test)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{DIVIDER}")
    print(f"  CONFUSION MATRIX — {name}  (rows=True, cols=Predicted)")
    print(DIVIDER)

    # Header row
    col_w = 6
    header = ''.join(f"{s[:col_w]:>{col_w}}" for s in short_names)
    print(f"{'':>20}  {header}")
    print(f"{'':>20}  " + '-'*(col_w * n_classes))

    for i, row_label in enumerate(short_names):
        row_str = ''.join(f"{int(cm[i,j]):>{col_w}}" for j in range(n_classes))
        marker  = ' <--' if cm[i,:].sum() > 0 and cm[i, i] == cm[i,:].max() else ''
        print(f"  {row_label[:18]:>18}  {row_str}{marker}")

    # Per-class recall
    print(f"\n  Per-class Recall ({name}):")
    for i, sn in enumerate(short_names):
        total = cm[i, :].sum()
        correct = cm[i, i]
        recall_val = correct / total if total > 0 else 0.0
        bar = '█' * int(recall_val * 20)
        print(f"    {sn[:28]:>28}  {recall_val:.3f}  {bar}")

# ── 8. Final comparison table ──────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  FINAL COMPARISON TABLE")
print(DIVIDER)

# Column headers
cols = ['Model', 'Accuracy', 'F1(weighted)', 'F1(macro)', 'Precision(weighted)', 'Recall(weighted)']
col_widths = [18, 12, 14, 12, 22, 20]
header_row = ''.join(f"{c:<{w}}" for c, w in zip(cols, col_widths))
print(f"  {header_row}")
print("  " + "-" * sum(col_widths))

for r in sorted(results_summary, key=lambda x: x['F1(weighted)'], reverse=True):
    row = [
        r['Model'],
        f"{r['Accuracy']:.4f}",
        f"{r['F1(weighted)']:.4f}",
        f"{r['F1(macro)']:.4f}",
        f"{r['Precision(weighted)']:.4f}",
        f"{r['Recall(weighted)']:.4f}",
    ]
    row_str = ''.join(f"{v:<{w}}" for v, w in zip(row, col_widths))
    print(f"  {row_str}")

best = max(results_summary, key=lambda x: x['F1(weighted)'])
print(f"\n  Best model  : {best['Model']}")
print(f"  Accuracy    : {best['Accuracy']*100:.2f}%")
print(f"  F1(weighted): {best['F1(weighted)']:.4f}")
print(DIVIDER)
