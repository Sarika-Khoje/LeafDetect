"""
main.py
-------
End-to-end pipeline for Tomato Plant Disease Detection.

Usage
-----
# Full pipeline (train + evaluate all three models):
    python main.py --data_dir data/raw --results_dir results

# Limit images per class (fast smoke-test):
    python main.py --data_dir data/raw --max_per_class 100

# Use hyperparameter grid search:
    python main.py --data_dir data/raw --grid_search

# Run inference on a single image after training:
    python main.py --data_dir data/raw --predict path/to/leaf.jpg

# Skip training and load previously saved models:
    python main.py --data_dir data/raw --skip_training
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt

# ── project modules ────────────────────────────────────────────────────────
from src.preprocessing       import (load_dataset, split_data,
                                     get_class_distribution, TOMATO_CLASSES,
                                     LABEL_TO_CLASS)
from src.feature_extraction  import FeatureExtractor
from src.dimensionality_reduction import PCAReducer, select_n_components
from src.models              import (build_all_models, train_all_models,
                                     ModelWrapper)
from src.evaluation          import (full_evaluation, compare_models,
                                     plot_model_comparison,
                                     plot_training_times)
from src.predictor           import (DiseasePredictor, save_predictor,
                                     load_predictor)

warnings.filterwarnings("ignore")


# ============================================================================
# Pipeline steps
# ============================================================================

def step_load(args) -> tuple:
    """Load images and split into train / val / test."""
    print("\n" + "="*60)
    print("  STEP 1 — Data Loading & Splitting")
    print("="*60)

    images, labels, found_classes = load_dataset(
        data_dir      = args.data_dir,
        image_size    = (args.image_size, args.image_size),
        max_per_class = args.max_per_class,
    )

    print(f"\n  Total images loaded : {len(images)}")
    print(f"  Classes found       : {len(found_classes)}")
    dist = get_class_distribution(labels, list(TOMATO_CLASSES.keys()))
    print(f"  Class distribution  : {dist}")

    # Build class name list aligned with integer labels
    # (only include classes that were actually loaded)
    label_to_name = LABEL_TO_CLASS.copy()
    class_names   = [label_to_name[i] for i in sorted(label_to_name)
                     if label_to_name[i] in found_classes]
    # Re-map labels to a contiguous 0-based range
    unique_labels  = sorted(np.unique(labels))
    remap          = {old: new for new, old in enumerate(unique_labels)}
    labels_remapped = np.array([remap[l] for l in labels], dtype=np.int32)
    class_names_ordered = [label_to_name[old] for old in unique_labels]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        images, labels_remapped,
        test_size    = args.test_size,
        val_size     = args.val_size,
        random_state = args.seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names_ordered


def step_extract(X_train, X_val, X_test, args) -> tuple:
    """Extract hand-crafted features from all image splits."""
    print("\n" + "="*60)
    print("  STEP 2 — Feature Extraction")
    print("="*60)

    extractor = FeatureExtractor(
        use_color_histogram  = True,
        histogram_bins       = args.hist_bins,
        histogram_spaces     = ("rgb", "hsv", "lab"),
        use_color_statistics = True,
        use_glcm             = True,
        use_lbp              = True,
        use_hu_moments       = True,
    )

    print("\n  Feature groups:")
    for group, size in extractor.feature_group_sizes().items():
        print(f"    {group:<25} {size} dims")

    F_train = extractor.extract_batch(X_train, desc="Train")
    F_val   = extractor.extract_batch(X_val,   desc="Val")
    F_test  = extractor.extract_batch(X_test,  desc="Test")

    print(f"\n  Raw feature vector size : {F_train.shape[1]}")
    return F_train, F_val, F_test, extractor


def step_pca(F_train, F_val, F_test, args, results_dir) -> tuple:
    """Fit PCA on training features and transform all splits."""
    print("\n" + "="*60)
    print("  STEP 3 — PCA Dimensionality Reduction")
    print("="*60)

    if args.pca_components is not None and args.pca_components > 1:
        n_comp = int(args.pca_components)
    else:
        n_comp = args.pca_variance   # float → variance threshold

    reducer = PCAReducer(n_components=n_comp, random_state=args.seed)
    P_train = reducer.fit_transform(F_train)
    P_val   = reducer.transform(F_val)
    P_test  = reducer.transform(F_test)

    # Save explained variance plot
    reducer.plot_explained_variance(
        save_path = str(Path(results_dir) / "pca_explained_variance.png"),
        threshold = args.pca_variance,
    )
    plt.close("all")

    reducer.save(str(Path(args.models_dir) / "pca_reducer.pkl"))
    return P_train, P_val, P_test, reducer


def step_train(P_train, y_train, args) -> dict:
    """Train (or grid-search) KNN, SVM, and Random Forest."""
    print("\n" + "="*60)
    print("  STEP 4 — Model Training")
    print("="*60)

    models = build_all_models()
    models = train_all_models(
        models,
        P_train,
        y_train,
        use_grid_search = args.grid_search,
    )

    # Persist each model
    for key, wrapper in models.items():
        wrapper.save(args.models_dir)

    return models


def step_evaluate(models, P_val, P_test,
                  y_val, y_test,
                  class_names, results_dir) -> None:
    """Evaluate all models on validation and test sets."""
    print("\n" + "="*60)
    print("  STEP 5 — Evaluation")
    print("="*60)

    all_results = []

    for key, wrapper in models.items():
        print(f"\n  ── {wrapper.name} ──")
        # Validate on val set first
        print("  [Validation Set]")
        full_evaluation(
            wrapper, P_val, y_val,
            class_names = class_names,
            results_dir = str(Path(results_dir) / "val"),
        )
        # Final evaluation on held-out test set
        print("  [Test Set]")
        result = full_evaluation(
            wrapper, P_test, y_test,
            class_names = class_names,
            results_dir = str(Path(results_dir) / "test"),
        )
        all_results.append(result)
        plt.close("all")

    # Summary comparison
    df = compare_models(
        all_results,
        save_path = str(Path(results_dir) / "model_comparison.csv"),
    )
    plot_model_comparison(
        df,
        save_path = str(Path(results_dir) / "model_comparison.png"),
    )
    plot_training_times(
        models,
        save_path = str(Path(results_dir) / "training_times.png"),
    )
    plt.close("all")

    # Declare best model
    best_name = df["f1_weighted"].idxmax()
    print(f"\n  Best model by F1-weighted: {best_name} "
          f"({df.loc[best_name, 'f1_weighted']:.4f})")
    return best_name


def step_build_predictor(models, extractor, reducer,
                         class_names, best_name, args) -> DiseasePredictor:
    """Wrap the best model in a DiseasePredictor and persist it."""
    print("\n" + "="*60)
    print("  STEP 6 — Building Disease Predictor")
    print("="*60)

    best_wrapper = models[best_name]
    predictor = DiseasePredictor(
        model       = best_wrapper,
        extractor   = extractor,
        reducer     = reducer,
        class_names = class_names,
        image_size  = (args.image_size, args.image_size),
    )
    save_predictor(
        predictor,
        path = str(Path(args.models_dir) / "best_predictor.pkl"),
    )
    return predictor


def step_predict(predictor: DiseasePredictor, image_path: str) -> None:
    """Run inference on a single image and print results."""
    print("\n" + "="*60)
    print("  STEP 7 — Single Image Prediction")
    print("="*60)
    print(f"  Image : {image_path}")

    result = predictor.predict_image(image_path)
    top_k  = predictor.predict_top_k(image_path, k=3)
    print(predictor.format_prediction(result, top_k))


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Tomato Disease Detection — end-to-end pipeline"
    )

    # Data
    p.add_argument("--data_dir",      default="data/raw",
                   help="Root directory of PlantVillage tomato images")
    p.add_argument("--results_dir",   default="results",
                   help="Directory to save plots and CSV reports")
    p.add_argument("--models_dir",    default="models",
                   help="Directory to save fitted model files")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap images per class (None = all)")
    p.add_argument("--image_size",    type=int, default=128,
                   help="Resize images to (image_size × image_size)")
    p.add_argument("--test_size",     type=float, default=0.20)
    p.add_argument("--val_size",      type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)

    # Features
    p.add_argument("--hist_bins",     type=int, default=32,
                   help="Bins per channel for color histograms")

    # PCA
    p.add_argument("--pca_variance",  type=float, default=0.95,
                   help="Variance threshold for PCA component selection")
    p.add_argument("--pca_components", type=float, default=None,
                   help="Exact number of PCA components (overrides pca_variance)")

    # Training
    p.add_argument("--grid_search",   action="store_true",
                   help="Run hyperparameter grid search (slow but thorough)")
    p.add_argument("--skip_training", action="store_true",
                   help="Skip training; load existing models from models_dir")

    # Inference
    p.add_argument("--predict",       default=None,
                   help="Path to a single image for inference after training")

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Ensure output directories exist
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "█"*60)
    print("  Tomato Plant Disease Detection System")
    print("█"*60)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Models dir  : {args.models_dir}")
    print(f"  Grid search : {args.grid_search}")
    print(f"  PCA target  : {args.pca_variance*100:.0f}% variance")

    # ── 1. Load data ──────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_names) = step_load(args)

    # ── 2. Feature extraction ─────────────────────────────────────────
    F_train, F_val, F_test, extractor = step_extract(
        X_train, X_val, X_test, args
    )

    # ── 3. PCA ────────────────────────────────────────────────────────
    P_train, P_val, P_test, reducer = step_pca(
        F_train, F_val, F_test, args, args.results_dir
    )

    # ── 2D projection visualisation ───────────────────────────────────
    try:
        reducer.plot_2d_projection(
            F_train, y_train,
            class_names = class_names,
            save_path   = str(Path(args.results_dir) / "pca_2d_projection.png"),
        )
        plt.close("all")
    except Exception as e:
        print(f"  [WARN] 2D projection plot failed: {e}")

    # ── 4. Train / load models ────────────────────────────────────────
    if args.skip_training:
        print("\n  Loading pre-trained models from disk ...")
        model_files = {
            "KNN": "knn.pkl",
            "SVM": "svm.pkl",
            "RF":  "random_forest.pkl",
        }
        models = {}
        for key, fname in model_files.items():
            path = str(Path(args.models_dir) / fname)
            try:
                models[key] = ModelWrapper.load(path)
            except FileNotFoundError:
                print(f"  [WARN] Model file not found, skipping: {path}")
        if not models:
            print("  No models found. Re-running training ...")
            models = step_train(P_train, y_train, args)
    else:
        models = step_train(P_train, y_train, args)

    # ── 5. Evaluate ───────────────────────────────────────────────────
    best_name = step_evaluate(
        models, P_val, P_test, y_val, y_test,
        class_names, args.results_dir
    )

    # ── 6. Build predictor ────────────────────────────────────────────
    predictor = step_build_predictor(
        models, extractor, reducer, class_names, best_name, args
    )

    # ── 7. Optional single-image prediction ───────────────────────────
    if args.predict:
        step_predict(predictor, args.predict)

    print("\n" + "█"*60)
    print("  Pipeline complete.")
    print(f"  Results saved to : {args.results_dir}/")
    print(f"  Models saved to  : {args.models_dir}/")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
