"""
evaluation.py
-------------
Evaluation utilities for the tomato disease detection system:
  - Per-model metrics: accuracy, precision, recall, F1-score (macro + weighted)
  - Confusion matrix computation and plotting
  - Side-by-side model comparison table
  - ROC / AUC curves (one-vs-rest for multi-class)
  - Results persistence (CSV)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   model_name: str = "Model",
                   class_names: list = None) -> dict:
    """
    Compute classification metrics for a single model's predictions.

    Parameters
    ----------
    y_true      : ground-truth integer labels
    y_pred      : predicted integer labels
    model_name  : label used in printout
    class_names : list of human-readable class names (optional)

    Returns
    -------
    dict with keys:
        model, accuracy,
        precision_macro, recall_macro, f1_macro,
        precision_weighted, recall_weighted, f1_weighted,
        report (full sklearn classification_report string)
    """
    accuracy           = accuracy_score(y_true, y_pred)
    precision_macro    = precision_score(y_true, y_pred, average="macro",
                                         zero_division=0)
    recall_macro       = recall_score(y_true, y_pred, average="macro",
                                      zero_division=0)
    f1_macro           = f1_score(y_true, y_pred, average="macro",
                                  zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted",
                                         zero_division=0)
    recall_weighted    = recall_score(y_true, y_pred, average="weighted",
                                      zero_division=0)
    f1_weighted        = f1_score(y_true, y_pred, average="weighted",
                                  zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy           : {accuracy:.4f}")
    print(f"  Precision (macro)  : {precision_macro:.4f}")
    print(f"  Recall    (macro)  : {recall_macro:.4f}")
    print(f"  F1-score  (macro)  : {f1_macro:.4f}")
    print(f"  Precision (weighted): {precision_weighted:.4f}")
    print(f"  Recall    (weighted): {recall_weighted:.4f}")
    print(f"  F1-score  (weighted): {f1_weighted:.4f}")
    print(f"\n  Classification Report:\n")
    print(report)

    return {
        "model":               model_name,
        "accuracy":            accuracy,
        "precision_macro":     precision_macro,
        "recall_macro":        recall_macro,
        "f1_macro":            f1_macro,
        "precision_weighted":  precision_weighted,
        "recall_weighted":     recall_weighted,
        "f1_weighted":         f1_weighted,
        "report":              report,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list = None,
                          model_name: str = "Model",
                          normalize: bool = True,
                          save_path: str = None) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    class_names : list of display names; defaults to integer strings
    model_name  : figure title prefix
    normalize   : if True, show row-normalised (recall per class) values
    save_path   : optional path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names or [str(i) for i in range(cm.shape[0])]

    # Shorten long PlantVillage names for readability
    short_labels = [l.replace("Tomato__", "").replace("_", "\n")
                    for l in labels]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt = ".2f"
        title_suffix = "(Normalised)"
        vmin, vmax = 0.0, 1.0
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = "(Counts)"
        vmin, vmax = None, None

    n_classes = len(labels)
    fig_size  = max(8, n_classes * 1.2)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        linecolor="grey",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_title(f"{model_name} — Confusion Matrix {title_suffix}",
                 fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# ROC curves (multi-class one-vs-rest)
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   class_names: list = None,
                   model_name: str = "Model",
                   save_path: str = None) -> plt.Figure:
    """
    Plot one-vs-rest ROC curves for each class, plus the macro-average.

    Parameters
    ----------
    y_true      : integer labels  (N,)
    y_proba     : predicted probabilities  (N, n_classes)
    class_names : optional label names
    model_name  : figure title prefix
    save_path   : optional save path

    Returns
    -------
    matplotlib Figure
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    labels = class_names or [str(i) for i in classes]

    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes)

    fpr_all, tpr_all = [], []
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        short_name = labels[i].replace("Tomato__", "").replace("_", " ")
        ax.plot(fpr, tpr, color=cmap(i), alpha=0.7,
                label=f"{short_name} (AUC={roc_auc:.2f})")

    # Macro average
    all_fpr = np.unique(np.concatenate(fpr_all))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_all, tpr_all):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", linewidth=2,
            label=f"Macro Average (AUC={macro_auc:.2f})")

    ax.plot([0, 1], [0, 1], "grey", linestyle=":", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{model_name} — ROC Curves (One-vs-Rest)", fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved ROC curves to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(results: list, save_path: str = None) -> pd.DataFrame:
    """
    Build a comparison table from a list of evaluate_model() result dicts.

    Parameters
    ----------
    results   : list of dicts returned by evaluate_model()
    save_path : optional CSV path to save the table

    Returns
    -------
    pd.DataFrame
    """
    cols = ["model", "accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted"]
    rows = []
    for r in results:
        rows.append({c: r.get(c) for c in cols})

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("f1_weighted", ascending=False)

    print(f"\n{'='*60}")
    print("  Model Comparison Summary")
    print(f"{'='*60}")
    print(df.to_string(float_format="{:.4f}".format))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print(f"\n  Saved comparison table to: {save_path}")

    return df


def plot_model_comparison(df: pd.DataFrame,
                          save_path: str = None) -> plt.Figure:
    """
    Bar-chart comparison of all three models across key metrics.

    Parameters
    ----------
    df        : DataFrame from compare_models()
    save_path : optional path to save the figure

    Returns
    -------
    matplotlib Figure
    """
    metrics = ["accuracy", "f1_macro", "f1_weighted",
               "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1 (Macro)", "F1 (Weighted)",
                     "Precision (Macro)", "Recall (Macro)"]

    n_metrics = len(metrics)
    n_models  = len(df)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, (model_name, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width=width * 0.9,
                      label=model_name, color=colors[i % len(colors)],
                      alpha=0.85, edgecolor="white")
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Classification Metrics", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved model comparison plot to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Training time comparison
# ---------------------------------------------------------------------------

def plot_training_times(models: dict, save_path: str = None) -> plt.Figure:
    """
    Horizontal bar chart of training times for each model.

    Parameters
    ----------
    models    : dict { name: ModelWrapper }  (must be fitted)
    save_path : optional path

    Returns
    -------
    matplotlib Figure
    """
    names  = list(models.keys())
    times  = [m.train_time_ or 0.0 for m in models.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    bars = ax.barh(names, times, color=colors[:len(names)], alpha=0.85,
                   edgecolor="white")

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=10)

    ax.set_xlabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Model Training Times", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training times plot to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Full evaluation pipeline for one model
# ---------------------------------------------------------------------------

def full_evaluation(model_wrapper,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    class_names: list = None,
                    results_dir: str = "results") -> dict:
    """
    Run a complete evaluation for a single ModelWrapper:
      1. Predict labels
      2. Compute all metrics
      3. Plot confusion matrix
      4. Plot ROC curves (if probabilities available)

    Parameters
    ----------
    model_wrapper : fitted ModelWrapper
    X_test        : test features
    y_test        : test labels
    class_names   : optional display names for classes
    results_dir   : directory to save all plots

    Returns
    -------
    dict from evaluate_model()
    """
    name   = model_wrapper.name
    y_pred = model_wrapper.predict(X_test)
    result = evaluate_model(y_test, y_pred, model_name=name,
                            class_names=class_names)

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        model_name=name,
        normalize=True,
        save_path=f"{results_dir}/{name.lower().replace(' ', '_')}_confusion.png",
    )

    # ROC curves
    try:
        y_proba = model_wrapper.predict_proba(X_test)
        plot_roc_curves(
            y_test, y_proba,
            class_names=class_names,
            model_name=name,
            save_path=f"{results_dir}/{name.lower().replace(' ', '_')}_roc.png",
        )
    except AttributeError:
        print(f"  [INFO] Skipping ROC plot for {name} "
              f"(no probability support).")

    return result
