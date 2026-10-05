"""
dimensionality_reduction.py
---------------------------
PCA-based dimensionality reduction with:
  - Automatic variance-threshold component selection
  - Feature standardisation (zero mean, unit variance) before PCA
  - Explained variance diagnostics and plotting helpers
  - Serialisation (save / load fitted pipeline)
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PCAReducer
# ---------------------------------------------------------------------------

class PCAReducer:
    """
    Wraps sklearn StandardScaler + PCA into a single, reusable pipeline.

    The scaler is always applied first so that PCA components are not
    dominated by features with large absolute magnitudes.

    Parameters
    ----------
    n_components : int | float | str | None
        • int   – keep exactly this many components
        • float – keep enough components to explain this fraction of variance
                  (e.g. 0.95 → 95 % of variance)
        • 'mle' – Minka's MLE automatic selection
        • None  – keep all components (effectively only standardises)
    random_state : int
    """

    def __init__(self, n_components=0.95, random_state: int = 42):
        self.n_components  = n_components
        self.random_state  = random_state

        self.scaler_: StandardScaler | None = None
        self.pca_:    PCA            | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """
        Fit the scaler and PCA on training features.

        Parameters
        ----------
        X : np.ndarray  shape (N, D)

        Returns
        -------
        self
        """
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            svd_solver="full",
        )
        self.pca_.fit(X_scaled)
        self._is_fitted = True

        print(f"\n  PCA Summary")
        print(f"    Original dimensionality : {X.shape[1]}")
        print(f"    Retained components     : {self.pca_.n_components_}")
        print(f"    Explained variance      : "
              f"{self.pca_.explained_variance_ratio_.sum() * 100:.2f} %")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted scaler + PCA to new data.

        Parameters
        ----------
        X : np.ndarray  shape (N, D)

        Returns
        -------
        np.ndarray  shape (N, n_components_)
        """
        self._check_fitted()
        X_scaled = self.scaler_.transform(X)
        return self.pca_.transform(X_scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return the transformed version."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Map reduced features back to the original (scaled) space.
        Useful for reconstruction error analysis.
        """
        self._check_fitted()
        X_scaled_approx = self.pca_.inverse_transform(X_reduced)
        return self.scaler_.inverse_transform(X_scaled_approx)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self.pca_.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> np.ndarray:
        self._check_fitted()
        return np.cumsum(self.pca_.explained_variance_ratio_)

    @property
    def n_components_(self) -> int:
        self._check_fitted()
        return int(self.pca_.n_components_)

    def components_needed_for(self, variance_threshold: float = 0.95) -> int:
        """
        Return the minimum number of components that explain at least
        `variance_threshold` of the total variance.
        """
        self._check_fitted()
        cum = self.cumulative_variance
        indices = np.where(cum >= variance_threshold)[0]
        if len(indices) == 0:
            return len(cum)
        return int(indices[0]) + 1

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error on dataset X."""
        self._check_fitted()
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        return float(np.mean((X - X_reconstructed) ** 2))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_explained_variance(self, save_path: str = None,
                                threshold: float = 0.95) -> plt.Figure:
        """
        Plot the cumulative explained variance curve.

        Parameters
        ----------
        save_path  : if given, the figure is saved here
        threshold  : mark the point where this variance fraction is reached

        Returns
        -------
        matplotlib Figure
        """
        self._check_fitted()

        ratios = self.explained_variance_ratio
        cumulative = self.cumulative_variance
        n = len(ratios)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ---- Left: per-component scree plot ----
        ax = axes[0]
        ax.bar(range(1, n + 1), ratios, color="steelblue", alpha=0.7,
               label="Individual")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Scree Plot")
        ax.legend()

        # ---- Right: cumulative curve ----
        ax = axes[1]
        ax.plot(range(1, n + 1), cumulative, marker="o", markersize=3,
                color="darkorange", linewidth=1.5, label="Cumulative")  
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1,
                   label=f"{threshold*100:.0f}% threshold")

        n_at_thresh = self.components_needed_for(threshold)
        ax.axvline(x=n_at_thresh, color="green", linestyle="--", linewidth=1,
                   label=f"{n_at_thresh} components")

        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Cumulative Explained Variance")
        ax.set_ylim(0, 1.05)
        ax.legend()

        fig.suptitle("PCA Explained Variance Analysis", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved explained variance plot to: {save_path}")

        return fig

    def plot_2d_projection(self, X: np.ndarray, labels: np.ndarray,
                           class_names: list = None,
                           save_path: str = None) -> plt.Figure:
        """
        Project data onto the first two principal components and scatter-plot,
        colour-coded by class label.

        Parameters
        ----------
        X           : raw feature matrix  (N, D)
        labels      : integer class labels (N,)
        class_names : list mapping label int -> display name
        save_path   : optional file path to save the figure
        """
        self._check_fitted()

        # Use the first 2 PCs regardless of how many were retained
        scaler_temp = self.scaler_
        X_scaled = scaler_temp.transform(X)

        pca2 = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca2.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 7))
        unique_labels = np.unique(labels)
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            name = (class_names[lbl] if class_names else f"Class {lbl}")
            # Shorten long PlantVillage names for readability
            name = name.replace("Tomato__", "").replace("_", " ")
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[cmap(i)], label=name,
                       alpha=0.5, s=20, edgecolors="none")

        ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("PCA 2-D Projection of Feature Space")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved 2-D projection plot to: {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted scaler and PCA to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler_, "pca": self.pca_,
                     "n_components": self.n_components}, path)
        print(f"  PCAReducer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "PCAReducer":
        """Load a previously saved PCAReducer from disk."""
        data = joblib.load(path)
        obj = cls(n_components=data["n_components"])
        obj.scaler_ = data["scaler"]
        obj.pca_    = data["pca"]
        obj._is_fitted = True
        print(f"  PCAReducer loaded from: {path}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "PCAReducer is not fitted yet. Call fit() or fit_transform() first."
            )

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return (f"PCAReducer(n_components={self.n_components}, "
                f"status={status})")


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def select_n_components(X_train: np.ndarray,
                        variance_thresholds: list = None) -> dict:
    """
    Fit a full PCA (all components) on X_train and return how many
    components are needed to reach each variance threshold.

    Useful for deciding the n_components argument before training.

    Parameters
    ----------
    X_train             : raw (unscaled) training features
    variance_thresholds : list of floats, e.g. [0.90, 0.95, 0.99]

    Returns
    -------
    dict  { threshold: n_components }
    """
    if variance_thresholds is None:
        variance_thresholds = [0.80, 0.90, 0.95, 0.99]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca_full = PCA(n_components=None, svd_solver="full")
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    result = {}
    for thresh in variance_thresholds:
        indices = np.where(cumvar >= thresh)[0]
        n = int(indices[0]) + 1 if len(indices) > 0 else len(cumvar)
        result[thresh] = n
        print(f"  {thresh*100:.0f}% variance → {n} components "
              f"(from {X_train.shape[1]})")

    return result
