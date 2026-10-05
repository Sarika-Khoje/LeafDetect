"""
models.py
---------
Defines, trains, tunes, and serialises the three classifiers:
  - K-Nearest Neighbours  (KNN)
  - Support Vector Machine (SVM, RBF kernel)
  - Random Forest          (RF)

Each classifier is wrapped in a ModelWrapper that provides a consistent
interface for training, prediction, hyperparameter search, and I/O.
"""

import numpy as np
import joblib
import time
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Default hyperparameter grids
# ---------------------------------------------------------------------------

KNN_PARAM_GRID = {
    "n_neighbors":  [3, 5, 7, 11, 15],
    "weights":      ["uniform", "distance"],
    "metric":       ["euclidean", "manhattan"],
}

SVM_PARAM_GRID = {
    "C":      [0.1, 1.0, 10.0, 100.0],
    "gamma":  ["scale", "auto"],
    "kernel": ["rbf", "poly"],
}

RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2"],
}


# ---------------------------------------------------------------------------
# ModelWrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    Thin wrapper around an sklearn estimator with training, prediction,
    grid-search, and persistence helpers.

    Parameters
    ----------
    name       : display name for logging / saving
    estimator  : unfitted sklearn-compatible classifier
    """

    def __init__(self, name: str, estimator):
        self.name       = name
        self.estimator  = estimator
        self._is_fitted = False
        self.train_time_: float | None = None
        self.best_params_: dict | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ModelWrapper":
        """
        Fit the estimator on the training set.

        Returns
        -------
        self
        """
        print(f"\n  Training {self.name} ...")
        t0 = time.perf_counter()
        self.estimator.fit(X_train, y_train)
        self.train_time_ = time.perf_counter() - t0
        self._is_fitted = True
        print(f"    Done in {self.train_time_:.2f}s")
        return self

    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                    param_grid: dict, cv: int = 5,
                    scoring: str = "f1_weighted",
                    n_jobs: int = -1) -> "ModelWrapper":
        """
        Run a stratified K-fold grid search and refit on the full training set
        using the best hyperparameters found.

        Parameters
        ----------
        X_train    : training features
        y_train    : training labels
        param_grid : dict of parameter grid
        cv         : number of folds
        scoring    : sklearn scoring metric
        n_jobs     : parallel jobs (-1 = all cores)

        Returns
        -------
        self  (with estimator replaced by the best estimator)
        """
        print(f"\n  Grid search for {self.name} "
              f"(cv={cv}, scoring={scoring}) ...")

        t0 = time.perf_counter()

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                      random_state=42)
        gs = GridSearchCV(
            self.estimator,
            param_grid,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        self.estimator    = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.train_time_  = elapsed
        self._is_fitted   = True

        print(f"    Best params : {self.best_params_}")
        print(f"    Best CV score ({scoring}): {gs.best_score_:.4f}")
        print(f"    Completed in {elapsed:.2f}s")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for X."""
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.
        Falls back to decision_function if predict_proba is unavailable.
        """
        self._check_fitted()
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        if hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X)
            # Softmax normalisation over decision scores
            e = np.exp(df - df.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        raise AttributeError(
            f"{self.name} does not support probability estimates."
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str, filename: str = None) -> str:
        """
        Persist the fitted estimator to disk with joblib.

        Parameters
        ----------
        directory : destination folder
        filename  : optional file name (defaults to <name>.pkl)

        Returns
        -------
        str  – path to the saved file
        """
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fname = filename or f"{self.name.lower().replace(' ', '_')}.pkl"
        path  = str(directory / fname)
        joblib.dump({
            "estimator":   self.estimator,
            "name":        self.name,
            "best_params": self.best_params_,
            "train_time":  self.train_time_,
        }, path)
        print(f"  Saved {self.name} to: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ModelWrapper":
        """Load a previously saved ModelWrapper from disk."""
        data = joblib.load(path)
        obj = cls(name=data["name"], estimator=data["estimator"])
        obj.best_params_ = data.get("best_params")
        obj.train_time_  = data.get("train_time")
        obj._is_fitted   = True
        print(f"  Loaded {obj.name} from: {path}")
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} is not fitted. Call fit() or grid_search() first."
            )

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return f"ModelWrapper(name='{self.name}', status={status})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_knn(n_neighbors: int = 5,
              weights: str = "distance",
              metric: str = "euclidean") -> ModelWrapper:
    """
    Create a KNN ModelWrapper with sensible defaults.

    Parameters
    ----------
    n_neighbors : number of nearest neighbours
    weights     : 'uniform' or 'distance'
    metric      : distance metric

    Returns
    -------
    ModelWrapper
    """
    estimator = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=-1,
    )
    return ModelWrapper(name="KNN", estimator=estimator)


def build_svm(C: float = 10.0,
              kernel: str = "rbf",
              gamma: str = "scale",
              probability: bool = True) -> ModelWrapper:
    """
    Create an SVM ModelWrapper.

    Parameters
    ----------
    C           : regularisation parameter
    kernel      : kernel type ('rbf', 'poly', 'linear')
    gamma       : kernel coefficient
    probability : enable probability estimates (adds slight overhead)

    Returns
    -------
    ModelWrapper
    """
    estimator = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=42,
        class_weight="balanced",
    )
    return ModelWrapper(name="SVM", estimator=estimator)


def build_random_forest(n_estimators: int = 200,
                        max_depth: int = None,
                        min_samples_split: int = 2,
                        max_features: str = "sqrt",
                        random_state: int = 42) -> ModelWrapper:
    """
    Create a Random Forest ModelWrapper.

    Parameters
    ----------
    n_estimators       : number of trees
    max_depth          : maximum depth of each tree (None = unlimited)
    min_samples_split  : minimum samples required to split a node
    max_features       : number of features for each split ('sqrt', 'log2')
    random_state       : reproducibility seed

    Returns
    -------
    ModelWrapper
    """
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return ModelWrapper(name="Random Forest", estimator=estimator)


# ---------------------------------------------------------------------------
# Build-all helper
# ---------------------------------------------------------------------------

def build_all_models() -> dict:
    """
    Instantiate all three classifiers with default hyperparameters.

    Returns
    -------
    dict  { 'KNN': ModelWrapper, 'SVM': ModelWrapper, 'RF': ModelWrapper }
    """
    return {
        "KNN": build_knn(),
        "SVM": build_svm(),
        "RF":  build_random_forest(),
    }


def train_all_models(models: dict,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     use_grid_search: bool = False,
                     param_grids: dict = None) -> dict:
    """
    Train (or grid-search) all models and return the fitted wrappers.

    Parameters
    ----------
    models          : dict of ModelWrapper instances (from build_all_models)
    X_train         : training features  (N, D)
    y_train         : training labels    (N,)
    use_grid_search : if True, run hyperparameter search for each model
    param_grids     : dict { model_key: param_grid } overrides defaults

    Returns
    -------
    dict  same structure as input, but fitted
    """
    default_grids = {
        "KNN": KNN_PARAM_GRID,
        "SVM": SVM_PARAM_GRID,
        "RF":  RF_PARAM_GRID,
    }
    if param_grids:
        default_grids.update(param_grids)

    for key, wrapper in models.items():
        if use_grid_search:
            wrapper.grid_search(X_train, y_train,
                                param_grid=default_grids.get(key, {}))
        else:
            wrapper.fit(X_train, y_train)

    return models
