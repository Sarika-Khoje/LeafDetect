"""
prediction.py
-------------
Standalone ML prediction module for Tomato Plant Disease Detection.

Loads the saved best_predictor.pkl artefact and exposes:
  - load_predictor(path)         → DiseasePredictor
  - DiseasePredictor.predict_image()  – classify a single image
  - DiseasePredictor.predict_top_k()  – top-k classes with confidence
  - DiseasePredictor.predict_batch()  – classify a list of images

All src.* dependencies are inlined so this file works without the src/ package.
"""

from __future__ import annotations

import numpy as np
import cv2
import joblib
from pathlib import Path

# ── scikit-image / sklearn used by the feature extractor ──────────────────
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ============================================================================
# Constants (from src/preprocessing.py)
# ============================================================================

IMAGE_SIZE = (128, 128)

TOMATO_CLASSES = {
    "Tomato___healthy":                              0,
    "Tomato___Tomato_mosaic_virus":                  1,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":         2,
    "Tomato___Target_Spot":                          3,
    "Tomato___Spider_mites Two-spotted_spider_mite": 4,
    "Tomato___Septoria_leaf_spot":                   5,
    "Tomato___Leaf_Mold":                            6,
    "Tomato___Late_blight":                          7,
    "Tomato___Early_blight":                         8,
    "Tomato___Bacterial_spot":                       9,
}

LABEL_TO_CLASS = {v: k for k, v in TOMATO_CLASSES.items()}


# ============================================================================
# Color-space helpers (from src/preprocessing.py)
# ============================================================================

def _to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert normalised RGB float32 image to HSV (all channels in [0,1])."""
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return hsv


def _to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert normalised RGB float32 image to CIE L*a*b* (all in [0,1])."""
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab /= 255.0
    return lab


def _to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Convert normalised RGB float32 image to grayscale float32."""
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


# ============================================================================
# Feature extraction helpers (from src/feature_extraction.py)
# ============================================================================

def _extract_color_histogram(
    image_rgb: np.ndarray,
    bins: int = 32,
    color_spaces: tuple = ("rgb", "hsv", "lab"),
) -> np.ndarray:
    """Flattened, L1-normalised color histograms across one or more color spaces."""
    space_map = {
        "rgb": image_rgb,
        "hsv": _to_hsv(image_rgb),
        "lab": _to_lab(image_rgb),
    }
    features = []
    for space in color_spaces:
        img = space_map[space]
        for ch in range(img.shape[2]):
            channel = img[:, :, ch].ravel()
            hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            features.append(hist)
    return np.concatenate(features)


def _extract_color_statistics(image_rgb: np.ndarray) -> np.ndarray:
    """Per-channel mean and std in RGB, HSV, and LAB — shape (18,)."""
    features = []
    for img in (image_rgb, _to_hsv(image_rgb), _to_lab(image_rgb)):
        for ch in range(img.shape[2]):
            channel = img[:, :, ch]
            features.append(float(channel.mean()))
            features.append(float(channel.std()))
    return np.array(features, dtype=np.float32)


def _extract_glcm_features(
    image_rgb: np.ndarray,
    distances: list = None,
    angles: list = None,
) -> np.ndarray:
    """GLCM texture descriptors: contrast, dissimilarity, homogeneity, energy, correlation, ASM."""
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    gray = _to_grayscale(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)

    glcm = graycomatrix(
        gray_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    properties = ["contrast", "dissimilarity", "homogeneity",
                  "energy", "correlation", "ASM"]
    features = []
    for prop in properties:
        features.extend(graycoprops(glcm, prop).ravel().tolist())
    return np.array(features, dtype=np.float32)


def _extract_lbp_features(
    image_rgb: np.ndarray,
    n_points: int = 24,
    radius: int = 3,
) -> np.ndarray:
    """Normalised LBP histogram for texture description."""
    gray = _to_grayscale(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def _extract_hu_moments(image_rgb: np.ndarray) -> np.ndarray:
    """7 log-transformed Hu invariant moments — shape (7,)."""
    gray = _to_grayscale(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)
    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).ravel()
    hu_log = np.sign(hu) * np.log1p(np.abs(hu))
    return hu_log.astype(np.float32)


def _extract_features(image_rgb: np.ndarray, extractor) -> np.ndarray:
    """
    Dispatch to the feature extractor object stored inside the predictor artefact.
    This handles both FeatureExtractor instances from src/ and raw callables.
    """
    return extractor.extract(image_rgb)


# ============================================================================
# DiseasePredictor
# ============================================================================

class DiseasePredictor:
    """
    End-to-end inference wrapper loaded from a saved artefact.

    Parameters
    ----------
    model       : fitted ModelWrapper
    extractor   : fitted FeatureExtractor
    reducer     : fitted PCAReducer (or None to skip PCA)
    class_names : list mapping integer label -> class name
    image_size  : (H, W) resize target before feature extraction
    """

    def __init__(
        self,
        model,
        extractor,
        reducer,
        class_names: list,
        image_size: tuple = IMAGE_SIZE,
    ):
        self.model       = model
        self.extractor   = extractor
        self.reducer     = reducer
        self.class_names = class_names
        self.image_size  = image_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image_input) -> np.ndarray:
        """
        Accept a file path (str/Path) or a numpy array.
        Returns float32 RGB image normalised to [0, 1].
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError(
                f"image_input must be a file path or numpy array, got {type(image_input)}"
            )

        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        img = cv2.resize(
            img,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        return img

    def _extract_and_reduce(self, image_rgb: np.ndarray) -> np.ndarray:
        """Feature extraction + optional PCA. Returns shape (1, D)."""
        features = self.extractor.extract(image_rgb).reshape(1, -1)
        if self.reducer is not None:
            features = self.reducer.transform(features)
        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, image_input) -> dict:
        """
        Classify a single tomato leaf image.

        Returns
        -------
        dict
            predicted_label  : int
            predicted_class  : str
            confidence       : float  (max probability, 0–1)
            probabilities    : dict   { class_name: probability }
        """
        img      = self._preprocess(image_input)
        features = self._extract_and_reduce(img)

        label = int(self.model.predict(features)[0])
        class_name = (
            self.class_names[label]
            if label < len(self.class_names)
            else str(label)
        )

        try:
            proba      = self.model.predict_proba(features)[0]
            confidence = float(proba.max())
            prob_dict  = {
                self.class_names[i]: float(proba[i])
                for i in range(len(proba))
                if i < len(self.class_names)
            }
        except (AttributeError, Exception):
            confidence = 1.0
            prob_dict  = {class_name: 1.0}

        return {
            "predicted_label": label,
            "predicted_class": class_name,
            "confidence":      confidence,
            "probabilities":   prob_dict,
        }

    def predict_top_k(self, image_input, k: int = 3) -> list:
        """
        Return the top-k most likely disease classes with confidence scores.

        Returns
        -------
        list of dicts sorted by descending confidence:
            [{"class": str, "label": int, "confidence": float}, ...]
        """
        result = self.predict_image(image_input)
        probs  = result["probabilities"]

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_k = []
        for class_name, conf in sorted_probs[:k]:
            label = (
                self.class_names.index(class_name)
                if class_name in self.class_names
                else -1
            )
            top_k.append({"class": class_name, "label": label, "confidence": conf})
        return top_k

    def predict_batch(self, image_inputs: list) -> list:
        """Classify a list of images. Returns list of predict_image() dicts."""
        return [self.predict_image(img) for img in image_inputs]

    def is_healthy(self, image_input) -> dict:
        """
        Binary convenience method.

        Returns
        -------
        dict  { is_healthy: bool, class: str, confidence: float }
        """
        result = self.predict_image(image_input)
        return {
            "is_healthy": "healthy" in result["predicted_class"].lower(),
            "class":      result["predicted_class"],
            "confidence": result["confidence"],
        }

    def format_prediction(self, result: dict, top_k: list = None) -> str:
        """Human-readable summary string for a prediction result."""
        short = (
            result["predicted_class"]
            .replace("Tomato__", "")
            .replace("_", " ")
        )
        lines = [
            "=" * 50,
            f"  Predicted Class : {short}",
            f"  Confidence      : {result['confidence']*100:.1f}%",
            "  Status          : "
            + ("HEALTHY ✓" if "healthy" in result["predicted_class"].lower()
               else "DISEASED ✗"),
        ]
        if top_k:
            lines.append("\n  Top Predictions:")
            for i, entry in enumerate(top_k, 1):
                sn = entry["class"].replace("Tomato__", "").replace("_", " ")
                lines.append(f"    {i}. {sn:<40} {entry['confidence']*100:.1f}%")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================================
# Serialisation helpers
# ============================================================================

def save_predictor(predictor: DiseasePredictor, path: str) -> None:
    """Persist a complete DiseasePredictor to a single .pkl file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":       predictor.model,
            "extractor":   predictor.extractor,
            "reducer":     predictor.reducer,
            "class_names": predictor.class_names,
            "image_size":  predictor.image_size,
        },
        path,
    )
    print(f"  DiseasePredictor saved to: {path}")


def load_predictor(path: str) -> DiseasePredictor:
    """
    Load a previously saved DiseasePredictor from disk.

    Parameters
    ----------
    path : .pkl file created by save_predictor() or main.py

    Returns
    -------
    DiseasePredictor
    """
    payload = joblib.load(path)
    predictor = DiseasePredictor(
        model       = payload["model"],
        extractor   = payload["extractor"],
        reducer     = payload["reducer"],
        class_names = payload["class_names"],
        image_size  = payload.get("image_size", IMAGE_SIZE),
    )
    print(f"  DiseasePredictor loaded from: {path}")
    return predictor
