"""
predictor.py
------------
High-level inference interface for the tomato disease detection system.

DiseasePredictor loads all persisted artefacts (feature extractor config,
PCA reducer, and one or more fitted models) and exposes:
  - predict_image()  – classify a single image from a file path or array
  - predict_batch()  – classify multiple images at once
  - predict_top_k()  – return the k most-likely classes with confidence scores
"""

import numpy as np
import cv2
import joblib
from pathlib import Path

from .preprocessing import IMAGE_SIZE, LABEL_TO_CLASS
from .feature_extraction import FeatureExtractor
from .dimensionality_reduction import PCAReducer
from .models import ModelWrapper


# ---------------------------------------------------------------------------
# DiseasePredictor
# ---------------------------------------------------------------------------

class DiseasePredictor:
    """
    End-to-end inference wrapper.

    Parameters
    ----------
    model       : fitted ModelWrapper
    extractor   : fitted FeatureExtractor
    reducer     : fitted PCAReducer  (pass None to skip PCA)
    class_names : list mapping integer label -> display name
    image_size  : (H, W) that images are resized to before feature extraction
    """

    def __init__(
        self,
        model: ModelWrapper,
        extractor: FeatureExtractor,
        reducer: PCAReducer | None,
        class_names: list,
        image_size: tuple = IMAGE_SIZE,
    ):
        self.model       = model
        self.extractor   = extractor
        self.reducer     = reducer
        self.class_names = class_names
        self.image_size  = image_size

    # ------------------------------------------------------------------
    # Core prediction helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, image_input) -> np.ndarray:
        """
        Accept either a file path (str / Path) or a raw numpy array.

        Returns
        -------
        np.ndarray  float32  (H, W, 3)  range [0, 1]
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(
                    f"Cannot read image: {image_input}"
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
            # If passed as uint8, normalise
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
        else:
            raise TypeError(
                f"image_input must be a file path or numpy array, "
                f"got {type(image_input)}"
            )

        # Ensure float32 in [0, 1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        # Resize to expected size
        img = cv2.resize(img,
                         (self.image_size[1], self.image_size[0]),
                         interpolation=cv2.INTER_AREA)
        return img

    def _extract_and_reduce(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Run feature extraction and optional PCA on a single image.

        Returns
        -------
        np.ndarray  shape (1, D_reduced)
        """
        features = self.extractor.extract(image_rgb)      # (D,)
        features = features.reshape(1, -1)                # (1, D)
        if self.reducer is not None:
            features = self.reducer.transform(features)   # (1, D_reduced)
        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, image_input) -> dict:
        """
        Classify a single tomato leaf image.

        Parameters
        ----------
        image_input : str | Path | np.ndarray
            File path to the image or a pre-loaded RGB/BGR array.

        Returns
        -------
        dict with keys:
            predicted_label  : int
            predicted_class  : str   (human-readable name)
            confidence       : float (max predicted probability)
            probabilities    : dict  { class_name: probability }
        """
        img      = self._preprocess_image(image_input)
        features = self._extract_and_reduce(img)

        label = int(self.model.predict(features)[0])
        class_name = self.class_names[label] if label < len(self.class_names) \
                     else str(label)

        # Try to get probabilities
        try:
            proba = self.model.predict_proba(features)[0]  # (n_classes,)
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

    def predict_batch(self, image_inputs: list) -> list:
        """
        Classify a list of images.

        Parameters
        ----------
        image_inputs : list of file paths or numpy arrays

        Returns
        -------
        list of dicts (same format as predict_image)
        """
        results = []
        for img_input in image_inputs:
            results.append(self.predict_image(img_input))
        return results

    def predict_top_k(self, image_input, k: int = 3) -> list:
        """
        Return the top-k most likely disease classes with confidence scores.

        Parameters
        ----------
        image_input : str | Path | np.ndarray
        k           : number of top classes to return

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
            label = self.class_names.index(class_name) \
                    if class_name in self.class_names else -1
            top_k.append({"class": class_name, "label": label,
                          "confidence": conf})
        return top_k

    def is_healthy(self, image_input) -> dict:
        """
        Binary convenience method — returns whether the leaf is healthy.

        Parameters
        ----------
        image_input : str | Path | np.ndarray

        Returns
        -------
        dict:
            is_healthy  : bool
            class       : str (full predicted class name)
            confidence  : float
        """
        result = self.predict_image(image_input)
        healthy = "healthy" in result["predicted_class"].lower()
        return {
            "is_healthy": healthy,
            "class":      result["predicted_class"],
            "confidence": result["confidence"],
        }

    # ------------------------------------------------------------------
    # Pretty-print helper
    # ------------------------------------------------------------------

    def format_prediction(self, result: dict, top_k: list = None) -> str:
        """
        Format a prediction result as a human-readable string.

        Parameters
        ----------
        result : dict from predict_image()
        top_k  : optional list from predict_top_k()

        Returns
        -------
        str
        """
        short_name = (result["predicted_class"]
                      .replace("Tomato__", "")
                      .replace("_", " "))
        lines = [
            "=" * 50,
            f"  Predicted Class : {short_name}",
            f"  Confidence      : {result['confidence']*100:.1f}%",
        ]
        if "healthy" in result["predicted_class"].lower():
            lines.append("  Status          : HEALTHY ✓")
        else:
            lines.append("  Status          : DISEASED ✗")

        if top_k:
            lines.append("\n  Top Predictions:")
            for i, entry in enumerate(top_k, 1):
                sn = entry["class"].replace("Tomato__", "").replace("_", " ")
                lines.append(f"    {i}. {sn:<40} {entry['confidence']*100:.1f}%")

        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_predictor(predictor: DiseasePredictor, path: str) -> None:
    """
    Persist a complete DiseasePredictor (all artefacts) to a single file.

    Parameters
    ----------
    predictor : fitted DiseasePredictor
    path      : destination .pkl path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":       predictor.model,
        "extractor":   predictor.extractor,
        "reducer":     predictor.reducer,
        "class_names": predictor.class_names,
        "image_size":  predictor.image_size,
    }
    joblib.dump(payload, path)
    print(f"  DiseasePredictor saved to: {path}")


def load_predictor(path: str) -> DiseasePredictor:
    """
    Load a previously saved DiseasePredictor from disk.

    Parameters
    ----------
    path : .pkl file created by save_predictor()

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
