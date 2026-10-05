"""
feature_extraction.py
---------------------
Extracts hand-crafted color and texture features from preprocessed images.

Color features:
  - RGB color histogram  (per-channel, configurable bins)
  - HSV color histogram  (per-channel, configurable bins)
  - LAB color histogram  (per-channel, configurable bins)
  - Color statistics     (mean + std per channel in RGB, HSV, LAB)

Texture features:
  - GLCM (Gray-Level Co-occurrence Matrix) statistics:
      contrast, dissimilarity, homogeneity, energy, correlation, ASM
  - LBP  (Local Binary Pattern) histogram
  - Hu Moments (shape descriptors)

All feature groups can be enabled/disabled individually via the
FeatureExtractor configuration.
"""

import numpy as np
import cv2
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .preprocessing import to_hsv, to_lab, to_grayscale


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def extract_color_histogram(image_rgb: np.ndarray,
                            bins: int = 32,
                            color_spaces: tuple = ("rgb", "hsv", "lab")) -> np.ndarray:
    """
    Compute flattened color histograms across one or more color spaces.

    Parameters
    ----------
    image_rgb   : float32 RGB image, shape (H, W, 3), range [0,1]
    bins        : number of histogram bins per channel
    color_spaces: which spaces to include

    Returns
    -------
    np.ndarray  1-D float32 feature vector
    """
    features = []

    space_map = {
        "rgb": image_rgb,
        "hsv": to_hsv(image_rgb),
        "lab": to_lab(image_rgb),
    }

    for space in color_spaces:
        img = space_map[space]
        for ch in range(img.shape[2]):
            channel = img[:, :, ch].ravel()
            hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
            # L1 normalise so absolute image count doesn't matter
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            features.append(hist)

    return np.concatenate(features)


def extract_color_statistics(image_rgb: np.ndarray) -> np.ndarray:
    """
    Compute per-channel mean and standard deviation in RGB, HSV, and LAB.

    Returns
    -------
    np.ndarray  shape (18,)  — 3 spaces × 3 channels × 2 stats
    """
    features = []
    for img in (image_rgb, to_hsv(image_rgb), to_lab(image_rgb)):
        for ch in range(img.shape[2]):
            channel = img[:, :, ch]
            features.append(float(channel.mean()))
            features.append(float(channel.std()))
    return np.array(features, dtype=np.float32)


def extract_glcm_features(image_rgb: np.ndarray,
                          distances: list = None,
                          angles: list = None) -> np.ndarray:
    """
    Compute GLCM texture descriptors on the grayscale version of the image.

    Properties extracted: contrast, dissimilarity, homogeneity,
                          energy, correlation, ASM

    Parameters
    ----------
    image_rgb : float32 RGB image, shape (H, W, 3), range [0,1]
    distances : pixel-pair distances  (default [1, 3])
    angles    : angles in radians     (default [0, π/4, π/2, 3π/4])

    Returns
    -------
    np.ndarray  1-D float32  length = 6 properties × len(distances) × len(angles)
    """
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    gray = to_grayscale(image_rgb)
    # Quantise to 256 levels for GLCM
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
        values = graycoprops(glcm, prop).ravel()   # shape: (len(dist), len(angles))
        features.extend(values.tolist())

    return np.array(features, dtype=np.float32)


def extract_lbp_features(image_rgb: np.ndarray,
                         n_points: int = 24,
                         radius: int = 3,
                         bins: int = 64) -> np.ndarray:
    """
    Compute a normalised LBP histogram for texture description.

    Parameters
    ----------
    image_rgb : float32 RGB image, shape (H, W, 3), range [0,1]
    n_points  : number of circularly symmetric neighbour set points
    radius    : radius of circle
    bins      : histogram bins

    Returns
    -------
    np.ndarray  shape (bins,)  float32
    """
    gray = to_grayscale(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)

    lbp = local_binary_pattern(gray_uint8, n_points, radius, method="uniform")
    # uniform LBP has n_points + 2 distinct patterns
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_hu_moments(image_rgb: np.ndarray) -> np.ndarray:
    """
    Compute the 7 Hu invariant moments from the grayscale image.

    Log-transformed to reduce magnitude differences.

    Returns
    -------
    np.ndarray  shape (7,)  float32
    """
    gray = to_grayscale(image_rgb)
    gray_uint8 = (gray * 255).astype(np.uint8)

    moments = cv2.moments(gray_uint8)
    hu = cv2.HuMoments(moments).ravel()

    # Log transform with sign preservation
    hu_log = np.sign(hu) * np.log1p(np.abs(hu))
    return hu_log.astype(np.float32)


# ---------------------------------------------------------------------------
# FeatureExtractor  –  configurable pipeline object
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Configurable feature extraction pipeline.

    Usage
    -----
    >>> extractor = FeatureExtractor(histogram_bins=32, use_lbp=True)
    >>> feature_vector = extractor.extract(image_rgb)
    >>> X = extractor.extract_batch(images)
    """

    def __init__(
        self,
        # Color histogram settings
        use_color_histogram: bool = True,
        histogram_bins: int = 32,
        histogram_spaces: tuple = ("rgb", "hsv", "lab"),
        # Color statistics
        use_color_statistics: bool = True,
        # GLCM settings
        use_glcm: bool = True,
        glcm_distances: list = None,
        glcm_angles: list = None,
        # LBP settings
        use_lbp: bool = True,
        lbp_n_points: int = 24,
        lbp_radius: int = 3,
        # Hu moments
        use_hu_moments: bool = True,
    ):
        self.use_color_histogram  = use_color_histogram
        self.histogram_bins       = histogram_bins
        self.histogram_spaces     = histogram_spaces
        self.use_color_statistics = use_color_statistics
        self.use_glcm             = use_glcm
        self.glcm_distances       = glcm_distances or [1, 3]
        self.glcm_angles          = glcm_angles or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.use_lbp              = use_lbp
        self.lbp_n_points         = lbp_n_points
        self.lbp_radius           = lbp_radius
        self.use_hu_moments       = use_hu_moments

        self._feature_dim = None   # populated after first extract()

    # ------------------------------------------------------------------
    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extract all enabled features from a single image.

        Parameters
        ----------
        image_rgb : np.ndarray  float32  (H, W, 3)  range [0,1]

        Returns
        -------
        np.ndarray  1-D float32 feature vector
        """
        parts = []

        if self.use_color_histogram:
            parts.append(extract_color_histogram(
                image_rgb,
                bins=self.histogram_bins,
                color_spaces=self.histogram_spaces,
            ))

        if self.use_color_statistics:
            parts.append(extract_color_statistics(image_rgb))

        if self.use_glcm:
            parts.append(extract_glcm_features(
                image_rgb,
                distances=self.glcm_distances,
                angles=self.glcm_angles,
            ))

        if self.use_lbp:
            parts.append(extract_lbp_features(
                image_rgb,
                n_points=self.lbp_n_points,
                radius=self.lbp_radius,
            ))

        if self.use_hu_moments:
            parts.append(extract_hu_moments(image_rgb))

        if not parts:
            raise ValueError("All feature extractors are disabled.")

        vec = np.concatenate(parts)
        self._feature_dim = len(vec)
        return vec

    # ------------------------------------------------------------------
    def extract_batch(self, images: np.ndarray,
                      desc: str = "Extracting features") -> np.ndarray:
        """
        Extract features for every image in a batch.

        Parameters
        ----------
        images : np.ndarray  shape (N, H, W, 3)

        Returns
        -------
        np.ndarray  shape (N, D)
        """
        features = []
        for img in tqdm(images, desc=f"  {desc}", leave=False):
            features.append(self.extract(img))
        X = np.array(features, dtype=np.float32)
        self._feature_dim = X.shape[1]
        return X

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector (available after first extract)."""
        if self._feature_dim is None:
            raise RuntimeError("Call extract() or extract_batch() first.")
        return self._feature_dim

    # ------------------------------------------------------------------
    def feature_group_sizes(self) -> dict:
        """
        Return a dict mapping feature group name -> vector length.
        Useful for understanding the composition of the feature vector.
        """
        sizes = {}
        if self.use_color_histogram:
            n_spaces  = len(self.histogram_spaces)
            n_channels = 3
            sizes["color_histogram"] = n_spaces * n_channels * self.histogram_bins

        if self.use_color_statistics:
            sizes["color_statistics"] = 18   # 3 spaces × 3 channels × 2 stats

        if self.use_glcm:
            n_props = 6
            n_combos = len(self.glcm_distances) * len(self.glcm_angles)
            sizes["glcm"] = n_props * n_combos

        if self.use_lbp:
            sizes["lbp"] = self.lbp_n_points + 2

        if self.use_hu_moments:
            sizes["hu_moments"] = 7

        return sizes

    # ------------------------------------------------------------------
    def __repr__(self):
        enabled = [k for k, v in {
            "histogram": self.use_color_histogram,
            "statistics": self.use_color_statistics,
            "glcm": self.use_glcm,
            "lbp": self.use_lbp,
            "hu_moments": self.use_hu_moments,
        }.items() if v]
        return (f"FeatureExtractor(enabled={enabled}, "
                f"hist_bins={self.histogram_bins})")
