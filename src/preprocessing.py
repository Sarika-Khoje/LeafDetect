"""
preprocessing.py
----------------
Handles all image loading and preprocessing steps:
  - Loading images from PlantVillage directory structure
  - Resizing to a fixed resolution
  - Normalization (pixel values to [0, 1])
  - Color-space conversions (BGR -> RGB, HSV, LAB)
  - Train/test splitting
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PlantVillage class names that relate to tomato leaves.
# The folder names inside the dataset directory must match these keys exactly.
TOMATO_CLASSES = {
    "Tomato___healthy": 0,
    "Tomato___Tomato_mosaic_virus": 1,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 2,
    "Tomato___Target_Spot": 3,
    "Tomato___Spider_mites Two-spotted_spider_mite": 4,
    "Tomato___Septoria_leaf_spot": 5,
    "Tomato___Leaf_Mold": 6,
    "Tomato___Late_blight": 7,
    "Tomato___Early_blight": 8,
    "Tomato___Bacterial_spot": 9,
}

# Reverse mapping: integer label -> human-readable name
LABEL_TO_CLASS = {v: k for k, v in TOMATO_CLASSES.items()}

# Target image size after resizing (height, width)
IMAGE_SIZE = (128, 128)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, image_size: tuple = IMAGE_SIZE,
                 classes: dict = None, max_per_class: int = None):
    """
    Walk a PlantVillage-style directory and load tomato leaf images.

    Expected layout:
        data_dir/
            Tomato__healthy/
                img1.jpg
                img2.jpg
            Tomato__Early_blight/
                ...

    Parameters
    ----------
    data_dir : str
        Root directory that contains one sub-folder per class.
    image_size : tuple
        (height, width) to resize every image to.
    classes : dict, optional
        Mapping of folder_name -> integer_label.  Defaults to TOMATO_CLASSES.
    max_per_class : int, optional
        Cap the number of images loaded per class (useful for quick tests).

    Returns
    -------
    images : np.ndarray  shape (N, H, W, 3)  dtype float32  range [0, 1]
    labels : np.ndarray  shape (N,)           dtype int
    class_names : list[str]
    """
    if classes is None:
        classes = TOMATO_CLASSES

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    images, labels = [], []
    found_classes = []

    for class_name, label in classes.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            # Try case-insensitive match
            matches = [d for d in data_path.iterdir()
                       if d.is_dir() and d.name.lower() == class_name.lower()]
            if not matches:
                print(f"  [WARN] Class folder not found, skipping: {class_name}")
                continue
            class_dir = matches[0]

        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if max_per_class:
            image_files = image_files[:max_per_class]

        loaded = 0
        for img_path in tqdm(image_files, desc=f"  Loading {class_name}", leave=False):
            img = _load_and_resize(str(img_path), image_size)
            if img is not None:
                images.append(img)
                labels.append(label)
                loaded += 1

        if loaded > 0:
            found_classes.append(class_name)
            print(f"  Loaded {loaded} images for class: {class_name}")

    if len(images) == 0:
        raise ValueError("No images were loaded. Check your data directory structure.")

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    return images_arr, labels_arr, found_classes


def _load_and_resize(path: str, image_size: tuple):
    """
    Load a single image with OpenCV, convert BGR -> RGB, resize, and
    normalise pixel values to [0, 1].

    Returns None if the file cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Could not read image: {path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size[1], image_size[0]),
                             interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized


# ---------------------------------------------------------------------------
# Color-space conversions
# ---------------------------------------------------------------------------

def to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a normalised RGB image (float32, [0,1]) to HSV.

    Returns
    -------
    np.ndarray  float32  H in [0,360], S and V in [0,1]
    """
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    # Normalise H to [0,1] as well
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] /= 180.0   # OpenCV H range is [0,180]
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return hsv


def to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a normalised RGB image (float32, [0,1]) to CIE L*a*b*.

    Returns
    -------
    np.ndarray  float32  all channels normalised to [0,1]
    """
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.float32)
    lab[:, :, 0] /= 255.0   # L channel [0,255] in OpenCV
    lab[:, :, 1] /= 255.0
    lab[:, :, 2] /= 255.0
    return lab


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a normalised RGB image to grayscale.

    Returns
    -------
    np.ndarray  float32  shape (H, W)
    """
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Batch conversions
# ---------------------------------------------------------------------------

def convert_batch(images: np.ndarray, color_space: str = "rgb") -> np.ndarray:
    """
    Apply a color-space conversion to an entire batch of images.

    Parameters
    ----------
    images : np.ndarray  shape (N, H, W, 3)
    color_space : str    one of "rgb", "hsv", "lab", "gray"

    Returns
    -------
    np.ndarray with same first dimension N
    """
    color_space = color_space.lower()
    converters = {
        "rgb":  lambda x: x,
        "hsv":  to_hsv,
        "lab":  to_lab,
        "gray": to_grayscale,
    }
    if color_space not in converters:
        raise ValueError(f"Unknown color space '{color_space}'. "
                         f"Choose from: {list(converters.keys())}")

    convert = converters[color_space]
    return np.array([convert(img) for img in tqdm(images,
                    desc=f"  Converting to {color_space.upper()}", leave=False)],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_data(images: np.ndarray, labels: np.ndarray,
               test_size: float = 0.20,
               val_size: float = 0.10,
               random_state: int = 42):
    """
    Stratified split into train / validation / test sets.

    Parameters
    ----------
    images : np.ndarray
    labels : np.ndarray
    test_size  : fraction of total data reserved for testing
    val_size   : fraction of total data reserved for validation
    random_state : int

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # From the remaining data, split off the validation set
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"\n  Dataset splits:")
    print(f"    Train : {len(X_train)} samples")
    print(f"    Val   : {len(X_val)}   samples")
    print(f"    Test  : {len(X_test)}  samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_class_distribution(labels: np.ndarray, class_names: list = None) -> dict:
    """Return a dict of label -> count."""
    unique, counts = np.unique(labels, return_counts=True)
    dist = {}
    for u, c in zip(unique, counts):
        key = class_names[u] if class_names else str(u)
        dist[key] = int(c)
    return dist
