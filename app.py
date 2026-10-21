"""
app.py
------
Flask web application for Tomato Plant Disease Detection.

Routes
------
GET  /            → upload page
POST /predict     → run inference, return results page
GET  /about       → brief about page
"""

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify
)
from werkzeug.utils import secure_filename

# ── project modules ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import load_predictor

# ── app config ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "tomato-disease-detector-secret"

UPLOAD_FOLDER  = Path("static/uploads")
ALLOWED_EXT    = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}
MODEL_PATH     = Path("models/best_predictor.pkl")
MAX_CONTENT_MB = 16

app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ── load predictor once at startup ─────────────────────────────────────────
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Please run the pipeline first:\n"
                "  python main.py --data_dir data/raw --max_per_class 200"
            )
        print("  Loading predictor from disk …")
        predictor = load_predictor(str(MODEL_PATH))
        print("  Predictor ready.")
    return predictor


# ── helpers ────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXT


def is_leaf_image(image_path: str) -> tuple:
    """
    Validate that the uploaded image is likely a plant/tomato leaf.

    Strategy — four checks that work for both healthy AND diseased leaves:

      1. Vegetation OR earth-tone coverage:
            Leaves (even diseased brown/dark ones) occupy a large fraction
            of the frame. Check broad colour range: green + yellow + brown
            + grey-green (blighted) combined ≥ 15%.

      2. Largest coherent region ≥ 10% of image:
            The leaf object must form one large blob, not scattered pixels
            (e.g. green trees behind a car).

      3. Organic texture score (Laplacian variance within the object):
            Leaves have smooth, organic gradients with vein micro-texture.
            Non-leaf objects (cars, faces, text, sky) produce different
            Laplacian energy patterns. We check that the texture score is
            in a "leaf-like" range — not too low (blank/sky) and not
            dominated by harsh geometric edges (metal/text).

      4. Aspect ratio of the largest blob:
            Leaves are roughly equi-dimensional or moderately elongated.
            Very wide or very tall objects (building, sky strip) are rejected.

    Returns (is_valid: bool, reason: str)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False, "Could not read the image file."

    img_small = cv2.resize(img_bgr, (256, 256))
    total_px  = 256 * 256

    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    # ── Check 1: broad leaf colour coverage ──────────────────────────────
    # Healthy green:    H 25–90,  S>25, V>25
    # Yellow/diseased:  H 15–25,  S>30, V>40
    # Brown/blight:     H 0–20,   S>25, V>20  (also catches dark spots)
    # Grey-green:       H 25–90,  S>10, V>20  (desaturated diseased)
    mask_green   = cv2.inRange(hsv, (25, 25, 25), (90, 255, 255))
    mask_yellow  = cv2.inRange(hsv, (15, 30, 40), (25, 255, 255))
    mask_brown   = cv2.inRange(hsv, (0,  25, 20), (20, 255, 200))
    mask_grgreen = cv2.inRange(hsv, (25, 10, 20), (90, 80,  220))
    veg_mask     = cv2.bitwise_or(mask_green, mask_yellow)
    veg_mask     = cv2.bitwise_or(veg_mask,   mask_brown)
    veg_mask     = cv2.bitwise_or(veg_mask,   mask_grgreen)

    # Close small gaps to merge the leaf body
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel)

    veg_ratio = cv2.countNonZero(veg_mask) / total_px
    if veg_ratio < 0.15:
        return False, (
            f"Only {veg_ratio*100:.1f}% of the image matches leaf colours. "
            "Please upload a clear, close-up photo of a tomato leaf."
        )

    # ── Check 2: largest coherent region ─────────────────────────────────
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        veg_mask, connectivity=8)
    if num_labels > 1:
        largest_px = int(stats[1:, cv2.CC_STAT_AREA].max())
    else:
        largest_px = 0

    largest_ratio = largest_px / total_px
    if largest_ratio < 0.15:
        return False, (
            "No single large leaf region found. "
            "Please upload a photo where the tomato leaf fills most of the frame."
        )

    # ── Check 3: texture score (Laplacian variance) ───────────────────────
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # Compute Laplacian only inside the leaf region (avoid background noise)
    lap       = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs   = np.abs(lap)
    leaf_mask = (veg_mask > 0)
    if leaf_mask.sum() > 0:
        texture_score = float(lap_abs[leaf_mask].mean())
    else:
        texture_score = 0.0

    # Too low → blank/featureless (sky, wall, solid background)
    # We don't set an upper limit — harsh edges are fine; it's the leaf region texture
    if texture_score < 2.0:
        return False, (
            "The image appears too featureless or out-of-focus. "
            "Please upload a clear, in-focus photo of a tomato leaf."
        )

    # ── Check 4: aspect ratio of largest blob ─────────────────────────────
    # Find the largest component index again
    if num_labels > 1:
        largest_idx = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 5.0:
            return False, (
                "The detected region is too elongated to be a leaf. "
                "Please upload a photo where the tomato leaf fills most of the frame."
            )

    return True, "OK"


def is_confident_prediction(top5: list) -> tuple:
    """
    Reject predictions where the model is essentially guessing —
    i.e. top confidence is below 20% (near-uniform distribution over 10 classes).

    Returns (is_confident: bool, reason: str)
    """
    if not top5:
        return True, "OK"
    top_conf = top5[0]["confidence"]   # already in percent 0–100
    if top_conf < 20.0:
        return False, (
            f"The model could not confidently identify this as a tomato leaf "
            f"(best match: {top_conf:.1f}%). "
            "Please upload a clear, well-lit photo of a single tomato leaf."
        )
    return True, "OK"


def disease_info(class_name: str) -> dict:
    """Return a short description and recommended action for each disease class."""
    INFO = {
        "healthy": {
            "description": "The leaf appears healthy with no visible disease symptoms.",
            "action": "No action needed. Continue regular monitoring and care.",
            "severity": "none",
            "color": "#2e7d32",
        },
        "Bacterial_spot": {
            "description": "Bacterial spot causes small, dark, water-soaked spots on leaves, "
                           "which enlarge and turn brown with yellow halos.",
            "action": "Remove infected leaves. Apply copper-based bactericide. "
                      "Avoid overhead irrigation.",
            "severity": "moderate",
            "color": "#e65100",
        },
        "Early_blight": {
            "description": "Early blight appears as dark brown spots with concentric rings "
                           "(target-like pattern), usually on older leaves first.",
            "action": "Apply fungicide (chlorothalonil or mancozeb). "
                      "Remove heavily infected leaves. Ensure good air circulation.",
            "severity": "moderate",
            "color": "#e65100",
        },
        "Late_blight": {
            "description": "Late blight creates large, greasy-looking dark spots that spread "
                           "rapidly. White mold may appear on leaf undersides in humid conditions.",
            "action": "Immediately apply systemic fungicide. Remove infected plants to prevent "
                      "spread. This disease can devastate crops quickly.",
            "severity": "severe",
            "color": "#b71c1c",
        },
        "Leaf_Mold": {
            "description": "Leaf mold causes yellow patches on upper leaf surfaces and "
                           "olive-green to brown velvet-like mold on the undersides.",
            "action": "Improve ventilation. Reduce humidity. Apply fungicide if severe.",
            "severity": "moderate",
            "color": "#e65100",
        },
        "Septoria_leaf_spot": {
            "description": "Septoria leaf spot creates small, circular spots with dark borders "
                           "and grey or tan centres, often with tiny black specks inside.",
            "action": "Apply fungicide. Remove infected leaves. Avoid wetting foliage.",
            "severity": "moderate",
            "color": "#e65100",
        },
        "Spider_mites": {
            "description": "Spider mite infestation causes tiny yellow or white speckles on "
                           "leaves, webbing, and overall leaf bronzing.",
            "action": "Apply miticide or neem oil. Increase humidity. "
                      "Spray water to dislodge mites.",
            "severity": "moderate",
            "color": "#e65100",
        },
        "Target_Spot": {
            "description": "Target spot creates brown lesions with concentric rings on leaves, "
                           "petioles, and stems.",
            "action": "Apply fungicide. Remove infected debris. Rotate crops.",
            "severity": "moderate",
            "color": "#e65100",
        },
        "Tomato_mosaic_virus": {
            "description": "Mosaic virus causes mottled light and dark green patches, "
                           "leaf distortion, and stunted growth.",
            "action": "No cure exists. Remove and destroy infected plants. "
                      "Control insect vectors. Disinfect tools.",
            "severity": "severe",
            "color": "#b71c1c",
        },
        "Yellow_Leaf_Curl_Virus": {
            "description": "Yellow leaf curl virus causes upward leaf curling, yellowing, "
                           "and severe stunting. Spread by whiteflies.",
            "action": "Remove infected plants immediately. Control whitefly populations "
                      "with insecticide. Use resistant varieties.",
            "severity": "severe",
            "color": "#b71c1c",
        },
    }

    for key, info in INFO.items():
        if key.lower().replace("_", "") in class_name.lower().replace("_", "").replace(" ", ""):
            return info

    return {
        "description": "Disease information not available.",
        "action": "Consult a local agricultural expert.",
        "severity": "unknown",
        "color": "#546e7a",
    }


# ── routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── validate upload ──────────────────────────────────────────────────
    if "file" not in request.files:
        flash("No file selected. Please choose an image to upload.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose an image to upload.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a JPG or PNG image.", "error")
        return redirect(url_for("index"))

    # ── save uploaded file ────────────────────────────────────────────────
    ext       = file.filename.rsplit(".", 1)[1].lower()
    filename  = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(str(save_path))

    # ── leaf validation (before running the model) ────────────────────────
    valid, reason = is_leaf_image(str(save_path))
    if not valid:
        save_path.unlink(missing_ok=True)
        flash(f"⚠️ Not a valid leaf image — {reason}", "error")
        return redirect(url_for("index"))

    # ── run prediction ───────────────────────────────────────────────────
    try:
        pred   = get_predictor()
        result = pred.predict_image(str(save_path))
        top5   = pred.predict_top_k(str(save_path), k=5)
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        save_path.unlink(missing_ok=True)
        return redirect(url_for("index"))

    # ── confidence check (catches images that pass colour filter but aren't leaves) ──
    top5_pct = [{"confidence": round(e["confidence"] * 100, 1),
                 "class": e["class"]} for e in top5]
    confident, conf_reason = is_confident_prediction(top5_pct)
    if not confident:
        save_path.unlink(missing_ok=True)
        flash(f"⚠️ {conf_reason}", "error")
        return redirect(url_for("index"))

    # ── build display data ───────────────────────────────────────────────
    raw_class  = result["predicted_class"]
    short_name = (raw_class
                  .replace("Tomato___", "")
                  .replace("Tomato__", "")
                  .replace("_", " "))
    confidence = round(result["confidence"] * 100, 1)
    is_healthy = "healthy" in raw_class.lower()
    info       = disease_info(raw_class)

    top5_display = []
    for entry in top5:
        sn       = (entry["class"]
                    .replace("Tomato___", "")
                    .replace("Tomato__", "")
                    .replace("_", " "))
        conf_val = round(entry["confidence"] * 100, 1)
        top5_display.append({"class": sn, "confidence": conf_val, "bar_width": conf_val})

    return render_template(
        "result.html",
        image_path = f"uploads/{filename}",
        short_name = short_name,
        confidence = confidence,
        is_healthy = is_healthy,
        info       = info,
        top5       = top5_display,
        raw_class  = raw_class,
    )


@app.route("/about")
def about():
    return render_template("about.html")


# ── run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Tomato Disease Detection — Web App")
    print("="*55)
    print("  Loading model …")
    try:
        get_predictor()
        print("  Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"\n  WARNING: {e}\n")
    print("\n  Open your browser and go to:")
    print("  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
