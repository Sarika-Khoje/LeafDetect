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

UPLOAD_FOLDER   = Path("static/uploads")
ALLOWED_EXT     = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}
MODEL_PATH      = Path("models/best_predictor.pkl")
MAX_CONTENT_MB  = 16

app.config["UPLOAD_FOLDER"]    = str(UPLOAD_FOLDER)
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


def disease_info(class_name: str) -> dict:
    """
    Return a short description and recommended action for each disease class.
    """
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

    # Match against any key found in the class name
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

    # ── save uploaded file with a unique name ────────────────────────────
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(str(save_path))

    # ── run prediction ───────────────────────────────────────────────────
    try:
        pred  = get_predictor()
        result = pred.predict_image(str(save_path))
        top5   = pred.predict_top_k(str(save_path), k=5)
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        save_path.unlink(missing_ok=True)
        return redirect(url_for("index"))

    # ── build display data ───────────────────────────────────────────────
    raw_class   = result["predicted_class"]
    short_name  = raw_class.replace("Tomato___", "").replace("Tomato__", "").replace("_", " ")
    confidence  = round(result["confidence"] * 100, 1)
    is_healthy  = "healthy" in raw_class.lower()
    info        = disease_info(raw_class)

    top5_display = []
    for entry in top5:
        sn = (entry["class"]
              .replace("Tomato___", "")
              .replace("Tomato__", "")
              .replace("_", " "))
        top5_display.append({
            "class":      sn,
            "confidence": round(entry["confidence"] * 100, 1),
            "bar_width":  round(entry["confidence"] * 100, 1),
        })

    return render_template(
        "result.html",
        image_path   = f"uploads/{filename}",
        short_name   = short_name,
        confidence   = confidence,
        is_healthy   = is_healthy,
        info         = info,
        top5         = top5_display,
        raw_class    = raw_class,
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
