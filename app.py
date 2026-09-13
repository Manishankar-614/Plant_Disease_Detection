import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import io
import json
from flask import Flask, render_template, request, jsonify
import time
import os
import traceback

# --- 1. APP CONFIGURATION ---
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_output_multi"
MODEL_PATH = MODEL_DIR / "final_multitask_model.h5"
DISEASE_LABEL_PATH = MODEL_DIR / "disease_labels.json"
PART_LABEL_PATH = MODEL_DIR / "part_labels.json"

# Default resolution fallback (supports 256 or 224 based on model weights)
DEFAULT_IMG_SIZE = 256

# Hierarchical Taxonomy: Part -> Allowed Disease Classes
PART_DISEASE_MAPPING = {
    "Leaves": {"Healthy leaf", "Leaf anthracnose", "Leaf rust", "Leaf spot", "Mosaic virus"},
    "Fruits": {"Brown rot", "Fruit Anthracnose", "Healthy fruit", "Powdery Mildew", "Scab"},
    "Stems": {"Black knot", "canker", "Gummosis", "Healthy stem", "Stem rust"}
}

# --- 2. MODEL LOADING ---
try:
    GLOBAL_MODEL = models.load_model(MODEL_PATH)
    with open(DISEASE_LABEL_PATH, "r") as f:
        GLOBAL_DISEASE_LABELS = json.load(f)
    with open(PART_LABEL_PATH, "r") as f:
        GLOBAL_PART_LABELS = json.load(f)
    
    # Auto-detect model input resolution
    if GLOBAL_MODEL.input_shape and len(GLOBAL_MODEL.input_shape) >= 3 and GLOBAL_MODEL.input_shape[1] is not None:
        IMG_SIZE = int(GLOBAL_MODEL.input_shape[1])
    else:
        IMG_SIZE = DEFAULT_IMG_SIZE

    print(f"--- Model and labels loaded successfully (Input Resolution: {IMG_SIZE}x{IMG_SIZE}). ---")
except Exception as e:
    print(f"--- FATAL ERROR: Could not load model on startup: {e} ---")
    traceback.print_exc()
    GLOBAL_MODEL = None
    GLOBAL_DISEASE_LABELS = []
    GLOBAL_PART_LABELS = []
    IMG_SIZE = DEFAULT_IMG_SIZE

# --- 3. IMAGE PREPROCESSING ---
def preprocess_image(image):
    """Preprocesses the uploaded image for MobileNetV2."""
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return preprocessed_img

# --- 4. FLASK ROUTES ---
@app.route("/")
def index():
    """Renders the main prediction page."""
    return render_template("index.html")

@app.route("/about")
def about_page():
    """Renders the about project page."""
    return render_template("about.html")

# --- 5. API ENDPOINTS ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload, runs multi-task prediction, applies
    hierarchical part-conditioned masking, and returns JSON.
    """
    if GLOBAL_MODEL is None:
        return jsonify({"error": "Model is not loaded. Please check server logs."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        preprocessed_img = preprocess_image(image)
        
        start_time = time.perf_counter()
        disease_pred_raw, part_pred_raw = GLOBAL_MODEL.predict(preprocessed_img)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # 1. Identify Plant Part
        p_index = int(np.argmax(part_pred_raw[0]))
        p_confidence = float(part_pred_raw[0][p_index] * 100)
        p_name = GLOBAL_PART_LABELS[p_index]
        
        result = {
            "part_name": p_name,
            "part_confidence": p_confidence,
            "disease_name": "N/A",
            "disease_confidence": 0.0,
            "inference_time_ms": inference_time_ms
        }
        
        # 2. Hierarchical Part-Conditioned Disease Masking
        if p_name in ("Not_a_plant", "Not a plant"):
            result["disease_name"] = "N/A (Non-plant specimen)"
            result["disease_confidence"] = 0.0
        else:
            # Filter candidate diseases based on predicted part
            valid_diseases = PART_DISEASE_MAPPING.get(p_name, set())
            valid_indices = [
                i for i, name in enumerate(GLOBAL_DISEASE_LABELS)
                if name in valid_diseases
            ]
            
            if valid_indices:
                sub_probs = disease_pred_raw[0][valid_indices]
                # Re-normalize softmax among candidate diseases for the detected part
                sub_probs_normalized = sub_probs / (np.sum(sub_probs) + 1e-9)
                best_sub_idx = int(np.argmax(sub_probs))
                
                d_index = valid_indices[best_sub_idx]
                d_confidence = float(sub_probs_normalized[best_sub_idx] * 100)
                d_name = GLOBAL_DISEASE_LABELS[d_index]
            else:
                d_index = int(np.argmax(disease_pred_raw[0]))
                d_confidence = float(disease_pred_raw[0][d_index] * 100)
                d_name = GLOBAL_DISEASE_LABELS[d_index]
            
            result["disease_name"] = d_name
            result["disease_confidence"] = d_confidence
        
        return jsonify(result)

    except Exception as e:
        print(f"--- ERROR DURING PREDICTION ---")
        traceback.print_exc()
        print(f"-------------------------------")
        return jsonify({"error": str(e)}), 500

# --- 6. RUN THE APP ---
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)