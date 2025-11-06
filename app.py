import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import io
import json
from flask import Flask, render_template, request, jsonify, g, send_file
import time
import os
import traceback  # For better error logging

# --- 1. APP CONFIGURATION ---
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_output_multi"
MODEL_PATH = MODEL_DIR / "final_multitask_model.h5"
DISEASE_LABEL_PATH = MODEL_DIR / "disease_labels.json"
PART_LABEL_PATH = MODEL_DIR / "part_labels.json"
IMG_SIZE = 224


# --- 2. MODEL LOADING (NEW GLOBAL METHOD) ---
# We load the model and labels into global variables ONCE when the app file is
# first read. This is critical for Gunicorn's --preload flag.
try:
    GLOBAL_MODEL = models.load_model(MODEL_PATH)
    with open(DISEASE_LABEL_PATH, "r") as f:
        GLOBAL_DISEASE_LABELS = json.load(f)
    with open(PART_LABEL_PATH, "r") as f:
        GLOBAL_PART_LABELS = json.load(f)
    print("--- Model and labels loaded successfully into GLOBAL scope. ---")
except Exception as e:
    print(f"--- FATAL ERROR: Could not load model on startup: {e} ---")
    traceback.print_exc()
    GLOBAL_MODEL = None
    GLOBAL_DISEASE_LABELS = []
    GLOBAL_PART_LABELS = []
# --- End of Change ---


# --- 3. DATABASE (REMOVED) ---
# (No database code)


# --- 4. IMAGE PREPROCESSING ---
def preprocess_image(image):
    """Preprocesses the uploaded image for the model."""
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return preprocessed_img

# --- 5. FLASK ROUTES (THE PAGES) ---

@app.route("/")
def index():
    """Renders the main prediction page."""
    return render_template("index.html")

@app.route("/about")
def about_page():
    """Renders the about project page."""
    return render_template("about.html")

# --- 6. API ENDPOINTS (THE LOGIC) ---

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the image upload, performs prediction,
    and returns the result as JSON.
    """
    # CHANGED: Check the new global variable
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
        # CHANGED: Use the new global model
        disease_pred_raw, part_pred_raw = GLOBAL_MODEL.predict(preprocessed_img)
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # CHANGED: Use the new global labels
        p_index = np.argmax(part_pred_raw[0])
        p_confidence = float(part_pred_raw[0][p_index] * 100)
        p_name = GLOBAL_PART_LABELS[p_index]
        
        result = {
            "part_name": p_name,
            "part_confidence": p_confidence,
            "disease_name": "N/A",
            "disease_confidence": 0.0,
            "inference_time_ms": inference_time_ms
        }
        
        if p_name != "Not_a_plant":
            d_index = np.argmax(disease_pred_raw[0])
            d_confidence = float(disease_pred_raw[0][d_index] * 100)
            # CHANGED: Use the new global labels
            d_name = GLOBAL_DISEASE_LABELS[d_index]
            result["disease_name"] = d_name
            result["disease_confidence"] = d_confidence
        
        return jsonify(result)

    except Exception as e:
        print(f"--- ERROR DURING PREDICTION ---")
        traceback.print_exc()
        print(f"-------------------------------")
        return jsonify({"error": str(e)}), 500

# --- 7. RUN THE APP ---
# This block is only for local testing ("python app.py")
if __name__ == "__main__":
    app.run(debug=True)