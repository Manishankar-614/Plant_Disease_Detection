import numpy as np
from PIL import Image
from pathlib import Path
import io
import json
from flask import Flask, render_template, request, jsonify
import time
import os
import traceback
import tflite_runtime.interpreter as tflite  # <-- NEW: Import TFLite

# --- 1. APP CONFIGURATION ---
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# --- UPDATED: Load TFLite model ---
TFLITE_MODEL_PATH = BASE_DIR / "model.tflite"
# We must re-load the labels. They are no longer in the .tflite model
DISEASE_LABEL_PATH = BASE_DIR / "model_output_multi" / "disease_labels.json"
PART_LABEL_PATH = BASE_DIR / "model_output_multi" / "part_labels.json"
IMG_SIZE = 224


# --- 2. MODEL LOADING (NEW GLOBAL TFLite METHOD) ---
try:
    # Load TFLite model and allocate tensors
    GLOBAL_INTERPRETER = tflite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    GLOBAL_INTERPRETER.allocate_tensors()
    
    # Get input and output details
    GLOBAL_INPUT_DETAILS = GLOBAL_INTERPRETER.get_input_details()
    GLOBAL_OUTPUT_DETAILS = GLOBAL_INTERPRETER.get_output_details()
    
    # Load labels
    with open(DISEASE_LABEL_PATH, "r") as f:
        GLOBAL_DISEASE_LABELS = json.load(f)
    with open(PART_LABEL_PATH, "r") as f:
        GLOBAL_PART_LABELS = json.load(f)
        
    # Find which output is which (safer than relying on order)
    if "disease_out" in GLOBAL_OUTPUT_DETAILS[0]['name']:
        GLOBAL_DISEASE_OUTPUT_INDEX = GLOBAL_OUTPUT_DETAILS[0]['index']
        GLOBAL_PART_OUTPUT_INDEX = GLOBAL_OUTPUT_DETAILS[1]['index']
    else:
        GLOBAL_DISEASE_OUTPUT_INDEX = GLOBAL_OUTPUT_DETAILS[1]['index']
        GLOBAL_PART_OUTPUT_INDEX = GLOBAL_OUTPUT_DETAILS[0]['index']
        
    print("--- TFLite Model and labels loaded successfully. ---")

except Exception as e:
    print(f"--- FATAL ERROR: Could not load TFLite model: {e} ---")
    traceback.print_exc()
    GLOBAL_INTERPRETER = None
# --- End of Change ---


# --- 3. DATABASE (REMOVED) ---
# (No database code)


# --- 4. IMAGE PREPROCESSING ---
def preprocess_image(image):
    """Preprocesses the uploaded image for the model."""
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # --- UPDATED: Manually normalize for TFLite ---
    # MobileNetV2 expects inputs from -1 to 1
    img_array = (img_array / 127.5) - 1.0
    
    # TFLite models often expect float32
    return img_array.astype(np.float32)

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
    if GLOBAL_INTERPRETER is None:
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
        
        # --- NEW TFLite Prediction ---
        GLOBAL_INTERPRETER.set_tensor(GLOBAL_INPUT_DETAILS[0]['index'], preprocessed_img)
        GLOBAL_INTERPRETER.invoke() # Run inference
        
        # Get results
        disease_pred_raw = GLOBAL_INTERPRETER.get_tensor(GLOBAL_DISEASE_OUTPUT_INDEX)
        part_pred_raw = GLOBAL_INTERPRETER.get_tensor(GLOBAL_PART_OUTPUT_INDEX)
        # --- End TFLite Prediction ---
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Apply softmax (TFLite models output raw logits)
        part_probs = tf.nn.softmax(part_pred_raw[0]).numpy()
        disease_probs = tf.nn.softmax(disease_pred_raw[0]).numpy()

        p_index = np.argmax(part_probs)
        p_confidence = float(part_probs[p_index] * 100)
        p_name = GLOBAL_PART_LABELS[p_index]
        
        result = {
            "part_name": p_name,
            "part_confidence": p_confidence,
            "disease_name": "N/A",
            "disease_confidence": 0.0,
            "inference_time_ms": inference_time_ms
        }
        
        if p_name != "Not_a_plant":
            d_index = np.argmax(disease_probs)
            d_confidence = float(disease_probs[d_index] * 100)
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
if __name__ == "__main__":
    app.run(debug=True)