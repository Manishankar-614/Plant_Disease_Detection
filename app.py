import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
from pathlib import Path
import io
import json
from flask import Flask, render_template, request, jsonify, g, send_file
import time  # <-- IMPORT TIME

# --- 1. APP & MODEL CONFIGURATION ---
# (No changes here)
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_output_multi"
MODEL_PATH = MODEL_DIR / "final_multitask_model.h5"
DISEASE_LABEL_PATH = MODEL_DIR / "disease_labels.json"
PART_LABEL_PATH = MODEL_DIR / "part_labels.json"
DB_PATH = BASE_DIR / "history.db"
IMG_SIZE = 224

# --- 2. MODEL & LABEL LOADING ---
# (No changes here)
def load_keras_model_and_labels():
    # ... (same as before) ...
    try:
        model = models.load_model(MODEL_PATH)
        with open(DISEASE_LABEL_PATH, "r") as f:
            disease_labels = json.load(f)
        with open(PART_LABEL_PATH, "r") as f:
            part_labels = json.load(f)
        print("Model and labels loaded successfully.")
        return model, disease_labels, part_labels
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None
model, disease_labels, part_labels = load_keras_model_and_labels()


# --- 3. DATABASE FUNCTIONS ---

def get_db():
    # (No changes here)
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    # (No changes here)
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database and creates the 'history' table."""
    try:
        with app.app_context():
            db = get_db()
            cursor = db.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,        -- <-- FIX: Changed DATETIME to TIMESTAMP
                    image_blob BLOB NOT NULL,
                    part_prediction TEXT,
                    part_confidence REAL,
                    disease_prediction TEXT,
                    disease_confidence REAL,
                    inference_time REAL                 -- <-- NEW: Added inference time
                )
            """)
            db.commit()
            print("Database initialized.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def add_to_history(image_bytes, part_pred, part_conf, disease_pred, disease_conf, inf_time): # <-- NEW: Added inf_time
    """Adds a new prediction record to the database."""
    try:
        db = get_db()
        cursor = db.cursor()
        timestamp = datetime.now()
        cursor.execute("""
            INSERT INTO history (timestamp, image_blob, part_prediction, part_confidence, disease_prediction, disease_confidence, inference_time)
            VALUES (?, ?, ?, ?, ?, ?, ?) -- <-- NEW: Added 7th placeholder
        """, (timestamp, image_bytes, part_pred, part_conf, disease_pred, disease_conf, inf_time)) # <-- NEW: Added inf_time
        db.commit()
    except Exception as e:
        print(f"Error saving to history: {e}")

def get_history():
    # (No changes here - this function is already correct)
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
        records = cursor.fetchall()
        return records
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []

# --- 4. IMAGE PREPROCESSING ---
# (No changes here)
def preprocess_image(image):
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return preprocessed_img

# --- 5. FLASK ROUTES (THE PAGES) ---
# (No changes in these routes)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history_page():
    records = get_history()
    return render_template("history.html", records=records)

@app.route("/about")
def about_page():
    return render_template("about.html")

# --- 6. API ENDPOINTS (THE LOGIC) ---

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the image upload, performs prediction, saves to history,
    and returns the result as JSON.
    """
    if model is None or 'file' not in request.files:
        return jsonify({"error": "Model not loaded or no file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        preprocessed_img = preprocess_image(image)
        
        # --- NEW: Start Timer ---
        start_time = time.perf_counter()
        
        # Predict
        disease_pred_raw, part_pred_raw = model.predict(preprocessed_img)
        
        # --- NEW: End Timer ---
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000 # Calculate in milliseconds
        
        # Process Part Prediction
        p_index = np.argmax(part_pred_raw[0])
        p_confidence = float(part_pred_raw[0][p_index] * 100)
        p_name = part_labels[p_index]
        
        result = {
            "part_name": p_name,
            "part_confidence": p_confidence,
            "disease_name": "N/A",
            "disease_confidence": 0.0,
            "inference_time_ms": inference_time_ms  # <-- NEW: Add to result
        }
        
        if p_name != "Not_a_plant":
            d_index = np.argmax(disease_pred_raw[0])
            d_confidence = float(disease_pred_raw[0][d_index] * 100)
            d_name = disease_labels[d_index]
            result["disease_name"] = d_name
            result["disease_confidence"] = d_confidence

        # Save to history
        add_to_history(
            image_bytes, 
            result["part_name"], 
            result["part_confidence"], 
            result["disease_name"], 
            result["disease_confidence"],
            result["inference_time_ms"] # <-- NEW: Pass to history
        )
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history/image/<int:record_id>')
def get_history_image(record_id):
    # (No changes here)
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT image_blob FROM history WHERE id = ?", (record_id,))
        blob = cursor.fetchone()
        
        if blob:
            return send_file(io.BytesIO(blob['image_blob']), mimetype='image/png')
        return "Image not found", 404
    except Exception as e:
        print(f"Error fetching image: {e}")
        return "Error fetching image", 500

# --- 7. RUN THE APP ---
if __name__ == "__main__":
    init_db()
    app.run(debug=True)