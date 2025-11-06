"""
predict.py
Loads the trained multi-task model and performs two actions:
1. Evaluates the model on the 'test' set, calculating and
   printing detailed metrics (accuracy, precision, recall, f1-score)
   and batch inference time.
2. Enters an interactive loop to predict on single, real-world
   images selected by the user from their device.
"""

import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import filedialog

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_OUT_DIR = BASE_DIR / "model_output_multi"
MODEL_PATH = MODEL_OUT_DIR / "final_multitask_model.h5"
DISEASE_LABEL_PATH = MODEL_OUT_DIR / "disease_labels.json"
PART_LABEL_PATH = MODEL_OUT_DIR / "part_labels.json"

TEST_DIR = BASE_DIR / "dataset_split" / "test"

IMG_SIZE = 224
BATCH_SIZE = 32
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------- HELPER FUNCTIONS ----------------

def load_model_and_labels():
    """Loads the H5 model and JSON label maps."""
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    if not DISEASE_LABEL_PATH.exists() or not PART_LABEL_PATH.exists():
        raise SystemExit(f"Label files not found in: {MODEL_OUT_DIR}")

    # Load model
    model = models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    # Load labels
    with open(DISEASE_LABEL_PATH, "r") as f:
        disease_labels = json.load(f)
    with open(PART_LABEL_PATH, "r") as f:
        part_labels = json.load(f)
    print(f"Loaded {len(disease_labels)} disease labels and {len(part_labels)} part labels.")
    
    return model, disease_labels, part_labels

def load_and_preprocess_image(image_path):
    """Loads and preprocesses a single image for prediction."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    # Add a batch dimension
    img = tf.expand_dims(img, axis=0)
    return img

def load_test_set(test_dir, disease_map, part_map):
    """Scans the test set directory to get all paths and true labels."""
    image_paths = []
    true_d_labels = []
    true_p_labels = []
    
    if not test_dir.exists():
        print(f"Warning: Test directory not found, skipping evaluation: {test_dir}")
        return [], [], []

    for part_dir in test_dir.iterdir():
        if part_dir.name not in part_map:
            continue
        part_idx = part_map[part_dir.name]
        
        for disease_dir in part_dir.iterdir():
            if disease_dir.name not in disease_map:
                continue
            disease_idx = disease_map[disease_dir.name]
            
            files = [p for p in disease_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT]
            for f in files:
                image_paths.append(str(f))
                true_d_labels.append(disease_idx)
                true_p_labels.append(part_idx)
                
    return image_paths, true_d_labels, true_p_labels

# ---------------- PART 1: EVALUATE ON TEST SET ----------------

def evaluate_on_test_set(model, disease_labels, part_labels, test_dir):
    """Runs predictions on the test set and prints metrics."""
    print(f"Scanning test set: {test_dir}...")
    
    # Create name-to-index maps
    disease_map = {name: i for i, name in enumerate(disease_labels)}
    part_map = {name: i for i, name in enumerate(part_labels)}
    
    image_paths, true_d_labels, true_p_labels = load_test_set(test_dir, disease_map, part_map)
    
    if not image_paths:
        print("No images found in the test set. Skipping evaluation.")
        return

    print(f"Found {len(image_paths)} images for evaluation.")

    # --- Build tf.data pipeline for efficient batch prediction ---
    def load_for_dataset(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    test_ds = test_ds.map(load_for_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Run prediction and measure time ---
    print("Running batch prediction on test set...")
    start_time = time.perf_counter()
    
    predictions = model.predict(test_ds)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_per_image = (total_time / len(image_paths)) * 1000  # in ms
    
    disease_preds_raw, part_preds_raw = predictions
    
    # Get the predicted class index
    pred_d_labels = np.argmax(disease_preds_raw, axis=1)
    pred_p_labels = np.argmax(part_preds_raw, axis=1)

    # --- Print Metrics ---
    print("\n" + "-"*50)
    print(" BATCH INFERENCE TIME ".center(50, "-"))
    print(f"Total time for {len(image_paths)} images: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time_per_image:.4f} ms")
    print("-" * 50)
    
    print("\n" + "-"*50)
    print("MobilenetV2 MODEL EVALUATION RESULTS".center(50, "-"))
    print(" DISEASE CLASSIFICATION METRICS ".center(50, "-"))
    print("-" * 50)
    print(classification_report(true_d_labels, pred_d_labels, target_names=disease_labels, zero_division=0))
    
    print("\n" + "-"*50)
    print(" PART CLASSIFICATION METRICS ".center(50, "-"))
    print("-" * 50)
    print(classification_report(true_p_labels, pred_p_labels, target_names=part_labels, zero_division=0))


# ---------------- PART 2: PREDICT ON SINGLE IMAGE ----------------

def predict_single_image(model, disease_labels, part_labels):
    """Opens a file dialog to let the user pick an image and predicts it."""
    # Set up Tkinter root window (and hide it)
    root = tk.Tk()
    root.withdraw()
    
    print("\n" + "="*50)
    print(" READY FOR SINGLE IMAGE PREDICTION ".center(50, "="))
    print("="*50)
    
    while True:
        print("\nPlease select an image file to predict...")
        
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not file_path:
            print("\nNo file selected. Exiting interactive mode.")
            break
            
        image_path = Path(file_path)
        
        try:
            # 1. Validate image
            with Image.open(image_path) as img:
                img.verify()
            
            # 2. Preprocess
            preprocessed_img = load_and_preprocess_image(image_path)
            
            # 3. Predict and measure time
            start_time = time.perf_counter()
            disease_pred, part_pred = model.predict(preprocessed_img, verbose=0) # verbose=0 to silence output
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # in ms
            
            # 4. Get results
            p_index = np.argmax(part_pred[0])
            p_confidence = part_pred[0][p_index] * 100
            p_name = part_labels[p_index]
            
            # 5. Print results
            print(f"\n--- Prediction for: {image_path.name} ---")
            
            # --- NEW LOGIC HERE ---
            if p_name == "Not_a_plant":
                print(f"  Result:     This does not appear to be a plant image.")
                print(f"  Confidence: {p_confidence:.2f}%")
            else:
                # It is a plant, so get the disease prediction
                d_index = np.argmax(disease_pred[0])
                d_confidence = disease_pred[0][d_index] * 100
                d_name = disease_labels[d_index]
                
                print(f"  Part:       {p_name} (Confidence: {p_confidence:.2f}%)")
                print(f"  Disease:    {d_name} (Confidence: {d_confidence:.2f}%)")
            # --- END NEW LOGIC ---

            print(f"  Inference Time: {inference_time:.2f} ms")
            print("-" * (21 + len(image_path.name)))

        except Exception as e:
            print(f"Could not process file {image_path}. Error: {e}")

# ---------------- MAIN EXECUTION ----------------

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    # 1. Load model and labels
    model, disease_labels, part_labels = load_model_and_labels()
    
    # 2. Run evaluation on the test set
    evaluate_on_test_set(model, disease_labels, part_labels, TEST_DIR)
    
    # 3. Start interactive prediction loop
    predict_single_image(model, disease_labels, part_labels)
    
    print("\n✅ Prediction script finished.")