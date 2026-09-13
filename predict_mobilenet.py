"""
predict_mobilenet.py
Loads the trained MobileNetV2 multi-task model and performs:
1. Evaluation on the test set with both Raw and Hierarchical Part-Masked metrics.
2. Interactive prediction loop on single user-selected images.
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

DEFAULT_IMG_SIZE = 256
BATCH_SIZE = 32
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

PART_DISEASE_MAPPING = {
    "Leaves": {"Healthy leaf", "Leaf anthracnose", "Leaf rust", "Leaf spot", "Mosaic virus"},
    "Fruits": {"Brown rot", "Fruit Anthracnose", "Healthy fruit", "Powdery Mildew", "Scab"},
    "Stems": {"Black knot", "canker", "Gummosis", "Healthy stem", "Stem rust"}
}

# ---------------- HELPER FUNCTIONS ----------------

def load_model_and_labels():
    """Loads the H5 model and JSON label maps."""
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    if not DISEASE_LABEL_PATH.exists() or not PART_LABEL_PATH.exists():
        raise SystemExit(f"Label files not found in: {MODEL_OUT_DIR}")

    model = models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    with open(DISEASE_LABEL_PATH, "r") as f:
        disease_labels = json.load(f)
    with open(PART_LABEL_PATH, "r") as f:
        part_labels = json.load(f)

    # Detect resolution
    if model.input_shape and len(model.input_shape) >= 3 and model.input_shape[1] is not None:
        img_size = int(model.input_shape[1])
    else:
        img_size = DEFAULT_IMG_SIZE

    print(f"Loaded {len(disease_labels)} disease labels and {len(part_labels)} part labels (Model Resolution: {img_size}x{img_size}).")
    return model, disease_labels, part_labels, img_size

def load_and_preprocess_image(image_path, img_size):
    """Loads and preprocesses a single image for MobileNetV2."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

def load_test_set(test_dir, disease_map, part_map):
    """Scans the test set directory to get all paths and true labels."""
    image_paths, true_d_labels, true_p_labels = [], [], []
    
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

def evaluate_on_test_set(model, disease_labels, part_labels, test_dir, img_size):
    """Runs predictions on the test set and prints metrics."""
    print(f"\nScanning test set: {test_dir}...")
    
    disease_map = {name: i for i, name in enumerate(disease_labels)}
    part_map = {name: i for i, name in enumerate(part_labels)}
    
    image_paths, true_d_labels, true_p_labels = load_test_set(test_dir, disease_map, part_map)
    if not image_paths:
        print("No images found in the test set. Skipping evaluation.")
        return

    print(f"Found {len(image_paths)} images for evaluation.")

    def load_for_dataset(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    test_ds = test_ds.map(load_for_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Running batch prediction on test set...")
    start_time = time.perf_counter()
    disease_preds_raw, part_preds_raw = model.predict(test_ds)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_image = (total_time / len(image_paths)) * 1000

    # 1. Raw Predictions
    pred_d_raw = np.argmax(disease_preds_raw, axis=1)
    pred_p_labels = np.argmax(part_preds_raw, axis=1)

    # 2. Hierarchical Part-Masked Predictions
    pred_d_masked = []
    for i in range(len(image_paths)):
        p_name = part_labels[pred_p_labels[i]]
        if p_name in ("Not_a_plant", "Not a plant"):
            not_plant_idx = disease_map.get("Not a plant", disease_map.get("Not_a_plant", pred_d_raw[i]))
            pred_d_masked.append(not_plant_idx)
        else:
            valid_diseases = PART_DISEASE_MAPPING.get(p_name, set())
            valid_indices = [idx for idx, name in enumerate(disease_labels) if name in valid_diseases]
            if valid_indices:
                sub_probs = disease_preds_raw[i][valid_indices]
                best_sub_idx = np.argmax(sub_probs)
                pred_d_masked.append(valid_indices[best_sub_idx])
            else:
                pred_d_masked.append(pred_d_raw[i])

    print("\n" + "="*60)
    print(" EVALUATION RESULTS ".center(60, "="))
    print(f"Average inference latency: {avg_time_per_image:.2f} ms / image")
    print(f"Part Accuracy: {accuracy_score(true_p_labels, pred_p_labels) * 100:.2f}%")
    print(f"Raw Disease Accuracy: {accuracy_score(true_d_labels, pred_d_raw) * 100:.2f}%")
    print(f"Hierarchical Masked Disease Accuracy: {accuracy_score(true_d_labels, pred_d_masked) * 100:.2f}%")
    print("="*60)

    print("\n--- HIERARCHICAL MASKED DISEASE CLASSIFICATION REPORT ---")
    print(classification_report(true_d_labels, pred_d_masked, target_names=disease_labels, zero_division=0))

    print("\n--- PART CLASSIFICATION REPORT ---")
    print(classification_report(true_p_labels, pred_p_labels, target_names=part_labels, zero_division=0))

# ---------------- PART 2: PREDICT ON SINGLE IMAGE ----------------

def predict_single_image(model, disease_labels, part_labels, img_size):
    """Opens a file dialog to let the user pick an image and predicts it."""
    root = tk.Tk()
    root.withdraw()
    
    print("\n" + "="*50)
    print(" INTERACTIVE SINGLE IMAGE PREDICTOR ".center(50, "="))
    print("="*50)
    
    while True:
        print("\nPlease select an image file to predict (or cancel to exit)...")
        file_path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            print("No file selected. Exiting.")
            break
            
        image_path = Path(file_path)
        try:
            with Image.open(image_path) as img:
                img.verify()
            
            preprocessed_img = load_and_preprocess_image(image_path, img_size)
            
            start_time = time.perf_counter()
            disease_pred_raw, part_pred_raw = model.predict(preprocessed_img, verbose=0)
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000
            
            p_index = int(np.argmax(part_pred_raw[0]))
            p_confidence = float(part_pred_raw[0][p_index] * 100)
            p_name = part_labels[p_index]
            
            print(f"\n--- Prediction: {image_path.name} ---")
            print(f"  Part: {p_name} ({p_confidence:.2f}%)")
            
            if p_name in ("Not_a_plant", "Not a plant"):
                print("  Disease: N/A (Non-plant specimen)")
            else:
                valid_diseases = PART_DISEASE_MAPPING.get(p_name, set())
                valid_indices = [idx for idx, name in enumerate(disease_labels) if name in valid_diseases]
                if valid_indices:
                    sub_probs = disease_pred_raw[0][valid_indices]
                    sub_probs_norm = sub_probs / (np.sum(sub_probs) + 1e-9)
                    best_sub = int(np.argmax(sub_probs))
                    d_index = valid_indices[best_sub]
                    d_confidence = float(sub_probs_norm[best_sub] * 100)
                    d_name = disease_labels[d_index]
                else:
                    d_index = int(np.argmax(disease_pred_raw[0]))
                    d_confidence = float(disease_pred_raw[0][d_index] * 100)
                    d_name = disease_labels[d_index]
                
                print(f"  Disease: {d_name} ({d_confidence:.2f}%)")
            
            print(f"  Inference Latency: {inference_time:.2f} ms")
            print("-" * 40)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    model, disease_labels, part_labels, img_size = load_model_and_labels()
    evaluate_on_test_set(model, disease_labels, part_labels, TEST_DIR, img_size)
    predict_single_image(model, disease_labels, part_labels, img_size)