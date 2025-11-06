import tensorflow as tf
from pathlib import Path

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_output_multi"
MODEL_PATH = MODEL_DIR / "final_multitask_model.h5"
TFLITE_MODEL_PATH = BASE_DIR / "model.tflite"

print(f"Loading H5 model from: {MODEL_PATH}")

# Load the Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Add optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite...")

# Perform the conversion
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Successfully converted and saved to: {TFLITE_MODEL_PATH}")