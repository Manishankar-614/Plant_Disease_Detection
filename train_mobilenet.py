"""
train_multitask.py
Trains a MobileNetV2-based multi-task model:
 - head 1 => disease classification (many classes)
 - head 2 => plant part classification (Fruits / Leaves / Stems)

This script reads from a pre-split dataset folder structured as:
dataset_split/
    ├── train/
    │   ├── Fruits/
    │   │   ├── Disease_A/
    │   │   └── Disease_B/
    │   ├── Leaves/
    │   └── Stems/
    ├── val/
    │   ├── Fruits/
    │   ├── Leaves/
    │   └── Stems/
    └── test/ (Note: this script only uses train and val)

Place this script next to your dataset_split/ folder and run:
python train_multitask_split.py
"""

import time
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# No train_test_split needed
from PIL import Image

# ---------------- AUTO CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
# ✅ POINT TO THE SPLIT DATASET
DATASET_DIR = BASE_DIR / "dataset_split"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
# TEST_DIR = DATASET_DIR / "test" # Not used in this training script

MODEL_OUT = BASE_DIR / "model_output_multi"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20
RANDOM_SEED = 42

print("=" * 60)
print(f"Dataset Root: {DATASET_DIR}")
print(f"Training Data: {TRAIN_DIR}")
print(f"Validation Data: {VAL_DIR}")
print(f"Model output: {MODEL_OUT}")
print(f"Batch: {BATCH_SIZE}, Img: {IMG_SIZE}, Epochs: {EPOCHS}")
print("=" * 60)

# ---------------- NEW: SCAN DATASET ----------------

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def build_label_maps(train_dir):
    """
    Scans the training directory to build a complete, sorted list
    of all disease and part classes.
    """
    if not train_dir.exists():
        raise SystemExit(f"Training folder not found: {train_dir}")
        
    part_name_list = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    disease_name_list = set()
    
    for part_dir in train_dir.iterdir():
        if not part_dir.is_dir():
            continue
        for disease_dir in part_dir.iterdir():
            if disease_dir.is_dir():
                disease_name_list.add(disease_dir.name)
    
    disease_name_list = sorted(list(disease_name_list))
    
    if not part_name_list or not disease_name_list:
        raise SystemExit(f"No classes found in {train_dir}. Check structure.")

    # Create map from name to index
    disease_map = {name: i for i, name in enumerate(disease_name_list)}
    part_map = {name: i for i, name in enumerate(part_name_list)}
    
    return disease_name_list, part_name_list, disease_map, part_map

def load_split_data(split_dir, disease_map, part_map):
    """
    Loads all image paths and corresponding labels for a given split
    (e.g., train or val) using the pre-built label maps.
    """
    image_paths = []
    disease_labels = []
    part_labels = []

    if not split_dir.exists():
        print(f"Warning: Directory not found, skipping: {split_dir}")
        return np.array([]), np.array([]), np.array([])

    for part_dir in split_dir.iterdir():
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
                disease_labels.append(disease_idx)
                part_labels.append(part_idx)

    return np.array(image_paths), np.array(disease_labels), np.array(part_labels)

# 1. Build label maps from the 'train' directory
print("Scanning 'train' directory to build label maps...")
disease_name_list, part_name_list, disease_map, part_map = build_label_maps(TRAIN_DIR)

print(f"Found {len(disease_name_list)} disease classes and {len(part_name_list)} part classes")

# 2. Load Train and Validation data
print("Loading training data...")
train_paths, train_d, train_p = load_split_data(TRAIN_DIR, disease_map, part_map)
print("Loading validation data...")
val_paths, val_d, val_p = load_split_data(VAL_DIR, disease_map, part_map)

if len(train_paths) == 0:
    raise SystemExit("No training images found. Check your dataset structure.")
if len(val_paths) == 0:
    print("Warning: No validation images found.")

# ---------------- VALIDATE IMAGES ----------------
def validate_image_data(paths, d_labels, p_labels):
    """Helper function to filter out corrupted images."""
    valid_paths, valid_d, valid_p, skipped = [], [], [], []
    for p, d, pt in zip(paths.tolist(), d_labels.tolist(), p_labels.tolist()):
        try:
            with Image.open(p) as img:
                img.verify()
            valid_paths.append(p)
            valid_d.append(d)
            valid_p.append(pt)
        except Exception:
            skipped.append(p)

    if skipped:
        print(f"Skipped {len(skipped)} corrupted images (examples):")
        for s in skipped[:5]:
            print(" -", s)
    
    return np.array(valid_paths), np.array(valid_d), np.array(valid_p)

print("Validating training images...")
train_paths, train_d, train_p = validate_image_data(train_paths, train_d, train_p)
print("Validating validation images...")
val_paths, val_d, val_p = validate_image_data(val_paths, val_d, val_p)

# ---------------- OLD TRAIN/VAL SPLIT REMOVED ----------------
# The data is already split, so no train_test_split is needed.

print(f"Train images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")


# ---------------- BUILD tf.data PIPELINE ----------------
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, d_label, p_label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, {"disease_out": tf.cast(d_label, tf.int32),
                 "part_out": tf.cast(p_label, tf.int32)}

def make_dataset(paths, d_labels, p_labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, d_labels, p_labels))
    ds = ds.map(lambda p, d, pt: load_and_preprocess(p, d, pt), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=RANDOM_SEED)
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_d, train_p, training=True)
val_ds = make_dataset(val_paths, val_d, val_p, training=False)

# ---------------- MODEL ----------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# Disease head
disease_dense = layers.Dense(256, activation="relu")(x)
disease_dense = layers.Dropout(0.3)(disease_dense)
disease_out = layers.Dense(len(disease_name_list), activation="softmax", name="disease_out")(disease_dense)

# Part head
part_dense = layers.Dense(128, activation="relu")(x)
part_dense = layers.Dropout(0.2)(part_dense)
part_out = layers.Dense(len(part_name_list), activation="softmax", name="part_out")(part_dense)

model = models.Model(inputs=inputs, outputs=[disease_out, part_out])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "disease_out": "sparse_categorical_crossentropy",
        "part_out": "sparse_categorical_crossentropy"
    },
    metrics={
        "disease_out": ["accuracy"],
        "part_out": ["accuracy"]
    }
)

model.summary()

# ---------------- CALLBACKS ----------------
stamp = time.strftime("%Y%m%d-%H%M%S")
ckpt_path = MODEL_OUT / f"best_multitask_{stamp}.h5"

cb_ckpt = ModelCheckpoint(
    ckpt_path,
    save_best_only=True,
    monitor="val_disease_out_loss",
    mode="min"
)

cb_early = EarlyStopping(
    monitor="val_disease_out_loss",
    patience=6,
    restore_best_weights=True,
    mode="min"
)

# ---------------- TRAIN ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[cb_ckpt, cb_early]
)

# ---------------- OPTIONAL: FINE-TUNE ----------------
UNFREEZE_FOR_FINETUNE = True
if UNFREEZE_FOR_FINETUNE:
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss={
            "disease_out": "sparse_categorical_crossentropy",
            "part_out": "sparse_categorical_crossentropy"
        },
        metrics={
            "disease_out": ["accuracy"],
            "part_out": ["accuracy"]
        }
    )
    print("Starting fine-tuning with reduced LR...")
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[cb_ckpt, cb_early]
    )

# ---------------- SAVE MODEL ----------------
final_model_path = MODEL_OUT / "final_multitask_model.h5"
model.save(final_model_path)
print(f"Saved model: {final_model_path}")

# Save label maps
with open(MODEL_OUT / "disease_labels.json", "w") as f:
    json.dump(disease_name_list, f)
with open(MODEL_OUT / "part_labels.json", "w") as f:
    json.dump(part_name_list, f)

print("Saved label maps: disease_labels.json, part_labels.json")
print("✅ Training complete.")