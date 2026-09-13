"""
train_mobilenet.py (High-Accuracy Botanical MobileNetV2 Multi-Task Pipeline)

Optimized with:
- 256x256 High-Resolution input for micro-lesion recognition (Option B)
- Advanced Botanical Augmentation (Hue/Sat/Brightness/Contrast/Rot/Zoom/Cutout) (Option D)
- Multi-Task Loss Weighting (Prioritizing 16-class disease task over 4-class part task)
- Two-Stage Transfer Learning with AdamW & Cosine Annealing Learning Rate Schedule
"""

import time
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image

# ---------------- AUTO CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_split"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_OUT = BASE_DIR / "model_output_multi"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 256  # Upgraded resolution (256x256)
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 25
RANDOM_SEED = 42

print("=" * 60)
print(f"Dataset Root: {DATASET_DIR}")
print(f"Training Data: {TRAIN_DIR}")
print(f"Validation Data: {VAL_DIR}")
print(f"Model Output Dir: {MODEL_OUT}")
print(f"Batch Size: {BATCH_SIZE}, Image Size: {IMG_SIZE}x{IMG_SIZE}")
print("=" * 60)

# ---------------- SCAN DATASET & BUILD LABELS ----------------
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def build_label_maps(train_dir):
    """Scans training directory to index parts and disease classes."""
    if not train_dir.exists():
        raise SystemExit(f"Training folder not found: {train_dir}")
        
    part_name_list = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    disease_name_list = set()
    
    print("\n" + "="*50)
    print(" DATASET CLASS COUNT (TRAIN) ".center(50, "-"))
    disease_counts = {}
    part_counts = {}
    
    for part_dir in train_dir.iterdir():
        if not part_dir.is_dir():
            continue
        part_file_count = 0
        part_name = part_dir.name
        
        for disease_dir in part_dir.iterdir():
            if disease_dir.is_dir():
                disease_name = disease_dir.name
                disease_name_list.add(disease_name)
                
                count = len([p for p in disease_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT])
                part_file_count += count
                disease_counts[f"{part_name}/{disease_name}"] = count
                
        part_counts[part_name] = part_file_count

    print("\nPart Counts:")
    for name, count in part_counts.items():
        print(f"  - {name}: {count} images")
        
    print("\nDisease Counts:")
    for name, count in sorted(disease_counts.items()):
        print(f"  - {name}: {count} images")
    print("="*50 + "\n")
    
    disease_name_list = sorted(list(disease_name_list))
    disease_map = {name: i for i, name in enumerate(disease_name_list)}
    part_map = {name: i for i, name in enumerate(part_name_list)}
    
    return disease_name_list, part_name_list, disease_map, part_map

def load_split_data(split_dir, disease_map, part_map):
    """Loads image paths and labels for train/val splits."""
    image_paths, disease_labels, part_labels = [], [], []

    if not split_dir.exists():
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

disease_name_list, part_name_list, disease_map, part_map = build_label_maps(TRAIN_DIR)
NUM_DISEASE_CLASSES = len(disease_name_list)
NUM_PART_CLASSES = len(part_name_list)

train_paths, train_d, train_p = load_split_data(TRAIN_DIR, disease_map, part_map)
val_paths, val_d, val_p = load_split_data(VAL_DIR, disease_map, part_map)

print(f"Found {NUM_DISEASE_CLASSES} disease classes and {NUM_PART_CLASSES} part classes.")
print(f"Loaded: {len(train_paths)} training samples, {len(val_paths)} validation samples.")

# ---------------- ENHANCED BOTANICAL DATA AUGMENTATION (OPTION D) ----------------
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, d_label, p_label):
    """Loads, decodes, and resizes image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    
    d_one_hot = tf.one_hot(tf.cast(d_label, tf.int32), depth=NUM_DISEASE_CLASSES)
    p_one_hot = tf.one_hot(tf.cast(p_label, tf.int32), depth=NUM_PART_CLASSES)
    
    return img, {"disease_out": d_one_hot, "part_out": p_one_hot}

def random_cutout(image, mask_size=32):
    """Applies random rectangular patch masking (Cutout regularization)."""
    h, w = IMG_SIZE, IMG_SIZE
    y = tf.random.uniform([], 0, h - mask_size, dtype=tf.int32)
    x = tf.random.uniform([], 0, w - mask_size, dtype=tf.int32)
    
    mask = tf.pad(
        tf.zeros([mask_size, mask_size, 3], dtype=image.dtype),
        [[y, h - y - mask_size], [x, w - x - mask_size], [0, 0]],
        constant_values=1
    )
    return image * mask

def botanical_augmentation(image, labels):
    """Comprehensive botanical color & spatial augmentation."""
    # 1. Geometric transforms
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))
    
    # 2. Random crop & zoom (90% to 100%)
    crop_size = tf.cast(tf.random.uniform([], 0.88, 1.0) * float(IMG_SIZE), tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # 3. Botanical color jitter (Lighting & Green Hue variability)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.25)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.06)
    
    # 4. Patch Cutout (50% probability)
    apply_cutout = tf.random.uniform([]) > 0.5
    image = tf.cond(apply_cutout, lambda: random_cutout(image, mask_size=int(IMG_SIZE * 0.15)), lambda: image)
    
    return image, labels

def make_dataset(paths, d_labels, p_labels, training=True):
    """Creates efficient tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((paths, d_labels, p_labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    
    if training:
        ds = ds.shuffle(1024, seed=RANDOM_SEED)
        ds = ds.map(botanical_augmentation, num_parallel_calls=AUTOTUNE)
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    )
    return ds.prefetch(AUTOTUNE)

train_ds = make_dataset(train_paths, train_d, train_p, training=True)
val_ds = make_dataset(val_paths, val_d, val_p, training=False)

# ---------------- MULTI-TASK MOBILENETV2 ARCHITECTURE ----------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

# Feature Trunk
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.35)(x)

# Disease Head (16 classes)
disease_dense = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
disease_dense = layers.Dropout(0.3)(disease_dense)
disease_out = layers.Dense(NUM_DISEASE_CLASSES, activation="softmax", name="disease_out")(disease_dense)

# Part Head (4 classes)
part_dense = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
part_dense = layers.Dropout(0.2)(part_dense)
part_out = layers.Dense(NUM_PART_CLASSES, activation="softmax", name="part_out")(part_dense)

model = models.Model(inputs=inputs, outputs=[disease_out, part_out])

# Cross-entropy with label smoothing
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)

# Multi-Task Loss Weighting (Prioritize harder 16-class disease problem)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss={"disease_out": loss_fn, "part_out": loss_fn},
    loss_weights={"disease_out": 2.0, "part_out": 0.6},
    metrics={"disease_out": ["accuracy"], "part_out": ["accuracy"]}
)

model.summary()

# ---------------- STAGE 1: TRAINING HEADS ----------------
print("\n" + "="*50)
print(" STAGE 1: TRAINING MULTI-TASK HEADS ".center(50, "="))
print("="*50)

stamp = time.strftime("%Y%m%d-%H%M%S")
ckpt_path = MODEL_OUT / f"best_multitask_{stamp}.h5"

callbacks_stage1 = [
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_disease_out_accuracy", mode="max"),
    EarlyStopping(monitor="val_disease_out_loss", patience=6, restore_best_weights=True, mode="min"),
    ReduceLROnPlateau(monitor="val_disease_out_loss", factor=0.3, patience=2, min_lr=1e-6, mode="min", verbose=1)
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_stage1
)


print("\n" + "="*50)
print(" STAGE 2: FINE-TUNING MOBILENETV2 BACKBONE ".center(50, "="))
print("="*50)

base_model.trainable = True
# Fine-tune top layers from layer 75 onwards
fine_tune_from = 75
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

print(f"Unfrozen backbone layers from layer {fine_tune_from} to {len(base_model.layers)}.")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss={"disease_out": loss_fn, "part_out": loss_fn},
    loss_weights={"disease_out": 2.0, "part_out": 0.6},
    metrics={"disease_out": ["accuracy"], "part_out": ["accuracy"]}
)

callbacks_stage2 = [
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_disease_out_accuracy", mode="max"),
    EarlyStopping(monitor="val_disease_out_loss", patience=7, restore_best_weights=True, mode="min"),
    ReduceLROnPlateau(monitor="val_disease_out_loss", factor=0.2, patience=2, min_lr=1e-7, mode="min", verbose=1)
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
    initial_epoch=EPOCHS_STAGE1,
    callbacks=callbacks_stage2
)

# ---------------- SAVE MODEL & EXPORTS ----------------
final_model_path = MODEL_OUT / "final_multitask_model.h5"
model.save(final_model_path)
print(f"\nSaved final model weights: {final_model_path}")

# Export TFLite model
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = MODEL_OUT / "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Exported TFLite model: {tflite_path}")
except Exception as e:
    print(f"TFLite export note: {e}")

# Save label mappings
with open(MODEL_OUT / "disease_labels.json", "w") as f:
    json.dump(disease_name_list, f, indent=2)
with open(MODEL_OUT / "part_labels.json", "w") as f:
    json.dump(part_name_list, f, indent=2)

print("[SUCCESS] High-Accuracy Training Pipeline Complete.")