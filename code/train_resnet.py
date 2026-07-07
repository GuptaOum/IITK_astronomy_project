"""
Train a ResNet50 bar-detection classifier on the rendered dataset
(transfer learning, TensorFlow/Keras).

Two-phase recipe:
  Phase 1 - freeze the ImageNet-pretrained ResNet50 base, train only the new
            classification head (fast, gets the head into a sane state).
  Phase 2 - unfreeze the top of the base and fine-tune everything above
            FINE_TUNE_AT with a very low learning rate.

Expects the folder layout produced by generate_dataset.py:
    dataset/{train,val,test}/{bar,no_bar}/*.png

Run from the windowpynbody2 folder:
    python code/train_resnet.py

Outputs:
    models/bar_resnet50.keras   trained model
    models/training_log.csv     per-epoch metrics
    models/test_report.txt      final held-out test metrics
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
# CONFIG
# -------------------------------
import sys
DATA_DIR = "dataset"
# Optional CLI: python train_resnet.py <seed> -> writes to models_seed<seed>/
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
MODEL_DIR = f"models_seed{SEED}" if len(sys.argv) > 1 else "models"
IMG_SIZE = (224, 224)
BATCH = 32
tf.keras.utils.set_random_seed(SEED)

EPOCHS_HEAD = 8         # phase 1
EPOCHS_FINETUNE = 18    # phase 2 (run1 was still improving at epoch 10)
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_AT = 143      # unfreeze from this layer up (start of conv5 block)

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# DATA
# -------------------------------
def load_split(name, shuffle):
    return keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, name),
        labels="inferred",
        label_mode="binary",        # bar=0, no_bar=1 (alphabetical)
        color_mode="rgb",           # grayscale PNGs replicated to 3 channels
        image_size=IMG_SIZE,        # ResNet50 wants 3-channel input
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED,
    )

train_ds = load_split("train", shuffle=True)
val_ds = load_split("val", shuffle=False)
test_ds = load_split("test", shuffle=False)
class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------------------------------
# MODEL
# -------------------------------
# In-plane spin was deliberately NOT rendered by generate_dataset.py -
# it is a pure 2D image rotation, so we do it here for free, every epoch.
augment = keras.Sequential([
    layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0),
], name="augmentation")

base = keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = keras.applications.resnet50.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

METRICS = [keras.metrics.BinaryAccuracy(name="acc"),
           keras.metrics.AUC(name="auc")]

callbacks = [
    keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, "training_log.csv"),
                              append=True),
    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                  restore_best_weights=True),
]

# -------------------------------
# PHASE 1: train the head
# -------------------------------
print("\n=== Phase 1: training classification head ===")
model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
              loss="binary_crossentropy", metrics=METRICS)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
          callbacks=callbacks)

# -------------------------------
# PHASE 2: fine-tune top of the base
# -------------------------------
print("\n=== Phase 2: fine-tuning ===")
base.trainable = True
for layer in base.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE),
              loss="binary_crossentropy", metrics=METRICS)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
          callbacks=callbacks)

# -------------------------------
# EVALUATE ON HELD-OUT SNAPSHOTS
# -------------------------------
print("\n=== Test set (snapshots never seen in training) ===")
results = model.evaluate(test_ds, return_dict=True)
print(results)

# Confusion matrix
y_true, y_prob = [], []
for xb, yb in test_ds:
    y_true.append(yb.numpy().ravel())
    y_prob.append(model.predict(xb, verbose=0).ravel())
y_true = np.concatenate(y_true)
y_pred = (np.concatenate(y_prob) > 0.5).astype(int)

tp = int(((y_pred == 1) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())

report = (f"Test loss={results['loss']:.4f} acc={results['acc']:.4f} "
          f"auc={results['auc']:.4f}\n"
          f"Confusion matrix (positive class = '{class_names[1]}'):\n"
          f"  TP={tp}  FP={fp}\n  FN={fn}  TN={tn}\n")
print(report)
with open(os.path.join(MODEL_DIR, "test_report.txt"), "w") as f:
    f.write(report)

model.save(os.path.join(MODEL_DIR, "bar_resnet50.keras"))
print(f"Model saved to {MODEL_DIR}/bar_resnet50.keras")
