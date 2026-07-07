"""
Regression training: predict the continuous A2/A0 bar strength from an image.

Uses BOTH datasets:
  dataset/      - the v3 binary layout ({split}/{bar,no_bar}/*.png), 122 snaps
  dataset_mid/  - flat mid-strength renders ({split}/*.png), 60 snaps
Labels for every image come from extras/bar_labels.csv via the snapshot id in
the filename. Target normalized to [0,1] as a2a0 / 0.30 (max in sim ~0.28).

Multi-architecture (pick via CLI) for a diverse ensemble:
    python3 train_regression.py resnet50
    python3 train_regression.py effnetv2b0
    python3 train_regression.py convnext_tiny

Two-phase transfer learning as always: frozen base + head (8 ep, lr 1e-3),
then fine-tune whole top (up to 18 ep, lr 1e-5, early stop on val MAE).
Loss = Huber (robust). Outputs models_reg_<arch>/ with weights + report:
  - regression MAE on test
  - classification accuracy on test using threshold: pred_a2a0 >= 0.19 -> bar
    (evaluated separately on the v3-test subset for apples-to-apples vs 86.4%)
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ARCH = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
OUT = f"models_reg_{ARCH}"
IMG = (224, 224)
BATCH = 32
A2A0_SCALE = 0.30          # normalize target to ~[0,1]
CLS_THR = 0.19 / A2A0_SCALE
os.makedirs(OUT, exist_ok=True)
tf.keras.utils.set_random_seed(42)

BACKBONES = {
    "resnet50":      (keras.applications.ResNet50,
                      keras.applications.resnet50.preprocess_input),
    "effnetv2b0":    (keras.applications.EfficientNetV2B0,
                      keras.applications.efficientnet_v2.preprocess_input),
    "convnext_tiny": (keras.applications.ConvNeXtTiny,
                      keras.applications.convnext.preprocess_input),
}

# ---------------- data: file list + continuous labels ----------------
labels_df = pd.read_csv("extras/bar_labels.csv").set_index("snapshot")

def collect(split):
    files = sorted(glob.glob(f"dataset/{split}/*/*.png")) + \
            sorted(glob.glob(f"dataset_mid/{split}/*.png"))
    y = np.array([labels_df.loc[re.search(r"snapshot_\d+", os.path.basename(f)).group(),
                                "peak_a2a0"] / A2A0_SCALE for f in files],
                 dtype=np.float32)
    return files, y

def make_ds(files, y, training):
    def load(path, target):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, IMG)
        return tf.cast(img, tf.float32), target
    ds = tf.data.Dataset.from_tensor_slices((files, y)).map(
        load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(4096, seed=42)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

tr_f, tr_y = collect("train"); va_f, va_y = collect("val"); te_f, te_y = collect("test")
print(f"train={len(tr_f)} val={len(va_f)} test={len(te_f)} images "
      f"(target range {tr_y.min():.2f}-{tr_y.max():.2f})")
train_ds, val_ds = make_ds(tr_f, tr_y, True), make_ds(va_f, va_y, False)
test_ds = make_ds(te_f, te_y, False)

# ---------------- model ----------------
Backbone, preprocess = BACKBONES[ARCH]
base = Backbone(weights="imagenet", include_top=False, input_shape=IMG + (3,))
base.trainable = False

augment = keras.Sequential([
    layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0),
])

inputs = keras.Input(shape=IMG + (3,))
x = augment(inputs)
x = preprocess(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)   # target is in [0,1]
model = keras.Model(inputs, outputs)

cb = [keras.callbacks.CSVLogger(f"{OUT}/training_log.csv", append=True),
      keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=6,
                                    restore_best_weights=True)]

print(f"\n=== {ARCH} phase 1: head ===")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(),
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=cb)

print(f"\n=== {ARCH} phase 2: fine-tune ===")
base.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=keras.losses.Huber(),
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(train_ds, validation_data=val_ds, epochs=18, callbacks=cb)

# ---------------- evaluate ----------------
pred = model.predict(test_ds, verbose=0).ravel()
mae = float(np.abs(pred - te_y).mean()) * A2A0_SCALE
# classification on the v3-test subset only (files under dataset/test/...)
is_v3 = np.array(["dataset/" in f or "dataset\\" in f for f in te_f])
true_cls = (te_y[is_v3] >= CLS_THR)          # bar = True
pred_cls = (pred[is_v3] >= CLS_THR)
acc = float((true_cls == pred_cls).mean())

rep = (f"{ARCH} regression: test MAE = {mae:.4f} (in A2/A0 units)\n"
       f"classification via threshold on v3-test subset "
       f"({is_v3.sum()} imgs): acc = {acc:.4f}\n")
print("\n" + rep)
np.save(f"{OUT}/test_pred.npy", pred)
with open(f"{OUT}/test_files.txt", "w") as fh:
    fh.write("\n".join(te_f))
with open(f"{OUT}/report.txt", "w") as fh:
    fh.write(rep)
model.save_weights(f"{OUT}/model.weights.h5")   # always works
try:
    model.save(f"{OUT}/model.keras")             # full model, may fail on
except Exception as e:                            # preprocess-layer serialization
    print("full-model save failed (weights are saved):", e)
print(f"saved {OUT}/")
