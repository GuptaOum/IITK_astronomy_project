"""
Evaluate the FINAL model (regression ResNet50) on the v3 test set:
fresh inference, overall metrics + per-tilt breakdown.

This is the script behind the reported numbers:
  96.0% accuracy, 92.0% bar-recall, MAE 0.0217 (v3 subset), and the per-tilt
  table (100% face-on -> 91.4% at 40 deg).

Run from windowpynbody2 (CPU ok, ~3 min):
    python code/eval_regression.py
"""
import glob, os, re
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

WEIGHTS = "results/regression_resnet50/model.weights.h5"
TEST_DIR = "data/dataset_v3_binary/test"
IMG = (224, 224)
A2A0_SCALE = 0.30
CLS_THR = 0.19


def build_model():
    base = keras.applications.ResNet50(weights=None, include_top=False,
                                       input_shape=IMG + (3,))
    inp = keras.Input(IMG + (3,))
    x = keras.Sequential([layers.Identity()])(inp)   # augment placeholder
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.load_weights(WEIGHTS)
    return m


model = build_model()
print("model loaded")

files, y_cls = [], []
for cls, lab in (("bar", 1), ("no_bar", 0)):
    for f in sorted(glob.glob(f"{TEST_DIR}/{cls}/*.png")):
        files.append(f); y_cls.append(lab)
y_cls = np.array(y_cls)
labdf = pd.read_csv("extras/bar_labels.csv").set_index("snapshot")
y_a2a0 = np.array([labdf.loc[re.search(r"snapshot_\d+", os.path.basename(f)).group(),
                             "peak_a2a0"] for f in files])
imgs = np.stack([keras.utils.img_to_array(
    keras.utils.load_img(f, color_mode="rgb", target_size=IMG)) for f in files])
print(f"loaded {len(files)} test images, predicting...")

pred = model.predict(imgs, batch_size=32, verbose=0).ravel() * A2A0_SCALE
pred_cls = (pred >= CLS_THR).astype(int)
lines = [f"Regression ResNet50 on v3 test ({len(files)} imgs):",
         f"classification acc = {(pred_cls == y_cls).mean()*100:.1f}%  "
         f"bar-recall = {pred_cls[y_cls == 1].mean()*100:.1f}%  "
         f"MAE = {np.abs(pred - y_a2a0).mean():.4f}", "", "per-tilt:"]

tilts = np.array([max(int(re.search(r"_X(\d+)_Y(\d+)", f).group(1)),
                      int(re.search(r"_X(\d+)_Y(\d+)", f).group(2))) for f in files])
for t in sorted(set(tilts)):
    m = tilts == t
    lines.append(f"  tilt {t:2d}: acc={((pred_cls[m]==y_cls[m]).mean())*100:5.1f}%  "
                 f"MAE={np.abs(pred[m]-y_a2a0[m]).mean():.4f}  n={m.sum()}")

report = "\n".join(lines)
print("\n" + report)
with open("results/regression_resnet50/eval_v3_report.txt", "w") as f:
    f.write(report + "\n")
print("\nsaved results/regression_resnet50/eval_v3_report.txt")
