"""
Test-time augmentation (TTA) for the v3 model - free accuracy, no retraining.

Idea: a bar looks like a bar under rotation and reflection. So at prediction
time we show the model several rotated/flipped copies of each test image and
average the probabilities. Averaging cancels view-specific noise and usually
nudges accuracy up 1-2 pts.

Compares: single-image (baseline) vs TTA, on the v3 test set.
Run from windowpynbody2:  python code/tta_eval.py
"""
import glob, os, re
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL = "models_v3/bar_resnet50.keras"
TEST = "dataset/test"
IMG = (224, 224)

model = keras.models.load_model(MODEL)

files, labels = [], []
for cls, lab in (("bar", 0), ("no_bar", 1)):
    for f in sorted(glob.glob(os.path.join(TEST, cls, "*.png"))):
        files.append(f); labels.append(lab)
labels = np.array(labels)
imgs = np.stack([keras.utils.img_to_array(
    keras.utils.load_img(f, color_mode="rgb", target_size=IMG)) for f in files])

# 8 label-preserving views: 4 rotations x optional horizontal flip
def views(x):
    out = []
    for k in range(4):                       # 0,90,180,270 deg
        r = np.rot90(x, k, axes=(1, 2))
        out.append(r)
        out.append(r[:, :, ::-1, :])         # + horizontal flip
    return out

base = model.predict(imgs, batch_size=32, verbose=0).ravel()
tta = np.mean([model.predict(v, batch_size=32, verbose=0).ravel()
               for v in views(imgs)], axis=0)

def report(name, prob):
    pred = (prob > 0.5).astype(int)
    acc = (pred == labels).mean()
    auc = keras.metrics.AUC()(labels, prob).numpy()
    bar_rec = (pred[labels == 0] == 0).mean()
    print(f"{name:18s} acc={acc*100:.1f}%  auc={auc:.3f}  bar-recall={bar_rec*100:.1f}%")

print(f"v3 test set: {len(files)} images\n")
report("single-image", base)
report("TTA (8 views)", tta)
