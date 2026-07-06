"""
Apples-to-apples: evaluate BOTH trained models on the SAME test set
(dataset v2's test split, tilt <= 40 deg).

Caveat noted in output: run1's model saw none of these snapshots in training?
Not guaranteed - v1 and v2 snapshot selections overlap, so a v2-test snapshot
may have been in run1's TRAINING set. The script checks and reports this.

Run from windowpynbody2:  python code/compare_models.py
"""

import glob
import os
import re
import numpy as np
from tensorflow import keras

IMG_SIZE = (224, 224)
TEST_DIR = "dataset/test"   # v2 test split

MODELS = {
    "run1 (trained tilt<=60)": "models_run1/bar_resnet50.keras",
    "run2 (trained tilt<=40)": "models_run2/bar_resnet50.keras",
}

# --- load the shared test set once ---
files, labels = [], []
for cls, lab in (("bar", 0), ("no_bar", 1)):
    fs = sorted(glob.glob(os.path.join(TEST_DIR, cls, "*.png")))
    files += fs
    labels += [lab] * len(fs)
labels = np.array(labels)
snaps_in_test = sorted({re.search(r"snapshot_\d+", f).group() for f in files})
print(f"Shared test set: {len(files)} images from snapshots {snaps_in_test}\n")

imgs = np.stack([
    keras.utils.img_to_array(
        keras.utils.load_img(f, color_mode="rgb", target_size=IMG_SIZE))
    for f in files
])

# --- leakage check for run1: were any of these snapshots in run1 training? ---
run1_train = sorted({re.search(r"snapshot_\d+", f).group()
                     for f in glob.glob("dataset_run1_tilt60/train/*/*.png")})
leak = [s for s in snaps_in_test if s in run1_train]
print(f"v2-test snapshots that were in run1 TRAINING data: {leak or 'none'}\n")

for name, path in MODELS.items():
    model = keras.models.load_model(path)
    prob = model.predict(imgs, batch_size=32, verbose=0).ravel()
    pred = (prob > 0.5).astype(int)
    acc = (pred == labels).mean()
    bar_rec = (pred[labels == 0] == 0).mean()
    nobar_rec = (pred[labels == 1] == 1).mean()
    # leakage-free subset for run1 fairness
    mask = np.array([re.search(r"snapshot_\d+", f).group() not in leak
                     for f in files])
    acc_clean = (pred[mask] == labels[mask]).mean()
    print(f"{name}: acc={acc:.4f} (excl. leaked snaps: {acc_clean:.4f}) "
          f"bar-recall={bar_rec:.4f} no_bar-recall={nobar_rec:.4f}")
