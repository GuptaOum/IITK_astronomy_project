"""
Post-training analysis (CPU, inference only - no retraining):

1. Accuracy vs tilt angle - predicts on every test image, groups results by
   the X/Y tilt encoded in the filename, and plots accuracy per inclination.
2. Decision-threshold tuning - sweeps the 0..1 cutoff on the VALIDATION set,
   picks the threshold that maximizes balanced accuracy, then reports test
   metrics at both 0.5 and the tuned threshold.

Run from the windowpynbody2 folder:
    python code/analyze_results.py

Outputs into models/:
    accuracy_vs_tilt.png, threshold_report.txt, per_image_predictions.csv
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

MODEL_PATH = "results/v3_binary/bar_resnet50.keras"
DATA_DIR = "data/dataset_v3_binary"
OUT_DIR = "models_v3"
IMG_SIZE = (224, 224)

model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")


def predict_split(split):
    """Return DataFrame: file, snapshot, rx, ry, true (1=no_bar), prob."""
    rows = []
    for cls, label in (("bar", 0), ("no_bar", 1)):
        files = sorted(glob.glob(os.path.join(DATA_DIR, split, cls, "*.png")))
        for f in files:
            m = re.search(r"(snapshot_\d+)_X(\d+)_Y(\d+)", os.path.basename(f))
            rows.append({"file": f, "snapshot": m.group(1),
                         "rx": int(m.group(2)), "ry": int(m.group(3)),
                         "true": label})
    df = pd.DataFrame(rows)
    imgs = np.stack([
        keras.utils.img_to_array(
            keras.utils.load_img(f, color_mode="rgb", target_size=IMG_SIZE))
        for f in df.file
    ])
    df["prob"] = model.predict(imgs, batch_size=32, verbose=1).ravel()
    return df


print("\nPredicting on validation set (for threshold tuning)...")
val = predict_split("val")
print("Predicting on test set...")
test = predict_split("test")
test.to_csv(os.path.join(OUT_DIR, "per_image_predictions.csv"), index=False)

# ------------------------------------------------------------------
# 1. Accuracy vs tilt (test set, threshold 0.5)
# ------------------------------------------------------------------
test["max_tilt"] = test[["rx", "ry"]].max(axis=1)
test["correct_05"] = ((test.prob > 0.5).astype(int) == test.true)

by_tilt = test.groupby("max_tilt").agg(
    accuracy=("correct_05", "mean"), n=("correct_05", "size"))
by_tilt_bar = test[test.true == 0].groupby("max_tilt")["correct_05"].mean()
by_tilt_nobar = test[test.true == 1].groupby("max_tilt")["correct_05"].mean()

print("\nAccuracy by max tilt angle (test set):")
print(by_tilt)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(by_tilt.index, by_tilt.accuracy, "o-", color="black", label="Overall")
ax.plot(by_tilt_bar.index, by_tilt_bar.values, "s--", color="crimson",
        label="Barred galaxies")
ax.plot(by_tilt_nobar.index, by_tilt_nobar.values, "^--", color="royalblue",
        label="Unbarred galaxies")
ax.set_xlabel("Max tilt angle  max(X, Y)  [degrees]")
ax.set_ylabel("Classification accuracy")
ax.set_title("CNN bar detection accuracy vs galaxy inclination")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random guessing")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "accuracy_vs_tilt.png"), dpi=150)
print(f"Saved {OUT_DIR}/accuracy_vs_tilt.png")

# ------------------------------------------------------------------
# 2. Threshold tuning on VALIDATION, evaluated on TEST
# ------------------------------------------------------------------
def balanced_acc(df, thr):
    pred = (df.prob > thr).astype(int)
    acc_bar = ((pred == 0) & (df.true == 0)).sum() / (df.true == 0).sum()
    acc_nobar = ((pred == 1) & (df.true == 1)).sum() / (df.true == 1).sum()
    return (acc_bar + acc_nobar) / 2, acc_bar, acc_nobar

thresholds = np.arange(0.05, 0.96, 0.01)
val_scores = [balanced_acc(val, t)[0] for t in thresholds]
best_thr = float(thresholds[int(np.argmax(val_scores))])

lines = [f"Best threshold on validation: {best_thr:.2f} "
         f"(balanced acc {max(val_scores):.4f})\n"]
for name, thr in (("default 0.50", 0.5), (f"tuned {best_thr:.2f}", best_thr)):
    bal, a_bar, a_nobar = balanced_acc(test, thr)
    overall = ((test.prob > thr).astype(int) == test.true).mean()
    lines.append(
        f"TEST @ {name}: overall={overall:.4f} balanced={bal:.4f} "
        f"bar-recall={a_bar:.4f} no_bar-recall={a_nobar:.4f}")
report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(OUT_DIR, "threshold_report.txt"), "w") as f:
    f.write(report + "\n")
print(f"\nSaved {OUT_DIR}/threshold_report.txt")
