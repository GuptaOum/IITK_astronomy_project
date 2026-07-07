"""
Fair v3 vs v4 comparison, per galaxy inclination.

v3 (grayscale x3) and v4 (morpho-kinematic 3-channel) take DIFFERENT image
representations, so we CANNOT feed one model the other's images. Instead each
model is scored on its OWN rendering of the SAME test snapshots/views (identical
seed-42 split), then we compare accuracy at each tilt by matching filenames.

Step 1: run the v4 model on v4's val+test images -> per-image predictions,
        threshold tuning, accuracy-vs-tilt (mirrors analyze_results.py for v3).
Step 2: merge v3 + v4 per-image predictions on filename, print per-tilt table,
        save a combined accuracy-vs-tilt comparison plot.

Run:  <venv>/python.exe compare_v3_v4_by_tilt.py
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

ROOT = r"E:/IIT KNP PROJ"
V4_MODEL = os.path.join(ROOT, "bar_cnn_v4", "models_v4", "bar_resnet50.keras")
V4_DATA = os.path.join(ROOT, "bar_cnn_v4", "dataset_v4")
V4_OUT = os.path.join(ROOT, "bar_cnn_v4", "models_v4")
V3_PRED = os.path.join(ROOT, "windowpynbody2", "models_v3", "per_image_predictions.csv")
IMG_SIZE = (224, 224)


def predict_split(model, data_dir, split):
    """DataFrame: file, snapshot, rx, ry, true (1=no_bar), prob."""
    rows = []
    for cls, label in (("bar", 0), ("no_bar", 1)):
        for f in sorted(glob.glob(os.path.join(data_dir, split, cls, "*.png"))):
            m = re.search(r"(snapshot_\d+)_X(\d+)_Y(\d+)", os.path.basename(f))
            rows.append({"file": os.path.basename(f), "snapshot": m.group(1),
                         "rx": int(m.group(2)), "ry": int(m.group(3)),
                         "true": label})
    df = pd.DataFrame(rows)
    imgs = np.stack([
        keras.utils.img_to_array(
            keras.utils.load_img(os.path.join(data_dir, split,
                                 ("bar", "no_bar")[r.true], r.file),
                                 color_mode="rgb", target_size=IMG_SIZE))
        for r in df.itertuples()])
    df["prob"] = model.predict(imgs, batch_size=32, verbose=1).ravel()
    return df


def balanced_acc(df, thr):
    pred = (df.prob > thr).astype(int)
    a_bar = ((pred == 0) & (df.true == 0)).sum() / (df.true == 0).sum()
    a_nobar = ((pred == 1) & (df.true == 1)).sum() / (df.true == 1).sum()
    return (a_bar + a_nobar) / 2, a_bar, a_nobar


# ---------------- Step 1: v4 inference ----------------
print("Loading v4 model...")
model = keras.models.load_model(V4_MODEL)
print("Predicting v4 val (threshold tuning)...")
v4_val = predict_split(model, V4_DATA, "val")
print("Predicting v4 test...")
v4_test = predict_split(model, V4_DATA, "test")
v4_test.to_csv(os.path.join(V4_OUT, "per_image_predictions.csv"), index=False)

thresholds = np.arange(0.05, 0.96, 0.01)
best_thr = float(thresholds[int(np.argmax([balanced_acc(v4_val, t)[0]
                                           for t in thresholds]))])
lines = [f"v4 best threshold on validation: {best_thr:.2f}"]
for name, thr in (("default 0.50", 0.5), (f"tuned {best_thr:.2f}", best_thr)):
    bal, a_bar, a_nobar = balanced_acc(v4_test, thr)
    overall = ((v4_test.prob > thr).astype(int) == v4_test.true).mean()
    lines.append(f"v4 TEST @ {name}: overall={overall:.4f} balanced={bal:.4f} "
                 f"bar-recall={a_bar:.4f} no_bar-recall={a_nobar:.4f}")
with open(os.path.join(V4_OUT, "threshold_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------- Step 2: merge with v3 and compare by tilt ----------------
v3 = pd.read_csv(V3_PRED)
v3["file"] = v3["file"].apply(lambda p: os.path.basename(str(p)))
v3 = v3.rename(columns={"prob": "prob_v3"})
v4 = v4_test.rename(columns={"prob": "prob_v4"})

m = v3.merge(v4[["file", "prob_v4"]], on="file", how="inner")
print(f"\nMatched {len(m)} test images common to v3 and v4 "
      f"(v3 test={len(v3)}, v4 test={len(v4)})")

m["max_tilt"] = m[["rx", "ry"]].max(axis=1)
m["ok_v3"] = (m.prob_v3 > 0.5).astype(int) == m.true
m["ok_v4"] = (m.prob_v4 > 0.5).astype(int) == m.true

tab = m.groupby("max_tilt").agg(n=("true", "size"),
                                v3_acc=("ok_v3", "mean"),
                                v4_acc=("ok_v4", "mean"))
tab["delta"] = tab.v4_acc - tab.v3_acc

print("\n=== Accuracy vs tilt (threshold 0.5, matched test images) ===")
print(tab.round(4).to_string())
print(f"\nOVERALL  v3={m.ok_v3.mean():.4f}  v4={m.ok_v4.mean():.4f}  "
      f"delta={m.ok_v4.mean()-m.ok_v3.mean():+.4f}")

# barred-only breakdown (kinematics should help most for real bars at high tilt)
bar = m[m.true == 0]
btab = bar.groupby("max_tilt").agg(n=("true", "size"),
                                   v3=("ok_v3", "mean"), v4=("ok_v4", "mean"))
btab["delta"] = btab.v4 - btab.v3
print("\n=== BARRED galaxies only, accuracy vs tilt ===")
print(btab.round(4).to_string())

# plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tab.index, tab.v3_acc, "o-", color="royalblue", label="v3 (grayscale)")
ax.plot(tab.index, tab.v4_acc, "s-", color="crimson", label="v4 (morpho-kinematic)")
ax.set_xlabel("Max tilt angle  max(X, Y)  [degrees]")
ax.set_ylabel("Classification accuracy (threshold 0.5)")
ax.set_title("v3 vs v4: bar-detection accuracy vs inclination (same test snapshots)")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out_png = os.path.join(ROOT, "bar_cnn_v4", "v3_vs_v4_accuracy_vs_tilt.png")
fig.savefig(out_png, dpi=150)
m.to_csv(os.path.join(ROOT, "bar_cnn_v4", "v3_vs_v4_matched.csv"), index=False)
print(f"\nSaved plot: {out_png}")
print(f"Saved matched CSV: {os.path.join(ROOT, 'bar_cnn_v4', 'v3_vs_v4_matched.csv')}")
