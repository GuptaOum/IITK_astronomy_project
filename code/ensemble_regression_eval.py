"""
Final regression ensemble + TTA evaluation. Run on the GPU box.

Loads the 3 trained regression models (resnet50 via rebuild+weights because of
the Keras save bug; the others from model.keras), predicts the test set with
8-view TTA, and evaluates every combination: each model alone/+TTA, ensembles
of all-3 and best-2, both plain and TTA. Classification = pred_a2a0 >= 0.19 on
the v3-test subset (450 imgs), directly comparable to the 86.4% binary baseline.

Also saves pred_vs_true.png (the report's scatter figure) + ensemble preds CSV.
"""
import glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG = (224, 224)
A2A0_SCALE = 0.30
CLS_THR = 0.19

BACKBONES = {
    "resnet50":      (keras.applications.ResNet50,
                      keras.applications.resnet50.preprocess_input),
    "effnetv2b0":    (keras.applications.EfficientNetV2B0,
                      keras.applications.efficientnet_v2.preprocess_input),
    "convnext_tiny": (keras.applications.ConvNeXtTiny,
                      keras.applications.convnext.preprocess_input),
}

def build(arch):
    Backbone, preprocess = BACKBONES[arch]
    base = Backbone(weights=None, include_top=False, input_shape=IMG + (3,))
    inputs = keras.Input(shape=IMG + (3,))
    x = preprocess(inputs)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)

def load_model(arch):
    d = f"models_reg_{arch}"
    if os.path.exists(f"{d}/model.keras"):
        try:
            return keras.models.load_model(f"{d}/model.keras")
        except Exception:
            pass
    # rebuild WITHOUT augmentation layer (inference), load weights by name
    m = build(arch)
    m.load_weights(f"{d}/model.weights.h5", skip_mismatch=True, by_name=False) \
        if False else None
    # weights were saved from a model containing an augmentation Sequential;
    # rebuild the same topology (augment is weightless, add identity in its place)
    Backbone, preprocess = BACKBONES[arch]
    base = Backbone(weights=None, include_top=False, input_shape=IMG + (3,))
    inputs = keras.Input(shape=IMG + (3,))
    x = keras.Sequential([layers.Identity()])(inputs)   # placeholder for augment
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inputs, out)
    m.load_weights(f"{d}/model.weights.h5")
    return m

# ---------------- test data ----------------
labels_df = pd.read_csv("extras/bar_labels.csv").set_index("snapshot")
files = sorted(glob.glob("dataset/test/*/*.png")) + sorted(glob.glob("dataset_mid/test/*.png"))
y = np.array([labels_df.loc[re.search(r"snapshot_\d+", os.path.basename(f)).group(),
                            "peak_a2a0"] for f in files], dtype=np.float32)
is_v3 = np.array([f.startswith("dataset/") or f.startswith("dataset\\") for f in files])
imgs = np.stack([keras.utils.img_to_array(
    keras.utils.load_img(f, color_mode="rgb", target_size=IMG)) for f in files])
print(f"test: {len(files)} imgs ({is_v3.sum()} v3-subset)")

def tta_views(x):
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        out.append(r); out.append(r[:, :, ::-1, :])
    return out

single, tta = {}, {}
for arch in BACKBONES:
    m = load_model(arch)
    single[arch] = m.predict(imgs, batch_size=32, verbose=0).ravel() * A2A0_SCALE
    tta[arch] = np.mean([m.predict(v, batch_size=32, verbose=0).ravel()
                         for v in tta_views(imgs)], axis=0) * A2A0_SCALE
    print(f"loaded+predicted {arch}")
    del m
    keras.backend.clear_session()

def report(name, pred):
    mae = np.abs(pred - y).mean()
    t, p = (y[is_v3] >= CLS_THR), (pred[is_v3] >= CLS_THR)
    acc = (t == p).mean()
    bar_rec = (p[t] == True).mean()
    line = f"{name:32s} MAE={mae:.4f}  cls-acc={acc*100:5.1f}%  bar-recall={bar_rec*100:.1f}%"
    print(line); return line

lines = []
print(); lines.append("--- single / +TTA ---")
for a in BACKBONES:
    lines.append(report(a, single[a]))
    lines.append(report(a + " +TTA", tta[a]))
print(); lines.append("--- ensembles ---")
lines.append(report("ens3 (all)", np.mean([single[a] for a in BACKBONES], axis=0)))
lines.append(report("ens3 +TTA", np.mean([tta[a] for a in BACKBONES], axis=0)))
best2 = ["resnet50", "convnext_tiny"]
lines.append(report("ens2 (rn50+convnext)", np.mean([single[a] for a in best2], axis=0)))
lines.append(report("ens2 +TTA", np.mean([tta[a] for a in best2], axis=0)))

with open("ensemble_regression_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# ---------------- scatter: predicted vs true A2/A0 ----------------
pred_best = np.mean([tta[a] for a in best2], axis=0)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y[~is_v3], pred_best[~is_v3], s=14, alpha=0.5, color="orange",
           label="mid-strength (regression-only)")
ax.scatter(y[is_v3], pred_best[is_v3], s=14, alpha=0.5, color="royalblue",
           label="v3 test (clear bar / no-bar)")
lim = [0, 0.30]
ax.plot(lim, lim, "k--", lw=1, label="perfect prediction")
ax.axvline(CLS_THR, color="gray", ls=":", lw=1)
ax.axhline(CLS_THR, color="gray", ls=":", lw=1)
ax.set_xlabel("True A2/A0 (physical measurement)")
ax.set_ylabel("CNN-predicted A2/A0 (ens2 + TTA)")
ax.set_title("CNN measures bar strength from a single image")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig("pred_vs_true.png", dpi=150)
np.save("ensemble_test_pred.npy", pred_best)
print("\nsaved ensemble_regression_report.txt, pred_vs_true.png")
