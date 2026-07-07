"""
Hybrid ensemble: best-2 BINARY models (v3 seed42 + seed1) + REGRESSION ResNet50.

The three answer differently (binary: P(no_bar) thr 0.5; regression: A2/A0 thr
0.19), so each model's TTA score is calibrated to a common P(bar) via Platt
scaling (1D logistic fit) on the v3 VALIDATION set, then averaged. Also reports
a simple 2-of-3 majority vote. Evaluated on the v3 test set (450 imgs) -
comparable to: binary v3 86.4, ens+TTA 91.8, regression rn50 96.0.

Run from windowpynbody2 (CPU, ~20 min):  python code/hybrid_ensemble_eval.py
"""
import glob, os
import numpy as np
from scipy.optimize import minimize
from tensorflow import keras
from tensorflow.keras import layers

IMG = (224, 224)

def load_files(split):
    files, labels = [], []
    for cls, lab in (("bar", 1), ("no_bar", 0)):     # bar = positive = 1 here
        for f in sorted(glob.glob(f"data/dataset_v3_binary/{split}/{cls}/*.png")):
            files.append(f); labels.append(lab)
    return files, np.array(labels)

def load_imgs(files):
    return np.stack([keras.utils.img_to_array(
        keras.utils.load_img(f, color_mode="rgb", target_size=IMG)) for f in files])

def tta_pred(model, imgs):
    outs = []
    for k in range(4):
        r = np.rot90(imgs, k, axes=(1, 2))
        outs.append(model.predict(r, batch_size=32, verbose=0).ravel())
        outs.append(model.predict(r[:, :, ::-1, :], batch_size=32, verbose=0).ravel())
    return np.mean(outs, axis=0)

def build_regression_rn50():
    base = keras.applications.ResNet50(weights=None, include_top=False,
                                       input_shape=IMG + (3,))
    inputs = keras.Input(shape=IMG + (3,))
    x = keras.Sequential([layers.Identity()])(inputs)   # augment placeholder
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inputs, out)
    m.load_weights("results/regression_resnet50/model.weights.h5")
    return m

def platt(scores, labels):
    """Fit P(bar) = sigmoid(a*score + b) on validation; return (a, b)."""
    def nll(p):
        z = np.clip(1/(1+np.exp(-(p[0]*scores + p[1]))), 1e-7, 1-1e-7)
        return -np.mean(labels*np.log(z) + (1-labels)*np.log(1-z))
    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return r.x

def apply_platt(scores, p):
    return 1/(1+np.exp(-(p[0]*scores + p[1])))

# ---------------- data ----------------
val_f, val_y = load_files("val"); test_f, test_y = load_files("test")
val_imgs, test_imgs = load_imgs(val_f), load_imgs(test_f)
print(f"val={len(val_f)} test={len(test_f)}")

# ---------------- models -> raw bar-scores (higher = more bar) ----------------
raw_val, raw_test, thr_native = {}, {}, {}
for name, path in (("binary_seed42", "results/v3_binary/bar_resnet50.keras"),
                   ("binary_seed1", "results/v3_binary_seed1/bar_resnet50.keras")):
    m = keras.models.load_model(path)
    raw_val[name] = 1 - tta_pred(m, val_imgs)      # 1 - P(no_bar)
    raw_test[name] = 1 - tta_pred(m, test_imgs)
    thr_native[name] = 0.5
    del m; keras.backend.clear_session(); print(f"{name} predicted")

m = build_regression_rn50()
raw_val["regression_rn50"] = tta_pred(m, val_imgs) * 0.30       # -> A2/A0 units
raw_test["regression_rn50"] = tta_pred(m, test_imgs) * 0.30
thr_native["regression_rn50"] = 0.19
del m; keras.backend.clear_session(); print("regression_rn50 predicted")

# ---------------- calibrate on val, evaluate on test ----------------
def acc(pred_bar, y): return (pred_bar == y).mean()
def rec(pred_bar, y): return pred_bar[y == 1].mean()

lines = []
cal_test = {}
for name in raw_val:
    p = platt(raw_val[name], val_y)
    cal_test[name] = apply_platt(raw_test[name], p)
    native = (raw_test[name] >= thr_native[name]).astype(int)
    lines.append(f"{name:18s} native-thr acc={acc(native, test_y)*100:5.1f}% "
                 f"bar-recall={rec(native, test_y)*100:5.1f}%")

hybrid = np.mean([cal_test[n] for n in cal_test], axis=0)
hy = (hybrid >= 0.5).astype(int)
lines.append(f"{'HYBRID avg-calibrated':18s} acc={acc(hy, test_y)*100:5.1f}% "
             f"bar-recall={rec(hy, test_y)*100:5.1f}%")

votes = np.sum([(raw_test[n] >= thr_native[n]).astype(int) for n in raw_test], axis=0)
mv = (votes >= 2).astype(int)
lines.append(f"{'HYBRID majority-vote':18s} acc={acc(mv, test_y)*100:5.1f}% "
             f"bar-recall={rec(mv, test_y)*100:5.1f}%")

print(); print("\n".join(lines))
with open("results/regression_ensemble/hybrid_ensemble_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nsaved results/regression_ensemble/hybrid_ensemble_report.txt")
