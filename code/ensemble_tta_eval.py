"""
Final accuracy push: 3-model ensemble + test-time augmentation (TTA).

Combines two independent noise-cancellers, evaluated on the v3 test set:
  - Ensemble: average predictions of 3 models trained on the same v3 dataset
    with different random seeds (42 = the v3 model, 1, 2). Their individual
    mistakes differ, so averaging cancels them.
  - TTA: for each image, average predictions over 8 rotations/flips (a bar is
    orientation-invariant), cancelling the model's per-orientation quirks.

Reports every combination so we can see what each technique buys.
Run from windowpynbody2:  python code/ensemble_tta_eval.py
"""
import glob, os
import numpy as np
from tensorflow import keras

MODELS = {
    "seed42(v3)": "models_v3/bar_resnet50.keras",
    "seed1":      "models_seed1/bar_resnet50.keras",
    "seed2":      "models_seed2/bar_resnet50.keras",
}
TEST = "dataset/test"
IMG = (224, 224)

# load test images once
files, labels = [], []
for cls, lab in (("bar", 0), ("no_bar", 1)):
    for f in sorted(glob.glob(os.path.join(TEST, cls, "*.png"))):
        files.append(f); labels.append(lab)
labels = np.array(labels)
imgs = np.stack([keras.utils.img_to_array(
    keras.utils.load_img(f, color_mode="rgb", target_size=IMG)) for f in files])
print(f"test set: {len(files)} images\n")

def views(x):
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        out.append(r); out.append(r[:, :, ::-1, :])
    return out

# per-model probabilities: single-image and TTA
single, tta = {}, {}
for name, path in MODELS.items():
    m = keras.models.load_model(path)
    single[name] = m.predict(imgs, batch_size=32, verbose=0).ravel()
    tta[name] = np.mean([m.predict(v, batch_size=32, verbose=0).ravel()
                         for v in views(imgs)], axis=0)
    print(f"loaded {name}")

def report(name, prob):
    pred = (prob > 0.5).astype(int)
    acc = (pred == labels).mean()
    auc = keras.metrics.AUC()(labels, prob).numpy()
    bar = (pred[labels == 0] == 0).mean()
    print(f"{name:34s} acc={acc*100:5.1f}%  auc={auc:.3f}  bar-recall={bar*100:.1f}%")

print("\n--- single model, single image ---")
for n in MODELS: report(n, single[n])
print("\n--- single model + TTA ---")
for n in MODELS: report(n + " +TTA", tta[n])
print("\n--- ENSEMBLE (avg of 3 models) ---")
report("ensemble, single image", np.mean(list(single.values()), axis=0))
report("ensemble + TTA", np.mean(list(tta.values()), axis=0))
