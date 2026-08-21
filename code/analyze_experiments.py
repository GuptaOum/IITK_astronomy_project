"""
Run the final regression model over the two heavy experiment image sets and
produce the figures. Runs on the EC2 box (after render_experiments.py).

  13_bar_evolution.png   CNN-predicted bar strength across simulation time,
                         overlaid on the true physical A2/A0 curve.
  14_tilt_breakdown.png  Accuracy + error from 0 to 90 degrees inclination,
                         with the <=40 training range shaded — shows exactly
                         where and how the model degrades beyond what it saw.

Also writes the raw predictions as CSV so the plots can be redrawn locally.
"""
import glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

IMG, A2A0_SCALE, THR = (224, 224), 0.30, 0.19
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def build_model():
    base = keras.applications.ResNet50(weights=None, include_top=False,
                                       input_shape=IMG + (3,))
    inp = keras.Input(IMG + (3,))
    x = keras.Sequential([layers.Identity()])(inp)
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.load_weights("regression_model.weights.h5")
    return m


def predict_dir(model, d):
    files = sorted(glob.glob(f"{d}/*.png"))
    if not files:
        return pd.DataFrame()
    imgs = np.stack([keras.utils.img_to_array(
        keras.utils.load_img(f, color_mode="rgb", target_size=IMG)) for f in files])
    pred = model.predict(imgs, batch_size=32, verbose=0).ravel() * A2A0_SCALE
    rows = []
    for f, p in zip(files, pred):
        b = os.path.basename(f)
        rows.append({"snapshot": re.search(r"snapshot_\d+", b).group(),
                     "tilt": int(re.search(r"_X(\d+)_", b).group(1)),
                     "pred": p})
    return pd.DataFrame(rows)


model = build_model()
print("model loaded", flush=True)
lab = pd.read_csv("extras/bar_labels.csv").set_index("snapshot")

# ============================================ EXPERIMENT A: time evolution
evo = predict_dir(model, "exp_evolution")
if len(evo):
    evo["true"] = [lab.loc[s, "peak_a2a0"] for s in evo.snapshot]
    evo["time_gyr"] = [lab.loc[s, "time_gyr"] if "time_gyr" in lab.columns
                       else int(s.split("_")[1]) for s in evo.snapshot]
    evo["num"] = [int(s.split("_")[1]) for s in evo.snapshot]
    evo = evo.sort_values("num")
    evo.to_csv(f"{OUT}/exp_evolution_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(evo.num, evo.true, "-", color="black", lw=2,
            label="true A2/A0 (physics computation)")
    ax.plot(evo.num, evo.pred, "o", color="crimson", ms=4, alpha=.7,
            label="CNN prediction (from image alone)")
    ax.axhline(THR, color="gray", ls=":", lw=1.2, label=f"bar threshold {THR}")
    ax.set_xlabel("snapshot number  (simulation time →)")
    ax.set_ylabel("bar strength  A2/A0")
    ax.set_title("Bar formation across the simulation — CNN vs physics\n"
                 f"(MAE {np.abs(evo.pred-evo.true).mean():.4f} over "
                 f"{len(evo)} snapshots)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/13_bar_evolution.png"); plt.close(fig)
    print("saved 13_bar_evolution", flush=True)

# ============================================== EXPERIMENT B: tilt sweep
sw = predict_dir(model, "exp_tiltsweep")
if len(sw):
    sw["true"] = [lab.loc[s, "peak_a2a0"] for s in sw.snapshot]
    sw["correct"] = (sw.pred >= THR) == (sw.true >= THR)
    sw["abs_err"] = (sw.pred - sw.true).abs()
    sw.to_csv(f"{OUT}/exp_tiltsweep_predictions.csv", index=False)
    g = sw.groupby("tilt").agg(acc=("correct", "mean"), mae=("abs_err", "mean"),
                               n=("correct", "size"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(-2, 40, color="green", alpha=.08)
    ax.text(18, 32, "trained on this range\n(0–40°)", ha="center",
            fontsize=9, color="green")
    ax.plot(g.index, g.acc * 100, "o-", color="black", lw=2, label="accuracy (%)")
    ax.axhline(50, color="gray", ls="--", lw=1, label="random guessing")
    ax.set_xlabel("galaxy inclination (degrees)  —  0° = face-on, 90° = edge-on")
    ax.set_ylabel("classification accuracy (%)")
    ax.set_ylim(30, 103); ax.set_xlim(-2, 92)
    ax2 = ax.twinx()
    ax2.plot(g.index, g.mae, "s--", color="crimson", label="MAE")
    ax2.set_ylabel("mean abs. error in A2/A0", color="crimson"); ax2.grid(False)
    ax.set_title("Where the model breaks — performance beyond its training range")
    ax.legend(loc="lower left"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/14_tilt_breakdown.png"); plt.close(fig)
    print("saved 14_tilt_breakdown", flush=True)
    print(g.round(3).to_string())

print("done")
