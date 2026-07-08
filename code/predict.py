"""
Bar prediction with the FINAL model: ResNet50 A2/A0 regression (96.0% acc,
MAE 0.027). Outputs the predicted bar STRENGTH plus a bar / no-bar verdict
(threshold A2/A0 >= 0.19).

Give it EITHER a raw GADGET snapshot OR a rendered grayscale density PNG:

    python code/predict.py gadget_snapshots/snapshot_501
    python code/predict.py data/dataset_v3_binary/test/bar/snapshot_474_X00_Y00.png
    python code/predict.py gadget_snapshots/snapshot_490 gadget_snapshots/snapshot_120

For a raw snapshot it renders the density image with the exact training
recipe (no training/serving skew). Prediction averages 8 TTA views
(rotations/flips) for a steadier strength estimate.
Run from the windowpynbody2 folder.
"""
import sys, os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

WEIGHTS = "results/regression_resnet50/model.weights.h5"
IMG_SIZE = 224
WIDTH = 40.0            # kpc, must match training
VMIN, VMAX = 1e7, 1e9   # density clip, must match training
A2A0_SCALE = 0.30
CLS_THR = 0.19


def build_model():
    """Rebuild the training topology (full .keras save hits a Keras bug)."""
    base = keras.applications.ResNet50(weights=None, include_top=False,
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.Sequential([layers.Identity()])(inputs)  # augment placeholder
    x = keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inputs, out)
    m.load_weights(WEIGHTS)
    return m


def render_snapshot_to_png(snap_path, out_png):
    import pynbody
    import pynbody.analysis.halo as halo
    import pynbody.analysis.angmom as angmom
    import pynbody.plot.sph as sph
    sim = pynbody.load(snap_path)
    sim.physical_units()
    sim["pos"] *= 1.0 / 1000.0
    halo.center(sim)
    angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
    im = sph.image(sim.star, qty="rho", width=WIDTH, resolution=IMG_SIZE,
                   log=False, noplot=True, show_cbar=False, approximate_fast=False)
    im = np.log10(np.clip(im, VMIN, VMAX))
    im = (im - np.log10(VMIN)) / (np.log10(VMAX) - np.log10(VMIN))
    plt.imsave(out_png, im, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")


def load_image(path):
    arr = keras.utils.img_to_array(
        keras.utils.load_img(path, color_mode="rgb", target_size=(IMG_SIZE, IMG_SIZE)))
    spread = np.abs(arr.max(axis=-1) - arr.min(axis=-1)).max()
    if spread > 2:
        print(f"  WARNING: '{os.path.basename(path)}' looks COLORED "
              f"(channel spread {spread:.0f}/255) - the model was trained on "
              f"grayscale density images; this prediction is UNRELIABLE.")
    return arr[None, ...]


def tta_views(x):
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        out.append(r); out.append(r[:, :, ::-1, :])
    return np.concatenate(out, axis=0)


def main(paths):
    print("Loading final regression model...")
    model = build_model()
    for path in paths:
        if path.lower().endswith(".png"):
            png, tmp = path, None
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            print(f"Rendering {path} ...")
            render_snapshot_to_png(path, tmp)
            png = tmp
        views = tta_views(load_image(png))
        a2a0 = float(np.mean(model.predict(views, batch_size=8, verbose=0))) * A2A0_SCALE
        verdict = "BAR" if a2a0 >= CLS_THR else "NO BAR"
        print(f"{os.path.basename(path):40s} -> {verdict:7s}  "
              f"(predicted bar strength A2/A0 = {a2a0:.3f}, threshold {CLS_THR})")
        if tmp:
            os.unlink(tmp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
