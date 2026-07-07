"""
End-to-end bar prediction with the final 3-seed ensemble + TTA (the 91.8% model).

Give it EITHER a raw GADGET snapshot OR an already-rendered PNG:

    python code/predict.py snapshot_501
    python code/predict.py dataset/test/bar/snapshot_474_X00_Y00.png
    python code/predict.py snapshot_490 snapshot_120   # several at once

For a raw snapshot it renders the density image exactly the way the training
data was made (same alignment, same 1e7-1e9 log clip, same grayscale) so there
is no training/serving skew. Then it runs the full pipeline:
  3 models (seed42/1/2)  x  8 TTA views (rotations+flips)  = 24 predictions,
averaged into one bar / no-bar decision with a confidence.

Output class convention: 0 = bar, 1 = no_bar (alphabetical, as trained).
Run from the windowpynbody2 folder.
"""
import sys, os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

MODELS = ["results/v3_binary/bar_resnet50.keras",
          "results/v3_binary_seed1/bar_resnet50.keras",
          "results/v3_binary_seed2/bar_resnet50.keras"]
IMG_SIZE = 224
WIDTH = 40.0            # kpc, must match training
VMIN, VMAX = 1e7, 1e9   # density clip, must match training


# ---- render a raw snapshot into the exact training image format ----
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
    """Load a PNG the same way training/eval did -> (1,224,224,3) float 0-255.

    Safety check: the model was trained on GRAYSCALE density images (all 3
    channels equal). A colored input (e.g. a berlin/viridis colormap render)
    encodes density in a way the model never saw -> unreliable predictions.
    """
    arr = keras.utils.img_to_array(
        keras.utils.load_img(path, color_mode="rgb", target_size=(IMG_SIZE, IMG_SIZE)))
    max_channel_spread = np.abs(arr.max(axis=-1) - arr.min(axis=-1)).max()
    if max_channel_spread > 2:   # tolerate tiny PNG compression artifacts
        print(f"  WARNING: '{os.path.basename(path)}' looks COLORED "
              f"(channel spread {max_channel_spread:.0f}/255). The model was "
              f"trained on grayscale density images - this prediction is "
              f"UNRELIABLE. Re-render the snapshot with predict.py instead.")
    return arr[None, ...]


def tta_views(x):
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        out.append(r); out.append(r[:, :, ::-1, :])
    return np.concatenate(out, axis=0)   # (8,224,224,3)


def main(paths):
    print("Loading 3 ensemble models...")
    models = [keras.models.load_model(m) for m in MODELS]

    for path in paths:
        # get a PNG: use directly if already an image, else render the snapshot
        if path.lower().endswith(".png"):
            png = path
            tmp = None
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            print(f"Rendering {path} ...")
            render_snapshot_to_png(path, tmp)
            png = tmp

        views = tta_views(load_image(png))            # 8 augmented views
        # 3 models x 8 views -> average of 24 probabilities of class "no_bar"
        probs = [m.predict(views, batch_size=8, verbose=0).ravel() for m in models]
        p_nobar = float(np.mean(probs))

        if p_nobar < 0.5:
            verdict, conf = "BAR", 1 - p_nobar
        else:
            verdict, conf = "NO BAR", p_nobar
        print(f"{os.path.basename(path):40s} -> {verdict:7s}  "
              f"(confidence {conf*100:.1f}%,  p_nobar={p_nobar:.3f})")

        if tmp:
            os.unlink(tmp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
