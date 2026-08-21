"""
Generate the complete figure pack for the report/presentation.

Produces every standard ML graph for the FINAL regression ResNet50 model,
plus the project-story figures. Run on the GPU EC2 box (or anywhere with
numpy/pandas/matplotlib + the small data files):

    python3 make_all_graphs.py

Outputs into figures/:
  A. training:   01_loss_curves, 02_metric_curves, 03_binary_training_curves
  B. classify:   04_confusion_matrix, 05_roc_curve, 06_precision_recall,
                 07_accuracy_vs_threshold
  C. regression: 08_pred_vs_true, 09_residuals, 10_error_histogram
  D. story:      11_results_ladder, 12_accuracy_vs_tilt
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
A2A0_SCALE, THR = 0.30, 0.19
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)
    print("saved", name)

# ---------------------------------------------------------------- load data
# works both locally (repo layout) and on EC2 (flat files in cwd)
def find(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(candidates[0])

R = "results/regression_resnet50"
log = pd.read_csv(find(f"{R}/training_log.csv", "training_log.csv"))
n_phase1 = 8                        # phase 1 = 8 epochs, then phase 2 restarts
log["step"] = range(1, len(log) + 1)

pred = np.load(find(f"{R}/test_pred.npy", "test_pred.npy")).ravel() * A2A0_SCALE
files = [f for f in open(find(f"{R}/test_files.txt", "test_files.txt")
                         ).read().split("\n") if f.strip()]
lab = pd.read_csv(find("extras/bar_labels.csv", "bar_labels.csv")).set_index("snapshot")
snaps = [re.search(r"snapshot_\d+", os.path.basename(f)).group() for f in files]
true = np.array([lab.loc[s, "peak_a2a0"] for s in snaps])
tilt = np.array([max(int(m.group(1)), int(m.group(2))) for m in
                 (re.search(r"_X(\d+)_Y(\d+)", f) for f in files)])
y_true = (true >= THR).astype(int)          # 1 = barred
y_pred = (pred >= THR).astype(int)
err = pred - true

# =============================================== A. TRAINING-PROCESS GRAPHS
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(log.step, log.loss, "o-", ms=3, label="training loss")
ax.plot(log.step, log.val_loss, "s-", ms=3, label="validation loss")
ax.axvline(n_phase1 + 0.5, color="gray", ls="--", lw=1)
ax.text(n_phase1 + 0.7, ax.get_ylim()[1] * 0.9, "phase 2:\nfine-tuning", fontsize=8)
ax.set_xlabel("epoch"); ax.set_ylabel("loss (Huber)")
ax.set_title("Training & validation loss — how wrong the model is over time")
ax.legend(); save(fig, "01_loss_curves")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(log.step, log.mae * A2A0_SCALE, "o-", ms=3, label="training MAE")
ax.plot(log.step, log.val_mae * A2A0_SCALE, "s-", ms=3, label="validation MAE")
ax.axvline(n_phase1 + 0.5, color="gray", ls="--", lw=1)
ax.set_xlabel("epoch"); ax.set_ylabel("mean absolute error (A2/A0 units)")
ax.set_title("Prediction error per epoch — lower is better")
ax.legend(); save(fig, "02_metric_curves")

BIN = next((p for p in ("results/v3_binary/training_log.csv",
                        "binary_training_log.csv") if os.path.exists(p)), None)
if BIN:
    b = pd.read_csv(BIN); b["step"] = range(1, len(b) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(b.step, b.loss, "o-", ms=3, label="train")
    axes[0].plot(b.step, b.val_loss, "s-", ms=3, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title("Binary model — loss")
    axes[1].plot(b.step, b.acc, "o-", ms=3, label="train acc")
    axes[1].plot(b.step, b.val_acc, "s-", ms=3, label="val acc")
    axes[1].plot(b.step, b.val_auc, "^--", ms=3, label="val AUC")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].legend()
    axes[1].set_title("Binary model — accuracy / AUC")
    save(fig, "03_binary_training_curves")

# ================================================ B. CLASSIFICATION GRAPHS
tp = int(((y_pred == 1) & (y_true == 1)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/cm.sum()*100:.1f}%)", ha="center",
                va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xticks([0, 1], ["predicted\nNO BAR", "predicted\nBAR"])
ax.set_yticks([0, 1], ["actually\nNO BAR", "actually\nBAR"])
ax.set_title(f"Confusion matrix — accuracy {(tp+tn)/cm.sum()*100:.1f}%")
ax.grid(False); save(fig, "04_confusion_matrix")

# ROC: sweep threshold over predicted strength
order = np.argsort(-pred)
ys = y_true[order]
tpr = np.cumsum(ys) / max(ys.sum(), 1)
fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
tpr, fpr = np.r_[0, tpr, 1], np.r_[0, fpr, 1]
auc = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr, fpr)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, lw=2, label=f"model (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guessing (AUC = 0.5)")
ax.set_xlabel("false-alarm rate"); ax.set_ylabel("bar-detection rate")
ax.set_title("ROC curve — catching bars vs raising false alarms")
ax.legend(loc="lower right"); save(fig, "05_roc_curve")

prec, rec = [], []
for t in np.linspace(pred.min(), pred.max(), 200):
    p = (pred >= t).astype(int)
    if p.sum() == 0: continue
    prec.append((p & y_true).sum() / p.sum()); rec.append((p & y_true).sum() / y_true.sum())
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(rec, prec, lw=2)
ax.axhline(y_true.mean(), color="k", ls="--", lw=1, label="always-guess-bar baseline")
ax.set_xlabel("recall (fraction of real bars found)")
ax.set_ylabel("precision (fraction of 'bar' calls that are right)")
ax.set_title("Precision–Recall curve"); ax.legend(); save(fig, "06_precision_recall")

ths = np.linspace(0.05, 0.28, 200)
accs = [((pred >= t).astype(int) == y_true).mean() for t in ths]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ths, np.array(accs) * 100, lw=2)
ax.axvline(THR, color="crimson", ls="--", lw=1.5,
           label=f"physical threshold {THR} → {((pred>=THR).astype(int)==y_true).mean()*100:.1f}%")
ax.set_xlabel("decision threshold on predicted A2/A0")
ax.set_ylabel("accuracy (%)")
ax.set_title("Accuracy vs decision threshold")
ax.legend(); save(fig, "07_accuracy_vs_threshold")

# =================================================== C. REGRESSION GRAPHS
is_v3 = np.array([f.startswith("dataset/") or f.startswith("dataset\\") for f in files])
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(true[~is_v3], pred[~is_v3], s=12, alpha=.5, color="orange",
           label="mid-strength (regression-only)")
ax.scatter(true[is_v3], pred[is_v3], s=12, alpha=.5, color="royalblue",
           label="clear bar / no-bar")
ax.plot([0, .3], [0, .3], "k--", lw=1, label="perfect prediction")
ax.axvline(THR, color="gray", ls=":", lw=1); ax.axhline(THR, color="gray", ls=":", lw=1)
ax.set_xlim(0, .3); ax.set_ylim(0, .3)
ax.set_xlabel("true A2/A0 (physics)"); ax.set_ylabel("predicted A2/A0 (CNN)")
ax.set_title(f"Predicted vs true bar strength — MAE {np.abs(err).mean():.4f}")
ax.legend(); save(fig, "08_pred_vs_true")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(true, err, s=12, alpha=.5, color="teal")
ax.axhline(0, color="k", ls="--", lw=1)
ax.axhline(err.mean(), color="crimson", ls=":", lw=1.5,
           label=f"mean bias = {err.mean():+.4f}")
ax.set_xlabel("true A2/A0"); ax.set_ylabel("prediction error (predicted − true)")
ax.set_title("Residual plot — is the model biased anywhere?")
ax.legend(); save(fig, "09_residuals")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(err, bins=45, color="steelblue", edgecolor="white")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("prediction error (A2/A0 units)"); ax.set_ylabel("number of images")
ax.set_title(f"Error distribution — {(np.abs(err) < 0.03).mean()*100:.0f}% of predictions within ±0.03")
save(fig, "10_error_histogram")

# ======================================================= D. STORY GRAPHS
names = ["v1\nbinary", "v2\nbinary", "v3\nbinary", "v3\n+TTA", "v3 +TTA\n+ensemble",
         "v4\nkinematic", "REGRESSION\n(final)"]
vals = [76.5, 86.8, 86.4, 90.2, 91.8, 83.3, 96.0]
cols = ["#b0bec5"]*6 + ["#2e7d32"]
fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.bar(names, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v}", ha="center", fontsize=9)
ax.set_ylim(70, 100); ax.set_ylabel("test accuracy (%)")
ax.set_title("Project journey — accuracy across all experiments")
save(fig, "11_results_ladder")

fig, ax = plt.subplots(figsize=(7, 4.5))
tl = sorted(set(tilt))
acc_t = [((y_pred == y_true)[tilt == t]).mean()*100 for t in tl]
mae_t = [np.abs(err)[tilt == t].mean() for t in tl]
ax.plot(tl, acc_t, "o-", color="black", label="accuracy (%)")
ax.set_xlabel("galaxy tilt angle (degrees)"); ax.set_ylabel("accuracy (%)")
ax.set_ylim(70, 102)
ax2 = ax.twinx(); ax2.plot(tl, mae_t, "s--", color="crimson", label="MAE")
ax2.set_ylabel("MAE (A2/A0)", color="crimson"); ax2.grid(False)
ax.set_title("Performance vs galaxy inclination — the physics limit")
ax.legend(loc="lower left"); ax2.legend(loc="upper right")
save(fig, "12_accuracy_vs_tilt")

print(f"\nAll figures written to {OUT}/")
