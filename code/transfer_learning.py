"""
Transfer Learning for Bar Detection in Galaxy Simulations
Based on Abraham et al. (2018) MNRAS 477, 894-903.

Key techniques from the paper:
  - Rotation-based data augmentation (label-preserving oversampling)
  - Occlusion test to verify CNN focuses on the bar region
  - ROC curve + AUC evaluation
  - Precision/Recall per class

Adaptations for single-simulation data:
  - Temporal-block split to prevent data leakage between consecutive snapshots
  - Offline augmentation to expand the small dataset (502 → thousands)
  - Transfer learning with ResNet18 (paper used AlexNet from scratch on 9346 galaxies;
    we use pretrained weights since our dataset is much smaller)

Reads:
  - cnn_images/snapshot_*.png
  - buckling_plots/bar_labels.csv

Outputs (in transfer_learning_results/):
  - best_model.pth
  - training_curves.png
  - confusion_matrix.png
  - roc_curve.png
  - occlusion_test.png
  - classification_report.txt
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageDraw
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

# ========================== CONFIG ==========================
IMAGE_DIR = "cnn_images"
LABEL_CSV = "buckling_plots/bar_labels.csv"
OUTPUT_DIR = "transfer_learning_results"
AUGMENTED_DIR = os.path.join(OUTPUT_DIR, "augmented_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 25            # Paper converged by ~16 epochs (Section 6)
LEARNING_RATE = 1e-2       # Paper used SGD with 0.01 base LR (Section 6)
VAL_SPLIT = 0.2            # 80/20 split (paper used 60/40 but had 9346 images;
                           # we only have 502 with 80 barred — need more for training)
RANDOM_SEED = 42
FREEZE_BACKBONE = True
FINE_TUNE_AFTER = 10       # Unfreeze earlier — fewer epochs available
FINE_TUNE_LR = 1e-5

# Rotation augmentation: paper rotated each image 360 times (1-degree steps).
# We use fewer since simulation renders have less noise than SDSS observations.
NUM_ROTATIONS = 36         # every 10 degrees (36 rotated copies per unbarred image)
AUGMENT_MINORITY_EXTRA = 5 # extra factor for barred (minority) class
                           # Rationale: ~60 barred in train, ~340 unbarred
                           # After augmentation: 60*(36*5)=10800 vs 340*36=12240 → balanced

# Temporal block split: group consecutive snapshots to prevent leakage.
# Adjacent snapshots are nearly identical in a single simulation.
BLOCK_SIZE = 10            # snapshots per block (e.g., 0-9, 10-19, ...)

# Transition zone: snapshots ~417-429 where A2/A0 oscillates around the 0.19 threshold.
# Labels flip-flop here (417=barred, 418-420=unbarred, 421-422=barred, 423-424=unbarred, 425+=barred).
# These ambiguous samples can confuse the classifier, so we exclude them from training.
EXCLUDE_TRANSITION = True
TRANSITION_RANGE = (415, 430)  # exclude snapshots in this range from training (keep in val for honesty)

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ========================== DATASET ==========================
class GalaxyBarDataset(Dataset):
    """Dataset that loads images from paths with optional transforms."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ========================== TRANSFORMS ==========================
# Paper Section 5: rotation is label-preserving for galaxies
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ========================== LOAD DATA ==========================
df = pd.read_csv(LABEL_CSV)
image_paths = []
labels = []
snapshot_indices = []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['snapshot'] + ".png")
    if os.path.exists(img_path):
        image_paths.append(img_path)
        labels.append(int(row['label']))
        # Extract snapshot number for temporal blocking
        snap_num = int(row['snapshot'].split('_')[1])
        snapshot_indices.append(snap_num)

labels = np.array(labels)
snapshot_indices = np.array(snapshot_indices)
n_barred = np.sum(labels == 1)
n_unbarred = np.sum(labels == 0)
print(f"Total samples: {len(labels)}  |  Not barred: {n_unbarred}  |  Barred: {n_barred}")
print(f"Class ratio: {n_unbarred/n_barred:.1f}:1 (unbarred:barred)")


# ========================== TEMPORAL BLOCK SPLIT ==========================
# Paper used random 60/40 split on independent galaxies (Section 6).
# Our snapshots are temporally correlated — consecutive frames look nearly identical.
# We group snapshots into blocks and split at the block level to prevent leakage.
block_ids = snapshot_indices // BLOCK_SIZE
unique_blocks = np.unique(block_ids)

# Identify transition zone blocks
transition_blocks = set()
if EXCLUDE_TRANSITION:
    for b in unique_blocks:
        snap_in_block = snapshot_indices[block_ids == b]
        if any(TRANSITION_RANGE[0] <= s <= TRANSITION_RANGE[1] for s in snap_in_block):
            transition_blocks.add(b)
    print(f"\nTransition zone blocks (snapshots {TRANSITION_RANGE[0]}-{TRANSITION_RANGE[1]}): "
          f"{sorted(transition_blocks)}")

# Assign each block a label: "barred" if majority of its snapshots are barred
block_labels = {}
for b in unique_blocks:
    mask = block_ids == b
    block_labels[b] = 1 if np.mean(labels[mask]) > 0.5 else 0

# Stratified split at block level (excluding transition blocks from train)
non_transition_blocks = [b for b in unique_blocks if b not in transition_blocks]
barred_blocks = [b for b in non_transition_blocks if block_labels[b] == 1]
unbarred_blocks = [b for b in non_transition_blocks if block_labels[b] == 0]

np.random.shuffle(barred_blocks)
np.random.shuffle(unbarred_blocks)

n_val_barred = max(1, int(len(barred_blocks) * VAL_SPLIT))
n_val_unbarred = max(1, int(len(unbarred_blocks) * VAL_SPLIT))

val_blocks = set(barred_blocks[:n_val_barred]) | set(unbarred_blocks[:n_val_unbarred])
# Transition blocks go to validation only (ambiguous samples test model robustness)
val_blocks |= transition_blocks
train_blocks = set(non_transition_blocks) - val_blocks

train_idx = np.array([i for i in range(len(labels)) if block_ids[i] in train_blocks])
val_idx = np.array([i for i in range(len(labels)) if block_ids[i] in val_blocks])

train_paths_orig = [image_paths[i] for i in train_idx]
val_paths = [image_paths[i] for i in val_idx]
train_labels_orig = labels[train_idx]
val_labels = labels[val_idx]

print(f"\nTemporal block split (block_size={BLOCK_SIZE}):")
print(f"  Train blocks: {sorted(train_blocks)}")
print(f"  Val blocks:   {sorted(val_blocks)}")
print(f"  Train: {len(train_labels_orig)} (barred={np.sum(train_labels_orig==1)}, "
      f"unbarred={np.sum(train_labels_orig==0)})")
print(f"  Val:   {len(val_labels)} (barred={np.sum(val_labels==1)}, "
      f"unbarred={np.sum(val_labels==0)})")
print(f"  Transition zone snapshots → validation only (ambiguous labels near threshold)")


# ========================== OFFLINE AUGMENTATION ==========================
# Paper Section 5: "we rotate each image by one degree 359 times so that our
# augmented set becomes 360 times larger than the original training set"
# This is critical for small datasets. We generate rotated copies on disk.
print(f"\nOffline augmentation: {NUM_ROTATIONS} rotations per image...")

augmented_paths = []
augmented_labels = []

for i, (path, label) in enumerate(zip(train_paths_orig, train_labels_orig)):
    # Always include original
    augmented_paths.append(path)
    augmented_labels.append(label)

    # Determine number of rotations (more for minority class)
    n_rot = NUM_ROTATIONS
    if label == 1:
        n_rot = NUM_ROTATIONS * AUGMENT_MINORITY_EXTRA

    basename = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path)

    angles = np.linspace(360 / n_rot, 360 - 360 / n_rot, n_rot - 1)
    for angle in angles:
        aug_name = f"{basename}_rot{int(angle):03d}.png"
        aug_path = os.path.join(AUGMENTED_DIR, aug_name)
        if not os.path.exists(aug_path):
            rotated = img.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
            rotated.save(aug_path)
        augmented_paths.append(aug_path)
        augmented_labels.append(label)

    if (i + 1) % 100 == 0:
        print(f"  Augmented {i+1}/{len(train_paths_orig)} images")

augmented_labels = np.array(augmented_labels)
n_aug_barred = np.sum(augmented_labels == 1)
n_aug_unbarred = np.sum(augmented_labels == 0)
print(f"  Augmented training set: {len(augmented_labels)} images")
print(f"    Barred:   {n_aug_barred} ({n_aug_barred/len(augmented_labels)*100:.1f}%)")
print(f"    Unbarred: {n_aug_unbarred} ({n_aug_unbarred/len(augmented_labels)*100:.1f}%)")
print(f"    Ratio:    {n_aug_unbarred/max(1,n_aug_barred):.1f}:1 "
      f"(was {n_unbarred/max(1,n_barred):.1f}:1 before augmentation)")


# ========================== DATA LOADERS ==========================
train_dataset = GalaxyBarDataset(augmented_paths, augmented_labels, transform=train_transform)
val_dataset = GalaxyBarDataset(val_paths, val_labels, transform=val_transform)

# Weighted sampler for remaining imbalance after augmentation
class_counts = np.bincount(augmented_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[augmented_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ========================== MODEL ==========================
# Paper used AlexNet (Section 4) trained from scratch on 9346 galaxies.
# With only ~500 images, we use transfer learning with ResNet18 pretrained on ImageNet.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

if FREEZE_BACKBONE:
    for param in model.parameters():
        param.requires_grad = False

# Replace final FC: paper used 3 FC layers with 50% dropout and softmax
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),          # 50% dropout like the paper (Section 4)
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
)
model = model.to(DEVICE)

# Weighted BCE loss
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Paper Section 6: "stochastic gradient algorithm with base learning rate of 0.01
# and a decay rate of 30 per cent"
optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

# Early stopping (Paper Section 6: "with early stopping we completed training at ~16 epochs")
EARLY_STOP_PATIENCE = 7


# ========================== TRAINING ==========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in loader:
        images = images.to(DEVICE)
        targets = labels_batch.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels_batch.to(DEVICE)).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(DEVICE)
            targets = labels_batch.float().to(DEVICE)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            probs = torch.sigmoid(outputs)

            running_loss += loss.item() * images.size(0)
            preds = (probs > 0.5).long()
            correct += (preds == labels_batch.to(DEVICE)).sum().item()
            total += images.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# Training loop
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')
epochs_no_improve = 0
actual_epochs = 0

print(f"\n{'='*60}")
print("TRAINING")
print(f"{'='*60}")

for epoch in range(NUM_EPOCHS):
    # Unfreeze backbone for fine-tuning (transfer learning phase 2)
    if FREEZE_BACKBONE and epoch == FINE_TUNE_AFTER:
        print(f"\n--- Unfreezing backbone at epoch {epoch} for fine-tuning ---")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model.parameters(), lr=FINE_TUNE_LR, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
        epochs_no_improve = 0  # reset patience after unfreezing

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_preds, val_true, val_probs = validate(model, val_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    actual_epochs = epoch + 1

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  |  "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  |  LR: {current_lr:.1e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
    else:
        epochs_no_improve += 1

    # Early stopping (Paper Section 6: "with early stopping ... at around 16 epochs")
    if epochs_no_improve >= EARLY_STOP_PATIENCE and epoch >= FINE_TUNE_AFTER:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
        break


# ========================== EVALUATION ==========================
print(f"\n{'='*60}")
print("EVALUATION (best model)")
print(f"{'='*60}")

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), weights_only=True))
_, final_acc, final_preds, final_true, final_probs = validate(model, val_loader, criterion)

# --- Classification report (Paper Table 3) ---
report = classification_report(final_true, final_preds, target_names=["Not Barred", "Barred"])
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("Classification Report (Paper Table 3 equivalent)\n")
    f.write("=" * 50 + "\n")
    f.write(report)

# --- Training curves (Paper Fig. 6) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history['train_loss'], label='Train', color='orange')
ax1.plot(history['val_loss'], label='Validation', color='blue', linestyle='--')
if FREEZE_BACKBONE and actual_epochs > FINE_TUNE_AFTER:
    ax1.axvline(x=FINE_TUNE_AFTER, color='gray', linestyle='--', alpha=0.7, label='Unfreeze backbone')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title(f'Training Loss — stopped at epoch {actual_epochs} (cf. Paper Fig. 6)')
ax1.legend()

ax2.plot(history['train_acc'], label='Train', color='orange')
ax2.plot(history['val_acc'], label='Validation', color='blue', linestyle='--')
if FREEZE_BACKBONE and actual_epochs > FINE_TUNE_AFTER:
    ax2.axvline(x=FINE_TUNE_AFTER, color='gray', linestyle='--', alpha=0.7, label='Unfreeze backbone')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()
#predict_single()
# --- Confusion matrix (Paper Fig. 9) ---
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(final_true, final_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Barred", "Barred"])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title("Confusion Matrix (cf. Paper Fig. 9)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# --- ROC curve + AUC (Paper Fig. 8) ---
fpr, tpr, thresholds = roc_curve(final_true, final_probs)
roc_auc = auc(fpr, tpr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random guess')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve (cf. Paper Fig. 8)')
ax1.legend(loc='lower right')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.05])

# Precision-Recall curve (additional metric for imbalanced data)
precision_arr, recall_arr, _ = precision_recall_curve(final_true, final_probs)
ap = average_precision_score(final_true, final_probs)

ax2.plot(recall_arr, precision_arr, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc='lower left')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()

print(f"\nROC AUC: {roc_auc:.4f}  (Paper achieved 0.94)")
print(f"Average Precision: {ap:.4f}")


# ========================== OCCLUSION TEST (Paper Section 7.1) ==========================
# "A circular mask is placed over the image cut-out such that it blocks the bar feature
#  by 25, 50, and 100 per cent." — Abraham et al. (2018)
# If the model truly detects bars, masking the centre should flip barred → unbarred.

print(f"\n{'='*60}")
print("OCCLUSION TEST (Paper Section 7.1)")
print(f"{'='*60}")


def predict_single(model, img_path, transform):
    """Return predicted probability of being barred for a single image."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze()).item()
    return prob


def occlude_center(img_path, fraction):
    """Apply circular mask over central fraction of the image (Paper Fig. 7)."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * fraction / 2)
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0))
    return img


# Find barred validation images for occlusion test
barred_val_idx = [i for i in range(len(val_labels)) if val_labels[i] == 1]
occlusion_levels = [0.0, 0.25, 0.50, 1.0]  # Paper Table 2

if len(barred_val_idx) > 0:
    # Test on up to 6 barred images
    test_indices = barred_val_idx[:min(6, len(barred_val_idx))]

    fig, axes = plt.subplots(len(test_indices), len(occlusion_levels),
                             figsize=(3 * len(occlusion_levels), 3 * len(test_indices)))
    if len(test_indices) == 1:
        axes = axes[np.newaxis, :]

    occlusion_results = []

    for row, vi in enumerate(test_indices):
        img_path = val_paths[vi]
        snap_name = os.path.basename(img_path)

        for col, frac in enumerate(occlusion_levels):
            if frac == 0.0:
                img = Image.open(img_path).convert("RGB")
            else:
                img = occlude_center(img_path, frac)

            # Get prediction on occluded image
            tensor = val_transform(img).unsqueeze(0).to(DEVICE)
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor).squeeze()).item()

            pred_label = "Barred" if prob > 0.5 else "Unbarred"
            color = 'green' if prob > 0.5 else 'red'

            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{int(frac*100)}%: {pred_label}\np={prob:.2f}",
                                     fontsize=9, color=color)
            axes[row, col].axis('off')

            if row == 0:
                occlusion_results.append({'occlusion': f"{int(frac*100)}%",
                                          'prediction': pred_label, 'prob': prob})

        print(f"  {snap_name}: " +
              " | ".join(f"{int(f*100)}%={predict_single(model, img_path, val_transform) if f == 0 else '...':.2f}"
                         for f in [0.0])
              + f" → see occlusion_test.png")

    plt.suptitle("Occlusion Test (cf. Paper Fig. 7 & Table 2)\n"
                 "If CNN detects bars, masking the centre should change prediction",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "occlusion_test.png"), dpi=150)
    plt.close()

    # Print summary table like Paper Table 2
    print("\n  Occlusion results (first barred image):")
    print(f"  {'Level':<20} {'Prediction':<15}")
    print(f"  {'-'*35}")
    for r in occlusion_results:
        print(f"  {r['occlusion']:<20} {r['prediction']:<15} (p={r['prob']:.3f})")
else:
    print("  No barred images in validation set — skipping occlusion test")


# ========================== PREDICTION ON ALL SNAPSHOTS ==========================
# Generate predictions for every snapshot (like Paper Section 9 catalogue)
print(f"\n{'='*60}")
print("FULL CATALOGUE PREDICTIONS")
print(f"{'='*60}")

all_results = []
model.eval()

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['snapshot'] + ".png")
    if not os.path.exists(img_path):
        continue

    prob = predict_single(model, img_path, val_transform)
    all_results.append({
        'snapshot': row['snapshot'],
        'time_gyr': row['time_gyr'],
        'peak_a2a0': row['peak_a2a0'],
        'true_label': row['label'],
        'cnn_prob_barred': round(prob, 4),
        'cnn_prediction': 1 if prob > 0.5 else 0,
    })

catalogue_df = pd.DataFrame(all_results)
catalogue_df.to_csv(os.path.join(OUTPUT_DIR, "full_catalogue.csv"), index=False)

# Compare CNN predictions vs A2/A0 labels
agreement = np.mean(catalogue_df['true_label'] == catalogue_df['cnn_prediction'])
print(f"CNN vs A2/A0 agreement: {agreement:.1%}")

# Plot CNN probability vs time (shows if model learned bar evolution)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(catalogue_df['time_gyr'], catalogue_df['peak_a2a0'], 'b-', lw=1, label='A2/A0 (Fourier)')
ax1.axhline(y=0.19, color='red', linestyle='--', alpha=0.7, label='Bar threshold (0.19)')
ax1.set_ylabel('Peak A2/A0')
ax1.set_title('Bar Strength: Fourier Analysis vs CNN Prediction')
ax1.legend()

ax2.plot(catalogue_df['time_gyr'], catalogue_df['cnn_prob_barred'], 'g-', lw=1, label='CNN P(barred)')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision boundary (0.5)')
ax2.fill_between(catalogue_df['time_gyr'], 0, 1,
                  where=catalogue_df['true_label'] == 1, alpha=0.15, color='orange', label='True barred region')
ax2.set_xlabel('Time (Gyr)')
ax2.set_ylabel('CNN P(barred)')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_vs_fourier.png"), dpi=150)
plt.close()

print(f"\nAll results saved to {OUTPUT_DIR}/:")
print(f"  - best_model.pth          (trained model)")
print(f"  - training_curves.png     (cf. Paper Fig. 6)")
print(f"  - confusion_matrix.png    (cf. Paper Fig. 9)")
print(f"  - roc_curve.png           (cf. Paper Fig. 8)")
print(f"  - occlusion_test.png      (cf. Paper Fig. 7)")
print(f"  - cnn_vs_fourier.png      (CNN predictions vs A2/A0 over time)")
print(f"  - full_catalogue.csv      (cf. Paper Table 4)")
print(f"  - classification_report.txt")
