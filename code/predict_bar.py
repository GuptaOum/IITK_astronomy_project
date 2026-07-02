"""
Load trained model from transfer_learning.py and predict if a galaxy image has a bar.
Usage:
    python predict_bar.py path/to/image.png
    python predict_bar.py path/to/folder/   (predicts on all PNGs in folder)
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ========================== CONFIG ==========================
MODEL_PATH = "transfer_learning_results/best_model.pth"
IMG_SIZE = 224
THRESHOLD = 0.5  # probability above this = barred
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== LOAD MODEL ==========================
def load_model(model_path=MODEL_PATH):
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


# ========================== TRANSFORM ==========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ========================== PREDICT ==========================
def predict(model, image_path):
    """Returns (label, probability) for a single image."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze()).item()
    label = "Barred" if prob > THRESHOLD else "Not Barred"
    return label, prob


# ========================== MAIN ==========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict_bar.py image.png")
        print("  python predict_bar.py folder/")
        sys.exit(1)

    path = sys.argv[1]

    # Load model once
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model()
    print("Model loaded.\n")

    # Collect image paths
    if os.path.isdir(path):
        images = sorted([os.path.join(path, f) for f in os.listdir(path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    else:
        images = [path]

    # Predict
    print(f"{'Image':<40} {'Prediction':<15} {'P(barred)':<10}")
    print("-" * 65)

    for img_path in images:
        label, prob = predict(model, img_path)
        name = os.path.basename(img_path)
        print(f"{name:<40} {label:<15} {prob:.4f}")
