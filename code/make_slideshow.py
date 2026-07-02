import imageio.v2 as imageio
import numpy as np
import os

START = 0
END = 501
FPS = 30          # output fps (higher because we're adding interpolated frames)
INTERP_FRAMES = 3  # number of blend frames inserted between each real frame

images = []
for i in range(START, END + 1):
    path = f"snapshot_{i:03d}.png"
    if not os.path.exists(path):
        print(f"Missing {path}, skipping.")
        continue
    images.append(imageio.imread(path))
    print(f"Loaded {path}")

# Build smoothed sequence with cross-fade interpolation between consecutive frames
smoothed = []
for idx in range(len(images)):
    smoothed.append(images[idx])
    if idx < len(images) - 1:
        img_a = images[idx].astype(np.float32)
        img_b = images[idx + 1].astype(np.float32)
        for t in range(1, INTERP_FRAMES + 1):
            alpha = t / (INTERP_FRAMES + 1)
            blended = ((1 - alpha) * img_a + alpha * img_b).astype(np.uint8)
            smoothed.append(blended)

output = "slideshow.mp4"
imageio.mimwrite(output, smoothed, fps=FPS)
print(f"\nDone! Saved {output} ({len(smoothed)} frames at {FPS} fps)")
