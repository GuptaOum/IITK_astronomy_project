import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

ROWS, COLS = 10, 10
IMAGES_PER_PAGE = ROWS * COLS
START, END = 0, 501
OUTPUT = "snapshot_grid2.pdf"

paths = []
for i in range(START, END + 1):
    p = f"snapshot_{i:03d}.png"
    if os.path.exists(p):
        paths.append(p)
    else:
        print(f"Missing {p}, skipping.")

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(OUTPUT) as pdf:
    for page in range(5):
        fig, axes = plt.subplots(ROWS, COLS, figsize=(25, 25))
        fig.subplots_adjust(wspace=0.02, hspace=0.02)
        for r in range(ROWS):
            for c in range(COLS):
                ax = axes[r][c]
                idx = page * IMAGES_PER_PAGE + r * COLS + c
                if idx < len(paths):
                    img = mpimg.imread(paths[idx])
                    ax.imshow(img)
                    ax.set_title(os.path.splitext(paths[idx])[0], fontsize=4, pad=1)
                ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
        print(f"Page {page + 1}/5 done")

print(f"\nSaved {OUTPUT}")
