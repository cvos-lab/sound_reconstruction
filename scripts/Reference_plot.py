import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

title_color = '#22667A'
center_dot_color = '#0072EF'   # Pure blue

img = Image.open('../data/raw_pattern/speckle.bmp')
img_np = np.array(img)

# --- Crop 256x256 region centered on the image ---
h, w = img_np.shape[:2]
center_x, center_y = w // 2, h // 2
half_patch = 128
crop_img = img_np[
    center_y - half_patch:center_y + half_patch,
    center_x - half_patch:center_x + half_patch
]

center_px, center_py = 128, 128
rect_size = 128
rect_corner = (center_px - rect_size // 2, center_py - rect_size // 2)  # (64,64)

fig, ax = plt.subplots(figsize=(4, 4), dpi=220)  # High dpi for display

ax.imshow(crop_img, cmap='gray' if crop_img.ndim == 2 else None)

# Red rectangle (128x128 centered)
rect = patches.Rectangle(
    rect_corner, rect_size, rect_size,
    linewidth=2.5, edgecolor='r', facecolor='none'
)
ax.add_patch(rect)

# Pure blue dot at center
ax.plot(center_px, center_py, 'o', color=center_dot_color, markersize=13)

# Bold, blue title
ax.set_title(
    "Reference pattern",
    fontsize=18,
    color=title_color,
    fontweight='bold',
    pad=18
)
ax.axis('off')
plt.tight_layout(pad=1.5)

# --- Save high-res figure (best for publication/zoom) ---
plt.savefig("reference_pattern_highres.png", dpi=300, bbox_inches='tight')

plt.show()

import matplotlib
print(matplotlib.rcParams['font.family'])
print(matplotlib.rcParams['font.sans-serif'])