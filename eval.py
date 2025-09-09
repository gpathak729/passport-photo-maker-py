import os
import numpy as np
from PIL import Image
from utils import remove_background_rgba

def iou(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / max(1, union)

images_dir = "data/images"
masks_dir = "data/masks"  # PNG: 255=person, 0=background

if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
    raise SystemExit("Expected data/images and data/masks folders. Please create them and add files.")

scores = []
for name in os.listdir(images_dir):
    if not name.lower().endswith((".jpg",".jpeg",".png")):
        continue
    base = os.path.splitext(name)[0]
    img_path = os.path.join(images_dir, name)
    mask_path = os.path.join(masks_dir, base + ".png")
    if not os.path.exists(mask_path):
        print(f"[skip] no mask for {name}")
        continue

    img = Image.open(img_path).convert("RGB")
    rgba = remove_background_rgba(img)                 # predicted matting
    pred_a = np.array(rgba.split()[-1])               # alpha channel

    gt = Image.open(mask_path).convert("L")
    gt_a = np.array(gt)

    scores.append(iou(pred_a, gt_a))

if scores:
    print(f"Mean IoU on {len(scores)} images: {np.mean(scores):.3f}")
else:
    print("No images evaluated. Put photos in data/images and masks in data/masks (same base names).")