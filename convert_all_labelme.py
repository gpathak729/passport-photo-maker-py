import os, subprocess, shutil

images_dir = "data/images"
masks_dir = "data/masks"
os.makedirs(masks_dir, exist_ok=True)

for name in os.listdir(images_dir):
    if name.lower().endswith(".json"):
        base = os.path.splitext(name)[0]
        src_json = os.path.join(images_dir, name)
        out_dir = os.path.join(masks_dir, f"{base}_out")
        subprocess.run(["labelme_json_to_dataset", src_json, "-o", out_dir], check=True)
        src_label = os.path.join(out_dir, "label.png")
        dst_mask  = os.path.join(masks_dir, f"{base}.png")
        if os.path.exists(src_label):
            shutil.move(src_label, dst_mask)
        shutil.rmtree(out_dir, ignore_errors=True)

print("Done. Masks saved to data/masks")