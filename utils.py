from typing import Dict, Optional
import numpy as np
from PIL import Image, ImageDraw

# ---------- background removal ----------
def remove_background_rgba(pil_rgb: Image.Image) -> Image.Image:
    """
    Uses rembg (U2Net/ONNX) to cut the person out.
    Returns an RGBA image where alpha represents the person mask.
    """
    try:
        from rembg import remove
    except Exception as e:
        raise RuntimeError("rembg is not installed. Run: pip install rembg") from e
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        raise RuntimeError("onnxruntime is required by rembg. Run: pip install onnxruntime")
    arr = np.array(pil_rgb)
    out = remove(arr)  # RGBA numpy array
    return Image.fromarray(out, mode="RGBA")

# ---------- face detection ----------
def detect_face_box(img_rgb: np.ndarray):
    """
    Uses MediaPipe FaceDetection. Returns dict(x,y,w,h) in pixel coords or None.
    """
    try:
        import mediapipe as mp
    except Exception:
        return None

    mp_fd = mp.solutions.face_detection
    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        results = fd.process(img_rgb)
        if not results.detections:
            return None
        d = results.detections[0].location_data.relative_bounding_box
        h, w = img_rgb.shape[:2]
        x = max(0, int(d.xmin * w))
        y = max(0, int(d.ymin * h))
        ww = int(d.width * w)
        hh = int(d.height * h)
        return dict(x=x, y=y, w=ww, h=hh)

# ---------- crop math ----------
def compute_crop_rect(imgW:int, imgH:int, face_box: Optional[Dict], params: Dict, scale_adj:int, eye_adj:int):
    """
    Compute a crop window so the head-height and eye-line match target ratios.
    """
    outW, outH = params["outW"], params["outH"]
    outAspect = outW / outH

    if face_box:
        fb = face_box
    else:
        # fallback: centered box
        fw, fh = int(imgW*0.30), int(imgH*0.35)
        fb = dict(x=int((imgW-fw)/2), y=int(imgH*0.2), w=fw, h=fh)

    headTop = max(0, fb["y"] - int(0.15 * fb["h"]))
    chin = min(imgH, fb["y"] + fb["h"] + int(0.10 * fb["h"]))
    headHeight = max(1, chin - headTop)

    targetHead = max(100, int(params["targetHead"] * (1 + scale_adj / 100.0)))

    cropH = int((headHeight * outH) / targetHead)
    cropW = int(cropH * outAspect)

    eyeY = headTop + int(0.40 * headHeight)
    eyeRatio = params["eyeRatio"] + (eye_adj / 100.0) * 0.15
    eyeRatio = min(0.75, max(0.45, eyeRatio))

    cropY = int(eyeY - eyeRatio * outH * (cropH / outH))
    cropX = int((fb["x"] + fb["w"] // 2) - cropW // 2)

    # clamp
    cropX = max(0, min(cropX, imgW - cropW))
    cropY = max(0, min(cropY, imgH - cropH))
    cropW = max(10, min(cropW, imgW))
    cropH = max(10, min(cropH, imgH))

    return dict(x=cropX, y=cropY, w=cropW, h=cropH)

# ---------- guides ----------
def draw_guides(pil_img: Image.Image, crop_rect: Dict, params: Dict) -> Image.Image:
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    x,y,w,h = crop_rect["x"], crop_rect["y"], crop_rect["w"], crop_rect["h"]
    # crop box
    draw.rectangle([x, y, x+w, y+h], outline=(0, 200, 255), width=3)
    # eye line
    eyeY = int(y + params["eyeRatio"] * h)
    draw.line([x, eyeY, x+w, eyeY], fill=(255, 80, 80), width=3)
    return img

# ---------- 4×6 sheet ----------
def layout_on_4x6(img_list):
    """
    Build a 4x6 inch (1800x1200 px @300dpi) sheet.
    - 3 columns x 2 rows (6 slots)
    - Auto-scales images DOWN to fit the grid (never upscales)
    - Centers each image in its cell so nothing gets cut
    """
    from PIL import Image

    SHEET_W, SHEET_H = 1800, 1200            # 6x4 inches landscape at 300dpi
    COLS, ROWS = 3, 2
    PAD_X, PAD_Y = 50, 50                    # outer padding
    GAP_X, GAP_Y = 40, 40                    # gap between cells (fixed, non-negative)

    sheet = Image.new("RGB", (SHEET_W, SHEET_H), (255, 255, 255))
    if not img_list:
        return sheet

    # Compute each cell's max drawable area
    drawable_w = SHEET_W - 2*PAD_X - (COLS - 1) * GAP_X
    drawable_h = SHEET_H - 2*PAD_Y - (ROWS - 1) * GAP_Y
    cell_w = drawable_w // COLS
    cell_h = drawable_h // ROWS

    # Use the first image size as reference; scale every instance to fit cell
    base = img_list[0].convert("RGB")
    bw, bh = base.size
    # downscale only
    scale = min(cell_w / bw, cell_h / bh, 1.0)
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    fitted = base.resize((new_w, new_h), Image.LANCZOS)

    # Paste 6 copies, centered in each cell
    for r in range(ROWS):
        for c in range(COLS):
            slot_x = PAD_X + c * (cell_w + GAP_X)
            slot_y = PAD_Y + r * (cell_h + GAP_Y)
            # center inside the cell
            px = slot_x + (cell_w - new_w) // 2
            py = slot_y + (cell_h - new_h) // 2
            sheet.paste(fitted, (px, py))

    return sheet
