from typing import Dict, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2  # opencv-python-headless

# ---------- background removal ----------
def remove_background_rgba(pil_rgb: Image.Image) -> Image.Image:
    """
    Uses rembg (U^2-Net / ONNX) to cut the person out.
    Returns RGBA with alpha = person matte.
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

# ---------- advanced matting refinement ----------
def refine_alpha_soft(alpha: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Softly feather edges (Gaussian) to reduce jaggies at hair.
    radius: 0 disables, 1-5 is typical.
    """
    if radius <= 0:
        return alpha
    # normalize to 0..1, blur, then re-stretch
    a = alpha.astype(np.float32) / 255.0
    k = max(1, int(2 * radius + 1))
    a = cv2.GaussianBlur(a, (k, k), sigmaX=radius, borderType=cv2.BORDER_REPLICATE)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)

def refine_alpha_morph(alpha: np.ndarray, open_px: int = 1, close_px: int = 1) -> np.ndarray:
    """
    Tiny morphological cleanup: opening removes specks; closing fills pinholes.
    Keep pixels small (0-2) to avoid shrinking hair.
    """
    a = alpha.copy()
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k)
    return a

# ---------- quality enhancement ----------
def enhance_rgb_quality(pil_rgb: Image.Image, denoise_strength: int = 2, sharpen_amount: int = 60) -> Image.Image:
    """
    - Bilateral denoise to keep edges (strength 0..10)
    - Unsharp mask for crisp detail (amount 0..150)
    """
    img = np.array(pil_rgb).astype(np.uint8)

    # denoise while preserving edges
    if denoise_strength > 0:
        # parameters tuned for portraits
        d = 5 + denoise_strength  # neighborhood
        sigma_c = 20 + 4 * denoise_strength
        sigma_s = 25 + 6 * denoise_strength
        img = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s)

    # unsharp mask
    if sharpen_amount > 0:
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        amt = sharpen_amount / 100.0  # 0..1.5 typical
        sharp = cv2.addWeighted(img, 1 + amt, blur, -amt, 0)
        img = np.clip(sharp, 0, 255).astype(np.uint8)

    return Image.fromarray(img, mode="RGB")

# ---------- face detection ----------
def detect_face_box(img_rgb: np.ndarray):
    """
    MediaPipe FaceDetection -> dict(x,y,w,h) in pixel coords, or None.
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
    Compute a crop window so the head-height and eye-line match the preset.
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
    draw.rectangle([x, y, x+w, y+h], outline=(0, 200, 255), width=3)
    eyeY = int(y + params["eyeRatio"] * h)
    draw.line([x, eyeY, x+w, eyeY], fill=(255, 80, 80), width=3)
    return img

# ---------- 4×6 layout (no overlap; auto-fit) ----------
def layout_on_4x6(img_list):
    """
    4x6 in @ 300 DPI = 1800x1200 px (landscape).
    3x2 grid, auto-scales down to fit each cell, centered. Always 6 copies.
    """
    SHEET_W, SHEET_H = 1800, 1200
    COLS, ROWS = 3, 2
    PAD_X, PAD_Y = 50, 50
    GAP_X, GAP_Y = 40, 40

    sheet = Image.new("RGB", (SHEET_W, SHEET_H), (255, 255, 255))
    if not img_list:
        return sheet

    base = img_list[0].convert("RGB")
    bw, bh = base.size

    drawable_w = SHEET_W - 2*PAD_X - (COLS - 1)*GAP_X
    drawable_h = SHEET_H - 2*PAD_Y - (ROWS - 1)*GAP_Y
    cell_w = drawable_w // COLS
    cell_h = drawable_h // ROWS

    scale = min(cell_w / bw, cell_h / bh, 1.0)  # downscale only
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    fitted = base.resize((new_w, new_h), Image.LANCZOS)

    for r in range(ROWS):
        for c in range(COLS):
            slot_x = PAD_X + c * (cell_w + GAP_X)
            slot_y = PAD_Y + r * (cell_h + GAP_Y)
            px = slot_x + (cell_w - new_w) // 2
            py = slot_y + (cell_h - new_h) // 2
            sheet.paste(fitted, (px, py))
    return sheet