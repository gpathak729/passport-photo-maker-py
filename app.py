import io
import numpy as np
from PIL import Image
import streamlit as st

from utils import (
    remove_background_rgba,
    detect_face_box,
    compute_crop_rect,
    draw_guides,
    layout_on_4x6,
)

st.set_page_config(page_title="Passport Photo Maker (Python)", page_icon="🪪", layout="wide")
st.title("Passport Photo Maker")
st.write("Upload a photo or take a camera snapshot, choose background, and export a passport-ready image.")

# -------------------- controls --------------------
colA, colB = st.columns([1, 1])
with colA:
    bg = st.selectbox("Background", ["white", "blue", "transparent"])
    preset = st.selectbox("Preset", ["2x2 in (600×600 px)", "35×45 mm (~413×531 px)"])
    fmt = st.selectbox("Download format", ["PNG", "JPG"])
with colB:
    head_scale = st.slider("Head size ± (%)", -10, 10, 0, 1)
    eye_nudge = st.slider("Eye line ± (relative)", -10, 10, 0, 1)

# -------------------- capture source (no WebRTC) --------------------
st.subheader("Capture source")
cap_mode = st.radio(
    "Choose how to take the picture",
    ["Upload", "Camera Snapshot"],
    captions=[
        "Pick a file from your computer",
        "Take a single still photo from your webcam",
    ],
    index=0,
)

use_cam = False
pil_from_cam = None

if cap_mode == "Camera Snapshot":
    snap = st.camera_input("Take a quick snapshot")
    if snap is not None:
        pil_from_cam = Image.open(snap).convert("RGB")
        use_cam = True

# ---------- Uploader ----------
uploaded = None
if cap_mode == "Upload":
    uploaded = st.file_uploader(
        "Upload a portrait image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
st.caption("Tip: neutral expression, even lighting, plain background if possible.")

# pick source image
if use_cam:
    pil = pil_from_cam
elif uploaded:
    pil = Image.open(uploaded).convert("RGB") if uploaded else None
else:
    pil = None

# -------------------- pipeline --------------------
if pil is not None:
    st.subheader("1) Input")
    st.image(pil)

    st.subheader("2) Background removal")
    try:
        rgba = remove_background_rgba(pil)  # RGBA with alpha for person
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.image(rgba, caption="Person on transparent background")

    # compose background
    if bg == "transparent":
        composed = rgba.copy()
    else:
        bg_rgb = (255, 255, 255) if bg == "white" else (47, 93, 170)
        canvas = Image.new("RGB", rgba.size, bg_rgb)
        composed = canvas.copy()
        composed.paste(rgba, mask=rgba.split()[-1])

    st.subheader("3) Face detection & crop")
    face_box = detect_face_box(np.array(pil))

    if "2x2" in preset:
        params = dict(outW=600, outH=600, targetHead=360, eyeRatio=0.60)
    else:
        params = dict(outW=413, outH=531, targetHead=330, eyeRatio=0.62)

    crop_rect = compute_crop_rect(
        imgW=composed.width,
        imgH=composed.height,
        face_box=face_box,
        params=params,
        scale_adj=head_scale,
        eye_adj=eye_nudge,
    )

    guided = draw_guides(composed, crop_rect, params)
    st.image(guided, caption="Crop preview (guides only)")

    # final render
    out = composed.crop(
        (
            crop_rect["x"],
            crop_rect["y"],
            crop_rect["x"] + crop_rect["w"],
            crop_rect["y"] + crop_rect["h"],
        )
    )
    out = out.resize((params["outW"], params["outH"]), Image.LANCZOS)
    if bg != "transparent":
        out = out.convert("RGB")

    st.subheader("4) Result")
    st.image(out, caption=f"{preset} — {bg} background")

    # 4×6 sheet (auto-fills 6 copies)
    with st.expander("Print sheet (optional): 4×6 inch with multiple copies"):
        if st.button("Create 4×6 sheet"):
            sheet = layout_on_4x6([out])  # fills all 6 slots with the same photo
            st.image(sheet, caption="4×6 layout preview")
            buf2 = io.BytesIO()
            sheet.save(buf2, format="JPEG", quality=95)
            st.download_button(
                "Download 4×6 JPG",
                data=buf2.getvalue(),
                file_name="sheet_4x6.jpg",
                mime="image/jpeg",
            )

    # download final photo
    buf = io.BytesIO()
    if fmt == "JPG":
        out.convert("RGB").save(buf, format="JPEG", quality=95)
        mime = "image/jpeg"; ext = "jpg"
    else:
        out.save(buf, format="PNG")
        mime = "image/png"; ext = "png"

    st.download_button(
        "Download final photo",
        data=buf.getvalue(),
        file_name=f"passport_{'2x2' if '2x2' in preset else '35x45'}.{ext}",
        mime=mime,
    )

    st.info("Note: Photo rules vary by country (head size, eye position). Use sliders to fine-tune and verify.")
else:
    st.write("Choose a capture method above, then take a photo or upload one to begin.")
