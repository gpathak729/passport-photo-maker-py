import io
import numpy as np
from PIL import Image
import streamlit as st

from utils import (
    remove_background_rgba,
    refine_alpha_soft,
    refine_alpha_morph,
    enhance_rgb_quality,
    detect_face_box,
    compute_crop_rect,
    draw_guides,
    layout_on_4x6,
)

st.set_page_config(page_title="Passport Photo Maker (Python)", page_icon="🪪", layout="wide")
st.title("Passport Photo Maker — Python/Streamlit")
st.write("Upload a photo or take a snapshot. The app removes background, refines hair edges, applies country presets, and exports a high-quality passport photo.")

# -------------------- country presets --------------------
PRESETS = {
    "United States — 2×2 in (600×600 px)": dict(outW=600, outH=600, targetHead=360, eyeRatio=0.60),
    "India — 35×45 mm (~413×531 px)":      dict(outW=413, outH=531, targetHead=330, eyeRatio=0.62),
    "EU — 35×45 mm (~413×531 px)":         dict(outW=413, outH=531, targetHead=330, eyeRatio=0.62),
    "UK — 35×45 mm (~413×531 px)":         dict(outW=413, outH=531, targetHead=320, eyeRatio=0.61),
}

# -------------------- controls --------------------
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    preset_name = st.selectbox("Country / size preset", list(PRESETS.keys()))
    bg = st.selectbox("Background", ["white", "blue", "transparent"])
    fmt = st.selectbox("Download format", ["PNG", "JPG"])
with colB:
    head_scale = st.slider("Head size ± (%)", -10, 10, 0, 1)
    eye_nudge = st.slider("Eye line ± (relative)", -10, 10, 0, 1)
with colC:
    denoise = st.slider("Denoise (bilateral)", 0, 10, 2, 1)
    sharpen = st.slider("Sharpen (unsharp mask)", 0, 150, 60, 5)
    edge_soft = st.slider("Edge feather (px)", 0, 10, 2, 1)

jpg_q = st.slider("JPG quality (when JPG)", 80, 100, 95, 1)

# -------------------- capture source (no WebRTC) --------------------
st.subheader("Capture source")
cap_mode = st.radio(
    "Choose how to take the picture",
    ["Upload", "Camera Snapshot"],
    captions=["Pick a file", "Take a single still photo from your webcam"],
    index=0,
)

pil = None
if cap_mode == "Camera Snapshot":
    snap = st.camera_input("Take a quick snapshot")
    if snap is not None:
        pil = Image.open(snap).convert("RGB")
else:
    uploaded = st.file_uploader(
        "Upload a portrait image (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")

st.caption("Tip: neutral expression, even lighting, hair visible, plain background if possible.")

params = PRESETS[preset_name]

def process(pil_img: Image.Image):
    # 1) person cut-out (RGBA)
    rgba = remove_background_rgba(pil_img)

    # 2) alpha refinement (advanced matting touch-ups)
    if edge_soft > 0:
        a = np.array(rgba.split()[-1])
        a = refine_alpha_soft(a, radius=edge_soft)         # soft feather for hair
        a = refine_alpha_morph(a, open_px=1, close_px=1)   # tiny clean-up on edges
        rgba = Image.merge("RGBA", (rgba.split()[0], rgba.split()[1], rgba.split()[2], Image.fromarray(a)))

    # 3) compose selected background
    if bg == "transparent":
        composed = rgba.copy()
    else:
        bg_rgb = (255, 255, 255) if bg == "white" else (47, 93, 170)
        canvas = Image.new("RGB", rgba.size, bg_rgb)
        composed = canvas.copy()
        composed.paste(rgba, mask=rgba.split()[-1])

    # 4) detect face & compute crop box from preset
    face_box = detect_face_box(np.array(pil_img))
    crop_rect = compute_crop_rect(
        imgW=composed.width, imgH=composed.height,
        face_box=face_box, params=params,
        scale_adj=head_scale, eye_adj=eye_nudge,
    )

    guided = draw_guides(composed, crop_rect, params)

    # 5) crop + resize (high-quality)
    out = composed.crop((
        crop_rect["x"], crop_rect["y"],
        crop_rect["x"] + crop_rect["w"], crop_rect["y"] + crop_rect["h"]
    ))
    out = out.resize((params["outW"], params["outH"]), Image.LANCZOS)

    # 6) quality enhancement on RGB
    if bg != "transparent":
        out = out.convert("RGB")
        out = enhance_rgb_quality(out, denoise_strength=denoise, sharpen_amount=sharpen)
    else:
        rgb = out.convert("RGB")
        rgb = enhance_rgb_quality(rgb, denoise_strength=denoise, sharpen_amount=sharpen)
        out = Image.merge("RGBA", (*rgb.split(), out.split()[-1]))

    return rgba, guided, out

# -------------------- pipeline --------------------
if pil is not None:
    st.subheader("1) Input")
    st.image(pil)

    st.subheader("2) Background removal + edge refinement")
    try:
        rgba, guided, out = process(pil)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.image(rgba, caption="Person on transparent background (refined edges)")

    st.subheader("3) Face detection & crop (preset applied)")
    st.image(guided, caption="Crop preview with guides")

    st.subheader("4) Final result")
    st.image(out, caption=f"{preset_name} — {bg} background")

    # 4×6 sheet (auto-fits, 6 copies)
    with st.expander("Print sheet (optional): 4×6 inch with multiple copies"):
        if st.button("Create 4×6 sheet"):
            sheet = layout_on_4x6([out])
            st.image(sheet, caption="4×6 layout preview")
            buf2 = io.BytesIO()
            sheet.save(buf2, format="JPEG", quality=95)
            st.download_button(
                "Download 4×6 JPG",
                data=buf2.getvalue(),
                file_name="sheet_4x6.jpg",
                mime="image/jpeg"
            )

    # download final photo
    buf = io.BytesIO()
    if fmt == "JPG":
        out.convert("RGB").save(buf, format="JPEG", quality=jpg_q, subsampling=0, optimize=True)
        mime = "image/jpeg"; ext = "jpg"
    else:
        out.save(buf, format="PNG", compress_level=6)
        mime = "image/png"; ext = "png"

    st.download_button(
        "Download final photo",
        data=buf.getvalue(),
        file_name=f"passport_{'2x2' if params['outW']==600 else '35x45'}.{ext}",
        mime=mime
    )

    st.info("Note: Country rules vary (head size, eye position). Use sliders to fine-tune and verify with your authority.")
else:
    st.write("Choose a capture method above, then take a photo or upload one to begin.")