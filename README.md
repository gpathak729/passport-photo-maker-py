# 🪪 Passport Photo Maker (AI Background Changer)

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-brightgreen?logo=streamlit)](https://gpathak729-passport-photo-maker-py-main.streamlit.app)

An open-source **Streamlit app** that generates passport-size photos instantly.  
It uses **AI-based background removal** (via `rembg`) and **face detection** (via MediaPipe) to create compliant 2×2 inch and 35×45 mm photos with selectable backgrounds (white / blue / transparent).  

---

## 🚀 Features
- 📤 Upload photo or 📷 capture a snapshot from your webcam
- ✂️ Automatic background removal using deep learning (`rembg`)
- 🎨 Background replacement: **white, blue, transparent**
- 📐 Auto crop to passport presets:  
  - **2×2 in (600×600 px)**  
  - **35×45 mm (~413×531 px)**
- ⚙️ Adjustable sliders for **head size** and **eye line**
- 📄 Download **PNG/JPG** final photo
- 🖼️ Create a **4×6 inch sheet** with 6 copies (ready to print)
- ✅ Works on both **desktop & mobile browsers**

---

## 📸 Demo
👉 **Live app:**  
https://gpathak729-passport-photo-maker-py-main.streamlit.app  

### Screenshots
(Add PNG/JPG images into a `docs/screenshots/` folder in your repo, then update the links here.)

1. **Input photo**  
   ![Input](docs/screenshots/input.png)

2. **Background removed**  
   ![Removed BG](docs/screenshots/removed.png)

3. **Crop preview with guides**  
   ![Guides](docs/screenshots/guides.png)

4. **Final passport photo**  
   ![Final](docs/screenshots/final.png)

5. **4×6 sheet layout**  
   ![Sheet](docs/screenshots/sheet.png)

### Demo video
(Add a short MP4/webm recording into `docs/demo/` and embed it:)

```markdown
[▶️ Watch demo video](docs/demo/demo.mp4)
