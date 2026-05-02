import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Object Detector", layout="wide")

# ✔️ Hindi binago title mo
st.title("🎥 Live Object Detection & Tracing")
st.write("Upload an image to detect objects using YOLOv8.")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-download

model = load_model()

st.success("✅ Model loaded successfully!")

# ===============================
# UPLOAD IMAGE
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload an image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"]
)

# ===============================
# DETECTION
# ===============================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(img_array, caption="Original Image", use_container_width=True)

    # Run YOLO detection
    results = model.predict(img_array, conf=0.25)

    # Annotated image
    for r in results:
        annotated_img = r.plot()

    st.subheader("📍 Detected Objects")
    st.image(annotated_img, caption="Detection Result", use_container_width=True)

    # Object count
    for r in results:
        boxes = r.boxes
        count = len(boxes)

    st.info(f"🔎 Objects Detected: {count}")
