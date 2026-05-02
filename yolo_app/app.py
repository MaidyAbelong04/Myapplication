import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# STEP 1: Page Configuration
st.set_page_config(page_title="AI Object Detector", layout="wide")

# STEP 2: Load Model (Cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")

# Sidebar Settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.20)

# STEP 3: Video frame callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    # YOLOv8 tracking
    results = model.track(img, persist=True, conf=conf_threshold, verbose=False)
    
    # Siguraduhing may detection bago mag-plot
    if results and len(results) > 0:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = img

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STEP 4: WebRTC Streamer - Direct Dictionary Config (WiFi Stable)
# Inalis natin ang anumang extra classes para maiwasan ang TypeError
webrtc_streamer(
    key="object-detection-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.divider()
st.info("Kung ayaw sa Wi-Fi: 1. I-refresh ang page. 2. Gamitin ang Incognito Mode. 3. I-off ang VPN kung meron.")
