import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# STEP 1: Page Setup
st.set_page_config(page_title="Universal AI Detector", layout="wide")

# STEP 2: Cache Model (Para mabilis mag-load)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 AI Object Detection (Data & Wi-Fi Ready)")

# Sidebar Settings
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# STEP 3: Detection Logic
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirror fix

    # Predict function (Mas stable sa labels)
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    # Ito ang nag-uutos na i-drawing ang labels (person, etc.)
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STEP 4: UNIVERSAL WEBRTC CONFIG (The Wi-Fi Fix)
# Gumagamit ng maraming Google at Twilio servers para sa maximum compatibility
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]
}

webrtc_streamer(
    key="universal-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION, # Ito ang pinaka-importante
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.divider()
st.info("💡 Tip: Kung itim pa rin ang screen sa Wi-Fi, i-refresh ang page at i-off ang anumang VPN.")
