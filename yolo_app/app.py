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

# UI Header
st.title("🎥 Live Object Detection & Tracing")
st.write("Current Status: Connecting via WebRTC...")

# Sidebar Settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# STEP 3: Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    results = model.track(img, persist=True, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STEP 4: The "Wi-Fi Fix" RTC Configuration
# Dinagdagan natin ng kilalang public STUN servers para sa redundancy
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]} # Dagdag na Twilio stun server
    ]
}

webrtc_streamer(
    key="wifi-fix-streamer",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    # Mahalaga ito para sa Wi-Fi connection handshake
    senders_buffer_size=1, 
)

st.divider()
st.warning("Kung naka-Wi-Fi at ayaw pa rin: I-off ang Firewall ng Windows o subukang gamitin ang Google Chrome 'Incognito Mode'.")
