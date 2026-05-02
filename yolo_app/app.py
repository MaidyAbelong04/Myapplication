import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# STEP 1: Page Configuration
st.set_page_config(page_title="AI Object Detector", layout="wide")

# STEP 2: I-cache ang model para mabilis (YOLOv8 Nano)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# UI Header
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Sidebar Settings
st.sidebar.header("Settings")
# Ginawa nating 0.25 ang default para mas madaling makadetect agad
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# STEP 3: Video frame callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Mirror Effect
    img = cv2.flip(img, 1)

    # Run YOLOv8 tracking (Mas stable ang .predict kung mahina ang Wi-Fi)
    results = model.track(
        img, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    # Object Counter logic
    if results[0].boxes is not None and results[0].boxes.id is not None:
        count = len(results[0].boxes.id)
        cv2.putText(annotated_frame, f"Objects Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STEP 4: WebRTC Streamer Setup na may pinalakas na Network Config
# Nagdagdag tayo ng maraming STUN servers para sa Wi-Fi stability
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]}
    ]}
)

webrtc_streamer(
    key="object-detection",
    mode=None, # Default mode
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    # Para siguradong kasya sa screen
    video_html_attrs={
        "style": {"width": "100%"},
        "controls": False,
        "autoPlay": True,
    },
)

st.divider()
st.info("Tips: 1. I-adjust ang slider sa 0.20. 2. Siguraduhing HTTPS ang URL. 3. Kung ayaw sa Wi-Fi, gamitin ang Mobile Data.")
