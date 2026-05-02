import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# STEP 1: Page Configuration
st.set_page_config(page_title="AI Object Detector", layout="wide")

# STEP 2: Cache the model
@st.cache_resource
def load_model():
    # Siguraduhing yolov8n.pt (Nano) ang gamit para magaan sa server
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Enhancement Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Mirror Flip
    img = cv2.flip(img, 1)

    # Simple Detection (Mas magaan ito kaysa sa .track())
    results = model.predict(img, conf=conf_threshold, verbose=False)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Object Counter (Simple version para iwas error)
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Detected: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STEP 4: WebRTC Streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True, # Mahalaga ito para sa performance
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

st.divider()
st.info("Kung itim pa rin ang screen: 1. Siguraduhing HTTPS ang link. 2. I-check kung may ibang app na gumagamit ng camera. 3. Subukan sa Incognito mode.")
