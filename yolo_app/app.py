import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

st.set_page_config(page_title="AI Object Detector", layout="wide")

@st.cache_resource
def load_model():
    # Siguraduhing nano model ang gamit para mabilis sa web
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")

# Sidebar - Default to 0.25 para mabilis lumabas ang labels
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.20)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    # GAMITIN ANG .predict PARA MAS STABLE ANG LABELS
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    # Ito ang nag-uutos sa code na i-drawing ang box at labels
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="labels-fix",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
