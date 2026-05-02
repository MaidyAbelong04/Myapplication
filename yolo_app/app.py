import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# STEP 1: DAPAT ITO ANG PINAKAUNANG STREAMLIT COMMAND
st.set_page_config(page_title="AI Object Detector", layout="wide")

# Cache the model para mabilis ang loading
@st.cache_resource
def load_model():
    # yolov8n.pt (Nano) ang gamit natin para hindi mag-lag sa deployment
    return YOLO("yolov8n.pt")

model = load_model()

# UI Header
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Enhancement: Sidebar para sa configuration
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Video frame callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # FIX: Mirror Effect (I-flip ang image para hindi ka tagilid)
    img = cv2.flip(img, 1)

    # Run YOLOv8 tracking
    results = model.track(
        img, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    # Annotate frame (Boxes and Labels)
    annotated_frame = results[0].plot()

    # Enhancement: Object Counter
    if results[0].boxes is not None and results[0].boxes.id is not None:
        count = len(results[0].boxes.id)
        cv2.putText(annotated_frame, f"Objects Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# WebRTC Streamer Configuration
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.divider()
st.info("Note: Make sure to allow camera access in your browser to start the detection.")
