import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# Cache the model to prevent reloading on every interaction
@st.cache_resource
def load_model():
    # Using the Nano version for faster performance on deployment servers
    return YOLO("yolov8n.pt")

model = load_model()

st.set_page_config(page_title="AI Object Detector", layout="wide")

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Enhancement: Sidebar settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.flip(img, 1) 

    # Run YOLOv8 tracking
    results = model.track(
        img, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    # Annotation
    annotated_frame = results[0].plot()

    # Enhancement: Object Counter Logic
    if results[0].boxes.id is not None:
        count = len(results[0].boxes.id)
        cv2.putText(annotated_frame, f"Objects Tracked: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.info("Note: The first time you start the camera, it may take a moment to initialize the AI model.")