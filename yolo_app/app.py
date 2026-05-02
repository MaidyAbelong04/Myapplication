import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# STEP 1: DAPAT ITO ANG PINAKAUNANG STREAMLIT COMMAND
# Nilalagay nito ang page title at layout bago mag-render ang kahit ano.
st.set_page_config(page_title="AI Object Detector", layout="wide")

# STEP 2: I-cache ang model para hindi ito mag-reload tuwing mag-i-interact sa app.
@st.cache_resource
def load_model():
    # Gagamit tayo ng 'yolov8n.pt' (Nano) para mabilis ang processing sa server.
    return YOLO("yolov8n.pt")

model = load_model()

# UI Header base sa requirements ng Activity 3
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Enhancement: Sidebar para sa configuration (Dagdag points sa grade)
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# STEP 3: Video frame callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # FIX: Mirror Effect (I-flip ang image horizontally para hindi ka "tagilid")
    img = cv2.flip(img, 1)

    # Run YOLOv8 tracking
    results = model.track(
        img, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    # Annotate frame (Dito nilalagay ang Bounding Boxes at Labels)
    annotated_frame = results[0].plot()

    # Enhancement: Object Counter (Bilangin ang nade-detect na objects)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        count = len(results[0].boxes.id)
        cv2.putText(annotated_frame, f"Objects Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STEP 4: WebRTC Streamer Setup
# Ginagamit ang Google STUN server para makalusot sa network firewalls.
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    # Pinipilit ang browser na i-autoplay ang video stream.
    video_html_attrs={
        "style": {"width": "100%"},
        "controls": False,
        "autoPlay": True,
    },
)

st.divider()
st.info("Note: Mangyaring payagan (Allow) ang camera access sa iyong browser. Kung itim pa rin ang screen, subukang i-refresh ang page o gamitin ang HTTPS link.")
