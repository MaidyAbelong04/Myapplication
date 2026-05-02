import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Object Detector", layout="wide")

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-download if not present

model = load_model()

# ===============================
# SIDEBAR SETTINGS
# ===============================
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.15  # mas mababa para mas responsive
)

# ===============================
# VIDEO PROCESSING FUNCTION
# ===============================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Mirror effect fix
    img = cv2.flip(img, 1)

    # YOLO Detection
    results = model.predict(img, conf=conf_threshold, verbose=False)

    object_count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            object_count += 1

            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Label
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            cv2.rectangle(img, (x1, y1 - 30), (x1 + 120, y1), (0, 255, 0), -1)

            # Label text
            cv2.putText(
                img,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    # Display object count
    cv2.putText(
        img,
        f"Objects Detected: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# WEBRTC CONFIG (FIX FOR WIFI)
# ===============================
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},

        # TURN servers (IMPORTANT sa Wi-Fi)
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
}

# ===============================
# WEBRTC STREAM
# ===============================
webrtc_streamer(
    key="object-detection-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    video_html_attrs={
        "style": {"width": "100%"},
        "autoPlay": True,
        "playsinline": True
    },
)
