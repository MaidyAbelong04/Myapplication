import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Object Detector", layout="wide")

# ORIGINAL TITLE (UNCHANGED)
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# ===============================
# VIDEO PROCESSING
# ===============================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Mirror effect
    img = cv2.flip(img, 1)

    # YOLO prediction
    results = model.predict(img, conf=conf_threshold, verbose=False)

    object_count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            object_count += 1

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Counter display
    cv2.putText(
        img,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# WEBRTC CONFIG (STABLE VERSION)
# ===============================
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

# ===============================
# STREAMLIT WEBRTC
# ===============================
webrtc_streamer(
    key="original-live-camera",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.caption("Allow camera permission in browser if black screen appears.")
