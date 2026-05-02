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
st.write("Real-time object detection using YOLOv8 + camera")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.success("✅ Model loaded")

# ===============================
# SETTINGS
# ===============================
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)

# ===============================
# VIDEO CALLBACK
# ===============================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    results = model.predict(img, conf=conf_threshold, verbose=False)

    count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            count += 1

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.putText(
        img,
        f"Objects: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# WEBRTC CONFIG (IMPORTANT FIX)
# ===============================
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
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
# STREAM
# ===============================
webrtc_streamer(
    key="live-yolo",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
