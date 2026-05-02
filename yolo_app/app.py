import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Object Detector", layout="wide")

st.title("🎥 Live Object Detection & Tracing (YOLOv8)")
st.write("Real-time object detection using camera + AI model")

st.info("📷 If black screen appears: allow camera permission or try another network (mobile hotspot).")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.success("✅ YOLO Model Loaded")

# ===============================
# CONFIDENCE
# ===============================
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# ===============================
# VIDEO CALLBACK
# ===============================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Mirror camera
    img = cv2.flip(img, 1)

    # YOLO detection
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

            # Label text
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Object counter
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
# ULTRA STABLE WEBRTC CONFIG
# ===============================
RTC_CONFIGURATION = {
    "iceServers": [
        # STUN servers (basic connection)
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},

        # TURN server (fallback relay - FIX FOR BLACK SCREEN)
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
}

# ===============================
# STREAMLIT WEBRTC
# ===============================
webrtc_streamer(
    key="ultra-stable-live-camera",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# ===============================
# DEBUG INFO
# ===============================
st.caption("✔ If camera does not appear, check browser permissions (🔒 icon) and refresh page.")
