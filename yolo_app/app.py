import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Object Detector", layout="wide")

# ORIGINAL TITLE (HINDI BINAGO)
st.title("🎥 Live Object Detection & Tracing")
st.write("Real-time object detection using YOLOv8 + Streamlit Cloud camera")

st.info("📷 If black screen appears: allow camera permission + try Chrome browser")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.success("✅ Model Loaded Successfully")

# ===============================
# SIDEBAR SETTINGS
# ===============================
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# ===============================
# VIDEO CALLBACK FUNCTION
# ===============================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Mirror camera
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

    # Object counter display
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
# CLOUD-STABLE WEBRTC CONFIG
# ===============================
RTC_CONFIGURATION = {
    "iceServers": [
        # STUN ONLY (most stable for Streamlit Cloud)
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

# ===============================
# WEBRTC STREAM
# ===============================
webrtc_streamer(
    key="final-cloud-live",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    async_processing=True,
)

# ===============================
# NOTE
# ===============================
st.caption("✔ Allow camera permission in browser. If black screen persists, switch network or use Chrome.")
