import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# STEP 1: Page Configuration
st.set_page_config(page_title="AI Object Detector", layout="wide")

# STEP 2: Load Model (YOLOv8 Nano)
@st.cache_resource
def load_model():
    # Gagamit ng yolov8n.pt para mabilis mag-load at mag-detect
    return YOLO("yolov8n.pt")

model = load_model()

# PAREHAS NA TITLE (HINDI BINAGO)
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Sidebar Settings
st.sidebar.header("Settings")
# Naka-set sa 0.20 para mas mabilis lumabas ang mga labels
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.20)

# STEP 3: Detection Logic (Manual Plotting para sigurado ang labels)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Mirror Effect fix
    img = cv2.flip(img, 1)

    # YOLO Detection
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    # MANUAL DRAWING: Ito ang sigurado na magpapakita ng boxes at labels
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Kunin ang coordinates ng box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Drawing ng Green Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Pagkuha at pag-sulat ng label (person, phone, etc.)
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# STEP 4: UNIVERSAL WEBRTC CONFIG (Fix para sa Wi-Fi at Data)
# Gagamit ng multiple servers para malusutan ang firewalls
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]
}

webrtc_streamer(
    key="universal-detection-v3",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    # Nakakatulong ito para hindi mag-black screen sa mahinang Wi-Fi
    video_html_attrs={
        "style": {"width": "100%"},
        "controls": False,
        "autoPlay": True,
    },
)

