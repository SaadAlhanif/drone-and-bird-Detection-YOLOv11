import os
import tempfile
import subprocess
import shutil

import streamlit as st
import cv2
import imageio_ffmpeg
import gdown
from ultralytics import YOLO


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Drone Detection", layout="wide")
st.title("🛸 Drone Detection (Video)")
st.write("ارفع فيديو، والنظام بيطلع لك فيديو عليه كشف (Drone/Bird) فوق البوكس.")


# =========================
# Model (Auto download)
# =========================
MODEL_PATH = "best.pt"

# 🔗 Google Drive file ID (من الرابط اللي أرسلته)
FILE_ID = "1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    # لو المودل غير موجود -> حمّله من درايف
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل المودل من Google Drive..."):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()


# =========================
# Controls
# =========================
st.sidebar.header("⚙️ Settings")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
iou_thres  = st.sidebar.slider("IoU",        0.05, 0.95, 0.50, 0.05)
show_conf  = st.sidebar.checkbox("Show confidence on label", value=True)
imgsz = st.sidebar.select_slider("Inference size (imgsz)", options=[320, 416, 512, 640], value=640)

uploaded = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi", "mkv"])


# =========================
# Helpers
# =========================
def convert_to_h264(src, dst):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-acodec", "aac",
        "-b:a", "128k",
        dst
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore"))

def get_color(name: str):
    n = name.lower()
    if "drone" in n:
        return (0, 0, 255)      # Red
    if "bird" in n:
        return (255, 0, 0)      # Blue
    return (0, 255, 0)


# ✅ فقط هذي الإضافة: نحدد الكلاسات المسموح فيها
ALLOWED_CLASSES = {"drone", "bird"}


# =========================
# Main
# =========================
if uploaded is None:
    st.info("ارفع فيديو عشان يبدأ الكشف.")
    st.stop()

tmp_dir = tempfile.mkdtemp()

try:
    # Save uploaded video
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.read())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ ما قدرت أفتح الفيديو. جرّب فيديو آخر.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

    raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    st.subheader("⏳ Processing")
    progress = st.progress(0)
    status = st.empty()

    frame_idx = 0
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.predict(
            frame,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=imgsz,
            verbose=False
        )[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = names.get(cls, str(cls))

            # ✅ فقط هذي الإضافة: فلترة الكلاسين (Drone/Bird)
            if name.lower() not in ALLOWED_CLASSES:
                continue

            label = f"{name} {conf:.2f}" if show_conf else f"{name}"
            color = get_color(name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

        writer.write(frame)

        if total_frames > 0:
            p = min(frame_idx / total_frames, 1.0)
            progress.progress(int(p * 100))
            status.write(f"Processing frame {frame_idx}/{total_frames} ...")

    cap.release()
    writer.release()

    progress.progress(100)
    status.write("✅ Finished!")

    # Convert to H.264
    final_output_path = os.path.join(tmp_dir, "output_h264.mp4")
    try:
        convert_to_h264(raw_output_path, final_output_path)
        playable_path = final_output_path
    except Exception:
        playable_path = raw_output_path

    st.subheader("📌 الفيديو الناتج")
    with open(playable_path, "rb") as f:
        out_bytes = f.read()

    st.video(out_bytes)

    st.download_button(
        "⬇️ Download result video",
        data=out_bytes,
        file_name="drone_detection_output.mp4",
        mime="video/mp4"
    )

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
