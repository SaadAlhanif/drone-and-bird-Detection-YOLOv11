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
st.set_page_config(page_title="Drone Detection")
st.title("🛸 Drone Detection (Video)")
st.write("ارفع فيديو، والنظام بيطلع لك فيديو عليه كشف (Drone/Bird).")

st.markdown("""
<style>
video {
    max-width: 200px !important;
    width: 100% !important;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Model (Auto download from Drive)
# =========================
MODEL_PATH = "best.pt"
FILE_ID = "1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل المودل من Google Drive..."):
            gdown.download(DRIVE_URL, MODEL_PATH, fuzzy=True, quiet=False)

    if not os.path.exists(MODEL_PATH):
        st.error("❌ فشل تحميل best.pt من Google Drive")
        st.stop()

    return YOLO(MODEL_PATH)

model = load_model()

names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
ALLOWED = {"drone", "bird"}


# =========================
# Controls (السلايدر)
# =========================
st.sidebar.header("⚙️ Settings")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
iou_thres  = st.sidebar.slider("IoU", 0.05, 0.95, 0.50, 0.05)

uploaded = st.file_uploader("📤 ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])

if uploaded is None:
    st.info("ارفع فيديو عشان نبدأ.")
    st.stop()


# =========================
# Save input
# =========================
tmp_dir = tempfile.mkdtemp()
input_path = os.path.join(tmp_dir, uploaded.name)

with open(input_path, "wb") as f:
    f.write(uploaded.read())

st.success("✅ تم رفع الفيديو")


# =========================
# Show input video (الأصلي فوق)
# =========================
st.subheader("🎬 الفيديو الأصلي")
st.video(input_path)
st.divider()


# =========================
# Read video
# =========================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    st.error("❌ ما قدرت أفتح الفيديو.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))


# =========================
# Processing (مثل اللي عندك + Status + Finished)
# =========================
st.subheader("⚙️ جاري المعالجة...")
progress = st.progress(0)
status = st.empty()

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = model.predict(
        frame,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False
    )

    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                name = names.get(cls, str(cls))

                # ✅ فلترة Drone/Bird فقط
                if name.lower() not in ALLOWED:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0

                # box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # label background + text (اسم الكلاس الحقيقي)
                txt = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y_top = max(y1 - th - 10, 0)
                cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y1), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    txt,
                    (x1 + 4, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

    writer.write(frame)

    if total_frames > 0:
        p = min(frame_idx / total_frames, 1.0)
        progress.progress(int(p * 100))
        status.write(f"Processing frame {frame_idx}/{total_frames} ...")
    else:
        if frame_idx % 30 == 0:
            status.write(f"Processing frame {frame_idx} ...")

cap.release()
writer.release()

progress.progress(100)
status.write("✅ Finished!")


# =========================
# Convert to H264
# =========================
final_output_path = os.path.join(tmp_dir, "output_h264.mp4")

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([
    ffmpeg, "-y",
    "-i", raw_output_path,
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-acodec", "aac",
    "-b:a", "128k",
    final_output_path
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# =========================
# Show output
# =========================
st.subheader("📌 الفيديو الناتج")
with open(final_output_path, "rb") as f:
    out_bytes = f.read()

st.video(out_bytes)

st.download_button(
    "⬇️ Download result video",
    data=out_bytes,
    file_name="drone_detection_output.mp4",
    mime="video/mp4"
)

shutil.rmtree(tmp_dir, ignore_errors=True)
