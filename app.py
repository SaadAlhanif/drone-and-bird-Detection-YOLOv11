import os
import tempfile
import subprocess

import streamlit as st
from ultralytics import YOLO
import cv2
import imageio_ffmpeg


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Drone Detection", layout="wide")
st.title("🛸 Drone Detection (Video)")
st.write("ارفع فيديو، والنظام بيطلع لك فيديو عليه كشف (Drone/Bird) فوق البوكس.")


# =========================
# Model
# =========================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ ما لقيت ملف المودل: {MODEL_PATH}\n"
            f"تأكد ان best.pt موجود بنفس مجلد app.py"
        )
    return YOLO(MODEL_PATH)

model = load_model()


# =========================
# Controls
# =========================
st.sidebar.header("⚙️ Settings")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
iou_thres  = st.sidebar.slider("IoU",        0.05, 0.95, 0.50, 0.05)
show_conf  = st.sidebar.checkbox("Show confidence on label", value=True)

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
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# =========================
# Main
# =========================
if uploaded is None:
    st.info("ارفع فيديو عشان يبدأ الكشف.")
    st.stop()

# Temp folder
tmp_dir = tempfile.mkdtemp()

# Save uploaded video to temp
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

# Writer (mp4v first, then we convert to h264)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

st.subheader("⏳ Processing")
progress = st.progress(0)
status = st.empty()

frame_idx = 0

# =========================
# Process frames
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Inference
    results = model(frame, conf=conf_thres, iou=iou_thres, verbose=False)[0]

    # Draw detections
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # اسم الكلاس من المودل (drone/bird)
        name = model.names.get(cls, str(cls))

        # النص المعروض
        if show_conf:
            label = f"{name} {conf:.2f}"
        else:
            label = f"{name}"

        # رسم البوكس + النص
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

    writer.write(frame)

    # Progress
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
# Convert to H.264 (better browser playback)
# =========================
final_output_path = os.path.join(tmp_dir, "output_h264.mp4")

try:
    convert_to_h264(raw_output_path, final_output_path)
    playable_path = final_output_path
except Exception:
    playable_path = raw_output_path


# =========================
# Show output
# =========================
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

st.caption("إذا الفيديو طويل جدًا ممكن ياخذ وقت بالمعالجة على Streamlit Cloud.")
