import os
import tempfile
import subprocess

import streamlit as st
from ultralytics import YOLO
import cv2
import imageio_ffmpeg
import gdown


# =========================
# Page UI (لا تغيير)
# =========================
st.set_page_config(page_title="Drone Detection", layout="wide")
st.title("🛸 Drone Detection (Video)")
st.write("ارفع فيديو، والنظام بيطلع لك فيديو عليه كشف الدرون + كلمة Drone فوقه.")


# =========================
# Model (تحميل تلقائي من Google Drive)
# =========================
MODEL_PATH = "best.pt"
GDRIVE_FILE_ID = "1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

@st.cache_resource
def load_model():
    # إذا الملف غير موجود في الريبو -> نزّله من Drive
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ best.pt غير موجود، جاري تحميله من Google Drive ...")
        try:
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            raise FileNotFoundError(
                "❌ ما قدرت أحمل best.pt من Google Drive.\n"
                "تأكد الرابط Public/Anyone with the link + حجم الملف مسموح.\n"
                f"تفاصيل الخطأ: {e}"
            )

    return YOLO(MODEL_PATH)

model = load_model()


# =========================
# Controls
# =========================
st.sidebar.header("⚙️ Settings")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
iou_thres  = st.sidebar.slider("IoU",        0.05, 0.95, 0.45, 0.05)

uploaded_file = st.file_uploader("📤 ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is None:
    st.stop()


# =========================
# Save upload to temp
# =========================
tmp_dir = tempfile.mkdtemp()
input_path = os.path.join(tmp_dir, uploaded_file.name)

with open(input_path, "wb") as f:
    f.write(uploaded_file.read())


# =========================
# Video IO init
# =========================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    st.error("❌ ما قدرت أفتح الفيديو.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (w, h))


# =========================
# UI progress
# =========================
progress = st.progress(0)
status = st.empty()


# =========================
# Helper: label mapping
# =========================
def get_label_name(cls_id: int) -> str:
    # أسماء الكلاسات من المودل (إذا موجودة)
    try:
        names = model.names  # dict or list
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        elif isinstance(names, list):
            return str(names[cls_id]) if cls_id < len(names) else str(cls_id)
    except Exception:
        pass

    # احتياط (حسب طلبك: Drone + Bird)
    if cls_id == 0:
        return "bird"
    if cls_id == 1:
        return "drone"
    return str(cls_id)


# =========================
# Process frames
# =========================
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # predict
    results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
    r = results[0]

    # draw boxes + labels (Drone/Bird)
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            conf   = float(b.conf[0].item()) if b.conf is not None else 0.0

            label_name = get_label_name(cls_id)
            label_text = f"{label_name} {conf:.2f}"

            # box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # label bg
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)

            # label text
            cv2.putText(frame, label_text, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    writer.write(frame)

    frame_idx += 1
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
