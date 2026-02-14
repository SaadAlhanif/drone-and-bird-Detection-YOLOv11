import os
import tempfile
import subprocess

import streamlit as st
from ultralytics import YOLO
import cv2
import imageio_ffmpeg
import requests


# =========================
# Google Drive download helper
# =========================
GDRIVE_URL = "https://drive.google.com/file/d/1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ/view?usp=sharing"
GDRIVE_FILE_ID = "1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ"  # extracted from your link


def download_from_gdrive(file_id: str, dest_path: str):
    """
    Download a file from Google Drive to dest_path (handles large file confirmation token).
    """
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"

    # 1) first request
    response = session.get(url, params={"id": file_id}, stream=True)
    response.raise_for_status()

    # 2) handle confirm token for large files
    confirm_token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
            break

    if confirm_token:
        response = session.get(url, params={"id": file_id, "confirm": confirm_token}, stream=True)
        response.raise_for_status()

    # 3) save
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Drone Detection")
st.title("🛸 Drone Detection (Video)")
st.write("ارفع فيديو، والنظام بيطلع لك فيديو عليه كشف الدرون + كلمة Drone فوقه.")

# ✅ ثابت: تصغير عرض الفيديو (Input/Output) إلى 700px
st.markdown("""
<style>
video {
    max-width: 200px !important;
    width: 100% !important;
    height: auto !important;
    display: block;
    margin-left: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Model
# =========================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    # ✅ إذا best.pt غير موجود: نزّله من Google Drive
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل best.pt من Google Drive ..."):
            try:
                download_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
            except Exception as e:
                raise FileNotFoundError(
                    "❌ ما قدرت أحمل best.pt من Google Drive.\n"
                    f"السبب: {e}\n"
                    "تأكد إن الرابط Public (Anyone with the link)."
                )

    # ✅ تأكد انه نزل فعلاً
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
        raise FileNotFoundError("❌ فشل تحميل best.pt أو الملف حجمه غير صحيح.")

    return YOLO(MODEL_PATH)

model = load_model()


# =========================
# Controls
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
    f.write(uploaded.getbuffer())

st.success("✅ تم رفع الفيديو")


# =========================
# Read video
# =========================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    st.error("❌ ما قدرت أفتح الفيديو.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
if width <= 0 or height <= 0:
    st.error("❌ ما قدرت أحدد أبعاد الفيديو.")
    cap.release()
    st.stop()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


# =========================
# Output writer
# =========================
raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

if not writer.isOpened():
    st.error("❌ ما قدرت أفتح VideoWriter. جرّب فيديو ثاني.")
    cap.release()
    st.stop()


# =========================
# Show input video
# =========================
st.subheader("🎬 الفيديو الأصلي")
with open(input_path, "rb") as f:
    st.video(f.read())

st.divider()


# =========================
# Processing
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
    results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)

    if results and len(results) > 0:
        r = results[0]

        # أسماء الكلاسات من المودل (مثل: ['drone','bird'] أو غيرها)
        names = r.names if hasattr(r, "names") and r.names is not None else model.names

        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0

                # ✅ يطلع اسم الكلاس الحقيقي (Drone / Bird ...)
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                label = (
                    names.get(cls_id, str(cls_id)) if isinstance(names, dict)
                    else (names[cls_id] if 0 <= cls_id < len(names) else str(cls_id))
                )

                # box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # label background + text
                txt = f"{label} {conf:.2f}"
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
