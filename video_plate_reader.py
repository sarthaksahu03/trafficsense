from ultralytics import YOLO
import easyocr
import cv2
import re
from collections import defaultdict

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Initialize EasyOCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# Video input
# -----------------------------
input_video = "test_video.mp4"
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise IOError("❌ Cannot open video")

# -----------------------------
# Output video
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output_with_plate_text.mp4",
    fourcc,
    fps,
    (width, height)
)

# -----------------------------
# Helpers
# -----------------------------
def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def is_valid_plate(text):
    return len(text) >= 6

# -----------------------------
# Store BEST OCR per plate
# key -> {text, confidence}
# -----------------------------
best_plate_read = defaultdict(lambda: {"text": "", "conf": 0.0})

print("🚀 Processing video with CONFIDENCE-LOCKED OCR...")

# -----------------------------
# Main loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, imgsz=640)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            key = (x1, y1, x2, y2)

            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            # ---------- OCR preprocessing ----------
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, gray = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # ---------- OCR with confidence ----------
            ocr_results = reader.readtext(
                gray,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            for (_, text, conf) in ocr_results:
                text = clean_text(text)

                if not is_valid_plate(text):
                    continue

                # 🔒 CONFIDENCE LOCK
                if conf > best_plate_read[key]["conf"]:
                    best_plate_read[key]["text"] = text
                    best_plate_read[key]["conf"] = conf

            final_text = best_plate_read[key]["text"]

            # ---------- Draw ----------
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            if final_text:
                cv2.putText(
                    frame,
                    final_text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    out.write(frame)

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

print("✅ Done!")
print("📁 Output saved as: output_with_plate_text.mp4")
