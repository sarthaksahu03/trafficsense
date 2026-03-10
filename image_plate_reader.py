from ultralytics import YOLO
import easyocr
import cv2
import re

# Load YOLO model
model = YOLO("best.pt")

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

# Read image
img = cv2.imread("test.jpg")
if img is None:
    raise IOError("❌ Cannot read image")

# Run YOLO detection
results = model(img, conf=0.25, imgsz=640)

def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        plate = img[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        # Preprocess for OCR
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # OCR
        ocr_result = reader.readtext(
            gray,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=0
        )

        plate_text = clean_text("".join(ocr_result))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text
        if plate_text:
            cv2.putText(
                img,
                plate_text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

# Save output image
cv2.imwrite("output_image.jpg", img)

print("✅ Done!")
print("📁 Output saved as: output_image.jpg")
