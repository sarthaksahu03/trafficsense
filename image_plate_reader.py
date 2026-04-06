from ultralytics import YOLO
import easyocr
import cv2
import re

# Load the plate detector
model = YOLO("best.pt")

# Set up OCR
reader = easyocr.Reader(['en'], gpu=False)

# Read the test image
img = cv2.imread("test.jpg")
if img is None:
    raise IOError("❌ Cannot read image")

# Run plate detection
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

        # Clean the crop a bit before OCR
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Read text from the crop
        ocr_result = reader.readtext(
            gray,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=0
        )

        plate_text = clean_text("".join(ocr_result))

        # Draw the detection
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the recognized plate text
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

# Save the annotated result
cv2.imwrite("output_image.jpg", img)

print("✅ Done!")
print("📁 Output saved as: output_image.jpg")
