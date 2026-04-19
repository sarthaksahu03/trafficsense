import os
import re

import cv2
import easyocr


_reader = None


def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)


def is_valid_plate(text):
    return len(text) >= 5


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def _read_plate_text(reader, image, min_conf=0.25):
    best_text = "UNKNOWN"
    best_conf = 0.0
    ocr_results = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    for _, text, conf in ocr_results:
        cleaned = clean_text(text)
        if is_valid_plate(cleaned) and conf >= min_conf and conf > best_conf:
            best_text = cleaned
            best_conf = conf

    parts = []
    for bbox, text, conf in ocr_results:
        cleaned = clean_text(text)
        if not cleaned or conf < min_conf:
            continue

        ys = [point[1] for point in bbox]
        xs = [point[0] for point in bbox]
        parts.append((min(ys), min(xs), cleaned, conf))

    if parts:
        parts.sort(key=lambda item: (item[0], item[1]))
        combined_text = clean_text("".join(part[2] for part in parts))
        combined_conf = sum(part[3] for part in parts) / len(parts)
        if is_valid_plate(combined_text) and combined_conf > best_conf:
            best_text = combined_text
            best_conf = combined_conf

    return best_text, best_conf


def read_plate_image(image_path):
    if not image_path or not os.path.exists(image_path):
        return "UNKNOWN", 0.0

    plate_crop = cv2.imread(image_path)
    if plate_crop is None or plate_crop.size == 0:
        return "UNKNOWN", 0.0

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 15:
        return "UNKNOWN", 0.0

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_text = "UNKNOWN"
    best_conf = 0.0
    for candidate in (gray, thresholded):
        text, conf = _read_plate_text(_get_reader(), candidate, min_conf=0.25)
        if conf > best_conf:
            best_text = text
            best_conf = conf

    return best_text, best_conf
