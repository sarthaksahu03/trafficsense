import cv2
import csv
import os
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO
import easyocr
import re
from image_utils import save_violation_crops
from violation_logger import ViolationLogger

# COCO class IDs we treat as vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]

def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def is_valid_plate(text):
    return len(text) >= 5

def read_plate_text_from_crop(reader, plate_crop, min_conf=0.4):
    if plate_crop is None or plate_crop.size == 0:
        return "UNKNOWN", 0.0

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 35:
        return "UNKNOWN", 0.0

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    candidates = [gray]
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(thresholded)

    best_text = "UNKNOWN"
    best_conf = 0.0

    for candidate in candidates:
        ocr_results = reader.readtext(candidate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        parts = []
        for bbox, text, conf in ocr_results:
            cleaned = clean_text(text)
            if not cleaned or conf < min_conf:
                continue

            ys = [point[1] for point in bbox]
            xs = [point[0] for point in bbox]
            parts.append((min(ys), min(xs), cleaned, conf))

        if not parts:
            continue

        parts.sort(key=lambda item: (item[0], item[1]))
        combined_text = clean_text("".join(part[2] for part in parts))
        combined_conf = sum(part[3] for part in parts) / len(parts)
        if is_valid_plate(combined_text) and combined_conf > best_conf:
            best_text = combined_text
            best_conf = combined_conf

        for _, _, text, conf in parts:
            if is_valid_plate(text) and conf > best_conf:
                best_text = text
                best_conf = conf

    return best_text, best_conf

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class TrafficLightDetector:
    def __init__(self, roi):
        self.roi = roi

    def get_state(self, frame):
        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "UNKNOWN"
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Red mask
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        
        # Green mask
        lower_green, upper_green = np.array([40, 50, 50]), np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        
        if red_pixels > 20 and red_pixels > green_pixels:
            return "RED"
        elif green_pixels > 20 and green_pixels > red_pixels:
            return "GREEN"
        return "UNKNOWN"

class StopLineDetector:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        
    def check_crossed(self, prev_pos, curr_pos):
        return intersect(self.pt1, self.pt2, prev_pos, curr_pos)

class PlateReader:
    def __init__(self, model_path="best.pt"):
        print("Initializing PlateReader...")
        self.plate_model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)

    def plate_crop_quality(self, plate_crop, detection_conf=0.0):
        if plate_crop is None or plate_crop.size == 0:
            return 0.0

        h, w = plate_crop.shape[:2]
        if w < 30 or h < 10:
            return 0.0

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = float(gray.std())
        area_factor = min((w * h) / 3000.0, 2.5)
        return (sharpness + contrast * 2.0) * max(area_factor, 0.5) * (1.0 + detection_conf)

    def plate_position_score(self, box, crop_w, crop_h):
        x1, y1, x2, y2 = box
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        if box_w == 0 or box_h == 0 or crop_w == 0 or crop_h == 0:
            return 0.0, None

        aspect_ratio = box_w / box_h
        if aspect_ratio < 0.7 or aspect_ratio > 8.0:
            return 0.0, None

        rel_cx = ((x1 + x2) / 2) / crop_w
        rel_cy = ((y1 + y2) / 2) / crop_h
        if rel_cx < 0.08 or rel_cx > 0.92 or rel_cy < 0.30 or rel_cy > 0.98:
            return 0.0, None

        center_score = 1.0 - min(abs(rel_cx - 0.5) / 0.5, 1.0) * 0.35
        vertical_score = 0.75 + min(max(rel_cy, 0.0), 1.0) * 0.25
        return center_score * vertical_score, (rel_cx, rel_cy)

    def is_same_plate_position(self, old_position, new_position, max_distance=0.30):
        if old_position is None or new_position is None:
            return old_position is None

        dx = old_position[0] - new_position[0]
        dy = old_position[1] - new_position[1]
        return (dx * dx + dy * dy) ** 0.5 <= max_distance

    def has_known_plate_text(self, plate_text):
        return plate_text not in ("", "-", "UNKNOWN", "Unknown", None)
        
    def read_plate(self, vehicle_crop):
        plate_text_disp = "UNKNOWN"
        best_conf = 0.0
        best_quality = 0.0
        best_position = None
        
        plate_results = self.plate_model(vehicle_crop, conf=0.25, imgsz=640, verbose=False, device='cpu')
        
        best_plate_crop = None
        for r in plate_results:
            for box in r.boxes:
                raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                box_w = max(0, raw_x2 - raw_x1)
                box_h = max(0, raw_y2 - raw_y1)
                if box_w == 0 or box_h == 0:
                    continue

                pad_x = max(2, int(box_w * 0.08))
                pad_y = max(3, int(box_h * 0.25))
                px1 = max(0, raw_x1 - pad_x)
                py1 = max(0, raw_y1 - pad_y)
                px2 = min(vehicle_crop.shape[1], raw_x2 + pad_x)
                py2 = min(vehicle_crop.shape[0], raw_y2 + pad_y)
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                
                if plate_crop.size > 0:
                    h, w = plate_crop.shape[:2]
                    if w < 30 or h < 10:
                        continue

                    detection_conf = float(box.conf[0]) if box.conf is not None else 0.0
                    position_score, position = self.plate_position_score(
                        (raw_x1, raw_y1, raw_x2, raw_y2),
                        vehicle_crop.shape[1],
                        vehicle_crop.shape[0]
                    )
                    if position_score == 0.0:
                        continue

                    quality = self.plate_crop_quality(plate_crop, detection_conf) * position_score
                    if quality > best_quality:
                        best_quality = quality
                        best_plate_crop = plate_crop.copy()
                        best_position = position
                        
                    plate_text, conf = read_plate_text_from_crop(self.reader, plate_crop, min_conf=0.4)
                    if conf > best_conf:
                        best_conf = conf
                        plate_text_disp = plate_text
                        best_plate_crop = plate_crop.copy()
                        best_position = position
        return plate_text_disp, best_plate_crop, best_conf, best_position

class RedLightSystem:
    def __init__(self, video_path, output_dir, stop_line_pts, traffic_light_roi):
        self.video_path = video_path
        self.output_dir = output_dir
        self.stop_line_pts = stop_line_pts
        self.traffic_light_roi = traffic_light_roi
        
        self.is_running = False
        
        print("Loading YOLOv8 Models for Red Light System...")
        self.vehicle_model = YOLO("vehicle.pt")
        self.plate_reader = PlateReader("best.pt")
        self.tl_detector = TrafficLightDetector(traffic_light_roi)
        self.stop_line = StopLineDetector(stop_line_pts[0], stop_line_pts[1])
        
    def stop(self):
        self.is_running = False

    def process_video(self):
        """
        Run the red-light detection loop and yield frame updates for the GUI.
        """
        self.is_running = True
        
        video_name = os.path.basename(self.video_path).split('.')[0]
        output_dir_violation = os.path.join(self.output_dir, "redlight_violations")
        output_video_path = os.path.join(self.output_dir, f"{video_name}_redlight_output.mp4")
        
        # Keep one logger per run
        self.logger = ViolationLogger(self.output_dir, video_name)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Could not open video source: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30
        fps = float(fps)
        evidence_window_frames = max(1, int(fps * 1.5))
        evidence_interval_frames = max(1, int(fps * 0.10))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        track_history = defaultdict(list)
        self.violated_records = {}  # track_id -> logged violation record
        
        frame_count = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            new_violations = []
                
            light_state = self.tl_detector.get_state(frame)
            state_color = (0, 0, 255) if light_state == "RED" else ((0, 255, 0) if light_state == "GREEN" else (0, 255, 255))
            
            # Draw the calibrated traffic-light region and current state
            tx1, ty1, tx2, ty2 = self.traffic_light_roi
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), state_color, 2)
            cv2.putText(frame, f"TL: {light_state}", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            # Draw the calibrated stop line
            sl_pt1, sl_pt2 = self.stop_line_pts
            cv2.line(frame, sl_pt1, sl_pt2, (0, 0, 255), 3)

            results = self.vehicle_model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.3, verbose=False, device='cpu')
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    xc = int((x1 + x2) / 2)
                    yc = int((y1 + y2) / 2)  # Track the box center
                    
                    current_pos = (xc, yc)
                    
                    crossed = False
                    if len(track_history[track_id]) > 0:
                        prev_pos = track_history[track_id][-1][1]
                        if self.stop_line.check_crossed(prev_pos, current_pos):
                            crossed = True
                            
                    track_history[track_id].append((frame_count, current_pos))
                    if len(track_history[track_id]) > fps * 2:
                        track_history[track_id].pop(0)

                    color = (0, 255, 0)
                    label = f"ID: {track_id}"
                    
                    if track_id in self.violated_records:
                        color = (0, 0, 255)
                        label += " VIOLATOR"
                        
                        # Try again later if the saved plate read is still weak
                        v_info = self.violated_records[track_id]
                        violation_frame = v_info.get("violation_frame", frame_count)
                        last_evidence_frame = v_info.get("last_evidence_frame", -evidence_interval_frames)
                        should_update_evidence = (
                            v_info.get("conf", 0.0) < 0.8
                            and frame_count - violation_frame <= evidence_window_frames
                            and frame_count - last_evidence_frame >= evidence_interval_frames
                        )

                        if should_update_evidence:
                            vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            plate_text, plate_crop, conf, plate_position = self.plate_reader.read_plate(vehicle_crop)
                            v_info["last_evidence_frame"] = frame_count
                            
                            existing_plate = v_info.get("plate")
                            has_known_existing_plate = self.plate_reader.has_known_plate_text(existing_plate)
                            has_same_plate_text = (
                                not has_known_existing_plate
                                or not self.plate_reader.has_known_plate_text(plate_text)
                                or plate_text == existing_plate
                            )
                            has_better_text = (
                                conf > v_info.get("conf", 0.0)
                                and has_same_plate_text
                                and self.plate_reader.has_known_plate_text(plate_text)
                            )
                            has_first_plate_image = plate_crop is not None and not v_info.get("plate_img")
                            has_same_plate_position = self.plate_reader.is_same_plate_position(
                                v_info.get("plate_position"),
                                plate_position
                            )
                            if (has_better_text or has_first_plate_image) and has_same_plate_position:
                                veh_path, plate_path = save_violation_crops(output_dir_violation, track_id, plate_text, vehicle_crop, plate_crop)
                                
                                # Update the cached record in place so the UI sees the better evidence
                                if has_better_text:
                                    v_info["plate"] = plate_text
                                v_info["vehicle_img"] = veh_path
                                v_info["plate_img"] = plate_path
                                v_info["conf"] = max(v_info.get("conf", 0.0), conf)
                                v_info["plate_position"] = plate_position or v_info.get("plate_position")
                                
                                # Persist the improved plate and image paths
                                self.logger.update_violation(track_id, v_info["plate"], veh_path, plate_path)
                                print(f"Upgraded evidence for ID: {track_id}, new plate: {v_info['plate']}")
                                
                    elif crossed and light_state == "RED" and track_id not in self.violated_records:
                        color = (0, 0, 255)
                        label += " VIOLATOR"
                        print(f"Violation detected! ID: {track_id}")
                        
                        vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        plate_text, plate_crop, conf, plate_position = self.plate_reader.read_plate(vehicle_crop)
                        
                        veh_path, plate_path = save_violation_crops(output_dir_violation, track_id, plate_text, vehicle_crop, plate_crop)
                        
                        violation_info = self.logger.log_violation(
                            vehicle_id=track_id, 
                            violation_type="Red Light Jump", 
                            plate_text=plate_text, 
                            vehicle_img_path=veh_path, 
                            plate_img_path=plate_path
                        )
                        
                        if violation_info:
                            violation_info["conf"] = conf
                            violation_info["plate_position"] = plate_position
                            violation_info["violation_frame"] = frame_count
                            violation_info["last_evidence_frame"] = frame_count
                            self.violated_records[track_id] = violation_info
                            new_violations.append(violation_info)

                    # Draw the tracked vehicle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            frame_count += 1
            
            # Hand the latest frame and violations back to the GUI loop
            yield {
                "frame": frame,
                "frame_count": frame_count,
                "new_violations": new_violations
            }

        cap.release()
        out.release()
        print("Processing complete!")

if __name__ == "__main__":
    # Quick local test
    VIDEO_PATH = "sample_input/test_video_1.mp4"
    OUTPUT_DIR = "output"
    
    # Update these for the clip you're testing with
    STOP_LINE_PTS = ((100, 600), (1100, 600)) 
    TRAFFIC_LIGHT_ROI = (600, 50, 650, 150)
    
    import sys
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    if os.path.exists(VIDEO_PATH):
        system = RedLightSystem(VIDEO_PATH, OUTPUT_DIR, STOP_LINE_PTS, TRAFFIC_LIGHT_ROI)
        for data in system.process_video():
            pass
    else:
        print(f"Test video {VIDEO_PATH} not found.")
