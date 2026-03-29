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

# COCO mapping
VEHICLE_CLASSES = [2, 3, 5, 7]

def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def is_valid_plate(text):
    return len(text) >= 5

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
        
        # Red range
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        
        # Green range
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
        
    def read_plate(self, vehicle_crop):
        plate_text_disp = "UNKNOWN"
        best_conf = 0.0
        
        plate_results = self.plate_model(vehicle_crop, conf=0.25, imgsz=640, verbose=False, device='cpu')
        
        best_plate_crop = None
        for r in plate_results:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                
                if plate_crop.size > 0:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    ocr_results = self.reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    
                    for (_, text, conf) in ocr_results:
                        cleaned = clean_text(text)
                        if is_valid_plate(cleaned) and conf > best_conf:
                            best_conf = conf
                            plate_text_disp = cleaned
                            best_plate_crop = plate_crop.copy()
        return plate_text_disp, best_plate_crop, best_conf

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
        Primary engine loop mapping YOLO bounds to physical crossing heuristics.
        Execution context bridges iteratively into `yield` buffers handling Tkinter thread locking implicitly.
        """
        self.is_running = True
        
        video_name = os.path.basename(self.video_path).split('.')[0]
        output_dir_violation = os.path.join(self.output_dir, "redlight_violations")
        output_video_path = os.path.join(self.output_dir, f"{video_name}_redlight_output.mp4")
        
        # Init logger
        self.logger = ViolationLogger(self.output_dir, video_name)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Could not open video source: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        track_history = defaultdict(list)
        self.violated_records = {} # Map track_id -> dict reference
        
        frame_count = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            new_violations = []
                
            light_state = self.tl_detector.get_state(frame)
            state_color = (0, 0, 255) if light_state == "RED" else ((0, 255, 0) if light_state == "GREEN" else (0, 255, 255))
            
            # Renders traffic state explicitly bounding the fixed calibration node bounds
            tx1, ty1, tx2, ty2 = self.traffic_light_roi
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), state_color, 2)
            cv2.putText(frame, f"TL: {light_state}", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            # Anchor evaluation segment bounds
            sl_pt1, sl_pt2 = self.stop_line_pts
            cv2.line(frame, sl_pt1, sl_pt2, (0, 0, 255), 3)

            results = self.vehicle_model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.3, verbose=False, device='cpu')
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    xc = int((x1 + x2) / 2)
                    yc = int((y1 + y2) / 2) # Use center
                    
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
                        
                        # Attempt refinement if plate OCR confidence is low
                        v_info = self.violated_records[track_id]
                        if v_info.get("conf", 0.0) < 0.8:
                            vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            plate_text, plate_crop, conf = self.plate_reader.read_plate(vehicle_crop)
                            
                            if conf > v_info.get("conf", 0.0):
                                veh_path, plate_path = save_violation_crops(output_dir_violation, track_id, plate_text, vehicle_crop, plate_crop)
                                
                                # By-reference dictionary mutations inherently cascade to active GUI modal threads
                                v_info["plate"] = plate_text
                                v_info["vehicle_img"] = veh_path
                                v_info["plate_img"] = plate_path
                                v_info["conf"] = conf
                                
                                # Instruct logger to rewrite CSV memory mapping ensuring persistent data validation
                                self.logger.update_violation(track_id, plate_text, veh_path, plate_path)
                                print(f"Upgraded evidence for ID: {track_id}, new plate: {plate_text}")
                                
                    elif crossed and light_state == "RED" and track_id not in self.violated_records:
                        color = (0, 0, 255)
                        label += " VIOLATOR"
                        print(f"Violation detected! ID: {track_id}")
                        
                        vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        plate_text, plate_crop, conf = self.plate_reader.read_plate(vehicle_crop)
                        
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
                            self.violated_records[track_id] = violation_info
                            new_violations.append(violation_info)

                    # Draw rect and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            frame_count += 1
            
            # Flush buffer array context into main process loop
            yield {
                "frame": frame,
                "frame_count": frame_count,
                "new_violations": new_violations
            }

        cap.release()
        out.release()
        print("Processing complete!")

if __name__ == "__main__":
    # Test block
    VIDEO_PATH = "sample_input/test_video_1.mp4"
    OUTPUT_DIR = "output"
    
    # Configure these according to the test video
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
