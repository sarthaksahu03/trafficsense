import cv2
import csv
import os
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO
import easyocr
import re
VEHICLE_CLASSES = [2, 3, 5, 7]

class ViewTransformer:
    """
    Maps points from the camera view into a top-down plane so speed can be estimated more consistently.
    """
    def __init__(self, source: np.ndarray, target_width: float, target_height: float):
        source = source.astype(np.float32)
        target = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ]).astype(np.float32)
        
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def is_valid_plate(text):
    return len(text) >= 5

class SpeedDetector:
    def __init__(self, video_path, output_dir, source_points, speed_limit=40, max_speed_threshold=200):
        self.video_path = video_path
        self.output_dir = output_dir
        self.source_points = source_points
        self.speed_limit = speed_limit
        self.max_speed_threshold = max_speed_threshold
        
        # Approximate real-world size of the calibrated region
        self.target_width = 20
        self.target_height = 50
        
        self.is_running = False
        
        print("Loading YOLOv8 Models...")
        self.vehicle_model = YOLO("vehicle.pt")
        self.plate_model = YOLO("best.pt")
        
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def stop(self):
        self.is_running = False

    def process_video(self):
        """
        Process the video frame by frame and yield data the GUI can consume.
        """
        self.is_running = True
        
        video_name = os.path.basename(self.video_path).split('.')[0]
        output_dir_all = os.path.join(self.output_dir, "all_vehicles")
        output_dir_violation = os.path.join(self.output_dir, "violations")
        os.makedirs(output_dir_all, exist_ok=True)
        os.makedirs(output_dir_violation, exist_ok=True)
        
        csv_path = os.path.join(self.output_dir, f"{video_name}_violations.csv")
        output_video_path = os.path.join(self.output_dir, f"{video_name}_output.mp4")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            yield {"error": f"Could not open video source: {self.video_path}"}
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        view_transformer = ViewTransformer(self.source_points, self.target_width, self.target_height)

        track_history = defaultdict(list)
        vehicle_speeds = {}
        saved_ids = set()
        violated_ids = set()

        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Vehicle_ID", "Speed_kmph", "Frame_Number", "License_Plate"])
        
        frame_count = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            new_violations = []

            results = self.vehicle_model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.3, verbose=False, device='cpu')
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                points_to_transform = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    xc = (x1 + x2) / 2
                    yc = y2 
                    points_to_transform.append([xc, yc])
                
                points_to_transform = np.array(points_to_transform)
                transformed_points = view_transformer.transform_points(points_to_transform)

                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    xc, yc = points_to_transform[i]

                    current_pos_transformed = transformed_points[i]
                    track_history[track_id].append((frame_count, current_pos_transformed))
                    
                    if len(track_history[track_id]) > fps * 2:
                         track_history[track_id].pop(0)

                    speed_val = 0
                    # Wait for enough tracking history (1/2 second) to stabilize bounding box jitter
                    history_required = max(5, int(fps / 2))
                    if len(track_history[track_id]) >= history_required: 
                        # Use a longer lookback, up to 1.5 seconds, to compute a smoother average speed
                        lookback_idx = max(0, len(track_history[track_id]) - int(fps * 1.5))
                        prev_frame, prev_pos = track_history[track_id][lookback_idx]
                        curr_frame, curr_pos = track_history[track_id][-1]
                        
                        distance_meters = np.linalg.norm(curr_pos - prev_pos)
                        time_seconds = (curr_frame - prev_frame) / fps
                        
                        if time_seconds > 0:
                            speed_mps = distance_meters / time_seconds
                            speed_kmph = speed_mps * 3.6
                            
                            if speed_kmph < self.max_speed_threshold:
                                if track_id in vehicle_speeds:
                                    # Lighter exponential smoothing since window is naturally larger
                                    speed_val = 0.5 * vehicle_speeds[track_id] + 0.5 * speed_kmph
                                else:
                                    speed_val = speed_kmph
                                
                                vehicle_speeds[track_id] = speed_val

                    color = (0, 255, 0)
                    label = f"ID: {track_id}"
                    
                    is_inside = cv2.pointPolygonTest(self.source_points, (xc, yc), False) >= 0

                    if is_inside and track_id in vehicle_speeds:
                        speed_disp = vehicle_speeds[track_id]
                        label += f" {speed_disp:.0f} km/h"
                        
                        if speed_disp > self.speed_limit:
                            color = (0, 0, 255)
                            
                            if track_id not in violated_ids:
                                violated_ids.add(track_id)
                                
                                plate_text_disp = "UNKNOWN"
                                best_conf = 0.0
                                
                                vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                                
                                if vehicle_crop.size > 0:
                                    plate_results = self.plate_model(vehicle_crop, conf=0.25, imgsz=640, verbose=False)
                                    
                                    for r in plate_results:
                                        for box in r.boxes:
                                            px1, py1, px2, py2 = map(int, box.xyxy[0])
                                            plate_crop = vehicle_crop[py1:py2, px1:px2]
                                            
                                            if plate_crop.size > 0:
                                                h, w = plate_crop.shape[:2]
                                                if w < 30 or h < 10: continue
                                                
                                                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                                
                                                # Skip OCR on blurry plate crops
                                                if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
                                                    continue

                                                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                                                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                                
                                                ocr_results = self.reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                                                
                                                for (_, text, conf) in ocr_results:
                                                    cleaned = clean_text(text)
                                                    # Only keep plate reads that clear a basic confidence bar
                                                    if is_valid_plate(cleaned) and conf > 0.4 and conf > best_conf:
                                                        best_conf = conf
                                                        plate_text_disp = cleaned
                                
                                csv_writer.writerow([track_id, f"{speed_disp:.2f}", frame_count, plate_text_disp])
                                csv_file.flush()  # Write each violation row immediately
                                
                                img_path = ""
                                if vehicle_crop.size > 0:
                                    img_name = f"violation_id_{track_id}_speed_{speed_disp:.0f}_plate_{plate_text_disp}.jpg"
                                    img_path = os.path.join(output_dir_violation, img_name)
                                    cv2.imwrite(img_path, vehicle_crop)
                                
                                violation_info = {
                                    "id": track_id,
                                    "speed": speed_disp,
                                    "plate": plate_text_disp,
                                    "image_path": img_path
                                }
                                new_violations.append(violation_info)

                    if track_id not in saved_ids and len(track_history[track_id]) > 10 and is_inside:
                        saved_ids.add(track_id)
                        img_name = f"vehicle_id_{track_id}.jpg"
                        img_path = os.path.join(output_dir_all, img_name)
                        cv2.imwrite(img_path, frame)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if track_id in violated_ids and label.endswith("km/h"):
                        cv2.putText(frame, "PLATE LOGGED", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.polylines(frame, [self.source_points.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            out.write(frame)
            frame_count += 1
            
            # Keep the payload format simple for the GUI loop
            yield {
                "frame": frame,
                "frame_count": frame_count,
                "new_violations": new_violations
            }

        cap.release()
        out.release()
        csv_file.close()

if __name__ == "__main__":
    # Quick local test
    VIDEO_PATH = "/home/burner/coding/yolo_test/sample_input/test_video_1.mp4"
    OUTPUT_DIR = "output"
    SOURCE_POINTS = np.array([[470, 257], [948, 270], [1274, 514], [334, 449]]).astype(np.float32)
    
    detector = SpeedDetector(VIDEO_PATH, OUTPUT_DIR, SOURCE_POINTS)
    print("Testing SpeedDetector generator (processing 10 frames)...")
    for i, data in enumerate(detector.process_video()):
        print(f"Frame {data['frame_count']} processed. Violations in this frame: {len(data['new_violations'])}")
        if i >= 10:
            detector.stop()
            break
