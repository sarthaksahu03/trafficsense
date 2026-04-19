import cv2
import csv
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from image_utils import save_violation_crops
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
        
    def stop(self):
        self.is_running = False

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

    def detect_plate_crop(self, vehicle_crop):
        """
        Return the clearest detected plate crop and its quality score.

        Speed violations do OCR only when a violation is opened from the UI.
        During frame processing we only keep improving the evidence image.
        """
        best_plate_crop = None
        best_quality = 0.0
        best_position = None

        if vehicle_crop is None or vehicle_crop.size == 0:
            return best_plate_crop, best_quality, best_position

        crop_h, crop_w = vehicle_crop.shape[:2]
        plate_results = self.plate_model(vehicle_crop, conf=0.25, imgsz=640, verbose=False, device='cpu')

        for r in plate_results:
            for box in r.boxes:
                raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                box_w = max(0, raw_x2 - raw_x1)
                box_h = max(0, raw_y2 - raw_y1)
                if box_w == 0 or box_h == 0:
                    continue

                position_score, position = self.plate_position_score(
                    (raw_x1, raw_y1, raw_x2, raw_y2),
                    crop_w,
                    crop_h
                )
                if position_score == 0.0:
                    continue

                pad_x = max(2, int(box_w * 0.08))
                pad_y = max(3, int(box_h * 0.25))
                px1 = max(0, raw_x1 - pad_x)
                py1 = max(0, raw_y1 - pad_y)
                px2 = min(crop_w, raw_x2 + pad_x)
                py2 = min(crop_h, raw_y2 + pad_y)

                plate_crop = vehicle_crop[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue

                h, w = plate_crop.shape[:2]
                if w < 30 or h < 10:
                    continue

                detection_conf = float(box.conf[0]) if box.conf is not None else 0.0
                quality = self.plate_crop_quality(plate_crop, detection_conf) * position_score
                if quality > best_quality:
                    best_quality = quality
                    best_plate_crop = plate_crop.copy()
                    best_position = position

        return best_plate_crop, best_quality, best_position

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
        if fps == 0 or np.isnan(fps):
            fps = 30
        fps = float(fps)
        evidence_window_frames = max(1, int(fps * 1.5))
        evidence_interval_frames = max(1, int(fps * 0.10))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        view_transformer = ViewTransformer(self.source_points, self.target_width, self.target_height)

        track_history = defaultdict(list)
        vehicle_speeds = {}
        saved_ids = set()
        violated_ids = set()
        violated_records = {}

        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_header = ["Vehicle_ID", "Speed_kmph", "Frame_Number", "License_Plate"]
        csv_writer.writerow(csv_header)
        violation_rows = {}
        
        frame_count = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            new_violations = []
            updated_violations = []

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
                                
                                vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                                plate_text_disp = "UNKNOWN"
                                plate_crop, plate_score, plate_position = self.detect_plate_crop(vehicle_crop)
                                
                                violation_rows[track_id] = [track_id, f"{speed_disp:.2f}", frame_count, plate_text_disp]
                                csv_writer.writerow(violation_rows[track_id])
                                csv_file.flush()  # Write each violation row immediately
                                
                                vehicle_img_path, plate_img_path = save_violation_crops(
                                    output_dir_violation,
                                    track_id,
                                    plate_text_disp,
                                    vehicle_crop,
                                    plate_crop
                                )
                                
                                violation_info = {
                                    "id": track_id,
                                    "speed": speed_disp,
                                    "plate": plate_text_disp,
                                    "image_path": vehicle_img_path,
                                    "vehicle_img": vehicle_img_path,
                                    "plate_img": plate_img_path,
                                    "conf": 0.0,
                                    "plate_score": plate_score,
                                    "plate_position": plate_position,
                                    "violation_frame": frame_count,
                                    "last_evidence_frame": frame_count
                                }
                                violated_records[track_id] = violation_info
                                new_violations.append(violation_info)

                    if track_id in violated_records:
                        violation_info = violated_records[track_id]
                        violation_frame = violation_info.get("violation_frame", frame_count)
                        last_evidence_frame = violation_info.get("last_evidence_frame", -evidence_interval_frames)
                        should_update_evidence = (
                            frame_count - violation_frame <= evidence_window_frames
                            and frame_count - last_evidence_frame >= evidence_interval_frames
                        )

                        if should_update_evidence:
                            vehicle_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            plate_crop, plate_score, plate_position = self.detect_plate_crop(vehicle_crop)
                            previous_score = violation_info.get("plate_score", 0.0)
                            previous_position = violation_info.get("plate_position")
                            violation_info["last_evidence_frame"] = frame_count

                            has_clearer_plate = plate_crop is not None and (
                                not violation_info.get("plate_img") or plate_score > previous_score * 1.10
                            )
                            has_same_plate_position = self.is_same_plate_position(previous_position, plate_position)
                            if has_clearer_plate and has_same_plate_position:
                                veh_path, plate_path = save_violation_crops(
                                    output_dir_violation,
                                    track_id,
                                    violation_info.get("plate", "UNKNOWN"),
                                    vehicle_crop,
                                    plate_crop
                                )

                                violation_info["vehicle_img"] = veh_path or violation_info.get("vehicle_img", "")
                                violation_info["image_path"] = violation_info["vehicle_img"]
                                violation_info["plate_img"] = plate_path or violation_info.get("plate_img", "")
                                violation_info["plate_score"] = plate_score
                                violation_info["plate_position"] = plate_position or previous_position
                                updated_violations.append(violation_info.copy())

                    if track_id not in saved_ids and len(track_history[track_id]) > 10 and is_inside:
                        saved_ids.add(track_id)
                        img_name = f"vehicle_id_{track_id}.jpg"
                        img_path = os.path.join(output_dir_all, img_name)
                        cv2.imwrite(img_path, frame)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            cv2.polylines(frame, [self.source_points.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            out.write(frame)
            frame_count += 1
            
            # Keep the payload format simple for the GUI loop
            yield {
                "frame": frame,
                "frame_count": frame_count,
                "new_violations": new_violations,
                "updated_violations": updated_violations
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
