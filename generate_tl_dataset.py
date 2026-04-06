import cv2
import os
import argparse
import numpy as np

def draw_traffic_light(frame, x, y, w, h, state):
    """Draw a simple traffic light overlay inside the given box."""
    if w <= 0 or h <= 0:
        return
        
    # Traffic light housing
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    
    # Split the box into three stacked light sections
    section_h = h / 3
    # Keep the circles inside the box
    max_r_w = w * 0.4
    max_r_h = section_h * 0.4
    radius = max(1, int(min(max_r_w, max_r_h)))
    
    # Circle centers
    cx = x + w // 2
    cy_red = y + int(section_h * 0.5)
    cy_yellow = y + int(section_h * 1.5)
    cy_green = y + int(section_h * 2.5)
    
    # Active colors in OpenCV's BGR format
    COLOR_RED_ACTIVE = (0, 0, 255)
    COLOR_YELLOW_ACTIVE = (0, 255, 255)
    COLOR_GREEN_ACTIVE = (0, 255, 0)
    
    # Color for inactive lights
    COLOR_OFF = (60, 60, 60)
    
    # Pick the active light
    r_color = COLOR_RED_ACTIVE if state == 'RED' else COLOR_OFF
    y_color = COLOR_YELLOW_ACTIVE if state == 'YELLOW' else COLOR_OFF
    g_color = COLOR_GREEN_ACTIVE if state == 'GREEN' else COLOR_OFF
    
    # Scale the outline with the light size
    thickness = max(1, int(radius * 0.15))
    
    # Draw the three lights
    cv2.circle(frame, (cx, cy_red), radius, r_color, -1)
    cv2.circle(frame, (cx, cy_yellow), radius, y_color, -1)
    cv2.circle(frame, (cx, cy_green), radius, g_color, -1)
    
    # Add a subtle outline
    cv2.circle(frame, (cx, cy_red), radius, (20, 20, 20), thickness)
    cv2.circle(frame, (cx, cy_yellow), radius, (20, 20, 20), thickness)
    cv2.circle(frame, (cx, cy_green), radius, (20, 20, 20), thickness)

def generate_dataset(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' not found.")
        return

    # Reuse the regular calibration helper to pick the box corners
    from get_coordinates import run_calibration
    
    print("Opening calibration window to define Traffic Light Box.")
    print("Please click exactly 2 points: Top-Left and Bottom-Right corners.")
    
    pts = run_calibration(video_path, num_points=2, title="Traffic Light Box")
    
    if pts is None:
        print("Error: Calibration was cancelled.")
        return

    x1, y1 = pts[0]
    x2, y2 = pts[1]

    x, y = int(min(x1, x2)), int(min(y1, y2))
    w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
    
    # Warn on tiny selections, but still allow them
    if w <= 5 or h <= 5:
        print("Warning: Selection is very small, but proceeding as requested.")

    print(f"Traffic Light Box Coordinates: x={x}, y={y}, w={w}, h={h}")

    # Open the source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    print(f"Traffic Light Box Coordinates: x={x}, y={y}, w={w}, h={h}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Basic signal timing
    green_frames = int(5 * fps)
    yellow_frames = int(2 * fps)
    red_frames = int(5 * fps)
    total_cycle = green_frames + yellow_frames + red_frames
    
    print(f"\nTiming Setup (FPS: {fps:.2f}):")
    print(f" - GREEN: 5s -> {green_frames} frames")
    print(f" - YELLOW: 2s -> {yellow_frames} frames")
    print(f" - RED: 5s -> {red_frames} frames")
    print("Processing video...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cv2.namedWindow("Dataset Generator (Press 'q' to quit)", cv2.WINDOW_NORMAL)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        cycle_pos = frame_count % total_cycle
        
        if cycle_pos < green_frames:
            state = 'GREEN'
        elif cycle_pos < green_frames + yellow_frames:
            state = 'YELLOW'
        else:
            state = 'RED'
            
        draw_traffic_light(frame, x, y, w, h, state)
        
        out.write(frame)
        cv2.imshow("Dataset Generator (Press 'q' to quit)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing safely interrupted by user.")
            break
            
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Traffic Light Dataset Generator")
    parser.add_argument("--input", "-i", type=str, default="/home/burner/coding/yolo_test/sample_input/14326971_2560_1440_60fps.mp4", help="Input video file path")
    parser.add_argument("--output", "-o", type=str, default="output_with_signal_2.mp4", help="Output video file path")
    
    args = parser.parse_args()
    generate_dataset(args.input, args.output)
