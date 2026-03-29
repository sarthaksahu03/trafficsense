"""
Interactive Video Coordinate Calibration Module
Validates discrete matrix mapping points matching arbitrary ROI bounds across asynchronous video streams.
"""

import cv2
import os
import numpy as np

# Global variables for the interactive window
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """Event hook anchoring 2D coordinates across local system GUI interactions."""
    global clicked_points
    num_points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < num_points:
            clicked_points.append([x, y])
            print(f"Clicked Point {len(clicked_points)}: ({x}, {y})")

def run_calibration(video_path, num_points=4, title="Calibration"):
    """Instantiates a blocking OpenCV frame query prompting `num_points` node anchors."""
    global clicked_points
    clicked_points = [] # Reset points for each run

    # Verify file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None

    # Setup Window
    window_name = f"{title} - Click {num_points} Points (Press 'q' or 'Enter' to confirm/exit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=num_points)

    print("\n" + "="*40)
    print(" INSTRUCTIONS")
    print("="*40)
    print(f"1. Click on {num_points} points.")
    print("   The window will close automatically after all clicks.")
    print("2. Or press 'q' or 'Enter' to exit early.")
    print("="*40 + "\n")

    while True:
        display_frame = frame.copy()
        
        # Draw points and lines as they are clicked
        for i, point in enumerate(clicked_points):
            cv2.circle(display_frame, tuple(point), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display_frame, tuple(clicked_points[i-1]), tuple(point), (0, 255, 0), 2)
        
        # Connect the last point to the first if enough points are clicked (optional visual)
        if len(clicked_points) == num_points and num_points > 2:
            cv2.line(display_frame, tuple(clicked_points[num_points-1]), tuple(clicked_points[0]), (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 13: # 'q' or Enter
            break
            
        if len(clicked_points) == num_points:
            # Give the user a moment to see the full polygon before closing automatically
            cv2.waitKey(500)
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(clicked_points) == num_points:
        return np.array(clicked_points).astype(np.float32)
    else:
        print(f"Warning: Only {len(clicked_points)} points clicked out of {num_points}. Returning None.")
        return None

if __name__ == "__main__":
    # Test block
    VIDEO_PATH = "/home/burner/coding/yolo_test/sample_input/test_video_1.mp4"
    pts = run_calibration(VIDEO_PATH)
    if pts is not None:
        print("Calibration successful. Points:")
        print(pts)

