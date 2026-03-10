"""
Coordinate Calibration Tool

This utility script allows the user to click on points in a video frame 
to determine pixel coordinates. It is used to define the Region of Interest (ROI)
for perspective transformation in the main speed detection script.

Usage:
    Run the script. A window will appear showing the first frame of the video.
    Click on the 4 corners of the road section you want to monitor.
    The coordinates will be printed to the console.
    Press 'q' to exit.
"""

import cv2
import os
import numpy as np

# Global variables for the interactive window
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """
    Callback function for mouse events.
    Records coordinates when the left mouse button is clicked.
    """
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Clicked Point {len(clicked_points)}: ({x}, {y})")

def run_calibration(video_path):
    """
    Opens the video and lets the user click 4 points to define the ROI.
    Returns:
        np.ndarray: Array of the 4 clicked points, or None if failed/cancelled.
    """
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
    window_name = "Calibration - Click 4 Points (Press 'q' or 'Enter' to confirm/exit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*40)
    print(" INSTRUCTIONS")
    print("="*40)
    print("1. Click on the 4 corners of the road section.")
    print("   (Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left)")
    print("   The window will close automatically after 4 clicks.")
    print("2. Or press 'q' or 'Enter' to exit early.")
    print("="*40 + "\n")

    while True:
        display_frame = frame.copy()
        
        # Draw points and lines as they are clicked
        for i, point in enumerate(clicked_points):
            cv2.circle(display_frame, tuple(point), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display_frame, tuple(clicked_points[i-1]), tuple(point), (0, 255, 0), 2)
        
        # Connect the last point to the first if 4 points are clicked
        if len(clicked_points) == 4:
            cv2.line(display_frame, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 13: # 'q' or Enter
            break
            
        if len(clicked_points) == 4:
            # Give the user a moment to see the full polygon before closing automatically
            cv2.waitKey(500)
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(clicked_points) == 4:
        return np.array(clicked_points).astype(np.float32)
    else:
        print(f"Warning: Only {len(clicked_points)} points clicked. Returning None.")
        return None

if __name__ == "__main__":
    # Test block
    VIDEO_PATH = "/home/burner/coding/yolo_test/sample_input/test_video_1.mp4"
    pts = run_calibration(VIDEO_PATH)
    if pts is not None:
        print("Calibration successful. Points:")
        print(pts)

