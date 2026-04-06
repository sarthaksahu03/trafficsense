"""
Simple OpenCV helper for picking points on the first frame of a video.
"""

import cv2
import os
import numpy as np

# State shared with the OpenCV mouse callback
clicked_points = []
frame_dims = (0, 0)  # (width, height)
display_scale = 1.0

def mouse_callback(event, x, y, flags, param):
    """Handle clicks in the calibration window."""
    global clicked_points, frame_dims, display_scale
    num_points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < num_points:
            # Convert the click from display space back to the original frame
            orig_x = int(x / display_scale)
            orig_y = int(y / display_scale)
            
            # Clamp the point so it stays inside the frame
            orig_w, orig_h = frame_dims
            orig_x = max(0, min(orig_w - 1, orig_x))
            orig_y = max(0, min(orig_h - 1, orig_y))
            
            clicked_points.append([orig_x, orig_y])
            print(f"Clicked Point {len(clicked_points)}: ({orig_x}, {orig_y})")

def run_calibration(video_path, num_points=4, title="Calibration"):
    """Open the first frame and collect a fixed number of points from the user."""
    global clicked_points, frame_dims, display_scale
    clicked_points = []  # Reset points for each run

    # Make sure the video path is valid
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None

    # Grab the first frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None

    # Remember the original frame size
    frame_dims = (frame.shape[1], frame.shape[0])

    # Scale large frames down so they fit on screen
    max_w, max_h = 1280, 720
    display_scale = min(max_w / frame_dims[0], max_h / frame_dims[1])
    if display_scale > 1.0:
        display_scale = 1.0  # Leave smaller videos at their original size
        
    display_w = int(frame_dims[0] * display_scale)
    display_h = int(frame_dims[1] * display_scale)
    
    # Resize once and reuse it while the user clicks
    scaled_frame = cv2.resize(frame, (display_w, display_h))

    # Use AUTOSIZE because the frame is already resized
    window_name = f"{title} - Click {num_points} Points (Press 'q' or 'Enter' to confirm/exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback, param=num_points)
    print("\n" + "="*40)
    print(" INSTRUCTIONS")
    print("="*40)
    print(f"1. Click on {num_points} points.")
    print("   The window will close automatically after all clicks.")
    print("2. You can resize this window if needed.")
    print("3. Or press 'q' or 'Enter' to exit early.")
    print("="*40 + "\n")

    while True:
        # Show the scaled frame so click positions stay consistent
        display_frame = scaled_frame.copy()
        
        # Draw the points and connecting lines as the user clicks
        for i, point in enumerate(clicked_points):
            disp_pt = (int(point[0] * display_scale), int(point[1] * display_scale))
            cv2.circle(display_frame, disp_pt, 5, (0, 0, 255), -1)
            if i > 0:
                prev_disp = (int(clicked_points[i-1][0] * display_scale), int(clicked_points[i-1][1] * display_scale))
                cv2.line(display_frame, prev_disp, disp_pt, (0, 255, 0), 2)
        
        # Close the polygon once all points are in
        if len(clicked_points) == num_points and num_points > 2:
            first_disp = (int(clicked_points[0][0] * display_scale), int(clicked_points[0][1] * display_scale))
            last_disp = (int(clicked_points[-1][0] * display_scale), int(clicked_points[-1][1] * display_scale))
            cv2.line(display_frame, last_disp, first_disp, (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 13:  # 'q' or Enter
            break
            
        if len(clicked_points) == num_points:
            # Briefly show the final shape before closing
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
    # Quick local test
    VIDEO_PATH = "/home/burner/coding/yolo_test/sample_input/test_video_1.mp4"
    pts = run_calibration(VIDEO_PATH)
    if pts is not None:
        print("Calibration successful. Points:")
        print(pts)
