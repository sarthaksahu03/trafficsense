import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import os
import numpy as np

# Import our refactored modules
from get_coordinates import run_calibration
from speed_detection import SpeedDetector

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("ALPR & Speed Detection System")
        self.geometry("1200x800")
        
        # Configure grid layout (1x2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- State Variables ---
        self.video_path = None
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.source_points = None
        self.is_processing = False
        self.detector = None
        self.processing_thread = None

        # --- UI Build ---
        self._build_sidebar()
        self._build_main_view()

    def _build_sidebar(self):
        # Sidebar Frame
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(9, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="System Setup", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Video Selection
        self.btn_select_video = ctk.CTkButton(self.sidebar, text="Select Input Video", command=self.select_video)
        self.btn_select_video.grid(row=1, column=0, padx=20, pady=10)
        self.lbl_video_path = ctk.CTkLabel(self.sidebar, text="No video selected", text_color="gray", wraplength=250)
        self.lbl_video_path.grid(row=2, column=0, padx=20, pady=(0, 10))

        # Output Dir Selection
        self.btn_select_output = ctk.CTkButton(self.sidebar, text="Select Output Folder", command=self.select_output)
        self.btn_select_output.grid(row=3, column=0, padx=20, pady=10)
        self.lbl_output_path = ctk.CTkLabel(self.sidebar, text=f"Output: {self.output_dir}", text_color="gray", wraplength=250)
        self.lbl_output_path.grid(row=4, column=0, padx=20, pady=(0, 10))

        # Speed Settings
        self.lbl_speed_limit = ctk.CTkLabel(self.sidebar, text="Speed Limit (km/h): 40")
        self.lbl_speed_limit.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_speed_limit = ctk.CTkSlider(self.sidebar, from_=10, to=150, number_of_steps=140, command=self.update_speed_label)
        self.slider_speed_limit.set(40)
        self.slider_speed_limit.grid(row=6, column=0, padx=20, pady=(0, 10))

        self.lbl_max_speed = ctk.CTkLabel(self.sidebar, text="Max Threshold (km/h): 200")
        self.lbl_max_speed.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_max_speed = ctk.CTkSlider(self.sidebar, from_=50, to=300, number_of_steps=250, command=self.update_max_speed_label)
        self.slider_max_speed.set(200)
        self.slider_max_speed.grid(row=8, column=0, padx=20, pady=(0, 10))

        # Actions (Bottom of sidebar)
        self.btn_calibrate = ctk.CTkButton(self.sidebar, text="Calibrate ROI", command=self.calibrate_roi, fg_color="orange", hover_color="#c97e00")
        self.btn_calibrate.grid(row=10, column=0, padx=20, pady=10)
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="Start Processing", command=self.toggle_processing, fg_color="green", hover_color="#006400", state="disabled")
        self.btn_start.grid(row=11, column=0, padx=20, pady=(10, 20))

    def _build_main_view(self):
        # Main Content Frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=3) # Video gets more space
        self.main_frame.grid_rowconfigure(1, weight=1) # Logs get less
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Display Canvas
        self.video_canvas = ctk.CTkLabel(self.main_frame, text="Video Feed", bg_color="black")
        self.video_canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Logs/Violations Area
        self.log_frame = ctk.CTkScrollableFrame(self.main_frame, label_text="Recent Violations")
        self.log_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    # --- UI Callbacks ---
    def select_video(self):
        filepath = ctk.filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filepath:
            self.video_path = filepath
            self.lbl_video_path.configure(text=f"...{filepath[-20:]}")
            self.check_ready_state()

    def select_output(self):
        dirpath = ctk.filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.output_dir = dirpath
            self.lbl_output_path.configure(text=f"Output: ...{dirpath[-20:]}")

    def update_speed_label(self, value):
        self.lbl_speed_limit.configure(text=f"Speed Limit (km/h): {int(value)}")

    def update_max_speed_label(self, value):
        self.lbl_max_speed.configure(text=f"Max Threshold (km/h): {int(value)}")

    def calibrate_roi(self):
        if not self.video_path:
            self.log_message("Please select a video first!")
            return
            
        self.log_message("Opening calibration window. Click 4 points.")
        # Run calibration from get_coordinates.py
        pts = run_calibration(self.video_path)
        
        if pts is not None:
            self.source_points = pts
            self.btn_calibrate.configure(fg_color="green", text="ROI Calibrated ✓")
            self.log_message("Calibration successful.")
            self.check_ready_state()
        else:
            self.log_message("Calibration failed or cancelled.")

    def check_ready_state(self):
        if self.video_path and self.source_points is not None:
            self.btn_start.configure(state="normal")

    def toggle_processing(self):
        if not self.is_processing:
            self._start_processing()
        else:
            self._stop_processing()

    def _start_processing(self):
        self.is_processing = True
        self.btn_start.configure(text="Stop Processing", fg_color="red", hover_color="#8b0000")
        
        # Disable inputs
        self.btn_select_video.configure(state="disabled")
        self.btn_calibrate.configure(state="disabled")
        self.slider_speed_limit.configure(state="disabled")
        
        # Initialize Detector
        speed_limit = self.slider_speed_limit.get()
        max_thresh = self.slider_max_speed.get()
        
        self.detector = SpeedDetector(
            video_path=self.video_path,
            output_dir=self.output_dir,
            source_points=self.source_points,
            speed_limit=speed_limit,
            max_speed_threshold=max_thresh
        )
        
        self.log_message(f"Started processing. Limit: {speed_limit}km/h")
        
        # Start Thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _stop_processing(self):
        self.is_processing = False
        if self.detector:
            self.detector.stop()
        
        self.btn_start.configure(text="Start Processing", fg_color="green", hover_color="#006400")
        
        # Re-enable inputs
        self.btn_select_video.configure(state="normal")
        self.btn_calibrate.configure(state="normal")
        self.slider_speed_limit.configure(state="normal")
        self.log_message("Processing stopped.")

    def _processing_loop(self):
        """Runs in separate thread"""
        for data in self.detector.process_video():
            if not self.is_processing:
                break
                
            if "error" in data:
                self.after(0, self.log_message, f"Error: {data['error']}")
                break
                
            # Safely update GUI from thread using self.after
            frame = data['frame']
            violations = data['new_violations']
            
            self.after(0, self._update_video_canvas, frame)
            
            for v in violations:
                msg = f"Violation! ID {v['id']} | {v['speed']:.1f} km/h | Plate: {v['plate']}"
                self.after(0, self.log_message, msg)
                
        # Processing finished naturally
        self.after(0, self._stop_processing)

    def _update_video_canvas(self, frame_bgr):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas roughly
        target_w = self.video_canvas.winfo_width()
        target_h = self.video_canvas.winfo_height()
        
        if target_w > 10 and target_h > 10: # Ensure valid dimensions
             h, w, _ = frame_rgb.shape
             scale = min(target_w/w, target_h/h)
             new_w, new_h = int(w*scale), int(h*scale)
             frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_rgb)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        
        self.video_canvas.configure(image=ctk_img, text="")

    def log_message(self, message):
        lbl = ctk.CTkLabel(self.log_frame, text=message, anchor="w", justify="left")
        lbl.pack(fill="x", padx=5, pady=2)
        # Keep only last 50
        if len(self.log_frame.winfo_children()) > 50:
            self.log_frame.winfo_children()[0].destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
