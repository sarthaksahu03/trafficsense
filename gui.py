import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import os
import numpy as np

# Local project modules
from get_coordinates import run_calibration
from speed_detection import SpeedDetector
from red_light import RedLightSystem
from gui_handler import ViolationDashboard
import time

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Main window setup
        self.title("ALPR & Speed/Red Light Detection System")
        self.geometry("1200x800")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Runtime state
        self.video_path = None
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.source_points = None
        self.stop_line_points = None
        self.tl_roi_points = None
        self.is_processing = False
        self.detector = None
        self.processing_thread = None
        self.current_mode = "Speed"
        self._last_frame_bgr = None

        # Build the initial layout
        self._build_sidebar()
        self._build_main_view()
        self.change_mode("Speed")

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(14, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="System Setup", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Mode switcher
        self.seg_button = ctk.CTkSegmentedButton(self.sidebar, values=["Speed", "Red Light"], command=self.change_mode)
        self.seg_button.grid(row=1, column=0, padx=20, pady=(0, 10))
        self.seg_button.set("Speed")

        self.btn_select_video = ctk.CTkButton(self.sidebar, text="Select Input Video", command=self.select_video)
        self.btn_select_video.grid(row=2, column=0, padx=20, pady=10)
        self.lbl_video_path = ctk.CTkLabel(self.sidebar, text="No video selected", text_color="gray", wraplength=250)
        self.lbl_video_path.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.btn_select_output = ctk.CTkButton(self.sidebar, text="Select Output Folder", command=self.select_output)
        self.btn_select_output.grid(row=4, column=0, padx=20, pady=10)
        self.lbl_output_path = ctk.CTkLabel(self.sidebar, text=f"Output: {self.output_dir}", text_color="gray", wraplength=250)
        self.lbl_output_path.grid(row=5, column=0, padx=20, pady=(0, 10))

        # Speed controls
        self.lbl_speed_limit = ctk.CTkLabel(self.sidebar, text="Speed Limit (km/h): 40")
        self.slider_speed_limit = ctk.CTkSlider(self.sidebar, from_=10, to=150, number_of_steps=140, command=self.update_speed_label)
        self.slider_speed_limit.set(40)

        self.lbl_max_speed = ctk.CTkLabel(self.sidebar, text="Max Threshold (km/h): 200")
        self.slider_max_speed = ctk.CTkSlider(self.sidebar, from_=50, to=300, number_of_steps=250, command=self.update_max_speed_label)
        self.slider_max_speed.set(200)

        # Calibration actions
        self.btn_calibrate = ctk.CTkButton(self.sidebar, text="Calibrate Speed ROI", command=self.calibrate_roi, fg_color="orange", hover_color="#c97e00")
        
        self.btn_calibrate_sl = ctk.CTkButton(self.sidebar, text="Calibrate Stop Line", command=self.calibrate_stop_line, fg_color="orange", hover_color="#c97e00")
        
        self.btn_calibrate_tl = ctk.CTkButton(self.sidebar, text="Calibrate TL ROI", command=self.calibrate_tl_roi, fg_color="orange", hover_color="#c97e00")
        
        # These buttons are shown based on the selected mode
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="Start Processing", command=self.toggle_processing, fg_color="green", hover_color="#006400", state="disabled")
        self.btn_start.grid(row=13, column=0, padx=20, pady=10)
        
        # System log
        self.sys_log = ctk.CTkTextbox(self.sidebar, height=100)
        self.sys_log.grid(row=14, column=0, padx=20, pady=10, sticky="nsew")
        self.sys_log.insert("end", "System Initialized.\n")

    def change_mode(self, mode):
        self.current_mode = mode
        if mode == "Speed":
            self.btn_calibrate_sl.grid_remove()
            self.btn_calibrate_tl.grid_remove()
            
            self.lbl_speed_limit.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
            self.slider_speed_limit.grid(row=7, column=0, padx=20, pady=(0, 10))
            self.lbl_max_speed.grid(row=8, column=0, padx=20, pady=(10, 0), sticky="w")
            self.slider_max_speed.grid(row=9, column=0, padx=20, pady=(0, 10))
            self.btn_calibrate.grid(row=10, column=0, padx=20, pady=10)
        else:
            self.lbl_speed_limit.grid_remove()
            self.slider_speed_limit.grid_remove()
            self.lbl_max_speed.grid_remove()
            self.slider_max_speed.grid_remove()
            self.btn_calibrate.grid_remove()
            
            self.btn_calibrate_sl.grid(row=10, column=0, padx=20, pady=5)
            self.btn_calibrate_tl.grid(row=11, column=0, padx=20, pady=5)
            
        self.check_ready_state()

    def _build_main_view(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.playback_pane = tk.PanedWindow(
            self.main_frame,
            orient=tk.VERTICAL,
            sashwidth=8,
            sashrelief=tk.RAISED,
            bg="#2b2b2b",
            bd=0,
            showhandle=True,
        )
        self.playback_pane.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.video_panel = ctk.CTkFrame(self.playback_pane, corner_radius=0)
        self.video_panel.grid_rowconfigure(0, weight=1)
        self.video_panel.grid_columnconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(self.video_panel, text="Video Feed", bg_color="black")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.bind("<Configure>", self._on_video_resize)

        self.dashboard = ViolationDashboard(self.playback_pane)
        self.playback_pane.add(self.video_panel, minsize=220, sticky="nsew", stretch="always")
        self.playback_pane.add(self.dashboard, minsize=140, sticky="nsew")
        self.after(100, self._set_initial_playback_sash)

    # UI event handlers
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
        if not self.video_path: return self.log_message("Select a video first!")
        self.log_message("Calibrating Speed ROI: Click 4 points.")
        pts = run_calibration(self.video_path, num_points=4, title="Speed ROI")
        if pts is not None:
            self.source_points = pts
            self.btn_calibrate.configure(fg_color="green", text="Speed ROI ✓")
            self.check_ready_state()

    def calibrate_stop_line(self):
        if not self.video_path: return self.log_message("Select a video first!")
        self.log_message("Calibrating Stop Line: Click 2 points.")
        pts = run_calibration(self.video_path, num_points=2, title="Stop Line")
        if pts is not None:
            self.stop_line_points = [tuple(p) for p in pts.astype(int)]
            self.btn_calibrate_sl.configure(fg_color="green", text="Stop Line ✓")
            self.check_ready_state()

    def calibrate_tl_roi(self):
        if not self.video_path: return self.log_message("Select a video first!")
        self.log_message("Calibrating TL ROI: Click 2 points (Top-Left, Bottom-Right).")
        pts = run_calibration(self.video_path, num_points=2, title="TL Bounding Box")
        if pts is not None:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            self.tl_roi_points = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
            self.btn_calibrate_tl.configure(fg_color="green", text="TL ROI ✓")
            self.check_ready_state()

    def check_ready_state(self):
        if not self.video_path: return
        
        ready = False
        if self.current_mode == "Speed" and self.source_points is not None:
            ready = True
        elif self.current_mode == "Red Light" and self.stop_line_points is not None and self.tl_roi_points is not None:
            ready = True
            
        if ready:
            self.btn_start.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")

    def toggle_processing(self):
        if not self.is_processing:
            self._start_processing()
        else:
            self._stop_processing()

    def _start_processing(self):
        self.is_processing = True
        self.btn_start.configure(text="Stop Processing", fg_color="red", hover_color="#8b0000")
        
        self.seg_button.configure(state="disabled")
        self.btn_select_video.configure(state="disabled")
        self.slider_speed_limit.configure(state="disabled")
        
        if self.current_mode == "Speed":
            speed_limit = self.slider_speed_limit.get()
            max_thresh = self.slider_max_speed.get()
            self.detector = SpeedDetector(
                video_path=self.video_path,
                output_dir=self.output_dir,
                source_points=self.source_points,
                speed_limit=speed_limit,
                max_speed_threshold=max_thresh
            )
            self.log_message(f"Started Speed Detection. Limit: {speed_limit}km/h")
        else:
            self.detector = RedLightSystem(
                video_path=self.video_path,
                output_dir=self.output_dir,
                stop_line_pts=self.stop_line_points,
                traffic_light_roi=self.tl_roi_points
            )
            self.log_message("Started Red Light Detection.")
        
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _stop_processing(self):
        self.is_processing = False
        if self.detector:
            self.detector.stop()
        
        self.btn_start.configure(text="Start Processing", fg_color="green", hover_color="#006400")
        self.seg_button.configure(state="normal")
        self.btn_select_video.configure(state="normal")
        self.slider_speed_limit.configure(state="normal")
        self.log_message("Processing stopped.")

    def _processing_loop(self):
        for data in self.detector.process_video():
            if not self.is_processing:
                break
                
            if "error" in data:
                self.after(0, self.log_message, f"Error: {data['error']}")
                break
                
            frame = data['frame']
            violations = data['new_violations']
            updated_violations = data.get('updated_violations', [])
            
            self.after(0, self._update_video_canvas, frame)
            
            for v in violations:
                if self.current_mode == "Red Light":
                    self.after(0, self.dashboard.add_violation_row, v)
                else:
                    formatted_v = {
                        "s_no": "-", 
                        "id": v.get('id', '-'),
                        "type": f"Speeding ({v.get('speed', 0):.0f}km/h)",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "plate": v.get('plate', '-'),
                        "vehicle_img": v.get('vehicle_img') or v.get('image_path', ''),
                        "plate_img": v.get('plate_img')
                    }
                    self.after(0, self.dashboard.add_violation_row, formatted_v)

            for v in updated_violations:
                if self.current_mode != "Red Light":
                    formatted_v = {
                        "id": v.get('id', '-'),
                        "type": f"Speeding ({v.get('speed', 0):.0f}km/h)",
                        "plate": v.get('plate', '-'),
                        "vehicle_img": v.get('vehicle_img') or v.get('image_path', ''),
                        "plate_img": v.get('plate_img')
                    }
                    self.after(0, self.dashboard.update_violation_row, formatted_v)
                
        self.after(0, self._stop_processing)

    def _update_video_canvas(self, frame_bgr):
        self._last_frame_bgr = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        target_w = self.video_canvas.winfo_width()
        target_h = self.video_canvas.winfo_height()
        
        if target_w > 10 and target_h > 10:
             h, w, _ = frame_rgb.shape
             scale = min(target_w/w, target_h/h)
             new_w, new_h = int(w*scale), int(h*scale)
             frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_rgb)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        
        self.video_canvas.configure(image=ctk_img, text="")

    def _on_video_resize(self, event=None):
        if self._last_frame_bgr is not None:
            self._update_video_canvas(self._last_frame_bgr)

    def _set_initial_playback_sash(self):
        height = self.playback_pane.winfo_height()
        if height > 400:
            self.playback_pane.sash_place(0, 0, int(height * 0.7))

    def log_message(self, message):
        self.sys_log.insert("end", message + "\n")
        self.sys_log.see("end")

if __name__ == "__main__":
    app = App()
    app.mainloop()
