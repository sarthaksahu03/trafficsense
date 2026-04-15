import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os
import time

# Local project modules
from get_coordinates import run_calibration
from speed_detection import SpeedDetector
from red_light import RedLightSystem
from analytics_page import AnalyticsDashboard


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("TrafficSense - Violation Detection")
        self.geometry("1100x700")
        self.minsize(900, 550)
        self.configure(bg="#f0f0f0")

        # Runtime state
        self.video_path = None
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.source_points = None
        self.stop_line_points = None
        self.tl_roi_points = None
        self.is_processing = False
        self.detector = None
        self.processing_thread = None
        self.current_mode = tk.StringVar(value="Speed")
        self._photo_ref = None  # Keep a reference to the current frame image
        self._last_frame_bgr = None
        self._violation_counter = 0
        self._violation_iids_by_id = {}
        self._plate_thumb_refs = {}

        self._setup_styles()
        self._build_ui()
        self._on_mode_change()

    # -- Styles --------------------------------------------------

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", font=("Segoe UI", 10), background="#f0f0f0")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", foreground="#333")
        style.configure("TButton", padding=(8, 4))

        style.configure("Sidebar.TFrame", background="#e8e8e8")
        style.configure("Sidebar.TLabel", background="#e8e8e8", foreground="#333")
        style.configure("SidebarHeading.TLabel", background="#e8e8e8",
                        foreground="#222", font=("Segoe UI", 12, "bold"))

        style.configure("Calibrate.TButton", foreground="#555")
        style.map("Calibrate.TButton",
                  foreground=[("active", "#333")])

        style.configure("Go.TButton", foreground="#fff", background="#4a7a4a")
        style.map("Go.TButton",
                  background=[("active", "#3d6b3d"), ("disabled", "#aaa")])

        style.configure("Stop.TButton", foreground="#fff", background="#8b3a3a")
        style.map("Stop.TButton", background=[("active", "#6e2e2e")])

        style.configure("StatusBar.TLabel", background="#ddd", foreground="#555",
                        font=("Segoe UI", 9), padding=(6, 3))
        style.configure("StatusBar.TFrame", background="#ddd")

        # Table styling
        style.configure("Violations.Treeview",
                        font=("Segoe UI", 9),
                        rowheight=62,
                        background="#fff",
                        fieldbackground="#fff",
                        foreground="#333")
        style.configure("Violations.Treeview.Heading",
                        font=("Segoe UI", 9, "bold"),
                        background="#e0e0e0",
                        foreground="#333")
        style.map("Violations.Treeview",
                  background=[("selected", "#cde")])

    # -- Build UI ------------------------------------------------

    def _build_ui(self):
        # Status bar at the bottom
        self._build_statusbar()

        # Main tab view
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        
        # Live detection tab
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="Live Detection")
        
        # Split live detection into sidebar and main content
        self.pane = ttk.PanedWindow(self.live_tab, orient=tk.HORIZONTAL)
        self.pane.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar()
        self._build_main_area()

        # Analytics tab
        self.analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_tab, text="Analytics Dashboard")
        
        self.analytics_db = AnalyticsDashboard(self.analytics_tab, db_path="traffic.db")
        self.analytics_db.pack(fill=tk.BOTH, expand=True)

    def _build_sidebar(self):
        sidebar = ttk.Frame(self.pane, style="Sidebar.TFrame", width=260)
        self.pane.add(sidebar, weight=0)

        # Left-aligned section title
        heading = ttk.Label(sidebar, text="Setup", style="SidebarHeading.TLabel")
        heading.pack(anchor="w", padx=12, pady=(14, 6))

        sep = ttk.Separator(sidebar, orient="horizontal")
        sep.pack(fill="x", padx=8, pady=(0, 10))

        # Detection mode switch
        mode_frame = ttk.Frame(sidebar, style="Sidebar.TFrame")
        mode_frame.pack(anchor="w", padx=12, pady=(0, 8))
        ttk.Label(mode_frame, text="Mode:", style="Sidebar.TLabel").pack(side="left")
        rb_speed = ttk.Radiobutton(mode_frame, text="Speed", variable=self.current_mode,
                                   value="Speed", command=self._on_mode_change)
        rb_speed.pack(side="left", padx=(6, 2))
        rb_rl = ttk.Radiobutton(mode_frame, text="Red Light", variable=self.current_mode,
                                value="Red Light", command=self._on_mode_change)
        rb_rl.pack(side="left", padx=2)

        # Input and output pickers
        self.btn_video = ttk.Button(sidebar, text="Select Video...", command=self._select_video)
        self.btn_video.pack(anchor="w", padx=12, pady=(6, 2))
        self.lbl_video = ttk.Label(sidebar, text="No video selected",
                                   style="Sidebar.TLabel", foreground="#888",
                                   font=("Segoe UI", 9))
        self.lbl_video.pack(anchor="w", padx=14, pady=(0, 6))

        self.btn_output = ttk.Button(sidebar, text="Output Folder...", command=self._select_output)
        self.btn_output.pack(anchor="w", padx=12, pady=(2, 2))
        self.lbl_output = ttk.Label(sidebar, text=f"-> {self._shorten(self.output_dir, 30)}",
                                    style="Sidebar.TLabel", foreground="#888",
                                    font=("Segoe UI", 9))
        self.lbl_output.pack(anchor="w", padx=14, pady=(0, 10))

        sep2 = ttk.Separator(sidebar, orient="horizontal")
        sep2.pack(fill="x", padx=8, pady=2)

        # Holds controls that change with the selected mode
        self.mode_container = ttk.Frame(sidebar, style="Sidebar.TFrame")
        self.mode_container.pack(fill="x")

        # Speed mode controls
        self.speed_frame = ttk.Frame(self.mode_container, style="Sidebar.TFrame")

        ttk.Label(self.speed_frame, text="Speed limit (km/h)",
                  style="Sidebar.TLabel").pack(anchor="w", padx=12, pady=(8, 0))
        sl_frame = ttk.Frame(self.speed_frame, style="Sidebar.TFrame")
        sl_frame.pack(anchor="w", padx=12, pady=(2, 4))
        self.speed_limit_var = tk.IntVar(value=40)
        self.scale_speed = tk.Scale(sl_frame, from_=10, to=150, orient="horizontal",
                                    variable=self.speed_limit_var, length=180,
                                    bg="#e8e8e8", highlightthickness=0,
                                    troughcolor="#ccc", sliderrelief="flat")
        self.scale_speed.pack(side="left")
        self.lbl_speed_val = ttk.Label(sl_frame, text="40", width=4,
                                       style="Sidebar.TLabel", font=("Segoe UI", 10, "bold"))
        self.lbl_speed_val.pack(side="left", padx=(4, 0))
        self.speed_limit_var.trace_add("write", lambda *_: self.lbl_speed_val.config(
            text=str(self.speed_limit_var.get())))

        ttk.Label(self.speed_frame, text="Max threshold (km/h)",
                  style="Sidebar.TLabel").pack(anchor="w", padx=12, pady=(6, 0))
        mx_frame = ttk.Frame(self.speed_frame, style="Sidebar.TFrame")
        mx_frame.pack(anchor="w", padx=12, pady=(2, 6))
        self.max_speed_var = tk.IntVar(value=200)
        self.scale_max = tk.Scale(mx_frame, from_=50, to=300, orient="horizontal",
                                  variable=self.max_speed_var, length=180,
                                  bg="#e8e8e8", highlightthickness=0,
                                  troughcolor="#ccc", sliderrelief="flat")
        self.scale_max.pack(side="left")
        self.lbl_max_val = ttk.Label(mx_frame, text="200", width=4,
                                     style="Sidebar.TLabel", font=("Segoe UI", 10, "bold"))
        self.lbl_max_val.pack(side="left", padx=(4, 0))
        self.max_speed_var.trace_add("write", lambda *_: self.lbl_max_val.config(
            text=str(self.max_speed_var.get())))

        self.btn_cal_roi = ttk.Button(self.speed_frame, text="Calibrate ROI",
                                      style="Calibrate.TButton",
                                      command=self._calibrate_roi)
        self.btn_cal_roi.pack(anchor="w", padx=12, pady=(4, 8))

        # Red-light mode controls
        self.rl_frame = ttk.Frame(self.mode_container, style="Sidebar.TFrame")

        self.btn_cal_stop = ttk.Button(self.rl_frame, text="Calibrate Stop Line",
                                       style="Calibrate.TButton",
                                       command=self._calibrate_stop_line)
        self.btn_cal_stop.pack(anchor="w", padx=12, pady=(10, 4))

        self.btn_cal_tl = ttk.Button(self.rl_frame, text="Calibrate TL Region",
                                     style="Calibrate.TButton",
                                     command=self._calibrate_tl_roi)
        self.btn_cal_tl.pack(anchor="w", padx=12, pady=(2, 8))

        # Spacer to keep the start button near the bottom
        spacer = ttk.Frame(sidebar, style="Sidebar.TFrame")
        spacer.pack(fill="both", expand=True)

        # Start/stop button
        self.btn_start = ttk.Button(sidebar, text="Start", style="Go.TButton",
                                    command=self._toggle_processing, state="disabled")
        self.btn_start.pack(fill="x", padx=12, pady=(4, 10))

        # Small log panel
        log_label = ttk.Label(sidebar, text="Log", style="Sidebar.TLabel",
                              font=("Segoe UI", 9, "bold"))
        log_label.pack(anchor="w", padx=12, pady=(0, 2))
        self.log_text = tk.Text(sidebar, height=6, font=("Consolas", 9),
                                bg="#e2e2e2", fg="#444", relief="flat",
                                bd=0, wrap="word", padx=6, pady=4)
        self.log_text.pack(fill="x", padx=12, pady=(0, 10))
        self.log_text.insert("end", "Ready.\n")
        self.log_text.config(state="disabled")

    def _build_main_area(self):
        main = ttk.Frame(self.pane)
        self.pane.add(main, weight=1)

        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        self.playback_pane = ttk.PanedWindow(main, orient=tk.VERTICAL)
        self.playback_pane.grid(row=0, column=0, sticky="nsew", padx=(4, 8), pady=(8, 6))

        video_frame = ttk.Frame(self.playback_pane)
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        self.playback_pane.add(video_frame, weight=3)

        # Video preview
        self.video_label = tk.Label(video_frame, text="No video feed", bg="#1a1a1a",
                                    fg="#777", font=("Segoe UI", 11),
                                    relief="solid", bd=1)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Configure>", self._on_video_resize)

        # Violation list
        table_frame = ttk.Frame(self.playback_pane)
        self.playback_pane.add(table_frame, weight=1)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        cols = ("sno", "type", "timestamp")
        self.tree = ttk.Treeview(table_frame, columns=cols, show=("tree", "headings"),
                                 style="Violations.Treeview", selectmode="browse")
        self.tree.heading("#0", text="Plate")
        self.tree.heading("sno", text="#")
        self.tree.heading("type", text="Violation")
        self.tree.heading("timestamp", text="Time")

        self.tree.column("#0", width=170, minwidth=140, stretch=False, anchor="center")
        self.tree.column("sno", width=36, minwidth=30, stretch=False, anchor="center")
        self.tree.column("type", width=160, minwidth=100)
        self.tree.column("timestamp", width=150, minwidth=100)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.tree.tag_configure("oddrow", background="#f5f5f5")
        self.tree.tag_configure("evenrow", background="#fff")
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        # Cache row data for the detail popup
        self._violation_data = {}
        self.after(100, self._set_initial_playback_sash)

    def _build_statusbar(self):
        bar = ttk.Frame(self, style="StatusBar.TFrame")
        bar.pack(fill="x", side="bottom")
        self.status_label = ttk.Label(bar, text="Idle", style="StatusBar.TLabel")
        self.status_label.pack(side="left")
        self.status_mode = ttk.Label(bar, text="Mode: Speed", style="StatusBar.TLabel")
        self.status_mode.pack(side="right", padx=(0, 4))

    # Mode switching

    def _on_mode_change(self):
        mode = self.current_mode.get()
        self.status_mode.config(text=f"Mode: {mode}")

        if mode == "Speed":
            self.rl_frame.pack_forget()
            self.speed_frame.pack(in_=self.mode_container, fill="x")
        else:
            self.speed_frame.pack_forget()
            self.rl_frame.pack(in_=self.mode_container, fill="x")
        self._check_ready()

    # File dialogs

    def _select_video(self):
        fp = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if fp:
            self.video_path = fp
            name = os.path.basename(fp)
            self.lbl_video.config(text=name if len(name) < 35 else "..." + name[-32:])
            self._log(f"Video: {name}")
            self._check_ready()

    def _select_output(self):
        dp = filedialog.askdirectory(title="Select Output Directory")
        if dp:
            self.output_dir = dp
            self.lbl_output.config(text=f"-> {self._shorten(dp, 30)}")

    # Calibration helpers

    def _calibrate_roi(self):
        if not self.video_path:
            self._log("Select a video first.")
            return
        self._log("Calibrating Speed ROI - click 4 points...")
        pts = run_calibration(self.video_path, num_points=4, title="Speed ROI")
        if pts is not None:
            self.source_points = pts
            self.btn_cal_roi.config(text="ROI OK (recalibrate)")
            self._log("Speed ROI set.")
            self._check_ready()

    def _calibrate_stop_line(self):
        if not self.video_path:
            self._log("Select a video first.")
            return
        self._log("Calibrating stop line - click 2 points...")
        pts = run_calibration(self.video_path, num_points=2, title="Stop Line")
        if pts is not None:
            self.stop_line_points = [tuple(p) for p in pts.astype(int)]
            self.btn_cal_stop.config(text="Stop Line OK (recalibrate)")
            self._log("Stop line set.")
            self._check_ready()

    def _calibrate_tl_roi(self):
        if not self.video_path:
            self._log("Select a video first.")
            return
        self._log("Calibrating TL ROI - click 2 corners...")
        pts = run_calibration(self.video_path, num_points=2, title="TL Bounding Box")
        if pts is not None:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            self.tl_roi_points = (int(min(x1, x2)), int(min(y1, y2)),
                                  int(max(x1, x2)), int(max(y1, y2)))
            self.btn_cal_tl.config(text="TL ROI OK (recalibrate)")
            self._log("TL ROI set.")
            self._check_ready()

    # Start button state

    def _check_ready(self):
        if not self.video_path:
            self.btn_start.config(state="disabled")
            return

        mode = self.current_mode.get()
        ready = False
        if mode == "Speed" and self.source_points is not None:
            ready = True
        elif mode == "Red Light" and self.stop_line_points and self.tl_roi_points:
            ready = True

        self.btn_start.config(state="normal" if ready else "disabled")

    # Processing control

    def _toggle_processing(self):
        if not self.is_processing:
            self._start()
        else:
            self._stop()

    def _start(self):
        self.is_processing = True
        self.btn_start.config(text="Stop", style="Stop.TButton")
        self.status_label.config(text="Processing...")
        self._violation_iids_by_id.clear()

        mode = self.current_mode.get()
        if mode == "Speed":
            sl = self.speed_limit_var.get()
            mx = self.max_speed_var.get()
            self.detector = SpeedDetector(
                video_path=self.video_path,
                output_dir=self.output_dir,
                source_points=self.source_points,
                speed_limit=sl,
                max_speed_threshold=mx
            )
            self._log(f"Speed detection started (limit {sl} km/h)")
        else:
            self.detector = RedLightSystem(
                video_path=self.video_path,
                output_dir=self.output_dir,
                stop_line_pts=self.stop_line_points,
                traffic_light_roi=self.tl_roi_points
            )
            self._log("Red-light detection started")

        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _stop(self):
        self.is_processing = False
        if self.detector:
            self.detector.stop()
        self.btn_start.config(text="Start", style="Go.TButton")
        self.status_label.config(text="Stopped")
        self._log("Processing stopped.")

    def _processing_loop(self):
        mode = self.current_mode.get()
        for data in self.detector.process_video():
            if not self.is_processing:
                break

            if "error" in data:
                self.after(0, self._log, f"Error: {data['error']}")
                break

            frame = data["frame"]
            violations = data["new_violations"]
            updated_violations = data.get("updated_violations", [])

            self.after(0, self._show_frame, frame)

            for v in violations:
                if mode != "Red Light":
                    v = {
                        "s_no": "-",
                        "id": v.get("id", "-"),
                        "type": f"Speeding ({v.get('speed', 0):.0f} km/h)",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "plate": v.get("plate", "-"),
                        "vehicle_img": v.get("vehicle_img") or v.get("image_path", ""),
                        "plate_img": v.get("plate_img"),
                    }
                self.after(0, self._add_violation, v)

            for v in updated_violations:
                if mode != "Red Light":
                    v = {
                        "id": v.get("id", "-"),
                        "type": f"Speeding ({v.get('speed', 0):.0f} km/h)",
                        "plate": v.get("plate", "-"),
                        "vehicle_img": v.get("vehicle_img") or v.get("image_path", ""),
                        "plate_img": v.get("plate_img"),
                    }
                    self.after(0, self._update_violation, v)

        self.after(0, self._stop)

    # Video preview

    def _show_frame(self, frame_bgr):
        self._last_frame_bgr = frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        lw = self.video_label.winfo_width()
        lh = self.video_label.winfo_height()
        if lw > 10 and lh > 10:
            h, w = rgb.shape[:2]
            scale = min(lw / w, lh / h)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=photo, text="")
        self._photo_ref = photo  # Keep Tk from dropping the image

    def _on_video_resize(self, event=None):
        if self._last_frame_bgr is not None:
            self._show_frame(self._last_frame_bgr)

    def _set_initial_playback_sash(self):
        height = self.playback_pane.winfo_height()
        if height > 400:
            self.playback_pane.sashpos(0, int(height * 0.7))

    # Violation table

    def _add_violation(self, v):
        self._violation_counter += 1
        sno = self._violation_counter
        tag = "oddrow" if sno % 2 else "evenrow"

        plate_thumb = self._load_plate_thumb(v.get("plate_img"))
        iid = self.tree.insert("", "end", values=(
            sno,
            v.get("type", "Unknown"),
            v.get("timestamp", "-"),
        ), tags=(tag,), image=plate_thumb or "", text="" if plate_thumb else "No plate image")

        self._violation_data[iid] = v
        if plate_thumb:
            self._plate_thumb_refs[iid] = plate_thumb
        if v.get("id", "-") != "-":
            self._violation_iids_by_id[v["id"]] = iid
        self._log_to_sqlite(v)

    def _update_violation(self, v):
        iid = self._violation_iids_by_id.get(v.get("id"))
        if not iid:
            v.setdefault("timestamp", time.strftime("%H:%M:%S"))
            self._add_violation(v)
            return

        existing = self._violation_data.get(iid, {})
        existing.update({k: value for k, value in v.items() if value not in (None, "")})
        self._violation_data[iid] = existing

        current_values = list(self.tree.item(iid, "values"))
        if len(current_values) >= 3:
            current_values[1] = existing.get("type", current_values[1])
            self.tree.item(iid, values=current_values)
        plate_thumb = self._load_plate_thumb(existing.get("plate_img"))
        if plate_thumb:
            self._plate_thumb_refs[iid] = plate_thumb
            self.tree.item(iid, image=plate_thumb, text="")
        self._update_sqlite_violation(existing)

    def _ensure_violations_schema(self, cursor):
        cursor.execute('''CREATE TABLE IF NOT EXISTS violations 
                         (id INTEGER PRIMARY KEY, plate_number TEXT, vehicle_type TEXT, 
                          violation_type TEXT, speed REAL, timestamp TEXT, location TEXT, image_path TEXT,
                          vehicle_id TEXT, plate_img_path TEXT)''')

        cursor.execute("PRAGMA table_info(violations)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "vehicle_id" not in existing_columns:
            cursor.execute("ALTER TABLE violations ADD COLUMN vehicle_id TEXT")
        if "plate_img_path" not in existing_columns:
            cursor.execute("ALTER TABLE violations ADD COLUMN plate_img_path TEXT")

    def _normalize_sqlite_violation(self, v):
        import time
        import re

        v_type_str = v.get("type", "")
        speed = 0.0
        v_type = "Unknown"

        if "Speeding" in v_type_str:
            v_type = "Speeding"
            m = re.search(r'(\d+)', v_type_str)
            if m:
                speed = float(m.group(1))
        elif "Red Light" in v_type_str or "Stop" in v_type_str:
            v_type = "Red Light"
        else:
            v_type = v_type_str

        ts = v.get("timestamp", "")
        if len(ts) <= 8 and ts != "-":  # Time-only values need today's date
            ts = f"{time.strftime('%Y-%m-%d')} {ts}"
        elif ts == "-":
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

        plate = v.get("plate", "Unknown")
        if not plate or plate == "-":
            plate = "Unknown"

        vehicle_id = str(v.get("id", "")).strip()
        if vehicle_id == "-":
            vehicle_id = ""

        return {
            "vehicle_id": vehicle_id,
            "plate": plate,
            "vehicle_type": "Vehicle",
            "violation_type": v_type,
            "speed": speed,
            "timestamp": ts,
            "location": "System",
            "vehicle_img": v.get("vehicle_img", ""),
            "plate_img": v.get("plate_img", ""),
        }

    def _log_to_sqlite(self, v):
        import sqlite3
        try:
            with sqlite3.connect("traffic.db") as conn:
                c = conn.cursor()
                self._ensure_violations_schema(c)
                row = self._normalize_sqlite_violation(v)
                
                c.execute('''INSERT INTO violations 
                             (plate_number, vehicle_type, violation_type, speed, timestamp, location,
                              image_path, vehicle_id, plate_img_path)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (row["plate"], row["vehicle_type"], row["violation_type"], row["speed"],
                           row["timestamp"], row["location"], row["vehicle_img"], row["vehicle_id"],
                           row["plate_img"]))
        except Exception as e:
            print(f"Error logging to SQLite: {e}")

    def _update_sqlite_violation(self, v):
        import sqlite3
        try:
            row = self._normalize_sqlite_violation(v)
            if not row["vehicle_id"]:
                return

            with sqlite3.connect("traffic.db") as conn:
                c = conn.cursor()
                self._ensure_violations_schema(c)
                c.execute('''
                    UPDATE violations
                    SET plate_number = ?,
                        image_path = ?,
                        plate_img_path = ?
                    WHERE id = (
                        SELECT id
                        FROM violations
                        WHERE vehicle_id = ?
                        ORDER BY timestamp DESC, id DESC
                        LIMIT 1
                    )
                ''', (row["plate"], row["vehicle_img"], row["plate_img"], row["vehicle_id"]))
        except Exception as e:
            print(f"Error updating SQLite violation: {e}")

    def _on_tree_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        v = self._violation_data.get(iid)
        if v:
            self._show_detail_popup(v)

    # Detail popup

    def _show_detail_popup(self, v):
        win = tk.Toplevel(self)
        win.title(f"Violation - {v.get('type', '')}")
        win.geometry("520x420")
        win.resizable(False, False)
        win.transient(self)
        win.configure(bg="#f4f4f4")
        win.update_idletasks()
        win.after(50, lambda: self._safe_grab(win))

        # Image row
        img_frame = tk.Frame(win, bg="#f4f4f4")
        img_frame.pack(pady=(14, 8), padx=14, anchor="w")

        veh_photo = self._load_popup_image(v.get("vehicle_img"), (240, 200))
        plate_photo = self._load_popup_image(v.get("plate_img"), (200, 90))

        if veh_photo:
            lbl_v = tk.Label(img_frame, image=veh_photo, bg="#f4f4f4", relief="groove", bd=1)
            lbl_v.image = veh_photo
            lbl_v.pack(side="left", padx=(0, 10))
        else:
            tk.Label(img_frame, text="(no vehicle image)", bg="#f4f4f4",
                     fg="#999", font=("Segoe UI", 9)).pack(side="left", padx=(0, 10))

        if plate_photo:
            lbl_p = tk.Label(img_frame, image=plate_photo, bg="#f4f4f4", relief="groove", bd=1)
            lbl_p.image = plate_photo
            lbl_p.pack(side="left")
        else:
            tk.Label(img_frame, text="(no plate image)", bg="#f4f4f4",
                     fg="#999", font=("Segoe UI", 9)).pack(side="left")

        # Text details
        info = tk.Frame(win, bg="#f4f4f4")
        info.pack(anchor="w", padx=16, pady=(6, 4))

        tk.Label(info, text=f"Type:  {v.get('type', '-')}", bg="#f4f4f4",
                 font=("Segoe UI", 11, "bold"), fg="#333").pack(anchor="w", pady=2)
        tk.Label(info, text=f"Time:  {v.get('timestamp', '-')}", bg="#f4f4f4",
                 font=("Segoe UI", 10), fg="#555").pack(anchor="w", pady=1)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(12, 10))

    def _load_plate_thumb(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            img = Image.open(path)
            img.thumbnail((150, 52), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _load_popup_image(self, path, size):
        if not path or not os.path.exists(path):
            return None
        try:
            img = Image.open(path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # Small helpers

    def _safe_grab(self, win):
        """Attempt grab_set, retry once after a short delay if window isn't viewable yet."""
        try:
            win.grab_set()
        except tk.TclError:
            win.after(100, lambda: self._try_grab_once(win))

    def _try_grab_once(self, win):
        try:
            win.grab_set()
        except tk.TclError:
            pass  # Ignore it if the window still isn't ready

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    @staticmethod
    def _shorten(path, n):
        if len(path) <= n:
            return path
        return "..." + path[-(n - 1):]


if __name__ == "__main__":
    app = App()
    app.mainloop()
