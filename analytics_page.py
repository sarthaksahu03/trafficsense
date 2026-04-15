import sqlite3
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AnalyticsDashboard(ttk.Frame):
    UNKNOWN_PLATES = ("", "-", "UNKNOWN", "Unknown", "unknown")

    def __init__(self, parent, db_path="traffic.db", auto_refresh_ms=5000):
        """
        Small analytics view backed by a local SQLite database.
        """
        super().__init__(parent)
        self.db_path = db_path
        self.auto_refresh_ms = auto_refresh_ms
        self._thumb_refs = {}
        self._row_data = {}
        
        self.setup_ui()
        self.schedule_refresh()

    def setup_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')  # Cleaner default ttk theme
        style.configure("Analytics.Treeview", rowheight=66)
        
        # Frame padding
        self.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Filters
        filter_frame = tk.LabelFrame(self, text="Filters & Search", font=("Segoe UI", 10, "bold"), bg="#f8f9fa")
        filter_frame.pack(fill=tk.X, pady=(0, 15))
        
        inner_filter = tk.Frame(filter_frame, bg="#f8f9fa", pady=5)
        inner_filter.pack(fill=tk.X, padx=5)

        tk.Label(inner_filter, text="Start (YYYY-MM-DD):", bg="#f8f9fa").pack(side=tk.LEFT, padx=(0, 5))
        self.start_date_var = tk.StringVar()
        ttk.Entry(inner_filter, textvariable=self.start_date_var, width=12).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(inner_filter, text="End:", bg="#f8f9fa").pack(side=tk.LEFT, padx=(0, 5))
        self.end_date_var = tk.StringVar()
        ttk.Entry(inner_filter, textvariable=self.end_date_var, width=12).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(inner_filter, text="Violation Type:", bg="#f8f9fa").pack(side=tk.LEFT, padx=(0, 5))
        self.type_var = tk.StringVar(value="All")
        self.type_combo = ttk.Combobox(inner_filter, textvariable=self.type_var, width=12, state="readonly")
        # Default types shown in the dropdown
        self.type_combo['values'] = ("All", "Red Light", "Speeding", "Unregistered")
        self.type_combo.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(inner_filter, text="Plate Search:", bg="#f8f9fa").pack(side=tk.LEFT, padx=(0, 5))
        self.plate_var = tk.StringVar()
        ttk.Entry(inner_filter, textvariable=self.plate_var, width=15).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Button(inner_filter, text="Apply Filters", command=self.refresh_data).pack(side=tk.LEFT, padx=(10, 0))

        # Summary cards
        self.cards_frame = tk.Frame(self)
        self.cards_frame.pack(fill=tk.X, pady=(0, 15))
        for i in range(4):
            self.cards_frame.columnconfigure(i, weight=1)

        self.lbl_tot_vol = self.create_card(self.cards_frame, "Total Violations Today", 0)
        self.lbl_tot_veh = self.create_card(self.cards_frame, "Distinct Vehicles", 1)
        self.lbl_com_vio = self.create_card(self.cards_frame, "Most Common Offense", 2)
        self.lbl_avg_spd = self.create_card(self.cards_frame, "Average Speed", 3)

        # Charts
        self.charts_frame = tk.Frame(self, bg="#ffffff", bd=1, relief="ridge")
        self.charts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Basic chart setup
        self.fig = Figure(figsize=(10, 3.5), dpi=100, facecolor='#ffffff')
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        self.fig.subplots_adjust(bottom=0.25, top=0.85, wspace=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Recent violations table
        table_container = tk.LabelFrame(self, text="Recent 10 Violations", font=("Segoe UI", 10, "bold"), bg="#f8f9fa")
        table_container.pack(fill=tk.X)

        cols = ("vehicle_id", "type", "speed", "timestamp", "status")
        self.tree = ttk.Treeview(table_container, columns=cols, show=("tree", "headings"), height=8,
                                 style="Analytics.Treeview")
        
        self.tree.heading("#0", text="Evidence")
        self.tree.column("#0", width=160, minwidth=130, stretch=False, anchor=tk.CENTER)

        headings = ["Vehicle ID", "Violation Type", "Speed", "Timestamp", "Offense Status"]
        widths = [90, 180, 80, 180, 130]
        
        for col, head, w in zip(cols, headings, widths):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=w, anchor=tk.CENTER)

        # Highlight repeat offenders
        self.tree.tag_configure('repeat', background='#ffebeb', foreground='#c0392b')

        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.tree.bind("<Double-1>", self._on_tree_open)
        self.tree.bind("<ButtonRelease-1>", self._on_evidence_click)

    def ensure_schema(self, cursor):
        """Create or migrate the violations table used by the analytics dashboard."""
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

    def create_card(self, parent, title, col):
        card = tk.Frame(parent, bg='white', relief="ridge", bd=1)
        card.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")
        
        title_lbl = tk.Label(card, text=title, font=("Segoe UI", 11), bg='white', fg='#7f8c8d')
        title_lbl.pack(pady=(15, 5))
        
        value_lbl = tk.Label(card, text="-", font=("Segoe UI", 20, "bold"), bg='white', fg='#2c3e50')
        value_lbl.pack(pady=(0, 15))
        return value_lbl

    def _build_query_clauses(self):
        """Build the SQL WHERE clause from the current filters."""
        clauses = ["1=1"]
        params = []

        start_d = self.start_date_var.get().strip()
        end_d = self.end_date_var.get().strip()
        v_type = self.type_var.get().strip()
        plate = self.plate_var.get().strip()

        if start_d:
            clauses.append("date(timestamp) >= ?")
            params.append(start_d)
        if end_d:
            clauses.append("date(timestamp) <= ?")
            params.append(end_d)
        if v_type and v_type != "All":
            clauses.append("violation_type = ?")
            params.append(v_type)
        if plate:
            clauses.append("plate_number LIKE ?")
            params.append(f"%{plate}%")
            
        return "WHERE " + " AND ".join(clauses), params

    def compute_statistics(self, cursor, where_clause, params):
        """Compute the values shown in the summary cards."""
        # Total violations
        cursor.execute(f"SELECT COUNT(*) FROM violations {where_clause}", params)
        total = cursor.fetchone()[0]

        # Distinct vehicles: prefer track IDs; fall back to usable OCR text only.
        distinct_vehicle_expr = """
            CASE
                WHEN vehicle_id IS NOT NULL AND TRIM(vehicle_id) NOT IN ('', '-') THEN 'id:' || vehicle_id
                WHEN plate_number IS NOT NULL
                     AND TRIM(plate_number) != ''
                     AND UPPER(TRIM(plate_number)) NOT IN ('UNKNOWN', '-')
                THEN 'plate:' || plate_number
                ELSE NULL
            END
        """
        cursor.execute(f"SELECT COUNT(DISTINCT {distinct_vehicle_expr}) FROM violations {where_clause}", params)
        vehicles = cursor.fetchone()[0]

        # Most common violation
        cursor.execute(f"SELECT violation_type, COUNT(*) as c FROM violations {where_clause} GROUP BY violation_type ORDER BY c DESC LIMIT 1", params)
        v_row = cursor.fetchone()
        common = v_row[0] if v_row else "None"

        # Average speed
        cursor.execute(f"SELECT AVG(CAST(speed AS FLOAT)) FROM violations {where_clause}", params)
        s_row = cursor.fetchone()
        avg_speed = f"{s_row[0]:.1f} km/h" if s_row and s_row[0] is not None else "0 km/h"

        return total, vehicles, common, avg_speed

    def draw_charts(self, cursor, where_clause, params):
        """Query grouped stats and redraw the charts."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Chart colors
        bar_color = '#3498db'
        line_color = '#e74c3c'
        pie_colors = ['#2ecc71', '#f1c40f', '#9b59b6', '#34495e', '#e67e22']

        # Violations by type
        cursor.execute(f"SELECT violation_type, COUNT(*) FROM violations {where_clause} GROUP BY violation_type", params)
        types_data = cursor.fetchall()
        if types_data:
            labels = [r[0] for r in types_data]
            counts = [r[1] for r in types_data]
            self.ax1.bar(labels, counts, color=bar_color, alpha=0.9)
            self.ax1.tick_params(axis='x', rotation=15)
        self.ax1.set_title("Violations by Type", fontsize=11, fontweight='bold', color='#333333')
        self.ax1.spines[['top', 'right']].set_visible(False)

        # Timeline by hour
        cursor.execute(f"SELECT strftime('%H', timestamp) as h, COUNT(*) FROM violations {where_clause} GROUP BY h ORDER BY h", params)
        hourly_data = cursor.fetchall()
        if hourly_data and hourly_data[0][0] is not None:
            hours = [r[0] for r in hourly_data]
            counts = [r[1] for r in hourly_data]
            self.ax2.plot(hours, counts, marker='o', color=line_color, linewidth=2)
            self.ax2.set_xlabel("Hour of Day", fontsize=9)
            self.ax2.grid(True, linestyle='--', alpha=0.4)
        self.ax2.set_title("Timeline (Hourly)", fontsize=11, fontweight='bold', color='#333333')
        self.ax2.spines[['top', 'right']].set_visible(False)

        # Vehicle breakdown
        cursor.execute(f"SELECT vehicle_type, COUNT(*) FROM violations {where_clause} GROUP BY vehicle_type", params)
        veh_data = cursor.fetchall()
        if veh_data and veh_data[0][0] is not None:
            labels = [r[0] for r in veh_data]
            counts = [r[1] for r in veh_data]
            self.ax3.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors, textprops={'fontsize': 9})
        self.ax3.set_title("Vehicle Breakdown", fontsize=11, fontweight='bold', color='#333333')

        self.fig.tight_layout()
        self.canvas.draw()

    def fetch_data(self, cursor, where_clause, params):
        """Fetch the latest rows and note which plates appear more than once."""
        # Latest 10 rows
        cursor.execute(f"""
            SELECT vehicle_id, plate_number, violation_type, speed, timestamp, image_path, plate_img_path
            FROM violations {where_clause}
            ORDER BY timestamp DESC
            LIMIT 10
        """, params)
        table_data = cursor.fetchall()
        
        # Plates that show up more than once in the filtered result. Unknown OCR is ignored.
        cursor.execute(f"""
            SELECT plate_number
            FROM violations {where_clause}
              AND plate_number IS NOT NULL
              AND TRIM(plate_number) != ''
              AND UPPER(TRIM(plate_number)) NOT IN ('UNKNOWN', '-')
            GROUP BY plate_number
            HAVING COUNT(*) > 1
        """, params)
        repeated_plates = set(row[0] for row in cursor.fetchall())
        
        return table_data, repeated_plates

    def _load_evidence_thumb(self, plate_img_path, vehicle_img_path):
        """Prefer the plate crop; fall back to the vehicle crop."""
        for path, size in ((plate_img_path, (150, 58)), (vehicle_img_path, (150, 58))):
            if not path or not os.path.exists(path):
                continue
            try:
                img = Image.open(path)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)
            except Exception:
                continue
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

    def _on_evidence_click(self, event):
        if self.tree.identify_region(event.x, event.y) != "tree":
            return
        item_id = self.tree.identify_row(event.y)
        if item_id:
            self._open_evidence_popup(item_id)

    def _on_tree_open(self, event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            self._open_evidence_popup(item_id)

    def _open_evidence_popup(self, item_id):
        data = self._row_data.get(item_id)
        if not data:
            return

        win = tk.Toplevel(self)
        win.title("Violation Evidence")
        win.geometry("760x460")
        win.minsize(620, 380)
        win.transient(self.winfo_toplevel())

        body = tk.Frame(win, bg="#f4f4f4", padx=14, pady=14)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        title = tk.Label(
            body,
            text=f"{data['violation_type']} | Vehicle ID: {data['vehicle_id'] or '-'}",
            bg="#f4f4f4",
            fg="#222",
            font=("Segoe UI", 13, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        vehicle_frame = tk.LabelFrame(body, text="Vehicle Image", bg="#f4f4f4", padx=8, pady=8)
        vehicle_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        vehicle_frame.columnconfigure(0, weight=1)
        vehicle_frame.rowconfigure(0, weight=1)

        vehicle_photo = self._load_popup_image(data["vehicle_img"], (460, 320))
        if vehicle_photo:
            vehicle_label = tk.Label(vehicle_frame, image=vehicle_photo, bg="#ffffff", relief="groove", bd=1)
            vehicle_label.image = vehicle_photo
            vehicle_label.grid(row=0, column=0, sticky="nsew")
        else:
            tk.Label(vehicle_frame, text="No vehicle image", bg="#ffffff", fg="#777").grid(row=0, column=0, sticky="nsew")

        plate_frame = tk.LabelFrame(body, text="Number Plate", bg="#f4f4f4", padx=8, pady=8)
        plate_frame.grid(row=1, column=1, sticky="nsew")
        plate_frame.columnconfigure(0, weight=1)

        plate_photo = self._load_popup_image(data["plate_img"], (250, 120))
        if plate_photo:
            plate_label = tk.Label(plate_frame, image=plate_photo, bg="#ffffff", relief="groove", bd=1)
            plate_label.image = plate_photo
            plate_label.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        else:
            tk.Label(plate_frame, text="No plate crop", bg="#ffffff", fg="#777", height=5).grid(
                row=0, column=0, sticky="ew", pady=(0, 10)
            )

        plate_text = data["plate"] if data["plate"] not in self.UNKNOWN_PLATES else "Unknown"
        details = [
            ("Plate Text", plate_text),
            ("Speed", data["speed"]),
            ("Time", data["timestamp"]),
            ("Status", data["status"]),
        ]
        for idx, (label, value) in enumerate(details, start=1):
            tk.Label(plate_frame, text=f"{label}: {value}", bg="#f4f4f4", fg="#333",
                     font=("Segoe UI", 10)).grid(row=idx, column=0, sticky="w", pady=2)

        ttk.Button(body, text="Close", command=win.destroy).grid(row=2, column=1, sticky="e", pady=(12, 0))

    def refresh_data(self):
        """Refresh the summary cards, charts, and table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                self.ensure_schema(cursor)

                where_clause, params = self._build_query_clauses()
                
                # Update summary cards
                total, vehicles, common, avg_speed = self.compute_statistics(cursor, where_clause, params)
                self.lbl_tot_vol.config(text=str(total))
                self.lbl_tot_veh.config(text=str(vehicles))
                self.lbl_com_vio.config(text=str(common))
                self.lbl_avg_spd.config(text=str(avg_speed))

                # Redraw charts
                self.draw_charts(cursor, where_clause, params)

                # Refresh the table
                table_data, repeated_plates = self.fetch_data(cursor, where_clause, params)
                
                for item in self.tree.get_children():
                    self.tree.delete(item)
                self._thumb_refs.clear()
                self._row_data.clear()
                
                for row in table_data:
                    vehicle_id, plate, v_type, speed, timestamp, vehicle_img, plate_img = row
                    is_repeat = plate not in self.UNKNOWN_PLATES and plate in repeated_plates
                    status = "Repeat Offender" if is_repeat else "First Offense"
                    evidence_thumb = self._load_evidence_thumb(plate_img, vehicle_img)
                    evidence_text = "" if evidence_thumb else "No image"
                    
                    item_id = self.tree.insert(
                        "",
                        "end",
                        text=evidence_text,
                        image=evidence_thumb or "",
                        values=(vehicle_id or "-", v_type, speed, timestamp, status),
                    )
                    if evidence_thumb:
                        self._thumb_refs[item_id] = evidence_thumb
                    self._row_data[item_id] = {
                        "vehicle_id": vehicle_id,
                        "plate": plate,
                        "violation_type": v_type,
                        "speed": speed,
                        "timestamp": timestamp,
                        "status": status,
                        "vehicle_img": vehicle_img,
                        "plate_img": plate_img,
                    }
                    if is_repeat:
                        self.tree.item(item_id, tags=('repeat',))

        except sqlite3.OperationalError as e:
            print(f"[Analytics] Database execution error: {e}")
        except Exception as e:
            print(f"[Analytics] Error updating interface: {e}")

    def schedule_refresh(self):
        """Refresh the dashboard on a timer."""
        self.refresh_data()
        self.after(self.auto_refresh_ms, self.schedule_refresh)

# Simple self-test
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Traffic Analytics Test")
    root.geometry("1000x700")
    app = AnalyticsDashboard(root)
    root.mainloop()
