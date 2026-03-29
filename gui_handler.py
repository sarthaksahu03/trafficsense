import customtkinter as ctk
from image_utils import load_image_for_gui

class ViolationDashboard(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.header_frame.grid_columnconfigure((0,1,2,3), weight=1)
        
        ctk.CTkLabel(self.header_frame, text="S.No", font=("Arial", 14, "bold")).grid(row=0, column=0)
        ctk.CTkLabel(self.header_frame, text="Type", font=("Arial", 14, "bold")).grid(row=0, column=1)
        ctk.CTkLabel(self.header_frame, text="Timestamp", font=("Arial", 14, "bold")).grid(row=0, column=2)
        ctk.CTkLabel(self.header_frame, text="Action", font=("Arial", 14, "bold")).grid(row=0, column=3)
        
        # Scrollable rows
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.scroll_frame.grid_columnconfigure((0,1,2,3), weight=1)
        
        self.row_count = 0

    def add_violation_row(self, v_data):
        """ Appends a row displaying violation summary. """
        if not v_data: return
        
        row_id = self.row_count
        self.row_count += 1
        
        # S.No
        ctk.CTkLabel(self.scroll_frame, text=str(v_data.get("s_no", "-"))).grid(row=row_id, column=0, pady=5)
        # Type
        ctk.CTkLabel(self.scroll_frame, text=v_data.get("type", "Unknown")).grid(row=row_id, column=1, pady=5)
        # Timestamp
        ctk.CTkLabel(self.scroll_frame, text=v_data.get("timestamp", "-")).grid(row=row_id, column=2, pady=5)
        # Button
        btn = ctk.CTkButton(self.scroll_frame, text="View Details", 
                            command=lambda v=v_data: self.show_details_modal(v))
        btn.grid(row=row_id, column=3, pady=5, padx=5)

    def show_details_modal(self, v_data):
        modal = ctk.CTkToplevel(self)
        modal.title(f"Violation Details - #{v_data.get('s_no')}")
        modal.geometry("600x550")
        modal.transient(self.winfo_toplevel()) # Keep on top of main window
        modal.wait_visibility() # Ensure window is visible before grabbing focus
        modal.grab_set() # Focus lock
        
        modal.grid_columnconfigure((0, 1), weight=1)
        
        # Images
        veh_img_path = v_data.get("vehicle_img")
        plate_img_path = v_data.get("plate_img")
        
        veh_ctk = load_image_for_gui(veh_img_path, size=(300, 300))
        plate_ctk = load_image_for_gui(plate_img_path, size=(250, 120))
        
        # Display Vehicle
        if veh_ctk:
            veh_lbl = ctk.CTkLabel(modal, image=veh_ctk, text="")
            veh_lbl.grid(row=0, column=0, padx=20, pady=20)
        else:
            ctk.CTkLabel(modal, text="No Vehicle Image").grid(row=0, column=0, padx=20, pady=20)
            
        # Display Plate
        if plate_ctk:
            plate_lbl = ctk.CTkLabel(modal, image=plate_ctk, text="")
            plate_lbl.grid(row=0, column=1, padx=20, pady=20)
        else:
            ctk.CTkLabel(modal, text="No Plate Image").grid(row=0, column=1, padx=20, pady=20)
            
        # Details texts
        info_frame = ctk.CTkFrame(modal, fg_color="transparent")
        info_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ctk.CTkLabel(info_frame, text=f"Violation Type: {v_data.get('type')}", font=("Arial", 16, "bold")).pack(pady=5)
        ctk.CTkLabel(info_frame, text=f"Timestamp: {v_data.get('timestamp')}", font=("Arial", 14)).pack(pady=5)
        ctk.CTkLabel(info_frame, text=f"Extracted License Plate: {v_data.get('plate')}", font=("Arial", 20, "bold"), text_color="yellow").pack(pady=10)
        
        close_btn = ctk.CTkButton(modal, text="Close", command=modal.destroy, fg_color="red", hover_color="#8b0000")
        close_btn.grid(row=2, column=0, columnspan=2, pady=20)
