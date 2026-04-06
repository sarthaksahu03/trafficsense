import csv
import os
import time

class ViolationLogger:
    """
    Writes violations to CSV and keeps one entry per tracked vehicle.
    """
    def __init__(self, output_dir, session_name="session"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.output_dir, f"{session_name}_violations.csv")
        self.logged_records = {}  # vehicle_id -> CSV row data
        
        # Start each run with a fresh header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S.No", "Vehicle_ID", "Timestamp", "Violation_Type", "License_Plate", "Vehicle_Image", "Plate_Image"])
            
        self.s_no = 0

    def _rewrite_csv(self):
        """Rewrite the CSV from the in-memory records."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S.No", "Vehicle_ID", "Timestamp", "Violation_Type", "License_Plate", "Vehicle_Image", "Plate_Image"])
            # Keep rows in the order they were first logged
            for row_data in sorted(self.logged_records.values(), key=lambda x: x[0]):
                writer.writerow(row_data)

    def log_violation(self, vehicle_id, violation_type, plate_text, vehicle_img_path, plate_img_path):
        """
        Log a new violation for this session.
        Returns a dict for the UI, or None if that vehicle was already logged.
        """
        if vehicle_id in self.logged_records:
            return None
            
        self.s_no += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        row_data = [self.s_no, vehicle_id, timestamp, violation_type, plate_text, vehicle_img_path, plate_img_path]
        self.logged_records[vehicle_id] = row_data
        
        # Persist after every new violation
        self._rewrite_csv()
            
        # Return the shape the UI expects
        return {
            "s_no": self.s_no,
            "id": vehicle_id,
            "timestamp": timestamp,
            "type": violation_type,
            "plate": plate_text,
            "vehicle_img": vehicle_img_path,
            "plate_img": plate_img_path
        }
        
    def update_violation(self, vehicle_id, new_plate, new_veh_img, new_plate_img):
        """
        Replace the saved plate and image paths when better evidence comes in.
        """
        if vehicle_id in self.logged_records:
            old_row = self.logged_records[vehicle_id]
            # Update plate text and image paths in the stored row
            old_row[4] = new_plate
            old_row[5] = new_veh_img
            old_row[6] = new_plate_img
            self._rewrite_csv()
