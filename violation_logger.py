import csv
import os
import time

class ViolationLogger:
    """
    Manages structured logging for traffic violations.
    Records to CSV and tracks unique vehicles to prevent duplicates.
    """
    def __init__(self, output_dir, session_name="session"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.output_dir, f"{session_name}_violations.csv")
        self.logged_records = {} # map vehicle_id -> row data (list)
        
        # Write header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S.No", "Vehicle_ID", "Timestamp", "Violation_Type", "License_Plate", "Vehicle_Image", "Plate_Image"])
            
        self.s_no = 0

    def _rewrite_csv(self):
        """Rewrites the entire CSV with the latest record updates in memory."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S.No", "Vehicle_ID", "Timestamp", "Violation_Type", "License_Plate", "Vehicle_Image", "Plate_Image"])
            # Sort by S.No to maintain chronological order
            for row_data in sorted(self.logged_records.values(), key=lambda x: x[0]):
                writer.writerow(row_data)

    def log_violation(self, vehicle_id, violation_type, plate_text, vehicle_img_path, plate_img_path):
        """
        Logs a violation if the vehicle ID hasn't been logged yet in this session.
        Returns the row data as a dictionary if successfully logged, else None.
        """
        if vehicle_id in self.logged_records:
            return None
            
        self.s_no += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        row_data = [self.s_no, vehicle_id, timestamp, violation_type, plate_text, vehicle_img_path, plate_img_path]
        self.logged_records[vehicle_id] = row_data
        
        # Write to file
        self._rewrite_csv()
            
        # Return structured data for GUI tables
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
        Overwrites existing violations in-memory cache and disk CSV 
        when higher quality evidence captures occur.
        """
        if vehicle_id in self.logged_records:
            old_row = self.logged_records[vehicle_id]
            # Update plate (idx 4), veh_img (idx 5), plate_img (idx 6)
            old_row[4] = new_plate
            old_row[5] = new_veh_img
            old_row[6] = new_plate_img
            self._rewrite_csv()
