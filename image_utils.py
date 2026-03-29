import cv2
import os
from PIL import Image
import customtkinter as ctk

def save_violation_crops(output_dir, vehicle_id, plate_text, vehicle_crop, plate_crop=None):
    """
    Saves the cropped vehicle and plate images to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vehicle_path = ""
    if vehicle_crop is not None and vehicle_crop.size > 0:
        vehicle_filename = f"violation_v{vehicle_id}_p{plate_text}_vehicle.jpg"
        vehicle_path = os.path.join(output_dir, vehicle_filename)
        cv2.imwrite(vehicle_path, vehicle_crop)
        
    plate_path = ""
    if plate_crop is not None and plate_crop.size > 0:
        plate_filename = f"violation_v{vehicle_id}_p{plate_text}_plate.jpg"
        plate_path = os.path.join(output_dir, plate_filename)
        cv2.imwrite(plate_path, plate_crop)
        
    return vehicle_path, plate_path

def load_image_for_gui(image_path, size=(300, 300)):
    """
    Loads an image from disk and scales it efficiently for CustomTkinter rendering.
    """
    if not image_path or not os.path.exists(image_path):
        return None
        
    try:
        img = Image.open(image_path)
        # Resize while maintaining aspect ratio
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        return ctk_img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
