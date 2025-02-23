import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Background Removal & Thresholding")
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.current_threshold = 127
        self.resize_percentage = 100
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Load Image Button
        self.load_btn = ttk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=5)
        
        # Remove Background Button
        self.remove_bg_btn = ttk.Button(self.button_frame, text="Remove Background", command=self.remove_background)
        self.remove_bg_btn.grid(row=0, column=1, padx=5)
        
        # Save Button
        self.save_btn = ttk.Button(self.button_frame, text="Save Image", command=self.save_image)
        self.save_btn.grid(row=0, column=2, padx=5)
        
        # Controls frame
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Image Controls", padding="5")
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Threshold slider
        self.threshold_label = ttk.Label(self.controls_frame, text="Threshold Value:")
        self.threshold_label.grid(row=0, column=0, pady=5)
        
        self.threshold_scale = ttk.Scale(self.controls_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                       command=self.update_threshold)
        self.threshold_scale.set(self.current_threshold)
        self.threshold_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Resize slider
        self.resize_label = ttk.Label(self.controls_frame, text="Resize (%):")
        self.resize_label.grid(row=1, column=0, pady=5)
        
        self.resize_scale = ttk.Scale(self.controls_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                    command=self.update_resize)
        self.resize_scale.set(self.resize_percentage)
        self.resize_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Resize value label
        self.resize_value_label = ttk.Label(self.controls_frame, text="100%")
        self.resize_value_label.grid(row=1, column=2, padx=5)
        
        # Image display areas
        self.original_label = ttk.Label(self.main_frame, text="Original Image")
        self.original_label.grid(row=2, column=0)
        
        self.processed_label = ttk.Label(self.main_frame, text="Processed Image")
        self.processed_label.grid(row=2, column=1)
        
        self.original_canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.original_canvas.grid(row=3, column=0, padx=5, pady=5)
        
        self.processed_canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.processed_canvas.grid(row=3, column=1, padx=5, pady=5)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_canvas)
                self.processed_image = self.original_image.copy()
                self.process_image()
    
    def remove_background(self):
        if self.original_image is None:
            return
        self.process_image()
    
    def process_image(self):
        if self.original_image is None:
            return
            
        # Resize image
        height, width = self.original_image.shape[:2]
        new_width = int(width * self.resize_percentage / 100)
        new_height = int(height * self.resize_percentage / 100)
        resized = cv2.resize(self.original_image, (new_width, new_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, self.current_threshold, 255, cv2.THRESH_BINARY)
        
        # Create mask
        mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to resized image
        self.processed_image = cv2.bitwise_and(resized, mask)
        
        # Display result
        self.display_image(self.processed_image, self.processed_canvas)
    
    def update_threshold(self, value):
        self.current_threshold = int(float(value))
        self.process_image()
    
    def update_resize(self, value):
        self.resize_percentage = int(float(value))
        self.resize_value_label.config(text=f"{self.resize_percentage}%")
        self.process_image()
    
    def save_image(self):
        if self.processed_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
    
    def display_image(self, image, canvas):
        if image is None:
            return
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit canvas while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        max_size = 400
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(max_size//2, max_size//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference to prevent garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()