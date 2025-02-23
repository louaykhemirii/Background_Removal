
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
        
        self.setup_ui()
        
    def setup_ui(self):
    
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
       
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.load_btn = ttk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=5)
        
        self.remove_bg_btn = ttk.Button(self.button_frame, text="Remove Background", command=self.remove_background)
        self.remove_bg_btn.grid(row=0, column=1, padx=5)
        
        self.save_btn = ttk.Button(self.button_frame, text="Save Image", command=self.save_image)
        self.save_btn.grid(row=0, column=2, padx=5)
        
        self.threshold_label = ttk.Label(self.main_frame, text="Threshold Value:")
        self.threshold_label.grid(row=1, column=0, pady=5)
        
        self.threshold_scale = ttk.Scale(self.main_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                       command=self.update_threshold)
        self.threshold_scale.set(self.current_threshold)
        self.threshold_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
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
                self.display_image(self.processed_image, self.processed_canvas)
    
    def remove_background(self):
        if self.original_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, self.current_threshold, 255, cv2.THRESH_BINARY)
        
        # Create mask
        mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to original image
        self.processed_image = cv2.bitwise_and(self.original_image, mask)
        
        # Display result
        self.display_image(self.processed_image, self.processed_canvas)
    
    def update_threshold(self, value):
        self.current_threshold = int(float(value))
        if self.original_image is not None:
            self.remove_background()
    
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
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit canvas while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        max_size = 400
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image_rgb, (new_width, new_height))
      
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        
        
        canvas.delete("all")
        canvas.create_image(max_size//2, max_size//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference to prevent garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()