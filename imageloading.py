import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.max_width = 1920  # Maximum width for slider
        self.max_height = 1080  # Maximum height for slider
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Load Image Button
        self.load_btn = ttk.Button(self.main_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.preview_tab = ttk.Frame(self.notebook)
        self.resize_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.preview_tab, text='Preview')
        self.notebook.add(self.resize_tab, text='Resize')
        
        # Setup Preview Tab
        self.setup_preview_tab()
        
        # Setup Resize Tab
        self.setup_resize_tab()
        
    def setup_preview_tab(self):
        # Image display area with label
        self.preview_label = ttk.Label(self.preview_tab, text="Original Image")
        self.preview_label.grid(row=0, column=0, pady=(5,0))
        
        self.preview_canvas = tk.Canvas(self.preview_tab, width=400, height=400, bg='lightgray')
        self.preview_canvas.grid(row=1, column=0, padx=5, pady=5)
        
        # Resolution info label
        self.resolution_label = ttk.Label(self.preview_tab, text="Resolution: -")
        self.resolution_label.grid(row=2, column=0, pady=5)
        
    def setup_resize_tab(self):
        # Resize controls frame
        self.resize_frame = ttk.LabelFrame(self.resize_tab, text="Resize Controls", padding="5")
        self.resize_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Width slider
        ttk.Label(self.resize_frame, text="Width:").grid(row=0, column=0, padx=5, pady=5)
        self.width_var = tk.IntVar()
        self.width_slider = ttk.Scale(
            self.resize_frame, 
            from_=1, 
            to=self.max_width,
            orient=tk.HORIZONTAL,
            variable=self.width_var,
            command=self.on_slider_change
        )
        self.width_slider.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.width_label = ttk.Label(self.resize_frame, text="0")
        self.width_label.grid(row=0, column=2, padx=5)
        
        # Height slider
        ttk.Label(self.resize_frame, text="Height:").grid(row=1, column=0, padx=5, pady=5)
        self.height_var = tk.IntVar()
        self.height_slider = ttk.Scale(
            self.resize_frame, 
            from_=1, 
            to=self.max_height,
            orient=tk.HORIZONTAL,
            variable=self.height_var,
            command=self.on_slider_change
        )
        self.height_slider.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.height_label = ttk.Label(self.resize_frame, text="0")
        self.height_label.grid(row=1, column=2, padx=5)
        
        # Save button
        self.save_btn = ttk.Button(self.resize_frame, text="Save Resized Image", command=self.save_image)
        self.save_btn.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Display areas for original and resized images
        # Original image
        ttk.Label(self.resize_tab, text="Original Image").grid(row=1, column=0, pady=(5,0))
        self.original_canvas = tk.Canvas(self.resize_tab, width=300, height=300, bg='lightgray')
        self.original_canvas.grid(row=2, column=0, padx=5, pady=5)
        
        # Resized image
        ttk.Label(self.resize_tab, text="Resized Image").grid(row=1, column=1, pady=(5,0))
        self.resized_canvas = tk.Canvas(self.resize_tab, width=300, height=300, bg='lightgray')
        self.resized_canvas.grid(row=2, column=1, padx=5, pady=5)
    
    def on_slider_change(self, _):
        self.width_label.config(text=str(self.width_var.get()))
        self.height_label.config(text=str(self.height_var.get()))
        self.resize_image()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                # Update preview tab
                self.display_image(self.original_image, self.preview_canvas, 400)
                
                # Get and set initial dimensions
                height, width = self.original_image.shape[:2]
                self.resolution_label.config(text=f"Resolution: {width}x{height}")
                
                # Update sliders with current dimensions
                self.width_var.set(width)
                self.height_var.set(height)
                self.width_label.config(text=str(width))
                self.height_label.config(text=str(height))
                
                # Update resize tab
                self.display_image(self.original_image, self.original_canvas, 300)
                self.resize_image()
    
    def resize_image(self):
        if self.original_image is None:
            return
            
        new_width = self.width_var.get()
        new_height = self.height_var.get()
        
        # Perform the actual resize using cv2
        self.processed_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # Display the resized image
        self.display_image(self.processed_image, self.resized_canvas, 300)
    
    def save_image(self):
        if self.processed_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
    
    def display_image(self, image, canvas, max_size):
        if image is None:
            return
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit canvas while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        scale = max_size / max(height, width)
        display_width = int(width * scale)
        display_height = int(height * scale)
        
        display_image = cv2.resize(image_rgb, (display_width, display_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(display_image))
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(
            max_size//2,
            max_size//2,
            image=photo,
            anchor=tk.CENTER
        )
        canvas.image = photo  # Keep a reference to prevent garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()