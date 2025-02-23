import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.thresholded_image = None
        self.max_width = 1920
        self.max_height = 1080
        self.threshold_value = tk.IntVar(value=127)
        self.max_value = tk.IntVar(value=255)
        self.comparison_value = tk.DoubleVar(value=50)
        
        # Threshold types
        self.threshold_types = {
            "Binary": cv2.THRESH_BINARY,
            "Binary Inverted": cv2.THRESH_BINARY_INV,
            "Truncate": cv2.THRESH_TRUNC,
            "To Zero": cv2.THRESH_TOZERO,
            "To Zero Inverted": cv2.THRESH_TOZERO_INV
        }
        self.current_threshold_type = tk.StringVar(value="Binary")
        
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
        self.threshold_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.preview_tab, text='Preview')
        self.notebook.add(self.threshold_tab, text='Threshold')
        self.notebook.add(self.comparison_tab, text='Compare')
        
        # Setup all tabs
        self.setup_preview_tab()
        self.setup_threshold_tab()
        self.setup_comparison_tab()

    def setup_preview_tab(self):
        # Image display area with label
        self.preview_label = ttk.Label(self.preview_tab, text="Original Image")
        self.preview_label.grid(row=0, column=0, pady=(5,0))
        
        self.preview_canvas = tk.Canvas(self.preview_tab, width=400, height=400, bg='lightgray')
        self.preview_canvas.grid(row=1, column=0, padx=5, pady=5)
        
        # Resolution info label
        self.resolution_label = ttk.Label(self.preview_tab, text="Resolution: -")
        self.resolution_label.grid(row=2, column=0, pady=5)

    def setup_threshold_tab(self):
        # Controls frame
        controls_frame = ttk.LabelFrame(self.threshold_tab, text="Threshold Controls", padding="5")
        controls_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Threshold type dropdown
        ttk.Label(controls_frame, text="Threshold Type:").grid(row=0, column=0, padx=5, pady=5)
        threshold_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.current_threshold_type,
            values=list(self.threshold_types.keys()),
            state="readonly"
        )
        threshold_combo.grid(row=0, column=1, padx=5, pady=5)
        threshold_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_threshold())
        
        # Threshold value slider
        ttk.Label(controls_frame, text="Threshold Value:").grid(row=1, column=0, padx=5, pady=5)
        threshold_slider = ttk.Scale(
            controls_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_value,
            command=lambda _: self.apply_threshold()
        )
        threshold_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Save button
        save_btn = ttk.Button(controls_frame, text="Save Thresholded Image", command=self.save_threshold_image)
        save_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Display areas
        ttk.Label(self.threshold_tab, text="Original Image").grid(row=1, column=0, pady=(5,0))
        self.threshold_original_canvas = tk.Canvas(self.threshold_tab, width=300, height=300, bg='lightgray')
        self.threshold_original_canvas.grid(row=2, column=0, padx=5, pady=5)
        
        ttk.Label(self.threshold_tab, text="Thresholded Image").grid(row=1, column=1, pady=(5,0))
        self.threshold_result_canvas = tk.Canvas(self.threshold_tab, width=300, height=300, bg='lightgray')
        self.threshold_result_canvas.grid(row=2, column=1, padx=5, pady=5)

    def setup_comparison_tab(self):
        # Frame for the comparison view
        self.comparison_frame = ttk.Frame(self.comparison_tab)
        self.comparison_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for comparison
        self.comparison_canvas = tk.Canvas(
            self.comparison_frame,
            width=600,
            height=400,
            bg='lightgray'
        )
        self.comparison_canvas.grid(row=0, column=0, padx=5, pady=5)
        
        # Slider for comparison
        self.comparison_slider = ttk.Scale(
            self.comparison_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.comparison_value,
            command=self.update_comparison
        )
        self.comparison_slider.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Bind mouse motion for interactive comparison
        self.comparison_canvas.bind('<B1-Motion>', self.on_comparison_drag)
        self.comparison_canvas.bind('<Button-1>', self.on_comparison_drag)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                # Update preview tab
                self.display_image(self.original_image, self.preview_canvas, 400)
                height, width = self.original_image.shape[:2]
                self.resolution_label.config(text=f"Resolution: {width}x{height}")
                
                # Update threshold tab displays
                self.display_image(self.original_image, self.threshold_original_canvas, 300)
                self.apply_threshold()

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
        canvas.image = photo

    def apply_threshold(self):
        if self.original_image is None:
            return
            
        # Convert to grayscale if not already
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image
            
        # Apply threshold
        _, self.thresholded_image = cv2.threshold(
            gray,
            self.threshold_value.get(),
            self.max_value.get(),
            self.threshold_types[self.current_threshold_type.get()]
        )
        
        # Convert back to BGR for display
        if len(self.original_image.shape) == 3:
            self.thresholded_image = cv2.cvtColor(self.thresholded_image, cv2.COLOR_GRAY2BGR)
            
        # Update displays
        self.display_image(self.thresholded_image, self.threshold_result_canvas, 300)
        self.update_comparison()

    def update_comparison(self, *args):
        if self.original_image is None or self.thresholded_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.comparison_canvas.winfo_width()
        canvas_height = self.comparison_canvas.winfo_height()
        
        # Prepare images for display
        original_display = self.prepare_image_for_display(self.original_image, canvas_width, canvas_height)
        thresholded_display = self.prepare_image_for_display(self.thresholded_image, canvas_width, canvas_height)
        
        # Calculate split position
        split_position = int((self.comparison_value.get() / 100) * canvas_width)
        
        # Create combined image
        combined = np.copy(original_display)
        combined[:, split_position:] = thresholded_display[:, split_position:]
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(Image.fromarray(combined))
        self.comparison_canvas.delete("all")
        self.comparison_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        self.comparison_canvas.image = photo
        
        # Draw slider line
        self.comparison_canvas.create_line(
            split_position, 0,
            split_position, canvas_height,
            fill='white',
            width=2
        )

    def prepare_image_for_display(self, image, canvas_width, canvas_height):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate scaling to fit canvas
        height, width = image_rgb.shape[:2]
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Create canvas-sized black background
        display = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Calculate position to center image
        y_offset = (canvas_height - new_height) // 2
        x_offset = (canvas_width - new_width) // 2
        
        # Place image on background
        display[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return display

    def on_comparison_drag(self, event):
        # Update slider based on mouse position
        canvas_width = self.comparison_canvas.winfo_width()
        value = max(0, min(100, (event.x / canvas_width) * 100))
        self.comparison_value.set(value)
        self.update_comparison()

    def save_threshold_image(self):
        if self.thresholded_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.thresholded_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()