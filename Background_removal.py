import cv2
import numpy as np
from tkinter import Tk, filedialog
import sys
print(sys.executable)


def select_image():
    """Allow the user to select an image file."""
    Tk().withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    print(f"Selected file path: {file_path}")
    return file_path

def resize_for_display(image, width=800):
    """Resize the image to a specified width while maintaining aspect ratio."""
    height, orig_width = image.shape[:2]
    aspect_ratio = height / orig_width
    new_height = int(width * aspect_ratio)
    return cv2.resize(image, (width, new_height))

def process_and_display(image_path):
    """Process the selected image and display results using GrabCut."""
    if not image_path:
        print("No image selected. Exiting...")
        return

    # Load the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image. File may be corrupted or unsupported.")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Image loaded successfully!")

    original = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the initial rectangle for GrabCut
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)  # Slight margin from image borders

    # Initialize models needed by GrabCut
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    # Refine the mask: Convert possible foreground/unknown to definite foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    # Apply the mask to the original image
    result = image * mask2[:, :, np.newaxis]

    # Replace background with white
    white_background = image.copy()
    white_background[mask2 == 0] = [255, 255, 255]

    # Resize images for display
    original_resized = resize_for_display(original)
    mask_resized = resize_for_display((mask2 * 255).astype(np.uint8))
    result_resized = resize_for_display(white_background)

    # Display results in resized OpenCV windows
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", original_resized)

    cv2.namedWindow("GrabCut Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("GrabCut Mask", mask_resized)

    cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Final Result", result_resized)

    # Save the processed result
    cv2.imwrite("grabcut_result.png", white_background)

    print("Press any key in the image windows to close them...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Main function to handle user interaction."""
    print("Select an image file to process...")
    image_path = select_image()
    process_and_display(image_path)

if __name__ == "__main__":
    main()
