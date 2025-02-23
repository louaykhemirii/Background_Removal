import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Load the segmentation model
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Replace the background with a solid color
def remove_background(image, background_color=(255, 255, 255)):
    # Convert BGR image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform segmentation
    result = segmenter.process(image_rgb)

    # Create a mask
    mask = result.segmentation_mask
    condition = mask > 0.5  # Threshold for the mask

    # Create a background with the given color
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = background_color

    # Combine the foreground and background using the mask
    output_image = np.where(condition[:, :, None], image, bg_image)
    return output_image

# Load an image
image_path = "data/test.jpg"
image = cv2.imread(image_path)

# Apply background removal
output = remove_background(image, background_color=(0, 255, 0))  # Green background

# Display the result
cv2.imshow("Original", image)
cv2.imshow("Background Removed", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
