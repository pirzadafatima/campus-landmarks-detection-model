import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

def adjust_lighting(img): #  Contrast Limited Adaptive Histogram Equalization for adjusting lighting/contrast adaptively
    # separating lightness (L) from color (A and B) in LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # enhanced L, and the original color channels merged back
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def random_blur(img):
  return cv2.GaussianBlur(img, (5, 5), 0) # 5 by 5 kernel and 0 sigmaX (automatically choose blur strength)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224 (square) as it is expected size for mobilenet
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch
    img_array = preprocess_input(img_array)  # Normalize for MobileNet (-1 to 1)
    return img_array