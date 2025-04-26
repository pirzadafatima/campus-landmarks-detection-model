FAST Buildings Image Classification Model

Phase 2 Members:
	- Hafsah Zulqarnain		21L-5315
	- Fatima Pirzada		21L-5487

	Group Number: 03

	Group Members: 
	- Muhammad Waleed Malik		21L-5248
	- Tania Waseem			21L-5480
	- Hafsah Zulqarnain		21L-5315
	- Fatima Pirzada		21L-5487
	- Muhammad Sami Khokher		21L-1868
	- Muhammad Arsalan Kashif	21L-7630
	- Saad Khan			21L-1867
	- Muhammad Faizan Majid		21L-5229

 
CV_Project_Phase2_Group3/
│
├── FAST_buildings_model.h5     	# Pre-trained model (saved in HDF5 format)
├── testimages.zip              	# Sample zip file of test images
├── CV_Project_Phase2_Group3.ipynb      # Python notebook with preprocessing, training, and inference code
├── preprocessing.py               	# Contains preprocessing functions (adjust_lighting, random_blur, preprocess_image)
├── Report.pdf             		# Summary of Model Details & Evaluation results (metrics & confusion matrix)
└── README.md  				# Explanation on usage of model

To use this model, follow the steps outlined below.

STEP 1: Install the required libraries
- pip install tensorflow opencv-python matplotlib pillow numpy

STEP 2: Unzip the Test Images
- Place any images you want to test in the project folder. Extract the images with the following code.

import zipfile

zip_path = 'testimages.zip'  # Path to the zip file
extract_folder = 'test_images'  # Folder to extract the images to

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

STEP 3: Load the Pre-trained Model
- Load the model saved as FAST_buildings_model.h5 using the following code.

from tensorflow.keras.models import load_model
model = load_model('FAST_buildings_model.h5')

STEP 4: Preprocess the Images and Predict
- To ensure the model receives testing images in the same format as during training, the images must be preprocessed first.
- Preprocessing steps include: lighting adjustment, optional random blur, resizing, expanding dimensions, and normalization which should be done using the functions available in preprocessing.py file.
- Use inverse_label_map for mapping predicted labels to building names

import os, cv2, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing import adjust_lighting, random_blur, preprocess_image

inverse_label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
image_folder = 'test_images'
image_files = []
for root, dirs, files in os.walk(image_folder):
    for f in files:
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(os.path.join(root, f))

for img_path in image_files:

    # Read image using OpenCV
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"Skipping unreadable image: {img_path}")
        continue

    # Apply lighting adjustment
    img_cv = adjust_lighting(img_cv)

    # Optionally apply random blur (30% of the time)
    if random.random() < 0.3:
        img_cv = random_blur(img_cv)

    # Save the temporary processed image for input
    temp_path = 'temp.jpg'
    cv2.imwrite(temp_path, img_cv)

    # Preprocess image for the model
    img_input = preprocess_image(temp_path)
    os.remove(temp_path)

    # Predict using the model
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = inverse_label_map[predicted_class]

    # Display the image with the predicted label
    plt.imshow(Image.open(img_path))
    plt.axis('off')
    plt.title(f'Predicted: {predicted_label}')
    plt.show()
    print(f"Predicted label: {predicted_label}")
    plt.pause(0.5)