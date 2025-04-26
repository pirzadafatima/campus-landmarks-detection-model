# FAST Buildings Landmark Recognition Model 

This project documents the design, training, and evaluation of a deep learning model for recognizing landmark buildings at FAST-NU Lahore. The model uses transfer learning with MobileNet architecture to achieve high accuracy.

## Introduction

- Recognizes images of different buildings (A, B, C, D, E, F) at FAST-NU Lahore.
- Built using transfer learning with **MobileNet** (pretrained on ImageNet).
- Fine-tuned for improved performance on a small custom dataset.

## Model Architecture

- **Base Model:** MobileNet
- **Input Size:** (224, 224, 3)
- **Modifications:**
  - Removed last 4 layers of MobileNet.
  - Reshaped Global Average Pooling output to (1024,).
  - Added a Dense layer with 6 units + softmax activation.
- **Fine-tuning:** Last 22 layers were unfrozen and trained on the custom dataset.

## Dataset

- Labeled images of 6 FAST-NU buildings (https://drive.google.com/drive/u/0/folders/1J-pwYV5NueoqHsO4rOExe46y2ZLrLyNz).
- Original dataset size: 305 images.
- Augmentation applied:
  - Random shifts, zooming, rotations (ImageDataGenerator)
  - Lighting adjustment (CLAHE from OpenCV)
  - Random blurring on 30% images
- Final dataset size: ~250 images per building.
- Preprocessing: Resizing, Normalization, Expanding Dimensions

## Technologies Used

- **Python 3**
- **TensorFlow / Keras** (Deep Learning Framework)
- **OpenCV** (Image processing and augmentation)
- **NumPy** (Numerical operations)
- **Matplotlib** (Visualization)
- **Pillow (PIL)** (Image file handling)
