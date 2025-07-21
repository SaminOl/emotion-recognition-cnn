import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model("D:/Computer vision/Files/Models/emotion_model22.h5")

# Load validation dataset to access class indices
test_datagen = ImageDataGenerator(rescale=1./255)
test_dataset = test_datagen.flow_from_directory(
    "D:/Computer vision/Files/Datasets/fer2013_2_classes/fer2013/validation",
    target_size=(48, 48),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Load and show the image
image = cv2.imread("D:/Computer vision/Files/Images/gabriel.png")
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Original image shape:", image.shape)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("D:/Computer vision/Files/Cascades/haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_cascade.detectMultiScale(image)

# Manually crop region of interest (ROI)
roi = image[40:40 + 128, 162:162 + 128]
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("ROI shape before resize:", roi.shape)

# Resize to match model input shape
roi = cv2.resize(roi, (48, 48))
cv2.imshow('Resized ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("ROI shape after resize:", roi.shape)

# Normalize the ROI
roi = roi / 255.0
roi = np.expand_dims(roi, axis=0)

# Predict emotion
predictions = model.predict(roi)
print("Prediction probabilities:", predictions)

# Get the predicted class
predicted_class_index = np.argmax(predictions)
print("Predicted class index:", predicted_class_index)

# Map index to label
class_labels = test_dataset.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # reverse the dictionary
predicted_label = class_labels[predicted_class_index]
print("Predicted emotion label:", predicted_label)
