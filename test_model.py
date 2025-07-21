# train_model.py
"""
Train a Convolutional Neural Network (CNN) to classify facial expressions
using the FER2013 dataset with 7 emotion classes.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import load_model

# Configuration
IMG_SIZE = 48
BATCH_SIZE = 64
NUM_CLASSES = 7
NUM_FILTERS = 32

# Paths
TRAIN_DIR = "datasets/fer2013/train"
VAL_DIR = "datasets/fer2013/validation"
MODEL_SAVE_PATH = "models/emotion_model22.h5"

# 1. Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_dataset = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 2. Build CNN Model
model = Sequential()

# Block 1
model.add(Conv2D(NUM_FILTERS, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(NUM_FILTERS * 2, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(NUM_FILTERS * 2, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(NUM_FILTERS * 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(NUM_FILTERS * 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(NUM_CLASSES, activation='softmax'))

# 3. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

# 5. Save the trained model
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)

# 6. Evaluate the model
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
