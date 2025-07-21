import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


try:
    img = tf.keras.preprocessing.image.load_img(
        "D:\\Computer vision\\Files\\Datasets\\cat_dog_2\\cat_dog_2\\training_set\\cat\\cat.1.jpg") # مثال تصویر گربه
    plt.imshow(img)
    plt.title("Example Cat Image")
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print("Example image not found. Please check the path after extraction.")


training_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=7,
    horizontal_flip=True,
    zoom_range=0.2
)

train_dataset = training_generator.flow_from_directory(
    "D:\\Computer vision\\Files\\Datasets\\cat_dog_2\\cat_dog_2\\training_set",
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory(
    "D:\\Computer vision\\Files\\Datasets\\cat_dog_2\\cat_dog_2\\test_set",
    target_size=(64, 64),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)


network = Sequential()
network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Flatten())

network.add(Dense(units=577, activation='relu'))
network.add(Dense(units=577, activation='relu'))
network.add(Dense(units=2, activation='softmax')) 

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training the CNN...")
history = network.fit(train_dataset, epochs=50) 


print("\nEvaluating the model on the test set...")
predictions = network.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)


true_labels = test_dataset.classes

print("\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(true_labels, predictions):.4f}")

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_dataset.class_indices.keys()),
            yticklabels=list(test_dataset.class_indices.keys()))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=list(test_dataset.class_indices.keys())))
