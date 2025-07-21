#  Emotion Detection with CNN (FER2013 - 7 Classes)

This project is a Convolutional Neural Network (CNN)-based emotion classifier trained on the FER2013 dataset.  
It can detect **7 facial emotions** from grayscale face images.

---

##  Emotions Detected

- Angry 
- Disgust   
- Fear   
- Happy   
- Neutral   
- Sad   
- Surprise 

---

##  Model Architecture

The model is built using TensorFlow and Keras with the following structure:

- Input shape: `(48, 48, 1)`
- Conv2D → ReLU → BatchNormalization → MaxPooling  
- Conv2D → ReLU → BatchNormalization → MaxPooling  
- Conv2D → ReLU → BatchNormalization → MaxPooling  
- Flatten → Dense → Dropout → Dense (output layer with 7 softmax neurons)

You can view a summary of the model using:

```python
model.summary()
```

---

##  Dataset: FER2013

The dataset contains 48x48 grayscale images of human faces, divided into 7 emotion classes as listed above.  
You can download it from: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

##  Installation

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install tensorflow opencv-python matplotlib seaborn numpy
```

---

##  How to Run

###  Train the Model

Make sure your dataset is structured like this:

```
dataset/
  └── train/
      ├── Angry/
      ├── Disgust/
      ├── Fear/
      ├── Happy/
      ├── Neutral/
      ├── Sad/
      ├── Surprise/
  └── test/
      ├── Angry/
      ├── Disgust/
      ├── Fear/
      ├── Happy/
      ├── Neutral/
      ├── Sad/
      ├── Surprise/
```

Then run:

```bash
python test_model.py
```

This will train the CNN and save the model (e.g., as `cnn_emotion_model.h5`).

---

###  Predict Emotion from an Image

Provide the path to an image of a face, and run the following code:

```python
import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("cnn_emotion_model.h5")

image = cv2.imread("face.png", 0)
resized = cv2.resize(image, (48, 48))
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 48, 48, 1))

pred = model.predict(reshaped)
emotion = np.argmax(pred)

print("Predicted emotion:", emotion)
```

---

##  Demo Output

<p align="center">
  <img src="demo1.png" width="300"/>
  <img src="demo2.png" width="300"/>
</p>

Predicted emotions displayed on input face images.

---

##  Notes

- You can tune the number of layers, dropout rate, batch size, and other hyperparameters to improve accuracy.  
- The model works best on frontal, clear face images.  
- Consider using face detection (e.g., Haar Cascade) before passing images to the CNN.

---

##  License

This project is open-source and free to use under the MIT license.
