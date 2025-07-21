# CNN Emotion Detection (FER2013 - 7 Classes)

This project trains a Convolutional Neural Network (CNN) to classify facial emotions using the FER2013 dataset with 7 classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Dataset

FER2013 dataset with 7 emotion classes. Make sure the dataset is preprocessed and split into `train`, `val`, and `test` folders. The directory structure should look like this:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
├── val/
│   ├── angry/
│   ├── disgust/
│   └── ...
├── test/
│   ├── angry/
│   ├── disgust/
│   └── ...
```

## Model Architecture

The CNN model includes the following:

- Convolutional layers with ReLU activation
- Batch normalization
- Max pooling layers
- Dropout for regularization
- Fully connected dense layers
- Output layer with softmax activation (7 units for 7 classes)

See the `train_model.py` for full implementation.

## Training the Model

To train the CNN model:

```bash
python train_model.py
```

The model will be saved in the `models/` directory as `emotion_model22.h5`.

## Evaluate the Model

To test the model accuracy on the test set:

```bash
python test_model.py
```

This script loads the trained model, evaluates it on the test data, and shows accuracy and confusion matrix.

## Predict Emotion from an Image

You can test the trained CNN model on a new image (e.g. `gabriel.png`) using the script `predict_image.py`.

This script performs the following:

- Loads the trained model (`emotion_model22.h5`)
- Reads an input image using OpenCV
- Detects the face using Haar cascades
- Extracts a region of interest (ROI) around the face
- Resizes and normalizes the image
- Predicts the emotion using the trained model
- Prints out the predicted class index and corresponding label

You can adjust the coordinates in `roi = image[...]` if the face is not properly cropped.

Make sure the following files/paths exist and are properly set:

- Trained model at: `models/emotion_model22.h5`
- Haar cascade at: `Cascades/haarcascade_frontalface_default.xml`
- Image for testing at: `Images/gabriel.png`

To run the script:

```bash
python predict_image.py
```

Example output:

```
Predicted class index: 3
Predicted emotion label: Happy
```

---

Let me know if you want a version in **Persian** too or want to add any screenshots or diagrams.
