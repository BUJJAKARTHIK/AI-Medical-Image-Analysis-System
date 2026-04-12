import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/model.h5")

def predict_image(img_path):
    # Check if file exists
    print("File exists:", os.path.exists(img_path))

    # Read image
    img = cv2.imread(img_path)

    # Check if image loaded properly
    if img is None:
        print("Error: Image not loaded. Check file path!")
        return "Image Error"

    # Resize image
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Reshape for model
    img = np.reshape(img, (1, 224, 224, 3))

    # Prediction
    prediction = model.predict(img)

    # Output result
    if prediction[0][0] > 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"