import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/model.h5")

st.title("AI Medical Image Analyzer 🏥")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, 224, 224, 3))

    prediction = model.predict(img_resized)

    if prediction[0][0] > 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"

    st.subheader(f"Prediction: {result}")