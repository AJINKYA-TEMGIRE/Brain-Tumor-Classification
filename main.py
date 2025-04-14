import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model('BrainTumorClassificationModel.h5')
classnames = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.header("🧠 Brain Tumor Classification")

file = st.file_uploader("📤 Upload your MRI Image here (jpg, png, jpeg)", type=['png', 'jpg', 'jpeg'])

def all(image):
    # Convert image to RGB to ensure 3 channels
    image = image.convert("RGB")
    img = np.array(image)

    # Normalize
    img = img / 255.0

    # Resize using cv2
    img = cv2.resize(img, (256, 256))

    # Reshape
    img = img.reshape((1, 256, 256, 3))

    # Predict
    pred = model.predict(img)[0]
    maxproba = np.max(pred)
    prediction = classnames[np.argmax(pred)].upper()

    if prediction != "NOTUMOR":
        prediction += " TUMOR"

    return f"🩺 THIS IS {prediction}, Probability: {maxproba:.2f}"


if file is None:
    st.write("📌 Please upload the MRI image to proceed.")
else:
    image = Image.open(file)
    st.image(image, width=300, caption="Uploaded MRI Image")
    st.header(all(image))
