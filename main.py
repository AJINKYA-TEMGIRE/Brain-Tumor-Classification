import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model('BrainTumorClassificationModel.h5')

classnames = ['glioma', 'meningioma', 'notumor', 'pituitary']


st.header("Brain Tumor Classification")

file = st.file_uploader("Upload your MRI Image here in form of (jgg, png or jpeg) " , type = ['png' , 'jpg' , 'jpeg'])

def all(image):
    img = np.asarray(image)

    img = img / 255.

    img1 = cv2.resize(img, (256, 256))

    a = img1.reshape((1, 256, 256, 3))

    maxproba = max(model.predict(a)[0])

    answer = classnames[np.argmax(model.predict(a))].upper()

    if answer != "NOTUMOR":
        answer = answer + " TUMOR"


    return  "THIS IS " +   answer + ", Probability: " + str(np.round(maxproba , 2))


if file is None:
    st.write("Please Upload the MRI Image")
else:
    image = Image.open(file)
    st.image(image , width = 300)
    st.header(all(image))



