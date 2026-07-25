import os
import urllib.request
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/apple_classifier.keras"
MODEL_URL = "https://github.com/PreciousJohn777/Apple_Disease_Classifier/releases/download/v1.0.0/apple_classifier.keras"


import os
import urllib.request

# Download if the file is missing OR if it's just the tiny Git LFS pointer
if (not os.path.exists(MODEL_PATH)) or os.path.getsize(MODEL_PATH) < 1000000:
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
classes = ["Formalin-mixed", "Rotten"]

# PAGE TITLE

st.markdown("""
Upload an image of an apple and the trained deep learning model will classify it as either:

- 🍎 Rotten Apple
- 🍏 Formalin-mixed Apple
""")
st.write("CE1 Mini Project")
st.write("Classify Apple Images into:")
st.write("- Rotten Apple")
st.write("- Formalin-mixed Apple")


# IMAGE UPLOADER
uploaded_file = st.file_uploader(
    "Upload an Apple Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence >= 0.5:
        st.success("Prediction: Rotten Apple")
        st.write(f"Confidence: {confidence * 100:.2f}%")
    else:
        st.success("Prediction: Formalin-mixed Apple")
        st.write(f"Confidence: {(1 - confidence) * 100:.2f}%")

import streamlit as st

st.write("Streamlit version:", st.__version__)