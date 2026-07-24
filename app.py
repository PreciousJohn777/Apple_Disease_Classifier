import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# LOAD TRAINED MODEL

model = tf.keras.models.load_model("models/apple_classifier.keras")

# Class names
classes = ["Formalin-mixed", "Rotten"]


# PAGE TITLE

st.title("🍎 Apple Disease Classification")
st.write("CE1 Mini Project")
st.write("Classify Apple Images into:")
st.write("- Rotten Apple")
st.write("- Formalin-mixed Apple")


# IMAGE UPLOADER
# ==========================
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
