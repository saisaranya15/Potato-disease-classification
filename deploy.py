import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Model Path
MODEL_PATH = "potato_disease_model.h5"
CLASS_LABELS = ["Early Blight", "Healthy", "Late Blight"]

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Function to Predict Disease
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return CLASS_LABELS[predicted_class]

# Streamlit UI
st.title("🥔 Potato Disease Classification")
st.write("Upload a potato leaf image to classify it as **Early Blight, Healthy, or Late Blight**.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image
    img_path = os.path.join("temp_image.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display Uploaded Image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Predict
    prediction = predict_image(img_path)
    st.success(f"### 🏷 Prediction: **{prediction}**")

    # Remove temp file
    os.remove(img_path)
