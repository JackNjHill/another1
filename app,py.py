import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Define emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load model (cached)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('emotion_detection_model.h5')
    return model

model = load_model()

# Streamlit app UI
st.title("😠😄 Emotion Detection from Facial Image")

# File uploader
file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

# Prediction function
def import_and_predict(image_data, model):
    size = (48, 48)  # Resize to match training input size
    image = ImageOps.grayscale(image_data)  # Convert to grayscale if needed
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype("float32") / 255.0  # Normalize
    img_reshape = img_array.reshape(1, 48, 48, 1)  # Add batch and channel dims
    prediction = model.predict(img_reshape)
    return prediction

# Handle prediction
if file is None:
    st.text("👈 Please upload an image file to detect emotion")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = import_and_predict(image, model)
    predicted_class = emotion_labels[np.argmax(prediction)]

    st.success(f"🎯 Predicted Emotion: **{predicted_class.upper()}**")
