import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "banana_leaf_inference.keras"

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(
    page_title="Banana Leaf Disease Classifier",
    page_icon="🍌"
)

st.title("🍌 Banana Leaf Disease Classification")
st.write("Upload a banana leaf image to predict its class.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a banana leaf image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict disease"):
        image = image.resize((224, 224))
        array = np.array(image, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

        probabilities = model.predict(array, verbose=0)[0]
        index = int(np.argmax(probabilities))

        st.subheader(f"Prediction: {CLASS_NAMES[index]}")
        st.write(
            f"Confidence: {probabilities[index] * 100:.2f}%"
        )

        st.write("Scores for all classes:")
        for name, score in zip(CLASS_NAMES, probabilities):
            st.write(f"**{name}:** {score * 100:.2f}%")