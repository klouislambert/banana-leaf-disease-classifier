import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "banana_leaf_inference.keras"
IMG_SIZE = (224, 224)          # must match training
CONFIDENCE_THRESHOLD = 0.60

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

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
        # Same preprocessing as training: tf.image.resize (bilinear), then scale to [0, 1].
        # PIL's image.resize() uses a different filter, which shifts the pixels the model sees.
        array = tf.image.resize(np.array(image, dtype=np.uint8), IMG_SIZE)
        array = tf.cast(array, tf.float32) / 255.0
        array = tf.expand_dims(array, axis=0)

        probabilities = model.predict(array, verbose=0)[0]
        index = int(np.argmax(probabilities))

        st.subheader(f"Prediction: {CLASS_NAMES[index]}")
        st.write(f"Confidence: {probabilities[index] * 100:.2f}%")
        if probabilities[index] < CONFIDENCE_THRESHOLD:
            st.warning("Low confidence. Try a clearer, well-lit photo of a single leaf.")

        st.write("Scores for all classes:")
        for name, score in zip(CLASS_NAMES, probabilities):
            st.write(f"**{name}:** {score * 100:.2f}%")
