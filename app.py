import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 224
classes = ["Normal", "Pneumonia"]

@st.cache_resource
def load_my_model():
    model = load_model("models/medical_model.h5")
    return model

model = load_my_model()

def preprocess_image(image):
    image = np.array(image)

    if image is None:
        raise ValueError("Invalid image")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict(image):
    processed = preprocess_image(image)

    prediction = model.predict(processed, verbose=0)

    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return classes[class_idx], confidence

st.set_page_config(page_title="AI Medical Scanner", layout="centered")

st.title("AI-Powered Medical Image Analysis")
st.write("Upload a Chest X-ray image and get instant diagnosis")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        try:
            label, conf = predict(image)

            st.markdown("## Diagnosis Result")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {conf:.2f}%")

            if label == "Pneumonia":
                st.error("Possible Lung Infection Detected")
            else:
                st.success("Lungs appear Normal")

        except Exception as e:
            st.error(f"Error: {str(e)}")
