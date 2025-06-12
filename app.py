import streamlit as st
from tensorflow.keras.models import load_model
from utils import preprocess_image
import numpy as np

st.set_page_config(page_title="Chest Disease Prediction", layout="centered")
model = load_model("model/chest_model.h5")

st.title("🩺 AI Chest X-ray Diagnosis")
st.write("Upload a chest X-ray image to predict Pneumonia.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded:
    with open("images/temp.jpg", "wb") as f:
        f.write(uploaded.read())

    st.image("images/temp.jpg", caption="Uploaded X-ray", width=300)
    img = preprocess_image("images/temp.jpg")
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error(f"Prediction: Pneumonia Detected ⚠️ (Confidence: {prediction:.2f})")
    else:
        st.success(f"Prediction: Normal ✅ (Confidence: {1 - prediction:.2f})")