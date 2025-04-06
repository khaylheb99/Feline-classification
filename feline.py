import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

MODEL_PATH = "Cat_model.h5"
DRIVE_FILE_ID = "1qsPJBrMfJWRixT0LSbIHMJcWVxDI4FcL"
CLASS_LABELS = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model_from_drive()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def download_model_from_drive():
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    st.info("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = CLASS_LABELS[class_idx]
    return label, confidence
