import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

MODEL_PATH = "Cat_model.h5"
DRIVE_FILE_ID = "1qsPJBrMfJWRixT0LSbIHMJcWVxDI4FcL"
CLASS_LABELS = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']

# @st.cache_resource
# import gdown
# import os
# import tensorflow as tf

# MODEL_PATH = "efficientnetv2_model.h5"
# FILE_ID = "1qsPJBrMfJWRixT0LSbIHMJcWVxDI4FcL"

@st.cache_resource(show_spinner=True)
def download_model():
    file_id = "1qsPJBrMfJWRixT0LSbIHMJcWVxDI4FcL"
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

# def load_model():
#     download_model()  # Ensure the model is downloaded
#     model = tf.keras.models.load_model("model")
#     return model
# def load_model():
#     model = tf.keras.models.load_model("model")
#     return model

# def load_model():
#     download_model()  # Ensure the model is downloaded
#     model = tf.keras.models.load_model("saved_model.pb")
#     return model

# def load_model():
#     download_model()
#     model = tf.keras.models.load_model(MODEL_PATH)
#     return model

# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         url = f"https://drive.google.com/uc?id={FILE_ID}"
#         gdown.download(url, MODEL_PATH, quiet=False)

def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         download_model_from_drive()
#     model = tf.keras.models.load_model(MODEL_PATH)
#     return model

# def download_model_from_drive():
#     url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
#     st.info("Downloading model from Google Drive...")
#     gdown.download(url, MODEL_PATH, quiet=False)

# def preprocess_image(image: Image.Image):
#     image = image.convert("RGB")
#     image = image.resize((224, 224))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

def predict_image(image, model):
    class_labels = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_labels[class_index], confidence
