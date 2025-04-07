import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

MODEL_PATH = "Cat_model.h5"
DRIVE_FILE_ID = "1qsPJBrMfJWRixT0LSbIHMJcWVxDI4FcL"
CLASS_LABELS = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']

@st.cache_resource(show_spinner=True)
def download_model():
    """Download the model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        st.info("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource(show_spinner=True)
def load_model():
    """Ensure the model is downloaded and load it."""
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict_image(image, model):
    """Preprocess the image and predict the class."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return CLASS_LABELS[class_index], confidence


# import streamlit as st
# from PIL import Image
# from feline import load_model, predict_image

# st.title("Big Cat Classifier 🐆")

# uploaded_file = st.file_uploader("Upload a big cat image", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     model = load_model()
#     label, confidence = predict_image(image, model)

#     st.success(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")


# # import streamlit as st
# # from PIL import Image
# # from feline import load_model, predict_image

# # st.set_page_config(page_title="Big Cat Classifier 🐆")
# # st.title("Big Cat Image Classifier 🐅")
# # st.write("Upload an image of a cheetah, jaguar, leopard, lion, or tiger and get a prediction.")

# # uploaded_file = st.file_uploader("Upload a big cat image...", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded Image", use_column_width=True)

# #     with st.spinner("Making prediction..."):
# #         model = load_model()
# #         label, confidence = predict_image(model, image)

# #     st.success(f"Prediction: **{label}** with **{confidence * 100:.2f}%** confidence.")
