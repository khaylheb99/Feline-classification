import tensorflow as tf
import numpy as np
import gdown  # For Google Drive downloads
from PIL import Image
import os

class_labels = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']


MODEL_PATH = "Cat_model.h5"
GDRIVE_URL = "https://drive.google.com/file/d/1qsPJBrMfJWRixT0LSbIHMJcWVxDI4FcL/view?usp=sharing
"

def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # normalize to [0,1]
    if img_array.shape[-1] == 4:  # remove alpha if present
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

def predict_image(model, image: Image.Image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class
