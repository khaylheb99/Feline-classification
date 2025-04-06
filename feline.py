import streamlit as st
from PIL import Image
from model import load_model, predict_image

st.set_page_config(page_title="🐆 Big Cat Classifier 🦁")
st.title("🐾 🐆 Big Cat Classifier 🦁")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = download_model("Cat_model.pth")
    prediction = predict_image(model, image)
    
    st.success(f"Prediction: {prediction}")
