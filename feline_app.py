import streamlit as st
from PIL import Image
from feline import load_model, predict_image

st.set_page_config(page_title="Big Cat Classifier 🐆")
st.title("Big Cat Image Classifier 🐅")
st.write("Upload an image of a cheetah, jaguar, leopard, lion, or tiger and get a prediction.")

uploaded_file = st.file_uploader("Upload a big cat image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Making prediction..."):
        model = load_model()
        label, confidence = predict_image(model, image)

    st.success(f"Prediction: **{label}** with **{confidence * 100:.2f}%** confidence.")
