import streamlit as st
import numpy as np
from PIL import Image
from deeplerningcorsera import model1

# Load your trained model
model = model1

st.title("🐶🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image path", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (resize, normalize, etc.)
    img = image.resize((128, 128))  # adjust to your model's input size
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)  # adjust shape as needed

    # Predict
    prediction = model.predict(img_array)
    label = "Dog" if prediction[0] > 0.5 else "Cat"
    st.write(f"Prediction: **{label}**")
