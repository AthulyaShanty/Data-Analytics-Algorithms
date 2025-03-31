import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model("cnn_model.h5")

st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit (28x28 grayscale) and get the prediction.")

uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=False)
    
    # Preprocess the image
    image = ImageOps.invert(image)  # MNIST digits are white on black
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # reshape for CNN input

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### Prediction: {predicted_class}")