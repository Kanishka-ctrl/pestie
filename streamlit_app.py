import streamlit as st
import pickle
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    with open('resnet50_0.497.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Preprocess the image for your specific model's needs
def preprocess_image(image):
    # Example preprocessing, change as per your model's requirements
    image = image.resize((224, 224))
    image = np.array(image).flatten()  # Flatten if needed
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Perform prediction
def predict(image, model):
    predictions = model.predict(image)
    return predictions

# Streamlit app interface
st.title("Pest Detection")

uploaded_file = st.file_uploader("Upload an Image of a Pest", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    processed_image = preprocess_image(image)
    predictions = predict(processed_image, model)

    st.subheader("Prediction Results")
    st.write(predictions)
