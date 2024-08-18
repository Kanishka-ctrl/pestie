import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pickle

# Function to load the model based on the type
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Try loading as PyTorch model
        with open('resnet50_0.497.pkl', 'rb') as file:
            model = torch.load(file)
        model.eval()  # Set to evaluation mode if it's a PyTorch model
        return model
    except AttributeError:
        # If loading as PyTorch model fails, try loading as a scikit-learn model
        with open('resnet50_0.497.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict using the model
def predict(image, model):
    try:
        # Try prediction as a PyTorch model
        with torch.no_grad():
            output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities
    except AttributeError:
        # If prediction fails, assume it's a scikit-learn model
        image = image.numpy().flatten().reshape(1, -1)  # Adjust if necessary
        predictions = model.predict(image)
        return predictions

# Streamlit App Interface
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
