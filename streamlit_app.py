import streamlit as st
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

@st.cache(allow_output_mutation=True)
def load_model():
    with open('resnet50_0.497.pkl', 'rb') as file:
        model = pickle.load(file)
    model.eval()  # Set the model to evaluation mode if it's a PyTorch model
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model):
    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

st.title("Pest Detection using ResNet50")

uploaded_file = st.file_uploader("Upload an Image of a Pest", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    processed_image = preprocess_image(image)
    predictions = predict(processed_image, model)

    st.subheader("Prediction Results")
    for idx, probability in enumerate(predictions):
        st.write(f"Class {idx}: {probability.item():.4f}")
