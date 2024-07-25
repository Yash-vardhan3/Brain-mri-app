import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from dotenv import load_dotenv
import os
import requests
import io

# Load environment variables from .env file
load_dotenv()

# Get model path and Azure credentials from environment variables
model_path = os.getenv('MODEL_PATH', 'final_model.h5')
azure_key = os.getenv('AZURE_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')

# Load the model
model = tf.keras.models.load_model(model_path)

# Define the image size
img_size = (224, 224)

def predict_image(img):
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    class_name = class_labels[class_idx]
    confidence = predictions[0][class_idx]
    
    return class_name, confidence

def visualize_prediction(img, class_name, confidence):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
    st.image(img, caption=f'Predicted: {class_name} ({confidence:.2f})', use_column_width=True)

def explain_prediction_lime(model, img_array, num_samples=1000):
    def model_predict(images):
        preds = model.predict(images)
        return preds

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array[0], model_predict, top_labels=1, hide_color=0, num_samples=num_samples)
    
    temp, mask = explanation.get_image_and_mask(label=0, positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp, mask)
    return img_boundry

def analyze_image_with_azure(img):
    analyze_url = azure_endpoint + "vision/v3.2/describe"

    headers = {
        "Ocp-Apim-Subscription-Key": azure_key,
        "Content-Type": "application/octet-stream",
    }

    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    response = requests.post(analyze_url, headers=headers, data=img_bytes)
    result = response.json()
    
    # Extract image properties
    width = result.get("metadata", {}).get("width", "Unknown")
    height = result.get("metadata", {}).get("height", "Unknown")
    
    # Convert PIL image to OpenCV format to calculate average color
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
    average_color = img_cv.mean(axis=(0, 1))  # Average color in BGR format

    return width, height, average_color

# Streamlit app
st.title('Brain MRI Tumor Classification')

st.write("Upload an MRI image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    img_array = np.array(img.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner('Predicting image with TensorFlow model...'):
        try:
            class_name, confidence = predict_image(img)
            st.write(f"Predicted Class: {class_name}")
            st.write(f"Confidence: {confidence:.2f}")
            visualize_prediction(img, class_name, confidence)

            if st.checkbox('Show LIME Explanation'):
                lime_img_boundaries = explain_prediction_lime(model, img_array)
                st.image(lime_img_boundaries, caption='LIME Explanation', use_column_width=True)

            if st.checkbox('Show Azure Image Details'):
                width, height, average_color = analyze_image_with_azure(img)
                st.write(f"Image Width: {width}")
                st.write(f"Image Height: {height}")
                st.write(f"Average Pixel Color (BGR): {average_color}")

        except Exception as e:
            st.error(f"Error processing image: {e}")
