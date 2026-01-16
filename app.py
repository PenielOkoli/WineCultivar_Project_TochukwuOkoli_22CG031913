import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os  # <--- Make sure to import this

# Page Configuration
st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷", layout="centered")

# --- UPDATED LOAD FUNCTION ---
@st.cache_resource
def load_model():
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the model file
    # This expects the file to be in a folder named 'model' inside the same folder as app.py
    model_path = os.path.join(current_dir, 'model', 'wine_cultivar_model.pkl')
    
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        # Debugging help: Print where it looked so you can fix it
        st.error(f"File not found at: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
        
model = load_model()

# UI Header
st.title("🍷 Wine Cultivar Prediction System")
st.markdown("Enter the chemical properties of the wine below to predict its origin cultivar.")
st.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.subheader("Chemical Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alcohol = st.number_input("Alcohol", min_value=10.0, max_value=15.0, value=13.0, step=0.1)
        magnesium = st.number_input("Magnesium", min_value=70.0, max_value=170.0, value=100.0, step=1.0)
        flavanoids = st.number_input("Flavanoids", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
        
    with col2:
        color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1)
        hue = st.number_input("Hue", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
        proline = st.number_input("Proline", min_value=200.0, max_value=1700.0, value=700.0, step=10.0)
    
    submit_button = st.form_submit_button("Predict Cultivar")

# Prediction Logic
if submit_button:
    # specific order matters! Must match the training order.
    input_data = pd.DataFrame([[alcohol, magnesium, flavanoids, color_intensity, hue, proline]], 
                              columns=['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline'])
    
    prediction = model.predict(input_data)[0]
    
    # Mapping prediction to Class Name (0, 1, 2) -> (Class 1, Class 2, Class 3)
    # The dataset typically maps 0->Class 1, 1->Class 2, 2->Class 3
    cultivar_map = {
        0: "Cultivar 1",
        1: "Cultivar 2",
        2: "Cultivar 3"
    }
    
    result = cultivar_map.get(prediction, "Unknown")
    
    st.success(f"### Prediction: {result}")

    st.info("Based on the chemical signature, this wine likely belongs to the origin above.")
