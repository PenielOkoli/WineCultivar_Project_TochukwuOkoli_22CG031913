import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷", layout="centered")

# --- MODEL HANDLING ---
@st.cache_resource
def get_model():
    # 1. Try to load the saved model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'wine_cultivar_model.pkl')
    
    try:
        model = joblib.load(model_path)
        # Verify it works by making a dummy prediction
        dummy_data = pd.DataFrame([[13.0, 100.0, 2.0, 5.0, 1.0, 700.0]], 
                                  columns=['alcohol', 'magnesium', 'flavanoids', 
                                           'color_intensity', 'hue', 'proline'])
        model.predict(dummy_data)
        return model
        
    except Exception as e:
        # 2. FALLBACK: If loading fails (Error 10), train a new model instantly
        # This ensures your assignment submission NEVER fails.
        st.warning(f"⚠️ Could not load saved model (Error: {e}). Training a fresh model instead...")
        
        # Load Data
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['cultivar'] = data.target
        
        # Select Features
        selected_features = ['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline']
        X = df[selected_features]
        y = df['cultivar']
        
        # Build Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X, y)
        return pipeline

# Load the model (or train it if loading fails)
model = get_model()
# ----------------------

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
    # Ensure input matches the training feature order exactly
    input_data = pd.DataFrame([[alcohol, magnesium, flavanoids, color_intensity, hue, proline]], 
                              columns=['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline'])
    
    prediction = model.predict(input_data)[0]
    
    cultivar_map = {
        0: "Cultivar 1",
        1: "Cultivar 2",
        2: "Cultivar 3"
    }
    
    result = cultivar_map.get(prediction, "Unknown")
    
    st.success(f"### Prediction: {result}")
    st.info("Based on the chemical signature, this wine likely belongs to the origin above.")
