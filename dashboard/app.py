# dashboard/app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Define paths
data_path = os.path.join(os.path.dirname(__file__), 'data/df_cleaned_with_model.csv')
model_path = os.path.join(os.path.dirname(__file__), 'models/best_rf_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'models/label_encoders.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')

# Load the dataset
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Load the model
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load encoders and scaler
@st.cache_resource
def load_encoders(path):
    with open(path, 'rb') as file:
        encoders = pickle.load(file)
    return encoders

@st.cache_resource
def load_scaler(path):
    with open(path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Load data, model, encoders, and scaler
df_cleaned = load_data(data_path)
best_rf_model = load_model(model_path)
label_encoder_authors, label_encoder_publisher = load_encoders(encoder_path)
scaler = load_scaler(scaler_path)

# Drop the 'title' column for prediction
if 'title' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['title'])

# Streamlit app
st.title('Book Rating Prediction')

# Get user input for numerical features
page_count = st.number_input('Page Count', min_value=1, max_value=2000)
ratings_count = st.number_input('Ratings Count', min_value=0, max_value=10000)
book_age = st.number_input('Book Age', min_value=0, max_value=50)

# Get user input for categorical features
authors = st.selectbox('Authors', options=label_encoder_authors.classes_)
publisher = st.selectbox('Publisher', options=label_encoder_publisher.classes_)

# Encode the categorical features
authors_encoded = label_encoder_authors.transform([authors])[0]
publisher_encoded = label_encoder_publisher.transform([publisher])[0]

# Dummy encode categories and language (assuming binary input)
categories_encoded = [0] * len([col for col in df_cleaned.columns if col.startswith('categories_')])
language_encoded = [0] * len([col for col in df_cleaned.columns if col.startswith('language_')])

# Combine all inputs into a single array
features = np.array([authors_encoded, publisher_encoded, page_count, ratings_count, book_age] + categories_encoded + language_encoded).reshape(1, -1)

# Ensure all features are numeric
features = features.astype(float)

# Predict and display the result
prediction = best_rf_model.predict(features)
st.write(f'Predicted Average Rating: {prediction[0]:.2f}')
