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
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except pickle.UnpicklingError:
        st.error("Error loading the model. The file might be corrupted or not a valid pickle file.")
        return None

# Load encoders and scaler
@st.cache_resource
def load_encoders(path):
    try:
        with open(path, 'rb') as file:
            encoders = pickle.load(file)
        return encoders
    except pickle.UnpicklingError:
        st.error("Error loading the encoders. The file might be corrupted or not a valid pickle file.")
        return None

@st.cache_resource
def load_scaler(path):
    try:
        with open(path, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except pickle.UnpicklingError:
        st.error("Error loading the scaler. The file might be corrupted or not a valid pickle file.")
        return None

# Load data, model, encoders, and scaler
df_cleaned = load_data(data_path)
best_rf_model = load_model(model_path)
label_encoders = load_encoders(encoder_path)
scaler = load_scaler(scaler_path)

if best_rf_model is None or label_encoders is None or scaler is None:
    st.stop()

# Extract encoders
label_encoder_authors, label_encoder_publisher = label_encoders

# Streamlit app
st.title('Book Rating Prediction')

# Get user input for numerical features
page_count = st.number_input('Page Count', min_value=1, max_value=2000)
ratings_count = st.number_input('Ratings Count', min_value=0, max_value=10000)
book_age = st.number_input('Book Age', min_value=0, max_value=50)

# Get user input for categorical features
authors = st.selectbox('Authors', options=label_encoder_authors.classes_)
publisher = st.selectbox('Publisher', options=label_encoder_publisher.classes_)
categories = st.multiselect('Categories', options=df_cleaned['categories'].unique())
language = st.selectbox('Language', options=df_cleaned['language'].unique())

# Encode the categorical features
authors_encoded = label_encoder_authors.transform([authors])[0]
publisher_encoded = label_encoder_publisher.transform([publisher])[0]

# Create a DataFrame for the input features
input_features = pd.DataFrame({
    'authors': [authors_encoded],
    'publisher': [publisher_encoded],
    'page_count': [page_count],
    'ratings_count': [ratings_count],
    'book_age': [book_age]
})

# One-hot encode the categories and language features
if categories:
    categories_encoded = pd.get_dummies(pd.DataFrame(categories, columns=['categories']).T, prefix='categories')
    categories_encoded = categories_encoded.reindex(columns=[col for col in df_cleaned.columns if col.startswith('categories_')], fill_value=0)
else:
    categories_encoded = pd.DataFrame(columns=[col for col in df_cleaned.columns if col.startswith('categories_')])

language_encoded = pd.get_dummies(pd.Series([language]), prefix='language')
language_encoded = language_encoded.reindex(columns=[col for col in df_cleaned.columns if col.startswith('language_')], fill_value=0)

# Combine the encoded features with the input features
input_features = pd.concat([input_features, categories_encoded, language_encoded], axis=1)

# Ensure all one-hot encoded columns are present
for col in df_cleaned.columns:
    if col not in input_features.columns:
        input_features[col] = 0

# Reorder columns to match the training set
input_features = input_features[df_cleaned.columns]

# Scale the numerical features
numerical_features = ['page_count', 'ratings_count', 'book_age']
input_features[numerical_features] = scaler.transform(input_features[numerical_features])

# Ensure the DataFrame has the same number of features as the model expects
st.write(f"Shape of input_features: {input_features.shape}")
st.write(f"Columns of input_features: {input_features.columns}")

# Prepare the feature array for prediction
features = input_features.values

# Predict and display the result
prediction = best_rf_model.predict(features)
st.write(f'Predicted Average Rating: {prediction[0]:.2f}')
