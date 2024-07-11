import pandas as pd
import os
import pickle
import streamlit as st

# Define paths
model_path = os.path.join(os.path.dirname(__file__), 'models/best_rf_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), 'models/feature_names.pkl')
unique_values_path = os.path.join(os.path.dirname(__file__), 'models/unique_values.pkl')
max_values_path = os.path.join(os.path.dirname(__file__), 'models/max_values.pkl')

# Load the best model
with open(model_path, 'rb') as file:
    best_model = pickle.load(file)

# Load the feature names
with open(features_path, 'rb') as file:
    feature_names = pickle.load(file)

# Load unique values
with open(unique_values_path, 'rb') as file:
    unique_values = pickle.load(file)

# Load maximum values
with open(max_values_path, 'rb') as file:
    max_values = pickle.load(file)

# Convert max_values to integers
max_page_count = int(max_values['page_count'])
max_book_age = int(max_values['book_age'])
max_ratings_count = int(max_values['ratings_count'])


# Streamlit app
st.title("Book Rating Prediction")

st.write("### Predict Book Rating")

title = st.selectbox("Title", unique_values['title'])
authors = st.selectbox("Authors", unique_values['authors'])
publisher = st.selectbox("Publisher", unique_values['publisher'])
categories = st.selectbox("Categories", unique_values['categories'])
description = st.text_area("Description (not used in prediction)")

page_count = st.slider("Page Count", min_value=1, max_value=max_page_count, value=1)
ratings_count = st.slider("Ratings Count", min_value=1, max_value=max_ratings_count, value=1)
book_age = st.slider("Book Age", min_value=0, max_value=max_book_age, value=0)

if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'title': [title],
        'authors': [authors],
        'publisher': [publisher],
        'categories': [categories],
        'description': [description],
        'page_count': [page_count],
        'ratings_count': [ratings_count],
        'book_age': [book_age]
    })

    # Exclude non-feature columns from prediction input
    prediction_input = input_data.drop(columns=['title', 'description'])

    # One-hot encode the input data
    prediction_input_encoded = pd.get_dummies(prediction_input)

    # Add missing columns with zeros
    for col in feature_names:
        if col not in prediction_input_encoded.columns:
            prediction_input_encoded[col] = 0

    # Ensure the column order matches the training data
    prediction_input_encoded = prediction_input_encoded[feature_names]

    # Make a prediction
    prediction = best_model.predict(prediction_input_encoded)[0]
    st.write(f"Predicted Average Rating: {prediction:.2f}")
