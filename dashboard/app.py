# dashboard/app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Define paths
data_path = os.path.join(os.path.dirname(__file__), 'data/df_cleaned_with_model.csv')
model_path = os.path.join(os.path.dirname(__file__), 'models/best_rf_model.pkl')

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

# Load data and model
df_cleaned = load_data(data_path)
best_rf_model = load_model(model_path)

# Streamlit app
st.title('Book Rating Prediction')

# Dropdown for selecting the book title
title = st.selectbox('Select Book Title', options=df_cleaned['title'])  # Ensure 'title' is in the DataFrame

# Fetch the details of the selected book
book_details = df_cleaned[df_cleaned['title'] == title].iloc[0]

# Display book details
st.write(f"Title: {title}")
st.write(f"Authors: {book_details['authors']}")
st.write(f"Publisher: {book_details['publisher']}")
st.write(f"Page Count: {book_details['page_count']}")
st.write(f"Ratings Count: {book_details['ratings_count']}")
st.write(f"Book Age: {book_details['book_age']}")
st.write(f"Categories: {', '.join([col for col in df_cleaned.columns if 'categories_' in col and book_details[col] == 1])}")
st.write(f"Language: {', '.join([col for col in df_cleaned.columns if 'language_' in col and book_details[col] == 1])}")

# Inputs for numerical features
page_count = st.number_input('Page Count', min_value=1, max_value=2000, value=int(book_details['page_count']))
ratings_count = st.number_input('Ratings Count', min_value=0, max_value=10000, value=int(book_details['ratings_count']))
book_age = st.number_input('Book Age', min_value=0, max_value=50, value=int(book_details['book_age']))

# Encode the categorical features
authors_encoded = book_details['authors']
publisher_encoded = book_details['publisher']

# One-hot encode categories and language (assuming binary input)
categories_encoded = [book_details[col] for col in df_cleaned.columns if col.startswith('categories_')]
language_encoded = [book_details[col] for col in df_cleaned.columns if col.startswith('language_')]

# Combine all inputs into a single array
features = np.array([authors_encoded, publisher_encoded, page_count, ratings_count, book_age] + categories_encoded + language_encoded).reshape(1, -1)

# Predict and display the result
prediction = best_rf_model.predict(features)
st.write(f'Predicted Average Rating: {prediction[0]:.2f}')


