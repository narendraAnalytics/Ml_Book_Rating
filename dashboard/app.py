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

# Custom CSS with Google Font, enhanced prediction result styling, and animations
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .main {
        background-color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background: #F0F2F6;
    }
    .css-1v3fvcr {
        color: #333333;
        font-size: 20px;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-family: 'Roboto', sans-serif;
    }
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        font-family: 'Roboto', sans-serif;
    }
    .stSlider .st-bo {
        color: #333333;
    }
    .stSlider .st-bz {
        color: #4CAF50;
    }
    .stSlider .st-cy .st-cu, .stSlider .st-cy .st-cv, .stSlider .st-cy .st-cx {
        background: #4CAF50;
    }
    .prediction-results {
        padding: 15px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        font-family: 'Roboto', sans-serif;
        animation: fadeIn 1s ease-in-out;
    }
    .prediction-results h3 {
        color: #4CAF50;
        margin-bottom: 15px;
    }
    .prediction-results ul {
        list-style-type: none;
        padding: 0;
    }
    .prediction-results ul li {
        margin-bottom: 10px;
        font-size: 16px;
    }
    .prediction-results ul li strong {
        color: #333333;
    }
    .prediction-results ul li span {
        color: #4CAF50;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app
st.title("Book Rating Prediction")
st.write("### Predict Book Rating")

title = st.selectbox("Title", unique_values['title'])
authors = st.selectbox("Authors", unique_values['authors'])
publisher = st.selectbox("Publisher", unique_values['publisher'])
categories = st.selectbox("Categories", unique_values['categories'])
description = st.text_area("Description (not used in prediction)")

page_count = st.slider("Page Count", min_value=1, max_value=max_page_count, value=1)
ratings_count = st.number_input("Ratings Count", min_value=1)
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
    
    # Display the user's input and prediction with enhanced styling and animation
    st.markdown(
        f"""
        <div class="prediction-results">
            <h3>Prediction Results</h3>
            <ul>
                <li><strong>Title:</strong> {title}</li>
                <li><strong>Authors:</strong> {authors}</li>
                <li><strong>Publisher:</strong> {publisher}</li>
                <li><strong>Categories:</strong> {categories}</li>
                <li><strong>Page Count:</strong> {page_count}</li>
                <li><strong>Ratings Count:</strong> {ratings_count}</li>
                <li><strong>Book Age:</strong> {book_age}</li>
                <li><strong>Predicted Average Rating:</strong> <span>{prediction:.2f}</span></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
