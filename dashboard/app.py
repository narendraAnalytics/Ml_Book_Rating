import os
import pandas as pd
import pickle
import joblib
import streamlit as st
import numpy as np
import sklearn

# Function to load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to load joblib file with detailed error logging
def load_joblib(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading joblib file: {e}")
        return None

# Function to convert pickle model to joblib
def convert_pickle_to_joblib(pickle_path, joblib_path):
    try:
        with open(pickle_path, 'rb') as file:
            model = pickle.load(file)
        joblib.dump(model, joblib_path)
        st.success("Model converted from pickle to joblib successfully.")
    except Exception as e:
        st.error(f"Error converting model: {e}")

# Streamlit app
def main():
    st.title('Book Rating Prediction')

    # Define base directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths to the files
    csv_file_path = os.path.join(base_dir, 'data', 'df_filtered_model.csv')
    pickle_file_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')
    joblib_file_path = os.path.join(base_dir, 'models', 'random_forest_model.joblib')

    # Convert pickle model to joblib if joblib file does not exist
    if not os.path.exists(joblib_file_path):
        st.write("Converting pickle model to joblib...")
        convert_pickle_to_joblib(pickle_file_path, joblib_file_path)

    # Load data
    st.header('Load Data')
    st.write('Loading CSV file...')
    data = load_csv(csv_file_path)
    st.write(data.head())

    st.write('Loading Model...')
    model = load_joblib(joblib_file_path)
    if model:
        st.write('Model loaded successfully.')
    else:
        st.write('Failed to load model. Check error logs above.')

if __name__ == '__main__':
    main()
