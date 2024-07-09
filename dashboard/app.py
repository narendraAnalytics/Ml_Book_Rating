import os
import pandas as pd
import pickle
import streamlit as st

# Function to load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to load pickle file with detailed error logging
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None

# Streamlit app
def main():
    st.title('Book Rating Prediction')

    # Define base directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths to the files
    csv_file_path = os.path.join(base_dir, 'data', 'df_filtered_model.csv')
    pkl_file_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')

    # Load data
    st.header('Load Data')
    st.write('Loading CSV file...')
    data = load_csv(csv_file_path)
    st.write(data.head())

    st.write('Loading Model...')
    model = load_pickle(pkl_file_path)
    if model:
        st.write('Model loaded successfully.')
    else:
        st.write('Failed to load model. Check error logs above.')

if __name__ == '__main__':
    main()
