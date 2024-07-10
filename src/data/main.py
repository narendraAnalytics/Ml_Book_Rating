# src/data/main.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Function to load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess the data
def preprocess_data(df):
    # Initialize label encoders
    label_encoder_authors = LabelEncoder()
    label_encoder_publisher = LabelEncoder()

    # Apply label encoding to 'authors' and 'publisher'
    df['authors'] = label_encoder_authors.fit_transform(df['authors'])
    df['publisher'] = label_encoder_publisher.fit_transform(df['publisher'])

    # Apply one-hot encoding to 'categories' and 'language'
    df = pd.get_dummies(df, columns=['categories', 'language'], drop_first=True)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Columns to scale
    numerical_features = ['page_count', 'ratings_count', 'book_age']

    # Fit and transform the numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, label_encoder_authors, label_encoder_publisher, scaler

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

# Define the file path
file_path = os.path.join(os.path.dirname(__file__), '../../data/raw/df_cleaned_with_model.csv')

# Load the dataset
df = load_data(file_path)

# Preprocess the data
df, label_encoder_authors, label_encoder_publisher, scaler = preprocess_data(df)

# Define the features (X) and the target variable (y)
X = df.drop(columns=['average_rating', 'description'])  # Exclude the target variable and text column
y = df['average_rating']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Evaluate each model and store results
results = []
for name, model in models.items():
    mse, mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append({
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    })

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Display the results DataFrame
print(results_df)

# Select the best model based on R2 score
best_model_name = results_df.loc[results_df['R2'].idxmax()]['Model']
best_model = models[best_model_name]

print(f"Best Model: {best_model_name}")

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=2)

# Fit the GridSearchCV model
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best R2 Score: {best_score}")

# Train the best model with the entire training set
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Metrics - MSE: {mse}, MAE: {mae}, R2: {r2}")

# Save the best model and the encoders
model_path = os.path.join(os.path.dirname(__file__), '../../models/best_rf_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), '../../models/label_encoders.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '../../models/scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(best_rf_model, f)
with open(encoder_path, 'wb') as f:
    pickle.dump((label_encoder_authors, label_encoder_publisher), f)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Best model and encoders saved successfully.")
