import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Define paths
data_path = os.path.join(os.path.dirname(__file__), '../../data/raw/df_cleaned_with_model.csv')
model_path = os.path.join(os.path.dirname(__file__), '../../models/best_rf_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), '../../models/feature_names.pkl')
unique_values_path = os.path.join(os.path.dirname(__file__), '../../models/unique_values.pkl')

# Load the dataset
df = pd.read_csv(data_path)

# Save unique values for authors, publisher, and categories
unique_values = {
    'authors': df['authors'].unique().tolist(),
    'publisher': df['publisher'].unique().tolist(),
    'categories': df['categories'].unique().tolist()
}
with open(unique_values_path, 'wb') as file:
    pickle.dump(unique_values, file)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['authors', 'publisher', 'categories', 'language'])

# Separate features and target variable
X = df_encoded.drop(columns=['average_rating', 'description'])
y = df_encoded['average_rating']

# Save the feature names
feature_names = X.columns.tolist()
with open(features_path, 'wb') as file:
    pickle.dump(feature_names, file)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_

# Train the best model
best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)

# Save the best model
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a DataFrame with the evaluation results
results = {"Model": ["Random Forest"], "MAE": [mae], "R2": [r2]}
results_df = pd.DataFrame(results)

# Print the results DataFrame
print("Model Evaluation Results:")
print(results_df)
