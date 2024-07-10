import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), '../../data/raw/df_cleaned_with_model.csv')
df = pd.read_csv(file_path)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['authors', 'publisher', 'categories', 'language'])

# Separate features and target variable
X = df_encoded.drop(columns=['average_rating', 'description'])
y = df_encoded['average_rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regressor": SVR()
}

# Evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "R2": r2})

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Print the results DataFrame
print("Model Evaluation Results:")
print(results_df)

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
model_path = os.path.join(os.path.dirname(__file__), '../../models/best_rf_models.pkl')
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