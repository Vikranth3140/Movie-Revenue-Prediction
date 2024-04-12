import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('../revised datasets\output.csv')

# Encode categorical features
le = LabelEncoder()
categorical_features = ['name', 'genre', 'director', 'star', 'country', 'company']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Define features and target
features = df[['name', 'genre', 'director', 'star', 'country', 'company', 'genre', 'runtime', 'score', 'budget', 'year', 'votes']]
target = df['gross']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1]
}

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(loss='squared_error', random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(features, target)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best R^2 Score:", best_score)

# Use the best parameters to train the final model
best_model = GradientBoostingRegressor(loss='squared_error', random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Predictions
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# Evaluation
train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print(f'Final Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Final Test Accuracy: {test_accuracy*100:.2f}%')

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train')
plt.scatter(y_test, test_predictions, color='red', label='Test')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()