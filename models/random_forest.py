import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('../new_updated_less-than-1b-dataset.csv')

# Encode categorical features
le = LabelEncoder()
categorical_features = ['name', 'genre', 'director', 'actor_2_name', 'actor_1_name']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Define features and target
features = df[['name', 'genre', 'score', 'director', 'actor_2_name', 'actor_1_name', 'budget']]
target = df['gross']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV with tqdm progress bar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])) as pbar:
    grid_search.fit(X_train, y_train)
    pbar.update()

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Fit the best model on the training data
best_model.fit(X_train, y_train)

# Make predictions on training and test sets
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# Calculate R-squared scores
train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Training Accuracy (R-squared): {train_accuracy*100:.2f}%')
print(f'Test Accuracy (R-squared): {test_accuracy*100:.2f}%')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train')
plt.scatter(y_test, test_predictions, color='red', label='Test')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()