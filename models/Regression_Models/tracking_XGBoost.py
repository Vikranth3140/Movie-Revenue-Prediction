import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np

# Loading our dataset
df = pd.read_csv('revised datasets\output.csv')

le = LabelEncoder()

categorical_features = ['released', 'writer', 'rating', 'name', 'genre', 'director', 'star', 'country', 'company']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Our features and target
features = df[['released', 'writer', 'rating', 'name', 'genre', 'director', 'star', 'country', 'company', 'runtime', 'score', 'budget', 'year', 'votes']]

target = df['gross']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a custom callback class to track the training R-squared score
train_r2_scores = []

class TrackR2Score(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # Calculate the training R-squared score
        pred = model.predict(xgb.DMatrix(X_train, label=y_train))
        train_r2 = r2_score(y_train, pred)
        train_r2_scores.append(train_r2)

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, callbacks=[TrackR2Score()]), param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(features, target)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best R^2 Score:", best_score)

best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
best_model.fit(X_train, y_train)
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# R2 scores and MAPE Calculation
train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)
print(f'\nFinal Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Final Test Accuracy: {test_accuracy*100:.2f}%')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

train_mape = mean_absolute_percentage_error(y_train, train_predictions)
test_mape = mean_absolute_percentage_error(y_test, test_predictions)
print(f'Train MAPE: {train_mape:.2f}%')
print(f'Test MAPE: {test_mape:.2f}%')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train')
plt.scatter(y_test, test_predictions, color='red', label='Test')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Plot the training R-squared score curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_r2_scores)+1), train_r2_scores)
plt.title('Training R-squared Score Curve')
plt.xlabel('Iterations')
plt.ylabel('R-squared Score')
plt.show()