import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import numpy as np

# Loading our dataset
df = pd.read_csv('revised datasets\output.csv')

le = LabelEncoder()

categorical_features = ['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company']
#categorical_features = ['name', 'genre', 'director', 'star', 'country', 'company']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Our features and target
features = df[['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company', 'runtime', 'score', 'budget', 'year', 'votes']]
#features = df[['name', 'director', 'star', 'country', 'company', 'genre', 'runtime', 'score', 'budget', 'year', 'votes']]

target = df['gross']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1]
}

# Implementing GridSearchCV 
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(loss='squared_error', random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           n_jobs=-1)

grid_search.fit(features, target)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best R^2 Score:", best_score)

best_model = GradientBoostingRegressor(loss='squared_error', random_state=42, **best_params)

best_model.fit(X_train, y_train)

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# R2 scores
train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print(f'\nFinal Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Final Test Accuracy: {test_accuracy*100:.2f}%')

# Plot actual vs predicted values with enhancements
plt.figure(figsize=(12, 8))
plt.scatter(y_train, train_predictions, color='blue', alpha=0.5, label=f'Train (R² = {train_accuracy:.2f})')
plt.scatter(y_test, test_predictions, color='red', alpha=0.5, label=f'Test (R² = {test_accuracy:.2f})')

z = np.polyfit(y_test, test_predictions, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='green', linestyle='--')

plt.title('Actual vs Predicted Values with Model Accuracy')
plt.xlabel('Actual Gross Values')
plt.ylabel('Predicted Gross Values')
plt.grid(True)
plt.legend()
plt.tight_layout()

# plt.savefig('model_accuracy_plot.png', dpi=300)

plt.show()