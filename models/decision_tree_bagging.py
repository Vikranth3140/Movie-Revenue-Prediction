import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('../revised datasets\output.csv')

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
categorical_features = ['name', 'genre', 'director', 'star', 'country', 'company']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Define features and target
features = df[['name', 'genre', 'director', 'star', 'country', 'company', 'genre', 'runtime', 'score', 'budget', 'year', 'votes']]
target = df['gross']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize DecisionTreeRegressor as the base estimator
base_estimator = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)

# Initialize BaggingRegressor with bagging parameters
model = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate R2 scores
train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

# Print the results
print(f'Training R2 Score: {train_accuracy*100:.2f}%')
print(f'Test R2 Score: {test_accuracy*100:.2f}%')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train')
plt.scatter(y_test, test_predictions, color='red', label='Test')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()