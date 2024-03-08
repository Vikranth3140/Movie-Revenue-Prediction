import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

df = pd.read_csv('final_dataset.csv')

le = LabelEncoder()

categorical_features = ['name', 'genre', 'director', 'actor_2_name', 'actor_1_name']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

features = df[['name', 'genre', 'score', 'director', 'actor_2_name', 'actor_1_name', 'budget']]
target = df['gross']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')

train_predictions = model.predict(X_train)

test_predictions = model.predict(X_test)

train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print(f'Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')

plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train')
plt.scatter(y_test, test_predictions, color='red', label='Test')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()