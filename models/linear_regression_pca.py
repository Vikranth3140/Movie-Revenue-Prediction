import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv('../revised datasets\output.csv')

le = LabelEncoder()

categorical_features = ['name', 'genre', 'director', 'star', 'country', 'company']
# categorical_features = ['name', 'genre', 'director', 'star']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# features = df[['name', 'genre', 'director', 'star', 'genre', 'score', 'budget', 'year']]
features = df[['name', 'genre', 'director', 'star', 'country', 'company', 'genre', 'runtime', 'score', 'budget', 'year', 'votes']]

target = df['gross']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=8)  # You can adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model = LinearRegression()

model.fit(X_train_pca, y_train)

predictions = model.predict(X_test_pca)

mse = mean_squared_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')

train_predictions = model.predict(X_train_pca)
test_predictions = model.predict(X_test_pca)

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