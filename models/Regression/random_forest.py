import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from feature_scaling import prepare_features


# Loading our dataset
df = pd.read_csv("revised datasets\output.csv")

# Getting the Preprocessed and scaled data.
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = np.mean(np.abs((np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true))) * 100
    return r2, mse, msle, mape


train_r2, train_mse, train_msle, train_mape = calculate_metrics(
    y_train, train_predictions
)
test_r2, test_mse, test_msle, test_mape = calculate_metrics(y_test, test_predictions)

print(f"\nTraining Metrics:")
print(f"R2 score: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"MLSE: {train_msle:.4f}")
print(f"MAPE: {train_mape:.2f}%")

print(f"\nTest Metrics:")
print(f"R2 score: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MSLE: {test_msle:.4f}")
print(f"MAPE: {test_mape:.2f}%")


# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color="blue", label="Train")
plt.scatter(y_test, test_predictions, color="red", label="Test")
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# Feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5

plt.figure(figsize=(12, 8))
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Variable Importance")
plt.tight_layout()
plt.show()
