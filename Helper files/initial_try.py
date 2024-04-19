import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

df = pd.read_csv('../revised datasets\output.csv')

numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

le = LabelEncoder()

# Encode categorical features
categorical_features = ['released','writer','rating','name', 'genre', 'director', 'star', 'country', 'company']

for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

df['budget_per_score'] = df['budget'] / df['score']

numeric_features = ['score', 'budget', 'budget_per_score']
categorical_features = categorical_cols.tolist()
features = df[numeric_features + categorical_features]
target = df['gross']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gb', GradientBoostingRegressor(random_state=42))
])

rf_param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 5],
    'rf__max_features': ['sqrt', None]
}

gb_param_grid = {
    'gb__n_estimators': [50, 100, 150],
    'gb__max_depth': [3, 6],
    'gb__learning_rate': [0.05, 0.1]
}

rf_gridsearch = GridSearchCV(rf_pipeline, param_grid=rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
gb_gridsearch = GridSearchCV(gb_pipeline, param_grid=gb_param_grid, cv=3, scoring='r2', n_jobs=-1)

with tqdm(total=len(rf_param_grid) * len(gb_param_grid)) as pbar:
    rf_gridsearch.fit(X_train, y_train)
    pbar.update()

    gb_gridsearch.fit(X_train, y_train)
    pbar.update()

ensemble_model = VotingRegressor([
    ('rf', rf_gridsearch.best_estimator_),
    ('gb', gb_gridsearch.best_estimator_)
])

ensemble_model.fit(X_train, y_train)

train_predictions = ensemble_model.predict(X_train)
test_predictions = ensemble_model.predict(X_test)

train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print()
print(f'Final Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Final Test Accuracy: {test_accuracy*100:.2f}%')

print()

mse = mean_squared_error(y_test, test_predictions)

print(f'Mean Squared Error: {mse}')
print()

plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color='blue', label='Train')
plt.scatter(y_test, test_predictions, color='red', label='Test')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()