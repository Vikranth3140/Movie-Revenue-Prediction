import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('output.csv')

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

# Impute numeric and categorical columns separately
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Encode categorical features
le = LabelEncoder()
for feature in categorical_cols:
    df[feature] = le.fit_transform(df[feature])

# Feature engineering
df['budget_per_score'] = df['budget'] / df['score']

# Define features and target
numeric_features = ['score', 'budget', 'budget_per_score']
categorical_features = categorical_cols.tolist()
features = df[numeric_features + categorical_features]
target = df['gross']

# Preprocessing pipelines
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model pipelines
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gb', GradientBoostingRegressor(random_state=42))
])

# Hyperparameter tuning with tqdm progress bar
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

# Reduce cross-validation folds
rf_gridsearch = GridSearchCV(rf_pipeline, param_grid=rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
gb_gridsearch = GridSearchCV(gb_pipeline, param_grid=gb_param_grid, cv=3, scoring='r2', n_jobs=-1)

# TQDM progress bar for grid search
with tqdm(total=len(rf_param_grid) * len(gb_param_grid)) as pbar:
    rf_gridsearch.fit(X_train, y_train)
    pbar.update()

    gb_gridsearch.fit(X_train, y_train)
    pbar.update()

# Ensemble model
ensemble_model = VotingRegressor([
    ('rf', rf_gridsearch.best_estimator_),
    ('gb', gb_gridsearch.best_estimator_)
])

ensemble_model.fit(X_train, y_train)

# Evaluation
train_predictions = ensemble_model.predict(X_train)
test_predictions = ensemble_model.predict(X_test)

train_accuracy = r2_score(y_train, train_predictions)
test_accuracy = r2_score(y_test, test_predictions)

print(f'Final Training R^2 Score: {train_accuracy*100:.2f}%')
print(f'Final Test R^2 Score: {test_accuracy*100:.2f}%')