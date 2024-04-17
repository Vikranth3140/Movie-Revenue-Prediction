import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder

# Read the CSV file containing the data
df = pd.read_csv('../../revised datasets\output.csv')

# Select relevant features and target
numerical_features = df[['year', 'score', 'votes', 'budget', 'runtime']]
categorical_features = df[['rating', 'genre', 'director', 'writer', 'star', 'country', 'company']]
target = df['gross']

# Perform one-hot encoding on categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(categorical_features)

# Convert feature names to strings and combine numerical and encoded categorical features
numerical_features.columns = numerical_features.columns.astype(str)
categorical_feature_names = [str(col) for col in encoder.get_feature_names_out(categorical_features.columns)]
feature_names = categorical_feature_names + list(numerical_features.columns)
features = pd.concat([pd.DataFrame(encoded_categorical), numerical_features], axis=1)
features.columns = feature_names

# Initialize SelectKBest with f_regression scoring function and fit it to the data
kbest = SelectKBest(score_func=f_regression, k='all')
kbest.fit(features, target)

# Get the feature scores
feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': kbest.scores_})

# Filter for features with scores greater than 100
significant_features = feature_scores[feature_scores['Score'] > 100]

# Print both the features and their corresponding scores
print(feature_scores)

# Write significant features and their scores to a text file
significant_features.to_csv('significant_features.txt', index=False)