import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../../revised datasets\output.csv")

numerical_features = df[["year", "score", "votes", "budget", "runtime"]]
categorical_features = df[
    ["rating", "genre", "director", "writer", "star", "country", "company"]
]
target = df["gross"]

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_categorical = encoder.fit_transform(categorical_features)

numerical_features.columns = numerical_features.columns.astype(str)
categorical_feature_names = [
    str(col) for col in encoder.get_feature_names_out(categorical_features.columns)
]
feature_names = categorical_feature_names + list(numerical_features.columns)

features = pd.concat([pd.DataFrame(encoded_categorical), numerical_features], axis=1)
features.columns = feature_names

kbest = SelectKBest(score_func=f_regression, k="all")

kbest.fit(features, target)

feature_scores = pd.DataFrame({"Feature": feature_names, "Score": kbest.scores_})

print(feature_scores)

feature_scores.to_csv("feature_scores.txt", index=False)
