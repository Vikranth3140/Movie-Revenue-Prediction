import pandas as pd

df = pd.read_csv('../revised datasets\output.csv')

null_counts = df.isnull().sum()

print(null_counts)