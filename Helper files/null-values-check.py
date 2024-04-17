import pandas as pd

# Read the CSV file
df = pd.read_csv('../revised datasets\output.csv')

# Count the number of null values for each column
null_counts = df.isnull().sum()

# Print the results
print(null_counts)