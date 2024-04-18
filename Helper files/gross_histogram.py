import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('revised datasets\output.csv')

# Define the gross categories
bins = [0, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
labels = ['0-$1M', '$1M-$10M', '$10M-$100M', '$100M-$1B']

# Categorize the 'gross' column into discrete categories
df['gross_category'] = pd.cut(df['gross'], bins=bins, labels=labels, right=False)

# Count the number of movies in each category
category_counts = df['gross_category'].value_counts().sort_index()

# Plot the bar histogram
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color=['red', 'green', 'blue', 'purple'])

# Set the title and labels
plt.title('Number of Movies by Gross Category')
plt.xlabel('Gross Category')
plt.ylabel('Number of Movies')

# Make the graph presentable
plt.xticks(rotation=45)
plt.tight_layout()


# Show the plot
plt.show()
