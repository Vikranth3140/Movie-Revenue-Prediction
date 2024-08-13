import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("revised datasets\output.csv")

bins = [0, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
labels = ["0-$1M", "$1M-$10M", "$10M-$100M", "$100M-$1B"]

df["gross_category"] = pd.cut(df["gross"], bins=bins, labels=labels, right=False)

category_counts = df["gross_category"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
category_counts.plot(
    kind="bar", width=0.45, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
)

plt.title("Number of Movies by Gross Category")
plt.xlabel("Gross Category")
plt.ylabel("Number of Movies")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
