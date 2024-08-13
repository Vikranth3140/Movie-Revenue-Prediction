import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../revised datasets/output.csv")

total_budget = df.groupby("name")["budget"].sum()
total_gross = df.groupby("name")["gross"].sum()

total_budget.index = total_budget.index.astype(str)
total_gross.index = total_gross.index.astype(str)

plt.figure(figsize=(10, 6))
plt.bar(total_budget.index, total_budget, label="Budget", width=0.4)
plt.bar(total_gross.index, total_gross, label="Gross", width=0.4, align="edge")
plt.xlabel("Title")
plt.ylabel("Amount ($)")
plt.title("Budget vs. Gross")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
