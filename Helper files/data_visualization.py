import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the file
data = pd.read_csv('revised datasets\output.csv')

# Set the seaborn style
sns.set_style('darkgrid')

# Create a figure with a larger size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the curve joining the top of the 'gross' values
ax.plot(data.index, data['gross'], label='Gross', linewidth=2)

# Set the title and labels with increased font size
ax.set_title('Index vs Gross', fontsize=16)
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('Gross', fontsize=14)

# Add grid lines
ax.grid(True, linestyle='--', linewidth=0.5)

# Increase the size of the tick labels
ax.tick_params(axis='both', labelsize=12)

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Add a legend with fancy box
legend = ax.legend(loc='upper left', shadow=True, fontsize='large', fancybox=True)

# Set the background color of the legend box to white
legend.get_frame().set_facecolor('white')

# Adjust the spacing between subplots
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

# Show the plot
plt.show()