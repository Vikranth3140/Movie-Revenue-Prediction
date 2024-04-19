import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../revised datasets\output.csv')

plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df.index, df['gross'], label='Gross', linewidth=2)

ax.set_title('Index vs Gross', fontsize=16)
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('Gross', fontsize=14)

ax.grid(True, linestyle='--', linewidth=0.5)

ax.tick_params(axis='both', labelsize=12)

plt.xticks(rotation=45)

legend = ax.legend(loc='upper left', shadow=True, fontsize='large', fancybox=True)

legend.get_frame().set_facecolor('white')

plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

plt.show()