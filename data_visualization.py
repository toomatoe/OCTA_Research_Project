import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV, skipping metadata rows
# Update this path to point to your actual data file
df = pd.read_csv(
    "data/resultsTable.csv",  # Place your CSV file in a 'data' folder
    skiprows=8,
    on_bad_lines='skip'  # Use this for pandas >= 1.3.0
)

# Rename columns for consistency
df.columns = ['Favorite', 'Model Number', 'Status', 'Model Type', 'Accuracy % (Validation)']

# Keep only relevant columns
df = df[['Model Type', 'Accuracy % (Validation)']]

# Group by model type and take the mean accuracy
df_grouped = df.groupby('Model Type')['Accuracy % (Validation)'].mean().reset_index()

# Sort by accuracy
df_grouped = df_grouped.sort_values('Accuracy % (Validation)', ascending=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Model Type', y='Accuracy % (Validation)', data=df_grouped, palette='viridis')
plt.xticks(rotation=45)
plt.title('Average Validation Accuracy by Model Type')
plt.ylabel('Accuracy (%)')
plt.xlabel('Model Type')
plt.tight_layout()
plt.show()