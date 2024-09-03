import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v2.csv')

# Ensure 'Dt_Customer' is in datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])

# Calculate Recency directly from the 'Recency' column provided in the dataset
# Calculate Frequency as the sum of all purchase types
data['Frequency'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

# Calculate Monetary as the sum of all money spent
data['Monetary'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

# Calculate quantiles for each RFM metric
quantiles = data[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.50, 0.75]).to_dict()

# Print the quantiles for each RFM metric
print("Quantiles for RFM scoring:")
for metric, values in quantiles.items():
    print(f"\n{metric} Quantiles:")
    for quantile, value in values.items():
        print(f"{quantile * 100}%: {value}")

# Scoring RFM metrics into quartiles
data['R_Score'] = pd.qcut(data['Recency'], q=4, labels=[4, 3, 2, 1])  # Note the reversed labels for Recency
data['F_Score'] = pd.qcut(data['Frequency'], q=4, labels=[1, 2, 3, 4])
data['M_Score'] = pd.qcut(data['Monetary'], q=4, labels=[1, 2, 3, 4])

# Combine the RFM scores into a single score (Optional)
data['RFM_Score'] = data['R_Score'].astype(str) + data['F_Score'].astype(str) + data['M_Score'].astype(str)

# Analyze the segments
print("\nRFM Score Distribution:")
print(data['RFM_Score'].value_counts().sort_index())

# Define age and income quantiles
age_quantiles = data['Year_Birth'].quantile([0.33, 0.66]).to_dict()
income_quantiles = data['Income'].quantile([0.33, 0.66]).to_dict()
print("\nAge Quantiles:")
print(age_quantiles)
print("\nIncome Quantiles:")
print(income_quantiles)

# Categorize Age into groups
def categorize_age(age):
    if age <= age_quantiles[0.33]:
        return 3
    elif age <= age_quantiles[0.66]:
        return 2
    else:
        return 1

data['Age_Group'] = data['Year_Birth'].apply(categorize_age)

# Categorize Income into groups
def categorize_income(income):
    if income <= income_quantiles[0.33]:
        return 1
    elif income <= income_quantiles[0.66]:
        return 2
    else:
        return 3

data['Income_Group'] = data['Income'].apply(categorize_income)

# Encode the channel
def encode_channel(channel):
    if channel == 'Catalog':
        return 1
    elif channel == 'Store':
        return 2
    else:
        return 3

# Assuming you have a column that indicates the most used channel
# This part is hypothetical as we need a method to determine the most used channel by a customer
data['Channel'] = data[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].idxmax(axis=1).apply(lambda x: x.replace('Num', '').replace('Purchases', ''))

data['Channel_Code'] = data['Channel'].apply(encode_channel)

# Combine the AIC scores into a single score
data['AIC_Score'] = data['Age_Group'].astype(str) + data['Income_Group'].astype(str) + data['Channel_Code'].astype(str)

# Calculate the purchase amounts by RFM and AIC Score
data['TotalPurchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']
purchase_responsiveness = data.groupby(['RFM_Score', 'AIC_Score'])['TotalPurchases'].sum().reset_index()

# Create a pivot table for the heatmap
pivot_table = purchase_responsiveness.pivot(index='AIC_Score', columns='RFM_Score', values='TotalPurchases')

# Fill missing values with 0 to avoid white cells in the heatmap
pivot_table = pivot_table.fillna(0)


# Define a function to create a heatmap for a specific "cmp accepted" value
def plot_heatmap_by_accepted_cmp(data, accepted_cmp_value, title):
  purchase_responsiveness_filtered = data[data['AcceptedCmp5'] == accepted_cmp_value]
  pivot_table = purchase_responsiveness_filtered.pivot(index='AIC_Score', columns='RFM_Score', values='TotalPurchases')
  pivot_table = pivot_table.fillna(0)
  plt.figure(figsize=(8, 6))
  sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='g', linewidths=0.5)
  plt.title(title, fontsize=12)
  plt.xlabel('RFM Score', fontsize=10)
  plt.ylabel('AIC Score', fontsize=10)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.show()

# Create separate heatmaps for each unique "cmp accepted" value
accepted_cmp_values = data['AcceptedCmp5'].unique()
for value in accepted_cmp_values:
  title = f"Total Purchases by AIC Score and RFM Score (AcceptedCmp5={value})"
  plot_heatmap_by_accepted_cmp(data.copy(), value, title)

# Arrange the subplots in a grid layout (optional)
plt.subplots_adjust(bottom=0.1)
cols = 2  # Adjust based on the number of unique accepted_cmp_values
rows = (len(accepted_cmp_values) + 1) // cols  # Adjust based on the number of accepted_cmp_values
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 6))

for i, value in enumerate(accepted_cmp_values):
  title = f"Total Purchases by AIC Score and RFM Score (AcceptedCmp5={value})"
  plot_heatmap_by_accepted_cmp(data.copy(), value, title, ax=axes.flat[i])

# Remove unused subplots (optional)
for ax in axes.flat[len(accepted_cmp_values):]:
    ax.axis('off')

plt.tight_layout()
plt.show()
