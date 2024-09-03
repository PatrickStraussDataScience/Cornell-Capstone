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

# Filter the data to include only the specified user IDs
specified_user_ids = [2201, 1250, 77, 1914, 559, 1504, 1958, 2030, 984, 1084, 1893, 376, 687, 1641, 575, 845, 1539, 676, 333, 2228, 1710, 2193, 1030, 1066, 1182, 1443, 15, 329, 591, 406, 1528, 2177, 763, 605, 1887, 126, 1922, 1111, 791, 1308, 787, 1334, 1280, 1809, 519, 53, 703, 1087, 644, 1272, 278]
filtered_data = data[data['ID'].isin(specified_user_ids)]

# Calculate the purchase amounts by RFM and AIC Score
filtered_data['TotalHits'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
purchase_responsiveness = filtered_data.groupby(['RFM_Score', 'AIC_Score'])['TotalHits'].sum().reset_index()

# Create a pivot table for the heatmap
pivot_table = purchase_responsiveness.pivot(index='AIC_Score', columns='RFM_Score', values='TotalHits')
pivot_table = pivot_table[::-1]

# Fill missing values with 0 to avoid white cells in the heatmap
pivot_table = pivot_table.fillna(0)

# Plotting heatmap for AIC and RFM scores
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='g', linewidths=0.5)
plt.title('AIC Score and RFM Score for Total Spent by Campaign 6 Acceptors', fontsize=15)  # Adjusted font size
plt.xlabel('RFM Score', fontsize=12)  # Adjusted font size
plt.ylabel('AIC Score', fontsize=12)  # Adjusted font size
plt.xticks(fontsize=10)  # Adjusted font size
plt.yticks(fontsize=10)  # Adjusted font size
plt.show()
