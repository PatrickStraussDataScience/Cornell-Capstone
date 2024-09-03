import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_V2.csv')

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

# Visualization of RFM segments
plt.figure(figsize=(12, 6))
sns.countplot(x='RFM_Score', data=data, order=sorted(data['RFM_Score'].unique()))
plt.title('RFM Segment Counts')
plt.xlabel('RFM Score')
plt.ylabel('Number of Customers')
plt.xticks(rotation=90)
plt.show()
