import pandas as pd
import numpy as np
import plotly.express as px

# Load the dataset
data = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v2.csv')

# Ensure 'Dt_Customer' is in datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])

# Calculate Recency directly from the 'Recency' column provided in the dataset
# Calculate Frequency as the sum of all purchase types
data['Frequency'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

# Calculate Monetary as the sum of all money spent
data['Monetary'] = (data['MntWines']) + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + (data['MntGoldProds']*2)

# Calculate quantiles for each RFM metric
quantiles = data[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.50, 0.75]).to_dict()

# Scoring RFM metrics into quartiles
data['R_Score'] = pd.qcut(data['Recency'], q=4, labels=[4, 3, 2, 1])  # Note the reversed labels for Recency
data['F_Score'] = pd.qcut(data['Frequency'], q=4, labels=[1, 2, 3, 4])
data['M_Score'] = pd.qcut(data['Monetary'], q=4, labels=[1, 2, 3, 4])

# Combine the RFM scores into a single score
data['RFM_Score'] = data['R_Score'].astype(str) + data['F_Score'].astype(str) + data['M_Score'].astype(str)

# Define age and income quantiles
age_quantiles = data['Year_Birth'].quantile([0.33, 0.66]).to_dict()
income_quantiles = data['Income'].quantile([0.33, 0.66]).to_dict()

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

# Determine the most used channel by a customer
data['Channel'] = data[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].idxmax(axis=1).apply(lambda x: x.replace('Num', '').replace('Purchases', ''))

data['Channel_Code'] = data['Channel'].apply(encode_channel)

# Combine the AIC scores into a single score
data['AIC_Score'] = data['Age_Group'].astype(str) + data['Income_Group'].astype(str) + data['Channel_Code'].astype(str)

# Calculate the purchase amounts and campaign hits by RFM and AIC Score
data['TotalPurchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']
purchase_responsiveness = data.groupby(['RFM_Score', 'AIC_Score']).agg({
    'AcceptedCmp1': 'sum', 'AcceptedCmp2': 'sum', 'AcceptedCmp3': 'sum', 'AcceptedCmp4': 'sum', 'AcceptedCmp5': 'sum', 'Response6': 'sum',
}).reset_index()

# Create a long-form DataFrame for 3D plotting
long_df = pd.melt(purchase_responsiveness, id_vars=['RFM_Score', 'AIC_Score'], value_vars=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response6'],
                   var_name='Campaign', value_name='Campaign_Acceptance')

# Add a numeric campaign number
long_df['Campaign_Number'] = long_df['Campaign'].str.extract('(\d)').astype(int)

# Create a mapping of RFM_Score and AIC_Score to numerical values
rfm_mapping = {rfm: idx for idx, rfm in enumerate(sorted(long_df['RFM_Score'].unique()), 1)}
aic_mapping = {aic: idx for idx, aic in enumerate(sorted(long_df['AIC_Score'].unique()), 1)}

# Map the scores to numerical values
long_df['RFM_Score_Num'] = long_df['RFM_Score'].map(rfm_mapping)
long_df['AIC_Score_Num'] = long_df['AIC_Score'].map(aic_mapping)

# Plotting 3D heatmap with Plotly
fig = px.scatter_3d(long_df, x='RFM_Score_Num', y='AIC_Score_Num', z='Campaign_Number', color='Campaign_Acceptance',
                    labels={'RFM_Score_Num': 'RFM Score', 'AIC_Score_Num': 'AIC Score', 'Campaign_Number': 'Campaign Number', 'Campaign_Acceptance': 'Campaign Acceptance'},
                    title='3D Heatmap of Campaign Number by RFM and AIC Scores with Campaign Acceptance')

# Update the color scale to be green for the most responsive and navy blue for the least responsive
fig.update_traces(marker=dict(colorscale='Viridis', colorbar=dict(title='Campaign Acceptance')))

fig.update_layout(scene=dict(
    xaxis_title='RFM Score',
    yaxis_title='AIC Score',
    zaxis_title='Campaign Number'
))

# Update x and y axis tick labels to show original RFM and AIC scores
fig.update_layout(scene=dict(
    xaxis=dict(tickmode='array', tickvals=list(rfm_mapping.values()), ticktext=list(rfm_mapping.keys())),
    yaxis=dict(tickmode='array', tickvals=list(aic_mapping.values()), ticktext=list(aic_mapping.keys()))
))

fig.update_scenes(zaxis_tickvals=[1, 2, 3, 4, 5, 6])

fig.show()
