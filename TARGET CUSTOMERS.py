import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# Load the new data
new_data = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v9.csv')

# Retain the user IDs
new_user_ids = new_data['ID']

# Load the scaler and the model from files
scaler = joblib.load(r'C:\Users\Patrick\downloads\scaler3.pkl')
log_model = joblib.load(r'C:\Users\Patrick\downloads\logistic_regression_model3.pkl')

# Scale the new data using the loaded scaler
cols = ['index', 'ID', 'Age', 'Income',
        'Recency', 'NumWebVisitsMonth', 'Customer_Days',
        'MntTotal', 'MntRegularProds', 'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
       ]

new_data[cols] = pd.DataFrame(scaler.transform(new_data[cols]), columns=new_data[cols].columns)

# Select the features to use in model
X_new = new_data.drop(['Response', 'index'], axis=1)  # Drop 'Response' and 'ID' from features

# Use the logistic regression model to get predictions on the new data
new_predictions = log_model.predict(X_new)

# Identify user IDs predicted to be "1"
predicted_1_new_user_ids = new_user_ids[new_predictions == 1]
print("New User IDs who should be targeted with Cmp6:")
print(predicted_1_new_user_ids.tolist())
