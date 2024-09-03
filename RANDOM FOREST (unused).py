import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the DataFrame
df_rf = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v9.csv')

# Retain the user IDs
user_ids = df_rf['index']

scaler = StandardScaler(with_mean=True, with_std=True)

cols = ['Age', 'Income',
        'Recency', 'NumWebVisitsMonth', 'Customer_Days',
        'MntTotal', 'MntRegularProds', 'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
       ]

df_rf[cols] = pd.DataFrame(scaler.fit_transform(df_rf[cols]), columns=df_rf[cols].columns)

# Random Forest
# Isolate the outcome variable
y = df_rf['Response']

# Select the features to use in the model
X = df_rf.drop(['Response', 'index'], axis=1)  # Drop 'ID' from features

# Split the data into training set and testing set
X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
    X, y, user_ids, test_size=0.25, stratify=y, random_state=42)

# Construct a Random Forest model and fit it to the training dataset
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_clf = rf_model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(rf_clf, r'C:\Users\Patrick\downloads\random_forest_model.pkl')
joblib.dump(scaler, r'C:\Users\Patrick\downloads\scaler_rf.pkl')

# Use the Random Forest model to get predictions on the test set
y_pred_rf = rf_clf.predict(X_test)

print('Random Forest Results on Test Data')
print("Accuracy:", "%.6f" % accuracy_score(y_test, y_pred_rf))
print("Precision:", "%.6f" % precision_score(y_test, y_pred_rf))
print("Recall:", "%.6f" % recall_score(y_test, y_pred_rf))
print("F1 Score:", "%.6f" % f1_score(y_test, y_pred_rf))

# Identify user IDs predicted to be "1"
predicted_1_user_ids_rf = user_ids_test[y_pred_rf == 1]
print("Rows where customer is predicted to accept the campaign:")
print(predicted_1_user_ids_rf.tolist())
