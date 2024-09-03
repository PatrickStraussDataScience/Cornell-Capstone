import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the DataFrame
df_xgb = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v9.csv')

# Retain the user IDs
user_ids = df_xgb['index']

scaler = StandardScaler(with_mean=True, with_std=True)

cols = ['Age', 'Income',
        'Recency', 'NumWebVisitsMonth', 'Customer_Days',
        'MntTotal', 'MntRegularProds', 'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
       ]

df_xgb[cols] = pd.DataFrame(scaler.fit_transform(df_xgb[cols]), columns=df_xgb[cols].columns)

# XGBoost
# Isolate the outcome variable
y = df_xgb['Response']

# Select the features to use in the model
X = df_xgb.drop(['Response', 'index'], axis=1)  # Drop 'ID' from features

# Split the data into training set and testing set
X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
    X, y, user_ids, test_size=0.25, stratify=y, random_state=42)

# Construct an XGBoost model and fit it to the training dataset
xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=10, learning_rate=0.1)
xgb_clf = xgb_model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(xgb_clf, r'C:\Users\Patrick\downloads\xgboost_model.pkl')
joblib.dump(scaler, r'C:\Users\Patrick\downloads\scaler_xgb.pkl')

# Use the XGBoost model to get predictions on the test set
y_pred_xgb = xgb_clf.predict(X_test)

print('XGBoost Results on Test Data')
print("Accuracy:", "%.6f" % accuracy_score(y_test, y_pred_xgb))
print("Precision:", "%.6f" % precision_score(y_test, y_pred_xgb))
print("Recall:", "%.6f" % recall_score(y_test, y_pred_xgb))
print("F1 Score:", "%.6f" % f1_score(y_test, y_pred_xgb))

# Identify user IDs predicted to be "1"
predicted_1_user_ids_xgb = user_ids_test[y_pred_xgb == 1]
print("Rows where customer is predicted to accept the campaign:")
print(predicted_1_user_ids_xgb.tolist())
