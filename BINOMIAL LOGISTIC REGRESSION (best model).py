import numpy as np
import pandas as pd
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import seaborn as sns

df_lr = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v9.csv')

scaler = StandardScaler(with_mean=True, with_std=True)

cols = ['Age','Income',
        'Recency', 'NumWebVisitsMonth', 'Customer_Days',
        'MntTotal', 'MntRegularProds','MntWines', 'MntFruits','MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts','MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases'
       ]

df_lr[cols] = pd.DataFrame(scaler.fit_transform(df_lr[cols]), columns=df_lr[cols].columns)

# Logistic Regression
# Isolate the outcome variable
y = df_lr['Response']

# Select the features to use in model
X = df_lr.drop('Response', axis=1)

# Split the data into training set and testing set
X_train, X_lr_test, y_train, y_lr_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Construct a logistic regression model and fit it to the training dataset
log_model = LogisticRegression(random_state=42, max_iter=2000)
log_clf = log_model.fit(X_train, y_train)

# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_lr_test)

print('Logistic Regression Results on Test Data')
print("Accuracy:", "%.6f" % accuracy_score(y_lr_test, y_pred))
print("Precision:", "%.6f" % precision_score(y_lr_test, y_pred))
print("Recall:", "%.6f" % recall_score(y_lr_test, y_pred))
print("F1 Score:", "%.6f" % f1_score(y_lr_test, y_pred))

# Print the confusion matrix
cm = confusion_matrix(y_lr_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
