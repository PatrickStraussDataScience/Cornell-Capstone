import numpy as np
import pandas as pd
import datetime

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data modeling
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from xgboost import XGBClassifier
#from xgboost import XGBRegressor
#from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

df = pd.read_csv(r'C:\Users\Patrick\downloads\marketing_campaign_YUH.csv')

# Update Missing Incomes
df.loc[((df['Income'].isna()) & (df['Education'] == 'Basic')),'Income'] = df[df['Education'] == 'Basic']['Income'].median()
df.loc[((df['Income'].isna()) & (df['Education'] == 'Graduation')),'Income'] = df[df['Education'] == 'Graduation']['Income'].median()
df.loc[((df['Income'].isna()) & (df['Education'] == '2n Cycle')),'Income'] = df[df['Education'] == '2n Cycle']['Income'].median()
df.loc[((df['Income'].isna()) & (df['Education'] == 'Master')),'Income'] = df[df['Education'] == 'Master']['Income'].median()
df.loc[((df['Income'].isna()) & (df['Education'] == 'PhD')),'Income'] = df[df['Education'] == 'PhD']['Income'].median()

df = df.rename(columns={'Education': 'education',
                         'Marital_Status': 'marital',
                         })

df = pd.get_dummies(df, drop_first = False, columns=['education', 'marital'])


# Change date customer joined to customer days since joined
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Customer_Days'] = (datetime.datetime.now() - df['Dt_Customer'])
df['Customer_Days'] = df['Customer_Days'].dt.days

df['Customer_Days'].describe()

# Add Age
df['Age'] = datetime.datetime.now().year - df['Year_Birth']

# Max age = 100
df.loc[df['Age'] > 100, 'Age'] = 100

# All products purchased
df['MntTotal'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

# Excluding Gold
df['MntRegularProds'] =  df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts']

df ['AcceptedCmpOverall'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']

# drop unnessary cols
df = df.drop(['Year_Birth', 'Z_CostContact', 'Z_Revenue'], axis=1)

#drop duplicates
df=df.drop_duplicates()

#reindex columns
df = df.reindex(columns=['ID', 'Age','education_2n Cycle', 'education_Basic', 'education_Graduation', 'education_Master',
                         'education_PhD', 'Income','marital_Single', 'marital_Married', 'marital_Together', 'marital_Divorced', 'marital_Widow',
                         'Kidhome', 'Teenhome', 'Recency', 'NumWebVisitsMonth', 'Customer_Days', 'Complain',
                         'MntTotal', 'MntRegularProds','MntWines', 'MntFruits','MntMeatProducts',
                         'MntFishProducts', 'MntSweetProducts','MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumCatalogPurchases', 'NumStorePurchases', 'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                         'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmpOverall','Response'
                        ]
               )


# Total Purchases
df['NumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']


# Average amount per purchase
df['AvgMntPerPurchase'] = df['MntTotal']/df['NumPurchases']
# Change infinit n. to 0
df.loc[df['AvgMntPerPurchase']==np.inf, 'AvgMntPerPurchase'] = 0


# Average amount per regular purchase
df['AvgRegMntPerPurchase'] = df['MntRegularProds']/df['NumPurchases']
df.loc[df['AvgRegMntPerPurchase']==np.inf, 'AvgRegMntPerPurchase'] = 0

# The influence of deals on number of purchases
df['DealPerPurchase'] = df['NumDealsPurchases']/df['NumPurchases']
df.loc[df['DealPerPurchase']==np.inf, 'DealPerPurchase'] = 0
df.loc[df['DealPerPurchase'].isna(), 'DealPerPurchase'] = 0

# outlier we will set to 1
df.loc[df['DealPerPurchase'] > 1, 'DealPerPurchase'] = 1
df[df['DealPerPurchase'] > 1].head()

# Drop col added that didn't seem promising
df = df.drop(['NumPurchases', 'AvgRegMntPerPurchase'], axis=1)

# Copy the df
df_lr = df.copy()
df_lr = df_lr.reset_index()

# Drop highy correlated independent variables
df_lr = df_lr.drop([ 'DealPerPurchase','AvgMntPerPurchase',
                    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                    'AcceptedCmp5'
                   ], axis=1)

df_lr.to_csv(r'C:\Users\Patrick\downloads\marketing_campaign_v9.csv')

print('Done!')






