import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

###
#st.title("Custom Model Deployment")
###

###
#st.title("Reading .csv")
data = pd.read_csv("churn_new.csv")
#st.table(data.head())
###

###
#st.title("Transpose Table")
#st.table(data.describe().transpose())
###

###
#st.title("N/A")
#st.write(data.isna().sum())
###

###
#st.title("Total Duplicated")
#st.text(data.duplicated().sum())
###

###
zeros = [x for x in data.Churn if x == 0]
ones = [x for x in data.Churn if x == 1]

plt.bar(0, len(zeros), width=0.333)
plt.bar(1, len(ones), width=0.333)
plt.show()
###

# ### Takes too much to load
# st.title("Pairplot")
sns.pairplot(data, hue="Churn", diag_kws={'bw': 0.5})
# ###

###
# st.title("")
import joblib

data = data.fillna(data.median())
scaler = joblib.load("scaler.pkl")

X, y = data.drop("Churn", axis=1), data["Churn"]
X_scaled = scaler.transform(X)
###

###
model = joblib.load("xgb_final.pkl")
print(model)
###

# ###
# from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, r2_score
#
# pred = model.predict(X_scaled)
#
# accuracy_xgb = accuracy_score(y, pred)
# precision_xgb = precision_score(y, pred)
# recall_xgb = recall_score(y, pred)
# f1_score_xgb = f1_score(y, pred)
# print(f"Accuracy: {accuracy_xgb}")
# print(f"Precison: {precision_xgb}")
# print(f"Recall: {recall_xgb}")
# print(f"F1 Score: {f1_score_xgb}")
# ###