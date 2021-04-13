import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

###
st.title("Custom Model Deployment")
###

###
st.title("Reading .csv")
data = pd.read_csv("churn_new.csv")
st.table(data.head())
###

###
st.title("Transpose Table")
st.table(data.describe().transpose())
###

###
st.title("N/A")
st.write(data.isna().sum())
###

###
st.title("Total Duplicated")
st.text(data.duplicated().sum())
###

###
zeros = [x for x in data.Churn if x == 0]
ones = [x for x in data.Churn if x == 1]

plt.bar(0, len(zeros), width=0.333)
plt.bar(1, len(ones), width=0.333)
st.pyplot(plt.show())
###

# ### Takes too much to load
# st.title("Pairplot")
# st.pyplot(sns.pairplot(data, hue="Churn", diag_kws={'bw': 0.5}))
# ###

###
st.title("Model")
import joblib

data = data.fillna(data.median())
scaler = joblib.load("scaler.pkl")

X, y = data.drop("Churn", axis=1), data["Churn"]
X_scaled = scaler.transform(X)

model = joblib.load("xgb_final.pkl")
st.write(model)
###

###
st.title("Model Details")

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, r2_score

pred = model.predict(X_scaled)

accuracy_xgb = accuracy_score(y, pred)
precision_xgb = precision_score(y, pred)
recall_xgb = recall_score(y, pred)
f1_score_xgb = f1_score(y, pred)
st.write(f"Accuracy: {accuracy_xgb}")
st.write(f"Precison: {precision_xgb}")
st.write(f"Recall: {recall_xgb}")
st.write(f"F1 Score: {f1_score_xgb}")
###

###
from sklearn.metrics import roc_curve, auc

y_pred_prop = model.predict_proba(X_scaled)[:, 1]

fpr_xgb, tpr_xgb, _ = roc_curve(y, y_pred_prop)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(14, 10))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange',
         label='ROC curve (area = %0.2f)' % roc_auc_xgb)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver Operating Characteristic', fontsize=22).set_position([.5, 1.02])
plt.legend(loc="lower right", fontsize=13)
st.pyplot(plt.show())
###

###
st.title("Confusion Matrix")
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(14, 10))

cm = confusion_matrix(y, pred)

ax = sns.heatmap(cm, square=True, annot=True, cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('True Labels', fontsize=15)
ax.set_title('Confusion Matrix', fontsize=16)
ax.xaxis.set_ticklabels(['Not Churn', 'Churn'], fontsize=12)
ax.yaxis.set_ticklabels(['Not Churn', 'Churn'], fontsize=12)
st.pyplot(plt.show())
###

### Opens in a new tab
st.title("Feature / Importance")

import plotly.express as px

fi_dt = pd.DataFrame({'Feature': X.columns,
                      'Importance': model.feature_importances_}).sort_values(by="Importance",
                                                                             ascending=True).reset_index(drop=True)

fig = px.bar(fi_dt, x='Importance', y='Feature', orientation='h', color='Importance')
st.plotly_chart(fig)
###

###
st.title("Shap Value")

import shap

shap_values = shap.TreeExplainer(model).shap_values(X)
st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar"))
###

###
st.title("Shap Summary Plot")

f = plt.figure()
st.pyplot(shap.summary_plot(shap_values, X))
###

###
import shap
import streamlit as st
import streamlit.components.v1 as components


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


st.title("SHAP in Streamlit")
st.markdown("`https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d`")

X_new = pd.DataFrame(X_scaled, columns=X.columns)
X_output = X_new.copy()

shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :]))
st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
###

###
st.title("User Interacted Output")

x_input = st.slider('Input X Value: ', 0, 99, 20)  # 20 is default value selected automatically to be changed

if model.predict(np.expand_dims(X_output.loc[x_input, :], axis=0)):
    st.markdown("`1 - YES`")
else:
    st.write("`0 - NO`")
###
