import numpy as np
import streamlit as st

from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

st.title("Streamlit Example")

st.write('''
# Explore different classifiers
Select from the menu below:
''')

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset", "Breast Cancer Dataset", "Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))


def load_dataset(dataset_name):
    if dataset_name.lower() == "iris dataset":
        data = datasets.load_iris()
    elif dataset_name.lower() == "breast cancer dataset":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X, y = data.data, data.target
    return X, y


X, y = load_dataset(dataset_name)
st.write("Shape of the dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = {}
    if clf_name.lower() == "knn":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name.lower() == "svm":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("C", 2, 15)
        params["max_depth"] = max_depth

        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators

    return params


params = add_parameter_ui(classifier_name)


def load_classifier(clf_name, params):
    if clf_name.lower() == "knn":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name.lower() == "svm":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"],
                                     random_state=1234)

    return clf

clf = load_classifier(clf_name=classifier_name, params=params)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(x_test)

# Displaying the accuracy and the model details
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier Name: {classifier_name}")
st.write(f"Accuracy: {acc}")


# Plotting
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)