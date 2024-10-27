import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
fruits = pd.read_table('fruit_data_with_colors.txt')

# Data preparation
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Logistic Regression Accuracy: {:.2f}'.format(logreg.score(X_test, y_test)))

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('Decision Tree Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('K-NN Accuracy: {:.2f}'.format(knn.score(X_test, y_test)))

# Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
print('SVM Accuracy: {:.2f}'.format(svm.score(X_test, y_test)))

# Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('LDA Accuracy: {:.2f}'.format(lda.score(X_test, y_test)))

# Gaussian Naive Bayes (GNB)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Gaussian Naive Bayes Accuracy: {:.2f}'.format(gnb.score(X_test, y_test)))

# Optional: Generate a classification report and confusion matrix for one of the models
pred = knn.predict(X_test)
print("\nConfusion Matrix for K-NN:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report for K-NN:")
print(classification_report(y_test, pred))