import pandas as pd # For data cleaning
import numpy as np # Numerical Methods
from sklearn.metrics import accuracy_score # For Checking Accuracy
from sklearn.model_selection import train_test_split # Splitting Data For Train Test
from sklearn import tree # ML Algo Decision Tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier # ML Algo KNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
df = pd.read_csv('trainDF.csv')
df.head()

# Separating Target & Other Columns 
XTT = df.drop(columns=['f_00'])
yTT = df['f_00']  
X_train, X_test, y_train, y_test = train_test_split(XTT, yTT, test_size=0.2)

modelKNN = KNeighborsClassifier(n_neighbors=5)
resultKNN = modelKNN.fit(X_train, y_train)
prediction_test = modelKNN.predict(X_test)
accuracyKNN = metrics.accuracy_score(y_test, prediction_test)
print("Model Accuracy (KNN):" "\n", accuracyKNN)
