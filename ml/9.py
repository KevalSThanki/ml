# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:30:36 2019

@author: students
"""
  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read dataset to pandas dataframe
dataset = pd.read_csv("9pg.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
plt.plot(X_train,y_train,'b.',X_test,y_test,'r.')

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

accuracy=classifier.score(X_test,y_test)
accuracy1=classifier.score(X_train,y_train)

print(accuracy)
print(accuracy1)

example=np.array([7.7,2.6,6.9,2.3])
example=example.reshape(1,-1)
print(example)

pred=classifier.predict(example)
print(pred)