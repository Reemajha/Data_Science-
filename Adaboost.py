# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Disease Prediction.csv")

data.head()
data.shape

map_diab={True:1, False:0}
data['diabetes']=data['diabetes'].map(map_diab)

data['diabetes'].value_counts()

x=data.drop('diabetes',axis=1)
y=data['diabetes']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=42)
 
from sklearn.ensemble import RandomForestClassifier
random_forest_model=RandomForestClassifier()

random_forest_model.fit(x_train, y_train)
predict_train_data=random_forest_model.predict(x_test)

from sklearn import metrics

print("Accuracy={0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

from sklearn.ensemble import AdaBoostClassifier
adaboost_model=AdaBoostClassifier()

adaboost_model.fit(x_train, y_train)
predict_train_data = adaboost_model.predict(x_test)

from sklearn import metrics

print("Accuracy={0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data))) 

##Hyperparameter Optimization







