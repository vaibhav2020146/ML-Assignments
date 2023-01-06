from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')
x=data.drop(['Class'],axis=1)
y=data['Class']
#normalizing the data
x=(x-x.mean())/x.std()
#classifying the data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Logistic Regression
lr=LogisticRegression(max_iter=1000)#max_iter=1000 is used to get the same result everytime and test size is 0.2 as we need 80:20 training testing split
lr.fit(x_train,y_train)
lr_score=lr.predict_proba(x_test)
print("Logistic Regression Accuracy: ",lr.score(x_test, y_test))
print("Logistic Regression recall: ",recall_score(y_test,lr.predict(x_test),average='weighted'))
print("Logistic Regression precision: ",precision_score(y_test,lr.predict(x_test),average='weighted'))
print("Logistic Regression f1: ",f1_score(y_test,lr.predict(x_test),average='weighted'))
