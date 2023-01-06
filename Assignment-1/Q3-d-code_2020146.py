import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')
#dropping the class feature as is a string and not a number
x=data.drop(['Class'],axis=1)
y=data['Class']
#normalizing the data
x=(x-x.mean())/x.std()
#classifying the data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#random_state=0 is used to get the same result everytime and test size is 0.2 as we need 80:20 training testing split

#Gaussion Naive Bayes
gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
print("Gausian Naive Bayes Accuracy: ",gaussian.score(x_test, y_test))
print("Gausian Naive Bayes recall: ",recall_score(y_test,gaussian.predict(x_test),average='weighted'))
print("Gaussian Naive Bayes precision: ",precision_score(y_test,gaussian.predict(x_test),average='weighted'))

print()

#Bernoulli Naive Bayes
bernoulli=BernoulliNB()
bernoulli.fit(x_train,y_train)
print("Bernoulli Naive Bayes Accuracy: ",bernoulli.score(x_test, y_test))
print("Bernoulli Naive Bayes recall: ",recall_score(y_test,bernoulli.predict(x_test),average='weighted'))
print("Bernoulli Naive Bayes precision: ",precision_score(y_test,bernoulli.predict(x_test),average='weighted'))
