import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')
#data=data.drop(['Class'],axis=1)#try without dropping it also
x=data.drop(['Class'],axis=1)
y=data['Class']
#normalizing the data
x=(x-x.mean())/x.std()
#classifying the data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Gaussion Naive Bayes used to calculate the ROC curve
gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
Gnb_score=gaussian.predict_proba(x_test)
#roc_curve is used to calculate the false positive rate and true positive rate
false_positivity_rate, true_positivity_rate,thresholds= roc_curve(y_test, Gnb_score[:,1], pos_label='SEKER')
plt.plot([0,1],[0,1],'k--') #plot the diagonal line
plt.plot(false_positivity_rate, true_positivity_rate, label='NB') #plot the ROC curve
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Curve Naive Bayes')
plt.show()