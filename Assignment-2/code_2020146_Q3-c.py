import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def score_calculation(y_true,y_pred):
    val=np.sum(y_true==y_pred)/len(y_true)
    return val
    
data=pd.read_csv('C://Users//91991//Desktop//ML//BitcoinHeistData.csv')
data=data.drop(['address'],axis=1)

#randomising the data
data=data.sample(frac=1)

x=data.drop(['label'],axis=1)
y=data['label']
#classifying data in training, testing and validation
m=x.shape[0]
train_split=int(0.7*m)
cut=int((m-train_split)/2)
test_split=m-train_split-cut

train_input=x[:train_split]
train_output=y[:train_split]
test_input=x[-test_split:]
test_output=y[-test_split:]
validation_input=x[train_split:-test_split]
validation_output=y[train_split:-test_split]


estimators=[4,8,10,15,20]
for i in estimators:
    abc_model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),n_estimators=i)
    abc_model.fit(train_input,train_output)
    abc_prediction=abc_model.predict(test_input)
    print('AdaBoost model score for estimators',i,'is',score_calculation(test_output,abc_prediction))

    rfc_model=RandomForestClassifier(n_estimators=i)
    rfc_model.fit(train_input,train_output)
    rfc_prediction=rfc_model.predict(test_input)
    print('RandomForest model score for estimators',i,'is',score_calculation(test_output,rfc_prediction))