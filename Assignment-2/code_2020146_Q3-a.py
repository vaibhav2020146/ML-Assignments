import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

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

#print(train_input)
#print(train_output)

#training the model
depths=[4,8,10,15,20]
entropy_testing=[]
entropy_validation=[]
gini_testing=[]
gini_validation=[]
for i in depths:
    entropy_model=DecisionTreeClassifier(criterion='entropy',max_depth=i)
    entropy_model.fit(train_input,train_output)
    ginni_model=DecisionTreeClassifier(criterion='gini',max_depth=i)
    ginni_model.fit(train_input,train_output)
    entropy_prediction_test=entropy_model.predict(test_input)
    gini_prediction_test=ginni_model.predict(test_input)
    entropy_prediction_validation=entropy_model.predict(validation_input)
    gini_prediction_validation=ginni_model.predict(validation_input)
    print('Entropy model score for depth',i, 'on testing data is :',score_calculation(test_output,entropy_prediction_test))
    entropy_testing.append(score_calculation(test_output,entropy_prediction_test))
    print('Gini model score for depth',i, 'on testing data is :',score_calculation(test_output,gini_prediction_test))
    gini_testing.append(score_calculation(test_output,gini_prediction_test))
    print('Entropy model score for depth',i, 'on validation data is :',score_calculation(validation_output,entropy_prediction_validation))
    entropy_validation.append(score_calculation(validation_output,entropy_prediction_validation))
    print('Gini model score for depth',i, 'on validation data is :',score_calculation(validation_output,gini_prediction_validation))
    gini_validation.append(score_calculation(validation_output,gini_prediction_validation))

print('Average Entropy testing scores is:',sum(entropy_testing)/len(entropy_testing))
print('Average Gini testing scores is:',sum(gini_testing)/len(gini_testing))
print('Average Entropy validation scores is:',sum(entropy_validation)/len(entropy_validation))
print('Average Gini validation scores is:',sum(gini_validation)/len(gini_validation))

#choose the model with best accuracy on test set