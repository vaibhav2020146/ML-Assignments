import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def score_calculation(y_true,y_pred):
    val=np.sum(y_true==y_pred)/len(y_true)
    return val
    '''val=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            val+=1
    val=val/len(y_true)
    return val'''


data=pd.read_csv('C://Users//91991//Desktop//ML//BitcoinHeistData.csv')
data=data.drop(['address'],axis=1)
#randomising the data
data_copy=data.copy()
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
test_output=y[-test_split:]#-test_split means from the end of the dataset
validation_input=x[train_split:-test_split]
validation_output=y[train_split:-test_split]

def split(dataset):
    #randomising the data
    dataset=dataset.sample(frac=1)
    x_help=dataset.drop(['label'],axis=1)
    y_help=dataset['label']
    m=x_help.shape[0]
    #randomly choosing the required dataset
    train_split=int(0.5*m)
    train_x=x_help[:train_split]
    train_y=y_help[:train_split]
    return train_x,train_y

prediction=[]
number_of_trees=100
#ensembling the data
for i in range(number_of_trees):
    x_new,y_new=split(data_copy)
    #criteria here used is entropy because as per previous parts it was found that entropy gives better results
    model=DecisionTreeClassifier(criterion='entropy',max_depth=3)
    model.fit(x_new,y_new)
    output=model.predict(train_input)
    prediction.append(output)

#prediction is a list of lists
prediction=np.array(prediction)
prediction=prediction.T
prediction=prediction.tolist()
#chossing the majority vote
prediction=[max(i,key=i.count) for i in prediction]
print('Random Forest model score is',score_calculation(train_output,prediction))