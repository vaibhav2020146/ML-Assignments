import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')
d=data.drop(['Class'],axis=1)
change_data=StandardScaler().fit_transform(d)
data_test=change_data
labels_test=data['Class']
#implementing PCA
pca=PCA(n_components=12)
pca_data=pca.fit_transform(data_test)
#storing the data in a new dataframe
pca_data=np.vstack((pca_data.T,labels_test)).T
#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(pca_data[:,:-1],pca_data[:,-1],test_size=0.2,random_state=0)
#implementing logistic regression
model=LogisticRegression(random_state=0,max_iter=1000)
model.fit(train_x,train_y)
#predicting the test data
predicted=model.predict(test_x)
print('Accuracy of Logistic Regression is:',model.score(test_x,test_y))
#calculating f1 score
print('F1 score of Logistic Regression is:',f1_score(test_y,predicted,average='weighted'))
#calculating precision
print('Precision of Logistic Regression is:',precision_score(test_y,predicted,average='weighted'))
#calculating recall
print('Recall of Logistic Regression is:',recall_score(test_y,predicted,average='weighted'))