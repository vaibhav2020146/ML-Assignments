import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
import math as mt
#inputing the data from the file
data=pd.read_csv('C://Users//91991//Desktop//ML//Real estate.csv')#set the location of the data
x0=data['X1 transaction date']
x1=data['X2 house age']
x2=data['X3 distance to the nearest MRT station']
x3=data['X4 number of convenience stores']
x4=data['X5 latitude']
x5=data['X6 longitude']
y=data['Y house price of unit area']

#normalizing the data
x0 = (x0 - x0.mean()) / x0.std()
x1 = (x1 - x1.mean()) / x1.std()
x2 = (x2 - x2.mean()) / x2.std()
x3 = (x3 - x3.mean()) / x3.std()
x4 = (x4 - x4.mean()) / x4.std()
x5 = (x5 - x5.mean()) / x5.std()

x=np.c_[np.ones(x0.shape[0]),x0,x1,x2,x3,x4,x5]
alpha = 0.1#learning rate
m = y.size  #no. of samples
theta = np.random.rand(7)#adding some random value in the theta list which will be updated later

def KFold(k,x,y):
    scores = []#for storing rsme error values
    fold_size = len(x)//k
    #creating the folds as required
    for i in range(k):
        x_test = x[i*fold_size:(i+1)*fold_size]
        y_test = y[i*fold_size:(i+1)*fold_size]
        
        if i>0:
            if i==k-1:
                x_train=x[:i*fold_size]
                y_train=y[:i*fold_size]

            else:
                p = x[(i-1)*fold_size:i*fold_size]
                n = x[(i+1)*fold_size:]
                x_train=np.concatenate((p,n))
                
                p = y[(i-1)*fold_size:i*fold_size]
                n = y[(i+1)*fold_size:]
                y_train=np.concatenate((p,n))
            
        else:
            x_train = x[(i+1)*fold_size:]
            y_train = y[(i+1)*fold_size:]
        
        #sending the fold for training of the model
        theta_list_train = gradient_descent(x_train, y_train, m, theta, alpha)#training the model
        theta_check=theta_list_train[-1]#taking the last theta values which are the most updated ones
        #print(theta_check)
        #computing the value that we get after training the model on the testing data
        prediction = np.dot(x_test, theta_check)
        #computing the error between the predicted and actual value
        error = prediction - y_test
        score=mt.sqrt(1/(2*m) * np.dot(error.T, error))#1/2m * sum of (y-y')^2
        #storing the error values for each fold
        scores.append(score)
    #print(scores)
    #returning the mean of the error values
    return (sum(scores)/len(scores))


def gradient_descent(x, y, m, theta, alpha):
    theta_list = []  #to store all values of theta/weights that are updated  
    for i in range(1000):
        prediction = np.dot(x, theta)   #predicted the y value
        error = prediction - y #error between predicted and actual value
        theta = theta - (alpha * (1/(2*m)) * np.dot(x.T, error))  #updating the theta values
        theta_list.append(theta)#adding the new theta values in the theta list
    return theta_list


mean=KFold(5,x,y)#can change the value of k here and find for different values
print(mean)