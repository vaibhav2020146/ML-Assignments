import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt

data=pd.read_csv('C://Users//91991//Desktop//ML//Real estate.csv')
x0=data['X1 transaction date']
x1=data['X2 house age']
x2=data['X3 distance to the nearest MRT station']
x3=data['X4 number of convenience stores']
x4=data['X5 latitude']
x5=data['X6 longitude']
y=data['Y house price of unit area']
x0 = (x0 - x0.mean()) / x0.std()#normalizing the data
x1 = (x1 - x1.mean()) / x1.std()
x2 = (x2 - x2.mean()) / x2.std()
x3 = (x3 - x3.mean()) / x3.std()
x4 = (x4 - x4.mean()) / x4.std()
x5 = (x5 - x5.mean()) / x5.std()
x=np.c_[np.ones(x0.shape[0]),x0,x1,x2,x3,x4,x5]
alpha = 0.1   #learning rate
m = y.size  #no. of samples
theta = np.random.rand(7)


def plot_cost_for_ridge(x_test, y_test, m, theta, alpha,lamda):
    cost_list=[]
    for i in range(1000):
        prediction = np.dot(x_test, theta)
        error = prediction - y_test
        ridge_regression_term=(lamda/2*m)*np.sum(np.dot(theta,theta.T))#ridge regression term
        cost = 1/(2*m) * np.dot(error.T, error)+ ridge_regression_term #extra term added for ridge regression
        cost_list.append(mt.sqrt(cost))#taking the square root of the cost
        theta = theta - (alpha * (1/m) * (np.dot(x_test.T, error)+ lamda*theta)) #updating the theta values
    plt.plot(cost_list)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()

def plot_cost_for_lasso(x_test, y_test, m, theta, alpha,lamda):
    cost_list=[]
    for i in range(1000):
        prediction = np.dot(x_test, theta)
        error = prediction - y_test
        lasso_regression_term=(lamda/2*m)*np.sum(np.absolute(theta))#lasso regression term
        cost = 1/(2*m) * np.dot(error.T, error)+ lasso_regression_term #extra term added for lasso regression
        cost_list.append(mt.sqrt(cost))
        theta = theta - (alpha * (1/m) * (np.dot(x_test.T, error)+ lamda*theta))#updating the theta values
    plt.plot(cost_list)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()


def KFold(k,x,y,lamda):
    fold_size = len(x)//k
    scores=[]
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
        
        #uncomment the type of regression you want to use along with the plot_cost function
        #theta_list_train = ridge_regression(x_train, y_train, m, theta, alpha,1)#training the model
        theta_list_train = lasso_regression(x_train, y_train, m, theta, alpha,0.01)#training the model
        theta_check=theta_list_train[-1]#taking the last theta values which are the most updated ones
        #common for both ridge and lasso
        prediction = np.dot(x_test, theta_check)
        error = prediction - y_test
        '''ridge_regression_term=(lamda/2*m)*np.sum(np.dot(theta,theta.T))
        cost = 1/(2*m) * np.dot(error.T, error)+ ridge_regression_term  
        scores.append(mt.sqrt(cost))'''

        
        lasso_regression_term=(lamda/2*m)*np.sum(np.absolute(theta))
        cost = 1/(2*m) * np.dot(error.T, error)+ lasso_regression_term
        scores.append(mt.sqrt(cost))
        #plot_cost_for_ridge(x_test, y_test, m, theta_check, alpha,1)
        plot_cost_for_lasso(x_train, y_train, m, theta_check, alpha,0.01)
    return scores


def ridge_regression(x, y, m, theta, alpha,lamda):
    theta_list = [] 
    for i in range(1000):
        prediction = np.dot(x, theta)
        error = prediction - y
        ridge_regression_term=(lamda/2*m)*np.sum(np.dot(theta,theta.T))#ridge regression term
        cost = 1/(2*m) * np.dot(error.T, error)+ ridge_regression_term  #extra term added due to regularisation
        theta = theta - (alpha * (1/m) * (np.dot(x.T, error)+ lamda*theta)) 
        theta_list.append(theta)
    return theta_list


def lasso_regression(x, y, m, theta, alpha,lamda):
    theta_list = [] 
    for i in range(1000):
        prediction = np.dot(x, theta)
        error = prediction - y
        lasso_regression_term=(lamda/2*m)*np.sum(np.absolute(theta))#lasso regression term
        cost = 1/(2*m) * np.dot(error.T, error)+ lasso_regression_term #extra term added due to regularisation
        theta = theta - (alpha * (1/m) * (np.dot(x.T, error)+ lamda*theta)) 
        theta_list.append(theta)
    return theta_list


scores=KFold(2,x,y,0.01)
#print(sum(scores)/len(scores))