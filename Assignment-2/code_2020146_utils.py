from ftplib import error_temp
import random
import numpy as np
import math
import matplotlib.pyplot as plt
class dataset:
    def __init__(self, number_of_points, center_x, center_y, radius):
        self.number_of_points = number_of_points
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
    

    def get(self,add_noise=False):
        points_x=[]
        points_y=[]
        for i in range(self.number_of_points):
            alpha=2*math.pi*random.random()
            r=self.radius*math.sqrt(random.random())
            x=r*math.cos(alpha)+self.center_x
            y=r*math.sin(alpha)+self.center_y
            if add_noise:
                x += random.gauss(0, 0.1)
                y += random.gauss(0, 0.1)
            points_x.append(x)
            points_y.append(y)
        return (points_x, points_y)


class Perceptron_Training_Classifier:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sgn(self,x):
        if x>=0:
            return 1
        else:
            return -1

    def fit(self, X, y):
        #X is a list of points
        #y is a list of labels
        self.weights = [0,0]
        self.bias = 0
        n=len(X)
        for i in range(self.epochs):#which means till it converges
            for j in range(n):
                error=y[j]-self.sgn(self.weights[0]*X[j][0] + self.weights[1]*X[j][1] + self.bias)
                if error != 0:
                    #update weights and bias
                    self.weights[0] += error*X[j][0]
                    self.weights[1] += error*X[j][1]
                    self.bias += error
        return (self.weights, self.bias)

    def predict(self, X):
        #X is a list of points
        #returns the predicted labels
        y_pred = []
        for i in range(len(X)):
            error_term=self.weights[0]*X[i][0] + self.weights[1]*X[i][1] + self.bias
            if error_term>= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return y_pred

    def score(self, X, y):
        #X is a list of points
        #y is a list of labels
        #returns the accuracy
        y_pred = self.predict(X)
        val=np.sum(y==y_pred)/len(y)
        return val

