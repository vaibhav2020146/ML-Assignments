import random
import numpy as np
import math
import matplotlib.pyplot as plt
import utils

label0=utils.dataset(10000, 0, 0, 1)
data_x, data_y = label0.get()

label1=utils.dataset(10000, 0, 3, 1)
data_x1, data_y1 = label1.get()

#training the classifier using PTA
x=[]
for i in range(len(data_x)):
    x.append((data_x[i], data_y[i]))

for i in range(len(data_x1)):
    x.append((data_x1[i], data_y1[i]))


y=[]
for i in range(len(data_x)):
    y.append(-1)
for i in range(len(data_x1)):
    y.append(1)


classifier = utils.Perceptron_Training_Classifier(0.01, 1000)
weights, bias = classifier.fit(x, y)
'''print("weights: ", weights)
print("bias: ", bias)
print("accuracy: ", classifier.score(x, y))'''


#plotting the decision boundary
x = np.linspace(-5,5)
y = -(weights[0]*x + bias)/weights[1]#there is a negative sign in front of weights[0] because the weights are calculated for the equation y = -w1x1 - w2x2 - b
plt.plot(x, y, c='g')
plt.scatter(data_x, data_y, c='r')
plt.scatter(data_x1, data_y1, c='b')
plt.show()

