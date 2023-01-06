import random
import numpy as np
import math
import matplotlib.pyplot as plt
import utils

label0=utils.dataset(10000, 0, 0, 1)
data_x, data_y = label0.get()

label1=utils.dataset(10000, 0, 3, 1)
data_x1, data_y1 = label1.get()

#print(data_x, data_y)

label0_with_noise=utils.dataset(10000, 0, 0, 1)
data_x_noise, data_y_noise = label0_with_noise.get(add_noise=True)

label1_with_noise=utils.dataset(10000, 0, 3, 1)
data_x1_noise, data_y1_noise = label1_with_noise.get(add_noise=True)

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
x = np.linspace(-5, 5)#where it is the range on the x axis
y = -(weights[0]*x + bias)/weights[1]#there is a negative sign in front of weights[0] because the weights are calculated for the equation y = -w1x1 - w2x2 - b
plt.plot(x, y, c='g')
plt.scatter(data_x, data_y, c='r')
plt.scatter(data_x1, data_y1, c='b')
plt.show()

x_noise=[]
for i in range(len(data_x_noise)):
    x_noise.append((data_x_noise[i], data_y_noise[i]))
    
for i in range(len(data_x1_noise)):
    x_noise.append((data_x1_noise[i], data_y1_noise[i]))

y_noise=[]
for i in range(len(data_x_noise)):
    y_noise.append(-1)

for i in range(len(data_x1_noise)):
    y_noise.append(1)

classifier = utils.Perceptron_Training_Classifier(0.01, 1000)
weights, bias = classifier.fit(x_noise, y_noise)
'''print("weights: ", weights)
print("bias: ", bias)
print("accuracy: ", classifier.score(x_noise, y_noise))'''

#plotting the decision boundary with noise
x = np.linspace(-5,5)
y = -(weights[0]*x + bias)/weights[1]
plt.plot(x, y, c='g')
plt.scatter(data_x_noise, data_y_noise, c='r')
plt.scatter(data_x1_noise, data_y1_noise, c='b')
plt.show()