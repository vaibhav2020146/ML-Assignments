import utils
import numpy as np
import matplotlib.pyplot as plt


label0=utils.dataset(10000, 0, 0, 1)
data_x, data_y = label0.get()

label1=utils.dataset(10000, 0, 3, 1)
data_x1, data_y1 = label1.get()

#print(data_x, data_y)

label0_with_noise=utils.dataset(10000, 0, 0, 1)
data_x_noise, data_y_noise = label0_with_noise.get(add_noise=True)

label1_with_noise=utils.dataset(10000, 0, 3, 1)
data_x1_noise, data_y1_noise = label1_with_noise.get(add_noise=True)

#plotting the datset
plt.scatter(data_x, data_y, c='r', label='label 0')
plt.scatter(data_x1, data_y1, c='b', label='label 1')
plt.show()

#plotting the datset with noise
plt.scatter(data_x_noise, data_y_noise, c='r', label='label 0')
plt.scatter(data_x1_noise, data_y1_noise, c='b', label='label 1')
plt.show() 