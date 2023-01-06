import numpy as np
import matplotlib.pyplot as plt
import gzip
import idx2numpy
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

image=gzip.open('C://Users//91991//Desktop//ML//data//fashion//train-images-idx3-ubyte.gz','r')
label=gzip.open('C://Users//91991//Desktop//ML//data//fashion//train-labels-idx1-ubyte.gz','r')

#converting them to numpy:
dataset_image=idx2numpy.convert_from_file(image)
dataset_label=idx2numpy.convert_from_file(label)

#reshaping the image data:
dataset_image=dataset_image.reshape(dataset_image.shape[0],dataset_image.shape[1]*dataset_image.shape[2])

#reshaping the label data:
dataset_label=dataset_label.reshape(dataset_label.shape[0],1)

#dividing the dataset into training and validation in ratio 85:15
training_image=dataset_image[:int(0.85*dataset_image.shape[0]),:]
training_label=dataset_label[:int(0.85*dataset_label.shape[0]),:]
validation_image=dataset_image[int(0.85*dataset_image.shape[0]):,:]
validation_label=dataset_label[int(0.85*dataset_label.shape[0]):,:]

#normalizing the dataset
train_images=training_image/255.0
test_images=validation_image/255.0

#converting 
#one hot encoding the labels it is used to convert the labels into binary form
#train_labels=tf.keras.utils.to_categorical(training_label)
#test_labels=tf.keras.utils.to_categorical(validation_label)

parameters = {'hidden_layer_sizes':np.arange(10,16),'activation':['softmax', 'tanh','relu','logistic'],'solver':['adam','sgd'],'alpha':[0.1,0.01,0.01],'max_iter':[80]}
grid = GridSearchCV(estimator=MLPClassifier(), param_grid = parameters, cv = 5, n_jobs=-1,scoring='accuracy')
grid.fit(train_images, training_label)

print(grid.score(test_images,validation_label))
print(grid.best_params_)