import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gzip
import idx2numpy
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

#one hot encoding the labels
train_labels=tf.keras.utils.to_categorical(training_label)
test_labels=tf.keras.utils.to_categorical(validation_label)

#on changing number of neurons in each layer, we observe that the accuracy increases with increase in number of neurons

model=MLPClassifier(hidden_layer_sizes=(128,8),activation='relu',max_iter=50,batch_size=128,validation_fraction=0.15,early_stopping=True)
model.fit(train_images,train_labels)
training_loss=model.loss_curve_
validation_score=model.validation_scores_
#as the validation score is not available in the model, we are calculating it manually

#calculating accuracy:
accuracy=model.score(test_images,test_labels)
print(accuracy)
plt.plot(training_loss)
plt.title('Loss Curve for Relu activation function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()