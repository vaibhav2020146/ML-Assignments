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

#defining the model as MLPClassifier
various_learning_rates=[0.1, 0.01, 0.001]
for l_rate in various_learning_rates:
    model=MLPClassifier(hidden_layer_sizes=(256,32),activation='relu',max_iter=30,batch_size=128,validation_fraction=0.15,early_stopping=True,learning_rate_init=l_rate)
    model.fit(train_images,train_labels)
    training_loss=model.loss_curve_
    validation_score=model.validation_scores_
    #as the validation score is not available in the model, we are calculating it manually

    for i in range(len(validation_score)):
        validation_score[i]=1-validation_score[i]#as the validation score is 1-accuracy
    #validation_loss_curve(func,validation_score)

    #plotting the loss curve and validation loss curve at same time
    plt.plot(training_loss)
    plt.plot(validation_score)
    plt.title('Loss Curve for learning rate '+str(l_rate))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Validation Loss'])
    plt.show()
    
    #printing the accuracy of the model
    print('Accuracy of the model for learning rate '+str(l_rate)+' is '+str(model.score(test_images,test_labels)))
