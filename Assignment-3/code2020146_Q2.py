import gzip
import idx2numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

training_loss=[]
validation_loss=[]

class NeuralNetwork:
    #takes a list as input specifying the number of neurons in each layer
    def __init__(self, layers,hidden_layer,learning_rate,activation_function,weight_initialization_function,number_of_epochs,batch_size):
        self.layers = layers
        self.size_of_each_layer = hidden_layer
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.weight_initialization_function = weight_initialization_function
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.weights = {}#dictionary to store weights
        self.biases = {}#dictionary to store biases
        self.help_A={} #dictionary to store help variables
        self.help_Z={}#dictionary to store help variables
        for i in range(1,len(self.size_of_each_layer)):
            wt_shape=(self.size_of_each_layer[i],self.size_of_each_layer[i-1])
            bias_shape=(self.size_of_each_layer[i],1)
            self.biases[i]=(np.zeros(bias_shape))
            if self.weight_initialization_function=='zero':
                self.weights[i]=(self.zero_init(wt_shape))
            elif self.weight_initialization_function=='random':
                self.weights[i]=(self.random_init(wt_shape))
            else:
                self.weights[i]=(self.normal_init(wt_shape))

    def zero_init(self,shape):
        return np.zeros(shape)

    def random_init(self,shape):
        return np.random.randn(*shape)

    def normal_init(self,shape):
        return np.random.normal(loc=0,scale=1,size=shape)
        

    def forward_propagation(self,X):
        self.help_A[0]=X.T
        for i in range(1,self.layers+2):
            #print(self.help_A)
            self.help_Z[i]=np.dot(self.weights[i],self.help_A[i-1])+self.biases[i]#calculating Z
            if i==self.layers+1:
                self.help_A[i]=self.activation_function_classify(self.help_Z[i],derivative=False)#calculating A
            else:
                self.help_A[i]=self.activation_function_classify(self.help_Z[i],derivative=False)#calculating A
        return self.help_A[self.layers+1]
        

    def backward_propagation(self,y,y_hat):
        delta={}
        #self.delta[self.layers+1]=(y_hat-y)*self.activation_function_classify(self.help[self.layers+1],derivative=False)
        change=y_hat-y.T
        for i in range(len(self.size_of_each_layer)-1,0,-1):
            delta[i]=np.dot(change,self.help_A[i-1].T)*1/y.shape[0]#assigning the derivative of the loss function
            delta[i]=np.sum(change,axis=1,keepdims=True)*1/y.shape[0]#summing over all the examples
            if i>1:
                if self.activation_function=='sigmoid':
                    change=np.dot(self.weights[i].T,change)*self.sigmoid_gradient(self.help_Z[i-1])
                elif self.activation_function=='relu':
                    change=np.dot(self.weights[i].T,change)*self.relu_gradient(self.help_Z[i-1])
                elif self.activation_function=='tanh':
                    change=np.dot(self.weights[i].T,change)*self.tanh_gradient(self.help_Z[i-1])
                elif self.activation_function=='linear':
                    change=np.dot(self.weights[i].T,change)*self.linear_gradient(self.help_Z[i-1])
                elif self.activation_function=='softmax':
                    change=np.dot(self.weights[i].T,change)*self.softmax_gradient(self.help_Z[i-1])
                elif self.activation_function=='leaky_relu':
                    change=np.dot(self.weights[i].T,change)*self.leaky_relu_gradient(self.help_Z[i-1])
                else:
                    print('Activation function not defined')
        return delta   

    def activation_function_classify(self,x,derivative=False):
        if self.activation_function=='sigmoid':
            return self.sigmoid(x,derivative)
        elif self.activation_function=='relu':
            return self.relu(x,derivative)
        elif self.activation_function=='tanh':
            return self.tanh(x,derivative)
        elif self.activation_function=='leaky_relu':
            return self.leaky_relu(x,derivative)
        elif self.activation_function=='linear':
            return self.linear(x,derivative)
        else:
            return self.softmax(x,derivative)

    def cross_entropy_loss(self,y,y_hat):
        #calculate cross entropy loss:
        if(self.activation_function=='tanh'):
            loss = np.sum(y.T*np.log(abs(y_hat)+1e-10))/y.shape[0]
        else:
            #print(y)
            #print(y_hat)
            loss = -np.sum(y.T*np.log(y_hat+1e-10))/y.shape[0]#1e-10 is added to avoid log(0)
        return loss

    def fit(self,X_Train,Y_Train,X_Test,Y_Test):
        for i in range(self.number_of_epochs):
            for j in range(0,X_Train.shape[0],self.batch_size):
                X=X_Train[j:j+self.batch_size]
                y=Y_Train[j:j+self.batch_size]
                y_hat=self.forward_propagation(X)
                delta=self.backward_propagation(y,y_hat)
                for k in range(1,self.layers+2):
                    self.weights[k]-=self.learning_rate*delta[k]
                    self.biases[k]-=self.learning_rate*delta[k]
            #if model gets stuck in local minima, reduce learning rate:
            #if i%5==0:
            #    self.learning_rate*=0.9
            
            #print("Training Accuracy for epoch:",i,"is->",self.score(X_Train,np.argmax(Y_Train,axis=-1)))
            #print("Testing Accuracy for epoch:",i,"is->",self.score(X_Test,np.argmax(Y_Test,axis=-1)))
            print("Training Accuracy for epoch:",i,"is->",self.score(training_loss,X_Train,np.argmax(Y_Train,axis=-1)))
            print("Testing Accuracy for epoch:",i,"is->",self.score(validation_loss,X_Test,np.argmax(Y_Test,axis=-1)))
            #printing the loss
            #printing the loss
            training_loss.append(self.cross_entropy_loss(Y_Train,self.forward_propagation(X_Train)))
            print("Training Loss for epoch:",i,"is->",self.cross_entropy_loss(Y_Train,self.forward_propagation(X_Train)))
            validation_loss.append(self.cross_entropy_loss(Y_Test,self.forward_propagation(X_Test)))
            print("Testing Loss for epoch:",i,"is->",self.cross_entropy_loss(Y_Test,self.forward_propagation(X_Test)))
        return self
        
    def predict(self,X):
        #predicts the class of the input
        temp=self.forward_propagation(X)
        return np.argmax(temp,axis=-1)
    
    def score(self,loss_list,X,Y):
        #returns the accuracy of the model
        pred = self.forward_propagation(X)
        result = np.mean(Y == np.argmax(pred.T, axis=-1))
        return result
    
    def predict_proba(self,X):
        #returns the probability of each class
        return self.forward_propagation(X)

    def sigmoid(self,x,derivative=False):
        if derivative:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def sigmoid_gradient(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self,x,derivative=False):
        if derivative:
            return 1*(x>0)
        return x*(x>0)

    def relu_gradient(self,x):
        return 1*(x>0)

    def softmax(self,x,derivative=False):
        if derivative:
            return x*(1-x)
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
    
    def softmax_gradient(self,x):
        return self.softmax(x)*(1-self.softmax(x))
        #return None
    
    def tanh(self,x,derivative=False):
        if derivative:
            return 1-x**2
        return np.tanh(x)
    
    def tanh_gradient(self,x):
        return 1-self.tanh(x)**2

    def leaky_relu(self,x,derivative=False):
        if derivative:
            return 1*(x>0)+0.01*(x<=0)
        return np.maximum(0.01*x,x)

    def leaky_relu_gradient(self,x):
        return x*(np.where(x>0,1,0.01))
    
    def linear(self,x,derivative=False):
        if derivative:
            return 1
        return x
    
    def linear_gradient(self,x):
        return np.ones(x.shape)


image=gzip.open('C://Users//91991//Desktop//ML//data-Q2//train-images-idx3-ubyte.gz','r')
label=gzip.open('C://Users//91991//Desktop//ML//data-Q2//train-labels-idx1-ubyte.gz','r')

dataset_image=idx2numpy.convert_from_file(image)
dataset_label=idx2numpy.convert_from_file(label)

#reshaping the image dataset:
dataset_image=dataset_image.reshape(dataset_image.shape[0],dataset_image.shape[1]*dataset_image.shape[2])

#normalizing the image dataset:
dataset_image=dataset_image/255

#one hot encoding without using sklearn:
dataset_label=dataset_label.reshape(dataset_label.shape[0],1)#reshaping the label dataset
dataset_label=dataset_label.astype(int)#converting the data type to int
dataset_label=np.eye(10)[dataset_label.reshape(-1)]#so as to assign the 10 classes to the 10 columns 



#splitting the dataset into train and test:
X_train,X_test,Y_train,Y_test=train_test_split(dataset_image,dataset_label,test_size=0.2,shuffle=True)

#creating the model:here 784 and 10 are the input and output dimensions respectively
model=NeuralNetwork(layers=4,hidden_layer=[784,256,128,64,32,10],learning_rate=0.01,activation_function='sigmoid',weight_initialization_function='normal',number_of_epochs=50,batch_size=128)

#fitting the model:
model.fit(X_train,Y_train,X_test,Y_test)

#predicting the class:
print(model.predict(X_test))

#predicting the probability:
print(model.predict_proba(X_test))

#calculating the accuracy:
print(model.score(training_loss,X_test,np.argmax(Y_test,axis=1)))

#plotting the loss:
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Reffered from Github, some part of implementation of the class.