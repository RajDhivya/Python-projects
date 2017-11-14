#1. Data pre processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3: -1].values
y = dataset.iloc[:, -1].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
x[:, 1] = labelencoder_country.fit_transform(x[:, 1])

labelencoder_gender = LabelEncoder()
x[:, 2] = labelencoder_gender.fit_transform(x[:, 2])

#creating the dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

#to remove one extra column of the dummy variable
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling to bring them all to the same range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#2. Building the ANN
#Importing Keras
import keras
from keras.models import Sequential #to initialise the ANN
from keras.layers import Dense #to create the layers in neural networks

#Initialising the ANN i.e. defining it as a sequence of layers
classifier = Sequential()

#Defining the input layer and the first hidden layer
#Tip - the number of nodes in the hidden layer = average of (nodes in input layer + nodes in output layer)
#Rectifier function used as activation function in hidden layers = relu
#Sigmoid function used as activation function in output layer
#input_dim is needed only for the first hidden layer to define the number of input nodes
classifier.add(Dense(units=6,init = 'uniform', activation='relu', input_dim = 11))

#Adding the second hidden layer
#input_dim is not needed since this is the second layer and the first layer already has 6 output spaces
classifier.add(Dense(units=6,init = 'uniform', activation='relu'))

#Adding the final output layer
classifier.add(Dense(units=1,init = 'uniform', activation='sigmoid'))

#compiling the ANN
#Stochastic GD = adam
#logarithmic loss used SGD = binary_crossentropy(for binary outcome)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
#a batch learning rather than reinforcement learning. so updates the weights after 10 entry rows are applied
#runs the whole set of input data 100 times
classifier.fit(x_train, y_train,batch_size=10,epochs=100)

#3. making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)

#the y_pred generated above is in terms of probability values
#converting the probabilities in y_pred to actual values of 0 and 1
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)