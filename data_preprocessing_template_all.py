#Data pre processing 

#Importing the libraries
import numpy as np #do the mathematics and work with arrays
import matplotlib.pyplot as plt #draw plots
import pandas as pd #to import dataset

#importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer#Scikit learn = sklearn, Imputer = class which can handle missing data
impute = Imputer(missing_values="NaN", strategy='mean',axis=0)
impute.fit(x[:,1:3])
x[:, 1:3] = impute.transform(x[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder#LabelEncoder = encodes categorical values, something like enum values
encoder_country = LabelEncoder()
x[:,0] = encoder_country.fit_transform(x[:,0])
#Label encoding only creates enum values but this numerical value automatically will give one label higher value than another one. 
#Dummy encoding
hotencoder = OneHotEncoder(categorical_features=[0])
x  = hotencoder.fit_transform(x).toarray()

encoder_final = LabelEncoder()
y = encoder_final.fit_transform(y)

#Splitting the data set into training and test data set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Feature scaling to make the whole data from all columns come into one scale one range. 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#We will not apply feature scaling to the y data because this is a classification problem and it will be in 1/0 format only
