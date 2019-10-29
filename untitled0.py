#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("kc_house_data.csv")
X = dataset.iloc[:, 3:].values
Y = dataset.iloc[:, 2].values

#split the dataset into trainining set and test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
HZ = LinearRegression()
model = HZ.fit(Xtrain, Ytrain)

#predicting the test set results
Ypred =  HZ.predict(Xtest)

#Visualizing the training set results
plt.scatter(Ytest, Ypred, color ="green")
plt.plot(Xtrain, HZ.predict(Xtrain), color ="red")
plt.title('True value vs Predicted value')
plt.xlabel("Price") 
plt.ylabel("Predicted value") 
plt.show()

