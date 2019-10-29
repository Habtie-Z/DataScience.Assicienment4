#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("kc_house_data.csv")
X = dataset.iloc[:, 3:].values
y = dataset.iloc[:, 2].values

#split the dataset into trainining set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.fit_transform(X_test) 



#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
model = LR.fit(X_train, y_train)

#predicting the test set results
y_pred =  LR.predict(X_test)

#Visualizing the training set results
plt.scatter(y_test, y_pred, color ="green")
plt.plot(X_train, LR.predict(X_train), color ="red")
plt.title('True value vs Predicted value', fontsize=18)
plt.xlabel("True value", fontsize=18) 
plt.ylabel("Predicted value",fontsize=14) 
print(plt.show())

#alpha and betas of multiple linear regression equation
print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')

