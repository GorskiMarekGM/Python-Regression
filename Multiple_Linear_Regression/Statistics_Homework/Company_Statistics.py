#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('30_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,5].values

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#STEP 1 Select a significance level to stay in the model S:=0.05
#We need to add 1 column of 1 for X0, y = B0 * X0+ b1 * X1 +..., where x0 is always 1
# np.ones(30, 1) - size of matrix of ones
#astype(int) - converting to int type
# we need to change order so we will add matrix at the beggining
#X = np.append(arr = X, values = np.ones(30, 1).astype(int), axis = 1)

X = np.append(arr = np.ones((30, 1)).astype(int), values = X , axis = 1)

#STEP 2 Fit the full model with all possible predictors
X_opt = X[:, [0,1,2,3,4]]

#arguments endog - dependent variable, exog - array with observations
#fiting
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#STEP 3 Consider the predictor with the highest P value. If P> SL go to STEP 4, otherwise go to FIN
#important info about regressor model
regressor_OLS.summary()

#STEP 4 Remove predictor with the highest R^2
X_opt = X[:, [0,2,3,4]]

#STEP 5 Fit model without this variable
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
