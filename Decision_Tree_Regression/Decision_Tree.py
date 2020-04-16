#Decision Tree

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#1:2 because we want X to always be matrix not vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the decision tree Regression to the dataset"
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#predicting results
y_pred = regressor.predict([[6.5]])

#Visualising Decision Tree Regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color= 'red')
plt.plot(X, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()