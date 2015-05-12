# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. You will need to complete the following 
#  functions in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

# Initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent


## ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

print(warmUpExercise()) # Prints object returned by WarmUpExercise 
input('Program paused. Press enter to continue.\n')


## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None]
y = np.array(data.y)[:,None]
m = len(y) # number of training examples

# Plot Data
fig = plotData(x,y)
fig.show()

input('Program paused. Press enter to continue.\n');

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

ones = np.ones_like(x)
X = np.hstack((ones,x)) # Add a column of ones to x
theta = np.zeros((2,1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print('Theta found by gradient descent: ');
print(theta[0], theta[1]);

"""
# Plot the linear fit
hold on; # keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off # don't overlay any more plots on this figure

# Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
print('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
print('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

print('Program paused. Press enter to continue.\n');
pause;i
"""





