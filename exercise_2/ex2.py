"""
Machine Learning Online Class - Exercise 2: Logistic Regression
"""

## Initialization
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex2_utils import *


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('ex2data1.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print("Plotting data with + indicating (y = 1) examples and o indicating",
" (y =0) examples.")

fig, ax = plotData(X, y)

# Specified in plot order
ax.legend(['Admitted', 'Not admitted'])


fig.show()

input('\nProgram paused. Press enter to continue.\n')

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term

# Add intercept term to x and X_test
X = np.hstack((np.ones_like(y)[:,None],X))

# Initialize fitting parameters
initial_theta = np.zeros(3)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): \n', cost)
print('Gradient at initial theta (zeros): \n',grad)

input('\nProgram paused. Press enter to continue.')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

res = minimize(costFunction,
	       initial_theta,
	       method='Newton-CG',
	       args=(X,y),
	       jac=True, 
	       options={'maxiter':400,
			'disp':True})

theta = res.x

# Print theta to screen
print('Cost at theta found by minimize: \n', res.fun)
print('theta: \n', theta)


# Plot Boundary
plotDecisionBoundary(theta, X, y)


input('\nProgram paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.dot([1,45,85],theta))
print('For a student with scores 45 and 85, we predict an ',
      'admission probability of ', prob)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: \n', np.mean(p==y)*100)

input('Program paused. Press enter to continue.\n')

