"""
  Machine Learning Online Class
  Exercise 1: Linear regression with multiple variables

  Instructions
  ------------

  This file contains code that helps you get started on the
  linear regression exercise. 

  You will need to complete the following functions in this 
  exericse:

     warmUpExercise.py
     plotData.py
     gradientDescent.py
     computeCost.py
     gradientDescentMulti.py
     computeCostMulti.py
     featureNormalize.py
     normalEqn.py

  For this part of the exercise, you will need to change some
  parts of the code below for various experiments (e.g., changing
  learning rates).
"""

## Initialization

## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

## Load Data
print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)
y = np.array(data.y)
m = length(y) # number of training examples


# Print out some data points
print('First 10 examples from the dataset: \n')
print(' x = [%.0f %.0f], y = %.0f \n', X[10] y[10])

input('Program paused. Press enter to continue.\n')
pause

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

[X mu sigma] = featureNormalize(X)

# Add intercept term to X
X = [ones(m, 1) X]


## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = zeros(3, 1)
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
figure
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2)
xlabel('Number of iterations')
ylabel('Cost J')

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(' %f \n', theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = 0 % You should change this


# ============================================================

print(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price)

print('Program paused. Press enter to continue.\n')
pause

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
data = csvread('ex1data2.txt')
X = data(:, 1:2)
y = data(:, 3)
m = length(y)

# Add intercept term to X
X = [ones(m, 1) X]

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(' %f \n', theta)
print('\n')


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = 0 # You should change this


# ============================================================

print(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price)

