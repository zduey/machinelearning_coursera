"""
Machine Learning Online Class - Exercise 2: Logistic Regression
"""

## Initialization
import pandas as pd
import numpy as np
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

plotData(X, y)

input('\nProgram paused. Press enter to continue.\n');

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term

# Add intercept term to x and X_test
X = np.hstack((np.ones_like(y),X))

# Initialize fitting parameters
initial_theta = np.zeros(3)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): \n', cost)
print('Gradient at initial theta (zeros): \n')
print('\n', grad)

input('\nProgram paused. Press enter to continue.\n')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

# Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

# Put some labels 
hold on;
# Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

# Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

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

prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

$ Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: #f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

