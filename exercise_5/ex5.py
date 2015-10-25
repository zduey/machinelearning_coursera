import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex5_utils import *
import scipy.io
import matplotlib.pyplot as plt

# Part 1 -- Loading and visualizing data
raw_mat = scipy.io.loadmat("ex5data1.mat")
X = raw_mat.get("X")
y = raw_mat.get("y")
ytest = raw_mat.get("ytest")
yval = raw_mat.get("yval")
Xtest = raw_mat.get("Xtest")
Xval = raw_mat.get("Xval")


plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

# Part 2 -- Regularized Linear Regression Cost
full_X = np.hstack((np.ones_like(y), X))
theta = np.array([1,1])
J, g = linearRegCostFunction(theta,full_X,y,0.0)


# Part 3 -- Reguliarized Linear Regression Gradient
J, g = linearRegCostFunction(theta,full_X,y,1.0)

# Part 4 -- Train Linear Regression
reg_param = 0
est_theta = trainLinearReg(full_X,y,reg_param)

# Plot linear fit based on estimated parameters
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.plot(X,np.dot(full_X,est_theta),'b-',linewidth=2)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

# Part 5 -- Learning Curve for Linear Regression
reg_param = 0.0
full_Xval = np.hstack((np.ones_like(yval),Xval))
error_train, error_val = learningCurve(full_X,y,full_Xval,yval,reg_param)

plt.plot(range(len(X)), error_train, range(len(X)), error_val);
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()

# Part 6 -- Feature Mapping for Polynomial Regression
p = 8
X_poly = polyFeatures(X,p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.hstack((np.ones_like(y),X_poly))

X_poly_test = polyFeatures(Xtest,p)
X_poly_test = np.divide(X_poly_test - mu, sigma)
X_poly_test = np.hstack((np.ones_like(ytest),X_poly_test))

X_poly_val = polyFeatures(Xval,p)
X_poly_val = np.divide(X_poly_val - mu, sigma)
X_poly_val = np.hstack((np.ones_like(yval),X_poly_val))

# Part 7 -- Learning Curve for Polynomial Regression
reg_param = 1.0
est_theta = trainLinearReg(X_poly,y,reg_param)
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(np.min(X), np.max(X), mu, sigma, est_theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

error_train, error_val = learningCurve(X_poly,y,X_poly_val,yval,reg_param)

plt.plot(range(len(X)), error_train, range(len(X)), error_val);
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()

# Part 8 -- Validation for selecting regularization parameter
lambda_vec, error_train, error_val = validationCurve(full_X,y,full_Xval,yval)

plt.plot(lambda_vec, error_train, lambda_vec, error_val);
plt.title('Selecting \lambda using a cross validation set')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()