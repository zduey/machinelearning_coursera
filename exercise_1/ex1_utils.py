#!/usr/bin/env python

# Contains all the helper functions necessary to run ex1.py and ex1_multi.py

# Imports
import numpy as np
import matplotlib.pyplot as plt

def warmUpExercise():
    """
    Example Function in Python
    Instructions: Return the 5x5 identity matrix. In Python,
                  define the object to be returned within the
		  function, and then return it at the bottom
		  with the "return" statement.
    """
    return np.eye(5)


def plotData(x, y):
    """
    plotData -- Plots the data points x and y into a new figure and gives
                the figure axes labels of population and profit. It returns
		at matplotlib figure.

     Instructions: Plot the training data into a figure by manipulating the
		   axes object created for you below. Set the axes labels using
                   the "xlabel" and "ylabel" commands. Assume the 
                   population and revenue data have been passed in
                   as the x and y arguments of this function.
    
     Hint: You can use the 'rx' option with plot to have the markers
           appear as red crosses. Furthermore, you can make the
           markers larger by using plot(..., 'rx', markersize=10)
    """
    
    fig, ax = plt.subplots() # create empty figure and set of axes
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")

    return fig


def normalEqn(X,y):
    """
    Computes the closed form least squares solution using normal
    equations

    theta = (X^T*X)^{-1}X^T*y
    Returns: Array of least-squares parameters
    """
    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    
    Performs gradient descent to learn theta by taking n_iters steps and
    updating theta with each step at a learning rate alpha

    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)

	# Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)
        print('Cost function has a value of: ', J_history[i])
    
    return (theta,J_History)


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    
    Performs gradient descent to learn theta by taking n_iters steps and
    updating theta with each step at a learning rate alpha

    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)

	# Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)
        print('Cost function as a value of: ',J_history[i])
    
    return (theta, J_history)


def featureNormalize(X):
    """
    Normalizes (mean=0, std=1) the features in design matrix X

    returns -- Normalized version of X where the mean of each
               value of each feature is 0 and the standard deviation
	       is 1. This will often help gradient descent learning
	       algorithms to converge more quickly.

    Instructions: First, for each feature dimension, compute the mean
                  of the feature and subtract it from the dataset,
		  storing the mean value in mu. Next, compute the 
		  standard deviation of each feature and divide
		  each feature by it's standard deviation, storing
		  the standard deviation in sigma. 
		  
		  Note that X is a matrix where each column is a 
		  feature and each row is an example. You need 
		  to perform the normalization separately for 
		  each feature. 
		  
		  Hint: You might find the 'mean' and 'std' functions useful.

    """
    
    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))


def computeCost(X, y, theta):
    """
    
    Compute cost using sum of square errors for linear 
    regression using theta as the parameter vector for 
    linear regression to fit the data points in X and y.

    Note: Requires numpy in order to run, but is not imported as part of this
	  script since it is imported in ex1.py and therefore numpy is part of
	  the namespace when the function is actually run.
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)

    return J

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
