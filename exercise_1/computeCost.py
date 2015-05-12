import numpy as np

def computeCost(X, y, theta):
    """
    
    Compute cost for linear regression using theta as the
    parameter vector for linear regression to fit the 
    data points in X and y

    """
    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)

    return J
