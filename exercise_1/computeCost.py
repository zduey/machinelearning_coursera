#!/usr/bin/env python

import numpy as np

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
