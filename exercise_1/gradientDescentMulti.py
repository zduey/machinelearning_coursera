#!/usr/bin/env python
import numpy as np
from computeCost import computeCost

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

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
