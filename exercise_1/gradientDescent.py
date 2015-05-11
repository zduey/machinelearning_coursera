#!/usr/bin/env python
def gradientDescent(X, y, theta, alpha, num_iters):
    """
    
    Performs gradient descent to learn theta by taking n_iters steps and
    updating theta with each step at a learning rate alpha

    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters, 1)

    for i in range(num_iters):
	theta_0 = theta[0]-(alpha/m)*np.sum((np.dot(np.dot(X,theta) - y)*X[0]))
	theta_1 = theta[1]-(alpha/m)*np.sum((np.dot(np.dot(X,theta) - y)*X[1]))		
	theta[0] = theta_0
	theta[1] = theta_1
	
	# Save the cost J in every iteration    
	J_history[i] = computeCost(X, y, theta);

    return theta

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
