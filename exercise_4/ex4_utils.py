# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import multiprocessing as mp

def displayData(X):
    """
    Displays 2D data stored in design matrix in a nice grid.
    """
    fig, ax = plt.subplots(10,10,sharex=True,sharey=True)
    img_num = 0
    for i in range(10):
        for j in range(10):
            # Convert column vector into 20x20 pixel matrix
            # You have to transpose to have them display correctly
            img = X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            img_num += 1

    return (fig, ax)

def displayImage(im):
    """
    Displays a single image stored as a column vector
    """
    fig2, ax2 = plt.subplots()
    image = im.reshape(20,20).T
    ax2.imshow(image,cmap='gray')
    
    return (fig2, ax2)

def sigmoid(z):
    """ Returns the value from using x in the sigmoid function """
    
    return 1.0/(1 +  np.e**(-z))

def predict(theta1,theta2,X):
    m = len(X) # number of samples

    if np.ndim(X) == 1:
        X = X.reshape((-1,1))
    
    D1 = np.hstack((np.ones((m,1)),X))# add column of ones
   
    # Calculate hidden layer from theta1 parameters
    hidden_pred = np.dot(D1,theta1.T) # (5000 x 401) x (401 x 25) = 5000 x 25
    
    # Add column of ones to new design matrix
    ones = np.ones((len(hidden_pred),1)) # 5000 x 1
    hidden_pred = sigmoid(hidden_pred)
    hidden_pred = np.hstack((ones,hidden_pred)) # 5000 x 26
    
    # Calculate output layer from new design matrix
    output_pred = np.dot(hidden_pred,theta2.T) # (5000 x 26) x (26 x 10)    
    output_pred = sigmoid(output_pred)
    # Get predictions
    p = np.argmax(output_pred,axis=1)
    
    return p

def nnCostFunction(nn_params,
	    	   input_layer_size,
		   hidden_layer_size,
		   num_labels,X,y,reg_param):
    """
    Computes loss using sum of square errors for a neural network
    using theta as the parameter vector for linear regression to fit 
    the data points in X and y with penalty reg_param.
    """

    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Reshape nn_params back into neural network
    theta1 = nn_params[:(hidden_layer_size * 
			 (input_layer_size + 1))].reshape((hidden_layer_size, 
							  input_layer_size + 1))
  
    theta2 = nn_params[-((hidden_layer_size + 1) * 
			 num_labels):].reshape((num_labels,
					        hidden_layer_size + 1))
   
    # Turn scalar y values into a matrix of binary outcomes
    init_y = np.zeros((len(y),num_labels)) # 5000 x 10
 
    for i in range(m):
        init_y[i][y[i]] = 1

    # Add column of ones to X
    ones = np.ones((m,1)) 
    d = np.hstack((ones,X))# add column of ones
 
    # Compute cost by doing feedforward propogation with theta1 and theta2
    cost = [0]*m 
    for i in range(m):
	# Feed Forward Propogation
        a1 = d[i][:,None] # 401 x 1
        z2 = np.dot(theta1,a1) # 25 x 1 
        a2 = sigmoid(z2) # 25 x 1
        a2 = np.vstack((np.ones(1),a2)) # 26 x 1
        z3 = np.dot(theta2,a2) #10 x 1
        h = sigmoid(z3) # 10 x 1

	# Calculate cost
        cost[i] = (np.sum((-init_y[i][:,None])*(np.log(h)) -
	          (1-init_y[i][:,None])*(np.log(1-h))))/m

    # Add regularization
    reg = (reg_param/(2*m))*((np.sum(theta1[:,1:]**2)) + 
	  (np.sum(theta2[:,1:]**2)))
    
    final_cost = sum(cost) + reg
    """
    # Gradient
    
    # Non-regularized 
    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    
    # Regularized
    grad_reg = grad_0 + (reg_param/m)*theta

    # Replace gradient for theta_0 with non-regularized gradient
    grad_reg[0] = grad_0[0] 
    
    # Don't bother with the gradient. Let scipy compute numerical
    # derivatives for you instead
    """

    return final_cost
