# Machine Learning Coursera -- Exercise 4

## Initialization
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex4_utils import *
import scipy.io


# Set up parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# Load training data
print("Loading training data...")

raw_mat = scipy.io.loadmat("ex4data1.mat")
X = raw_mat.get("X")
y = raw_mat.get("y").flatten()
y = (y - 1) % 10 # hack way of fixing conversion MATLAB 1-indexing

# Randomly select 100 datapoints to display
rand_indices = np.random.randint(0,len(X),100)
sel = X[rand_indices,:] 

# Display the data
digit_grid, ax = displayData(sel)
digit_grid.show()
input("Program paused, press enter to continue...")

print("Loading neural network parameters \n")

raw_params = scipy.io.loadmat("ex4weights.mat")
theta1 = raw_params.get("Theta1") # 25 x 401
theta2 = raw_params.get("Theta2") # 10 x 26

# Unroll Parameters
nn_params = np.append(theta1,theta2).reshape(-1)

# Compute Unregularized Cost
print("Checking cost function without regularization...")
reg_param = 0.0
g,cost = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
		                    X,y,reg_param)

# Test for correct cost
np.testing.assert_almost_equal(0.287629,cost,decimal=6, err_msg="Cost incorrect.")

input("Program paused, press enter to continue...")

# Compute Regularized Cost
print("Checking cost function with regularization...")
reg_param = 1.0
g,reg_cost = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
		            X,y,reg_param)
np.testing.assert_almost_equal(0.383770,reg_cost,decimal=6, err_msg="Regularized Cost incorrect.")
input("Program paused, press enter to continue...")

# Checking sigmoid gradient
print("Checking sigmoid gradient...")

vals = np.array([1,-0.5,0,0.5,1])
g = sigmoidGradient(vals)
np.testing.assert_almost_equal(0.25, g[2],decimal=2.err_msg="Sigmoid function incorrect")

# Initialize neural network parameters
print("Initializing neural network parameters...")
initial_theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size,num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_theta1,initial_theta2).reshape(-1)

reg_param = 0.0
g,initial_cost = nnCostFunction(initial_nn_params,input_layer_size,
                                    hidden_layer_size,num_labels,X,y,reg_param)

print("The initial cost after random initialization: ", initial_cost)

# Implement Backpropogation
# Check gradients
checkNNGradients(0.0)
input("Program paused, press enter to continue...")

# Implement Regularization
reg_param = 3.0
checkNNGradients(reg_param)

# Debugging value of the cost function
reg_param = 1.0
debug_J = nnCostFunction(initial_nn_params,input_layer_size,
                         hidden_layer_size,num_labels,X,y,reg_param)

print("Cost at fixed debugging parameters with lambda = 10 is: ", debug_J[1])


# Train NN Parameters
# Compute Numerical Gradient
def costfunc(p):
    grad, cost = nnCostFunction(p,input_layer_size,hidden_layer_size,
				num_labels,X,y,reg_param)
    return grad, cost

reg_param = 0.0
results = minimize(costfunc,
		   initial_nn_params,
                   jac=True,
		   tol=1e-6,
                   options={'maxiter':50})


